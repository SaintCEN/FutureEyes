import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import cv2, os, glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------- 自定义数据集类 ----------------------
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=(320, 320), preprocess=False, tol=7):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.bmp")))
        self.img_size = img_size
        self.preprocess = preprocess
        self.tol = tol  # 灰度裁剪阈值

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 读取图像
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        # 裁剪黑色边框
        image = self.crop_image_from_gray(image, self.tol)
        mask = self.crop_image_from_gray(mask, self.tol)

        # 预处理（可选）
        if self.preprocess:
            image = self._clahe_preprocess(image)

        # 调整尺寸
        image = cv2.resize(image, self.img_size)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        # 转换掩码为三分类（0-视盘，1-视杯，2-背景）
        mask = self._priority_mapping(mask)

        # 归一化并转为Tensor
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        mask = torch.from_numpy(mask).long()
        return image, mask

    def crop_image_from_gray(self, img, tol=7):
        if img.ndim == 2:  # 灰度图
            mask = img > tol
            return img[np.ix_(mask.any(1), mask.any(0))]
        elif img.ndim == 3:  # RGB图
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            mask = gray_img > tol
            check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
            if check_shape == 0:
                return img
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            return np.stack([img1, img2, img3], axis=-1)

    def _clahe_preprocess(self, image):
        """对比度受限的自适应直方图均衡化"""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def _priority_mapping(self, mask):
        """处理优先级覆盖关系：视盘(黑) > 视杯(灰) > 背景(白)"""
        priority_mask = np.zeros_like(mask)
        priority_mask[mask <= 50] = 0  # 视盘 (0-50)
        priority_mask[(mask > 50) & (mask < 200)] = 1  # 视杯 (50-200)
        priority_mask[mask >= 200] = 2  # 背景 (200-255)
        return priority_mask


# ---------------------- 模型架构 ----------------------
class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        att = self.gap(x).view(b, c)
        att = self.fc(att).view(b, c, 1, 1)
        return x * att.expand_as(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_att=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.att = AttentionBlock(out_channels) if use_att else None

    def forward(self, x):
        x = self.conv(x)
        if self.att is not None:
            x = self.att(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=16, depth=4, use_att=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.skip_channels = []

        for i in range(depth):
            out_channels = base_channels * (2 ** i)
            self.blocks.append(ConvBlock(in_channels, out_channels, use_att))
            self.pools.append(nn.MaxPool2d(2))
            self.skip_channels.append(out_channels)
            in_channels = out_channels

    def forward(self, x):
        skip_connections = []
        for block, pool in zip(self.blocks, self.pools):
            x = block(x)
            skip_connections.append(x)
            x = pool(x)
        return x, skip_connections


class Decoder(nn.Module):
    def __init__(self, in_channels, skip_channels, depth=4, use_att=True):
        super().__init__()
        self.upsamples = nn.ModuleList()
        self.blocks = nn.ModuleList()

        for i in reversed(range(depth)):
            out_channels = skip_channels[i]
            self.upsamples.append(nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2))
            self.blocks.append(ConvBlock(out_channels * 2, out_channels, use_att))
            in_channels = out_channels

    def forward(self, x, skip_connections):
        for upsample, block, skip in zip(self.upsamples, self.blocks, reversed(skip_connections)):
            x = upsample(x)
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        return x


class LWBNA_UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, base_channels=16, depth=4):
        super().__init__()
        self.encoder = Encoder(in_channels, base_channels, depth)
        self.mid = ConvBlock(self.encoder.skip_channels[-1], self.encoder.skip_channels[-1], use_att=True)
        self.decoder = Decoder(self.encoder.skip_channels[-1], self.encoder.skip_channels, depth)
        self.final_conv = nn.Conv2d(base_channels, num_classes, 1)

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.mid(x)
        x = self.decoder(x, skips)
        return self.final_conv(x)

# ---------------------- 损失函数 ----------------------
class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # 转换target为one-hot编码
        target_onehot = F.one_hot(target, num_classes=3).permute(0, 3, 1, 2).float()

        # 计算Dice系数
        intersection = (pred * target_onehot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()

        # 计算交叉熵损失
        ce_loss = F.cross_entropy(pred, target)
        return dice_loss + ce_loss

# ---------------------- 训练函数 ----------------------
def train_model(
        train_dir,
        mask_dir,
        img_size=(320, 320),
        batch_size=4,
        epochs=500,
        lr=1e-4,
        patience=40,
        save_path="best_model.pth",
        device=None
):
    # 自动检测设备
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化数据集
    full_dataset = SegmentationDataset(train_dir, mask_dir, img_size)

    # 数据集分割
    train_indices, val_indices = train_test_split(
            range(len(full_dataset)),
            test_size=0.2,
            shuffle=True,
            random_state=42
        )
    train_data = torch.utils.data.Subset(full_dataset, train_indices)
    val_data = torch.utils.data.Subset(full_dataset, val_indices)

    # 数据加载器
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True
    )

    # 初始化模型
    model = LWBNA_UNet(num_classes=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    criterion = DiceBCELoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience // 2, verbose=True)

    # 训练记录
    best_loss = float('inf')
    train_history = {'train_loss': [], 'val_loss': []}
    no_improve = 0

    # 训练循环
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        # 训练阶段（带进度条）
        with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch + 1}/{epochs}") as tepoch:
            for images, masks in tepoch:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                # 更新进度条显示
                tepoch.set_postfix(loss=loss.item())

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                outputs = model(images)
                val_loss += criterion(outputs, masks).item()

        # 计算平均损失
        train_loss = epoch_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        train_history['train_loss'].append(train_loss)
        train_history['val_loss'].append(val_loss)

        # 打印epoch结果
        print(
            f"\nEpoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # 早停与模型保存
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)
            no_improve = 0
            print(f"Validation loss improved. Model saved to {save_path}")
        else:
            no_improve += 1
            print(f"No improvement for {no_improve}/{patience} epochs")

        if no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

        # 学习率调整
        scheduler.step(val_loss)

    # 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_history['train_loss'], label='Train Loss')
    plt.plot(train_history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid()
    plt.show()
    return model


# ---------------------- 使用示例 ----------------------
if __name__ == "__main__":
    # 参数设置
    config = {
        'train_dir': 'C:/Users/SaintCHEN/Desktop/FutureEyes/dataset/Refuge/Train/Original/All',
        'mask_dir': 'C:/Users/SaintCHEN/Desktop/FutureEyes/dataset/Refuge/Train/Disc_Cup_Masks/All',
        'img_size': (320, 320),
        'batch_size': 4,
        'epochs': 500,
        'patience': 40,
        'save_path': 'C:/Users/SaintCHEN/Desktop/FutureEyes/models/LWBNA_unet_G.pth'
    }
    # 开始训练
    model = train_model(**config)