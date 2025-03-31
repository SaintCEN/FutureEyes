import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms, models
from tqdm import tqdm
import pandas as pd
import cv2
import numpy as np
from PIL import Image
import random

# -----------------------------
# 图像预处理与增强
# -----------------------------
def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img

def load_ben_color(image, sigmaX=10):
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (300, 300))
    return cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)

class PairTransform:

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, left, right):
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        left = self.transform(left)
        random.seed(seed)
        torch.manual_seed(seed)
        right = self.transform(right)
        return left, right

# 训练数据增强
train_transform = PairTransform(transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]))

# 验证/测试数据增强（无随机变换）
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -----------------------------
# 数据集类
# -----------------------------
class GDataset(Dataset):
    def __init__(self, data_pairs, transform=None, is_train=True):
        self.data_pairs = data_pairs
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        if self.is_train:
            left_path, right_path, label = self.data_pairs[idx]
            label = torch.tensor(label, dtype=torch.float32)
        else:
            left_path, right_path, img_id = self.data_pairs[idx]

        left_image = cv2.cvtColor(cv2.imread(left_path), cv2.COLOR_BGR2RGB)
        right_image = cv2.cvtColor(cv2.imread(right_path), cv2.COLOR_BGR2RGB)

        left_image = load_ben_color(left_image)
        right_image = load_ben_color(right_image)

        left_image = Image.fromarray(left_image)
        right_image = Image.fromarray(right_image)

        if self.transform:
            if isinstance(self.transform, PairTransform):
                left_image, right_image = self.transform(left_image, right_image)
            else:
                left_image = self.transform(left_image)
                right_image = self.transform(right_image)

        if self.is_train:
            return (left_image, right_image), label
        else:
            return (left_image, right_image), img_id

# -----------------------------
# 模型定义（双分支ResNet50）
# -----------------------------
class GResNet50(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.left_model = models.resnet50(pretrained=pretrained)
        self.right_model = models.resnet50(pretrained=pretrained)
        num_ftrs = self.left_model.fc.in_features
        self.left_model.fc = nn.Identity()
        self.right_model.fc = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(num_ftrs * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
    def forward(self, left, right):
        left_feat = self.left_model(left)
        right_feat = self.right_model(right)
        combined = torch.cat([left_feat, right_feat], dim=1)
        return self.fc(combined)

# -----------------------------
# 训练与验证
# -----------------------------
def load_train_val_data(root_dir, val_ratio=0.2):
    all_data = []
    for label_name in ['G', 'Normal']:
        label = 1 if label_name == 'G' else 0
        label_dir = os.path.join(root_dir, label_name)
        files = os.listdir(label_dir)
        left_images = {f.split('_')[0]: f for f in files if 'left' in f.lower()}
        right_images = {f.split('_')[0]: f for f in files if 'right' in f.lower()}
        for img_id in left_images:
            if img_id in right_images:
                left_path = os.path.join(label_dir, left_images[img_id])
                right_path = os.path.join(label_dir, right_images[img_id])
                all_data.append((left_path, right_path, label))

    train_data, val_data = train_test_split(
        all_data,
        test_size=val_ratio,
        random_state=42,
        stratify=[d[2] for d in all_data]
    )
    return train_data, val_data

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for (left, right), labels in loader:
            left, right, labels = left.to(device), right.to(device), labels.to(device)
            outputs = model(left, right).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total

def train_model(model, train_loader, val_loader, epochs, device):
    # 计算类别权重
    train_labels = [d[2] for d in train_data]
    pos_weight = torch.tensor([(len(train_labels) - sum(train_labels)) / sum(train_labels)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1)
    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for (left, right), labels in pbar:
            left, right, labels = left.to(device), right.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(left, right).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix({'Loss': loss.item()})
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        print(f"Epoch {epoch + 1}: "
              f"Train Loss: {train_loss / len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), '/web/models/best_model_G.pth')
            print(f"*** 保存最佳模型，验证集准确率: {best_val_acc:.4f} ***")
# -----------------------------
# 预测部分
# -----------------------------
class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_pairs = []

        files = os.listdir(root_dir)

        def get_img_id(filename):
            parts = filename.rsplit('_', 1)
            if len(parts) == 2:
                return parts[0].split('.')[0]
            return filename.split('.')[0]

        left_files = {get_img_id(f): f for f in files if 'left' in f.lower()}
        right_files = {get_img_id(f): f for f in files if 'right' in f.lower()}

        common_ids = set(left_files.keys()) & set(right_files.keys())
        for img_id in common_ids:
            left_path = os.path.join(root_dir, left_files[img_id])
            right_path = os.path.join(root_dir, right_files[img_id])
            self.image_pairs.append((left_path, right_path, img_id))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        left_path, right_path, img_id = self.image_pairs[idx]

        left_img = cv2.cvtColor(cv2.imread(left_path), cv2.COLOR_BGR2RGB)
        right_img = cv2.cvtColor(cv2.imread(right_path), cv2.COLOR_BGR2RGB)

        left_img = load_ben_color(left_img)
        right_img = load_ben_color(right_img)

        left_img = Image.fromarray(left_img)
        right_img = Image.fromarray(right_img)

        if self.transform:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)

        return (left_img, right_img), img_id


def predict(model_path='C:/Users/SaintCHEN/Desktop/FutureEyes/models/best_model_G.pth',
            test_dir='C:/Users/SaintCHEN/Desktop/FutureEyes/dataset/Test_All',
            output_csv='C:/Users/SaintCHEN/Desktop/FutureEyes/outputs/Saint_ODIR.csv'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GResNet50(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    df = pd.read_csv(output_csv)
    required_ids = set(df['ID'].astype(str))

    test_dataset = TestDataset(root_dir=test_dir, transform=test_transform)
    filtered_pairs = [
        (left, right, img_id)
        for (left, right, img_id) in test_dataset.image_pairs
        if str(img_id) in required_ids
    ]
    test_dataset.image_pairs = filtered_pairs

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    predictions = {}
    with torch.no_grad():
        for (left, right), img_ids in tqdm(test_loader, desc="Predicting"):
            left, right = left.to(device), right.to(device)
            outputs = model(left, right).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()

            for idx, img_id in enumerate(img_ids):
                predictions[int(img_id)] = float(probs[idx])

    # 将所有ID对应的预测概率填入G列
    df['G'] = df['ID'].map(predictions)
    df.to_csv(output_csv, index=False)


# -----------------------------
# 主程序
# -----------------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if True:
        # 加载数据
        train_data, val_data = load_train_val_data('C:/Users/SaintCHEN/Desktop/FutureEyes/dataset/Train_G', val_ratio=0.2)
        train_dataset = GDataset(train_data, transform=train_transform)
        val_dataset = GDataset(val_data, transform=test_transform)  # 验证集不使用数据增强
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
        # 初始化模型
        model = GResNet50(pretrained=True).to(device)
        # 训练
        train_model(model, train_loader, val_loader, epochs=20, device=device)
    # 预测
    predict()