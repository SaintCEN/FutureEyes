import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from segregation_model import LWBNA_UNet, DiceLoss

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

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, gt_dir, img_type='.jpg', gt_type='.bmp', transform=None, preprocess=False,
                     crop_black=True):
            self.image_paths = sorted(glob.glob(os.path.join(image_dir, f"*{img_type}")))
            self.gt_paths = sorted(glob.glob(os.path.join(gt_dir, f"*{gt_type}")))
            self.transform = transform
            self.preprocess = preprocess
            self.crop_black = crop_black
            self.target_size = (320, 320)

    def preprocess_image(self, image):
            if self.crop_black:
                image = crop_image_from_gray(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(image)
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            image = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
            return image

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)
        gt = cv2.imread(self.gt_paths[idx], cv2.IMREAD_GRAYSCALE)

        if self.crop_black:
            gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            mask = gray_img > 7
            crop_coords = np.ix_(mask.any(1), mask.any(0))
            image = image[crop_coords]
            gt = gt[crop_coords[0], crop_coords[1]]

        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        gt = cv2.resize(gt, self.target_size, interpolation=cv2.INTER_NEAREST)

        if self.preprocess:
            image = self.preprocess_image(image)
            gt_mask = np.zeros((*gt.shape, 3), dtype=np.float32)
            gt_mask[..., 0] = (gt == 0).astype(np.float32)  # Optic disc
            gt_mask[..., 1] = (gt == 128).astype(np.float32)  # Optic cup
            gt_mask[..., 2] = (gt == 255).astype(np.float32)  # Background

            if self.transform:
                image = self.transform(image)
                gt_mask = self.transform(gt_mask)
            else:
                image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
                gt_mask = torch.from_numpy(gt_mask).float().permute(2, 0, 1)

            return image, gt_mask
        else:
            if self.transform:
                image = self.transform(image)
                gt = torch.from_numpy(gt).float().unsqueeze(0)
            else:
                image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
                gt = torch.from_numpy(gt).float().unsqueeze(0)
            return image, gt


def train_model():
    torch.backends.cudnn.benchmark = True
    batch_size = 4
    epochs = 500
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 早停参数
    patience = 15
    best_val_loss = float('inf')
    no_improve_epochs = 0
    early_stop = False

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = SegmentationDataset(
        image_dir='../dataset/Refuge/Training400/Original/train-all',
        gt_dir='../dataset/Refuge/Training400/Disc_Cup_Masks/train-masks-all',
        transform=transform,
        preprocess=True,
        crop_black=True
    )
    val_dataset = SegmentationDataset(
        image_dir='../dataset/Refuge/REFUGE-Validation400',
        gt_dir='../dataset/Refuge/REFUGE-Validation400-GT/Disc_Cup_Masks',
        transform=transform,
        preprocess=True,
        crop_black=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2, pin_memory=True)

    model = LWBNA_UNet(in_channels=3, out_channels=3).to(device)
    criterion = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    for epoch in range(epochs):
        if early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, masks).item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        # 获取当前学习率
        current_lr = scheduler.get_last_lr()[0]

        # 早停逻辑
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), '../best_LWBNA_unet_G.pth')
            print(f"※ New best model saved (val_loss: {avg_val_loss:.4f}) ※")
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs}/{patience} epochs")
            if no_improve_epochs >= patience:
                early_stop = True

        print(f'Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss / len(train_loader):.4f} '
              f'- Val Loss: {avg_val_loss:.4f} - LR: {current_lr:.2e}')

    print(f'Training completed! Best validation loss: {best_val_loss:.4f}')


if __name__ == '__main__':
    train_model()