import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os

class MedicalDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=512, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.transform = transform
        self.files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.files[idx])
        mask_path = os.path.join(self.mask_dir, self.files[idx].replace('.jpg', '.bmp'))

        # 加载图像
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 数据增强
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # 默认调整大小
            image = cv2.resize(image, (self.img_size, self.img_size))
            mask = cv2.resize(mask, (self.img_size, self.img_size))

        # 归一化
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        # 转换为Tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask


def get_loaders(train_dir, val_dir, batch_size=4, num_workers=4):
    train_dataset = MedicalDataset(
        train_dir,
        train_dir,
        transform=get_train_augmentations()
    )
    val_dataset = MedicalDataset(val_dir, val_dir)

    return (
        DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_dataset, batch_size, num_workers=num_workers)
    )

def get_train_augmentations():
    from albumentations import (
        HorizontalFlip, VerticalFlip, RandomRotate90,
        ShiftScaleRotate, ElasticTransform, GridDistortion,
        RandomBrightnessContrast, Compose
    )
    return Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=30,
            p=0.5
        ),
        ElasticTransform(p=0.3),
        GridDistortion(p=0.3),
        RandomBrightnessContrast(p=0.3),
    ])