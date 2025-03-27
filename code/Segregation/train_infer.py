import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model_utils import DeepLabv3Plus, dense_crf_wrapper
from data_utils import get_loaders
import cv2
import numpy as np

class Trainer:
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=1e-4)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=3)

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        for images, masks in self.train_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in self.val_loader:
                outputs = self.model(images.to(self.device))
                loss = self.criterion(outputs, masks.to(self.device))
                val_loss += loss.item()
        return val_loss / len(self.val_loader)

    def train(self, epochs=50, save_path='best_model.pth'):
        best_loss = float('inf')

        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()

            self.scheduler.step(val_loss)

            print(f'Epoch {epoch + 1}/{epochs}')
            print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                print('Model saved!')

class Inferencer:
    def __init__(self, model_path, device='cuda'):
        self.model = DeepLabv3Plus().to(device)
        self.model.load_state_dict(torch.load(model_path))
        self.device = device
        self.model.eval()

    def predict(self, image_path, apply_crf=True):
        # 预处理
        image = cv2.imread(image_path)
        orig_size = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512)) / 255.0

        # 推理
        with torch.no_grad():
            tensor_img = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            logits = self.model(tensor_img)
            probs = torch.sigmoid(logits)

        # 后处理
        mask = (probs.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
        mask = cv2.resize(mask, orig_size[::-1])

        if apply_crf:
            return dense_crf_wrapper(tensor_img, logits)
        return mask

if __name__ == '__main__':
    # 训练模式
    train_loader, val_loader = get_loaders('data/train', 'data/val')
    model = DeepLabv3Plus(num_classes=1)
    trainer = Trainer(model, train_loader, val_loader)
    trainer.train(epochs=50)

    # 推理示例
    # infer = Inferencer('best_model.pth')
    # result = infer.predict('test_image.jpg')
    # cv2.imwrite('result.png', result)