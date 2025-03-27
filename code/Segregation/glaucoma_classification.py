import os
import glob
import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GlaucomaDataset(Dataset):
    def __init__(self, mask_dir):
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*.bmp')))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((320, 320))
        ])

    def __len__(self):
        return len(self.mask_paths)

    def __getitem__(self, idx):
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        return self.transform(mask), self.mask_paths[idx]


class MedicalFeatureExtractor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=5)
        self.imputer = SimpleImputer(strategy='mean')
        self.is_fitted = False
    def extract(self, mask):
        mask_np = (mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
        cup_mask = (mask_np < 50).astype(np.uint8)
        disc_mask = ((mask_np >= 50) & (mask_np < 200)).astype(np.uint8)
        c_contour = self._get_contour(cup_mask)
        d_contour = self._get_contour(disc_mask)
        features = []
        features += self._calc_cdr(c_contour, d_contour)
        isnt_features, isnt_rule = self._calc_isnt(disc_mask, d_contour)
        features += isnt_features
        features.append(self._calc_ddls(disc_mask, d_contour))
        features.append(isnt_rule)
        return np.nan_to_num(np.array(features, dtype=np.float32))

    def _get_contour(self, img):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return max(contours, key=cv2.contourArea) if contours else None

    def _calc_cdr(self, c, d):
        if c is None or d is None:
            return [0.3, 0.3, 0.3]
        _, _, cw, ch = cv2.boundingRect(c)
        _, _, dw, dh = cv2.boundingRect(d)
        return [
            min(cw / (dw + 1e-6), 1.0),
            min(ch / (dh + 1e-6), 1.0),
            min(cv2.contourArea(c) / (cv2.contourArea(d) + 1e-6), 1.0)
        ]

    def _calc_isnt(self, disc_mask, contour):
        if contour is None:
            return [0.5] * 4, 0

        x, y, w, h = cv2.boundingRect(contour)
        quadrants = [
            (x, x + w // 2, y + h // 2, y + h),
            (x, x + w // 2, y, y + h // 2),
            (x + w // 2, x + w, y, y + h // 2),
            (x, x + w // 2, y + h // 2, y + h)
        ]

        widths = []
        for q in quadrants:
            try:
                roi = disc_mask[q[2]:q[3], q[0]:q[1]]
                widths.append(np.mean(roi) / 255 if roi.size > 0 else 0.5)
            except:
                widths.append(0.5)

        isnt_rule = int((widths[0] > widths[1]) & (widths[1] > widths[2]) & (widths[2] > widths[3]))
        return widths, isnt_rule

    def _calc_ddls(self, disc_mask, contour):
        if contour is None:
            return 5.0

        try:
            (x, y), (diam, _), angle = cv2.minAreaRect(contour)
            dist = cv2.distanceTransform(disc_mask, cv2.DIST_L2, 3)
            min_rim = np.min(dist[dist > 0])
            rim_ratio = min_rim / (diam + 1e-6)

            if rim_ratio >= 0.4:
                return 1.0
            elif rim_ratio >= 0.3:
                return 2.0
            elif rim_ratio >= 0.2:
                return 3.0
            elif rim_ratio >= 0.1:
                return 4.0
            else:
                _, angles = self._calc_rim_angles(disc_mask, contour)
                gap = max(angles) - min(angles)
                if gap < 45:
                    return 6.0
                elif gap < 90:
                    return 7.0
                elif gap < 180:
                    return 8.0
                elif gap < 270:
                    return 9.0
                else:
                    return 10.0
        except:
            return 5.0

    def _calc_rim_angles(self, disc_mask, contour):
        mask = np.zeros_like(disc_mask)
        cv2.drawContours(mask, [contour], -1, 255, 1)
        yx = np.column_stack(np.where(mask.T))
        angles = np.arctan2(yx[:, 1] - contour[0][0][0], yx[:, 0] - contour[0][0][1]) * 180 / np.pi
        return angles % 360

    def fit_pca(self, features):
        features = np.vstack(features)
        features = self.imputer.fit_transform(features)
        self.scaler.fit(features)
        self.pca.fit(self.scaler.transform(features))
        self.is_fitted = True

    def transform_features(self, features):
        features = self.imputer.transform([features])
        pca_feat = self.pca.transform(self.scaler.transform(features))[0]
        gri = 6.8375 - 1.1325 * pca_feat[0] + 1.65 * pca_feat[1] + 2.7225 * pca_feat[2] + 0.675 * pca_feat[3] + 0.665 * \
              pca_feat[4]
        return np.concatenate([pca_feat, [gri]])

class GlaucomaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(6, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.register_buffer('gri_normal', torch.tensor([8.68, 1.67]))
        self.register_buffer('gri_abnormal', torch.tensor([4.84, 2.08]))

    def forward(self, x):
        gri = x[:, 5].unsqueeze(1)  # 确保gri是2D张量
        normal_p = torch.exp(-0.5 * ((gri - self.gri_normal[0]) / self.gri_normal[1]) ** 2)
        abnormal_p = torch.exp(-0.5 * ((gri - self.gri_abnormal[0]) / self.gri_abnormal[1]) ** 2)
        gri_pred = (abnormal_p > normal_p).float()
        return self.fc(x), gri_pred


def main():
    dataset = GlaucomaDataset('../dataset/Refuge/Annotation-Training400/Masks')
    extractor = MedicalFeatureExtractor()

    all_features = []
    labels = []
    for mask, path in tqdm(DataLoader(dataset, batch_size=1)):
        try:
            features = extractor.extract(mask)
            all_features.append(features)
            labels.append(1 if 'g' in path[0].lower() else 0)
        except Exception as e:
            print(f"Error processing {path[0]}: {str(e)}")
            continue

    extractor.fit_pca(all_features)

    # 转换为单个numpy数组再转tensor
    processed = np.array([extractor.transform_features(f) for f in all_features])
    X = torch.from_numpy(processed).float().to(device)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1).to(device)

    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X.cpu().numpy(), y.cpu().numpy(), test_size=0.2, stratify=labels, random_state=42
    )

    # 转换为tensor时确保使用torch.from_numpy
    X_train = torch.from_numpy(X_train).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    X_test = torch.from_numpy(X_test).float().to(device)
    y_test = torch.from_numpy(y_test).float().to(device)

    model = GlaucomaClassifier().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_acc = 0.0
    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        outputs, _ = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            preds, gri_pred = model(X_test)
            acc = ((preds > 0.5).float() == y_test).float().mean()
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), '../best_glaucoma_detector.pth')
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1:03d} | Loss: {loss.item():.4f} | Acc: {acc.item():.4f}')
    print(f'Best Test Accuracy: {best_acc.item():.4f}')

    joblib.dump(extractor.scaler, '../scaler.pkl')
    joblib.dump(extractor.imputer, '../imputer.pkl')
    joblib.dump(extractor.pca, '../pca.pkl')

if __name__ == '__main__':
    main()