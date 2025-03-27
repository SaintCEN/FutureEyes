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

# 自动检测设备（GPU/CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 青光眼数据集类：加载并预处理眼底结构分割掩码
class GlaucomaDataset(Dataset):
    def __init__(self, mask_dir):
        # 获取所有.bmp格式的掩码文件路径
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*.bmp')))
        # 定义预处理流程：转为Tensor并调整尺寸
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((320, 320))
        ])

    def __len__(self):
        return len(self.mask_paths)

    def __getitem__(self, idx):
        # 读取灰度掩码图像（取值范围0-255）
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        # 从文件名判断标签：文件名含'g'表示青光眼阳性（label=1）
        label = 1 if 'g' in os.path.basename(self.mask_paths[idx]).lower() else 0
        return self.transform(mask), label  # 返回预处理后的掩码和标签


# 医学特征提取器：从分割掩码中提取CDR、ISNT等临床特征
class MedicalFeatureExtractor:
    def __init__(self):
        # 初始化特征预处理工具
        self.scaler = StandardScaler()  # 标准化器
        self.pca = PCA(n_components=5)  # 主成分分析（降维至5维）
        self.imputer = SimpleImputer(strategy='mean')  # 缺失值填充器
        self.is_fitted = False  # 标记是否已完成PCA拟合

    # 核心方法：从单张掩码提取6维特征（CDR宽/高/面积比 + ISNT四象限 + DDLS评分 + ISNT规则）
    def extract(self, mask):
        # 将Tensor转为numpy数组，并确保为0-255的uint8格式
        mask_np = mask.squeeze().cpu().numpy()
        mask_np = (mask_np * 255).astype(np.uint8) if mask_np.max() <= 1.0 else mask_np.astype(np.uint8)

        # 提取视杯（<50）和视盘（50-200）区域
        cup_mask = (mask_np < 50).astype(np.uint8)
        disc_mask = ((mask_np >= 50) & (mask_np < 200)).astype(np.uint8)

        # 获取视杯和视盘轮廓（用于计算CDR）
        c_contour = self._get_contour(cup_mask)
        d_contour = self._get_contour(disc_mask)

        # 特征列表初始化
        features = []
        # 特征1-3：视杯视盘宽度比、高度比、面积比（CDR）
        features += self._calc_cdr(c_contour, d_contour)
        # 特征4-7：ISNT四象限宽度 + ISNT规则（0/1）
        isnt_features, isnt_rule = self._calc_isnt(disc_mask, d_contour)
        features += isnt_features
        # 特征8：DDLS评分（1-10）
        features.append(self._calc_ddls(disc_mask, d_contour))
        # 特征9：ISNT规则标记
        features.append(isnt_rule)

        # 处理缺失值并返回（6维特征）
        return np.nan_to_num(np.array(features, dtype=np.float32))

    # 辅助方法：获取最大轮廓
    def _get_contour(self, img):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return max(contours, key=cv2.contourArea) if contours else None

    # 计算杯盘比（Cup-to-Disc Ratio, CDR）
    def _calc_cdr(self, c, d):
        if c is None or d is None:
            return [0.3, 0.3, 0.3]  # 默认值（轮廓检测失败时）
        # 获取视杯和视盘的边界框参数
        _, _, cw, ch = cv2.boundingRect(c)
        _, _, dw, dh = cv2.boundingRect(d)
        # 计算宽度、高度、面积的比率（限制最大值为1.0）
        return [
            min(cw / (dw + 1e-6), 1.0),
            min(ch / (dh + 1e-6), 1.0),
            min(cv2.contourArea(c) / (cv2.contourArea(d) + 1e-6), 1.0)
        ]

    # 计算ISNT四象限特征（Inferior, Superior, Nasal, Temporal）
    def _calc_isnt(self, disc_mask, contour):
        if contour is None:
            return [0.5] * 4, 0  # 默认值（轮廓检测失败时）
        # 获取视盘边界框并划分四象限
        x, y, w, h = cv2.boundingRect(contour)
        quadrants = [
            (x, x + w // 2, y + h // 2, y + h),  # Inferior
            (x, x + w // 2, y, y + h // 2),  # Superior
            (x + w // 2, x + w, y, y + h // 2),  # Nasal
            (x, x + w // 2, y + h // 2, y + h)  # Temporal
        ]
        # 计算各象限的平均宽度（归一化到0-1）
        widths = []
        for q in quadrants:
            try:
                roi = disc_mask[q[2]:q[3], q[0]:q[1]]
                widths.append(np.mean(roi) / 255 if roi.size > 0 else 0.5)
            except:
                widths.append(0.5)  # 异常处理
        # ISNT规则：下>上>鼻>颞（符合则为1）
        isnt_rule = int(widths[0] > widths[1]) & (widths[1] > widths[2]) & (widths[2] > widths[3])
        return widths, isnt_rule

    # 计算DDLS评分（Disc Damage Likelihood Scale）
    def _calc_ddls(self, disc_mask, contour):
        if contour is None:
            return 5.0  # 默认值（轮廓检测失败时）
        try:
            # 获取视盘最小外接矩形参数
            (x, y), (diam, _), angle = cv2.minAreaRect(contour)
            # 计算视盘边缘到背景的距离变换
            dist = cv2.distanceTransform(disc_mask, cv2.DIST_L2, 3)
            min_rim = np.min(dist[dist > 0])  # 最小边缘宽度
            rim_ratio = min_rim / (diam + 1e-6)

            # 根据边缘宽度比率评分（1-4级）
            if rim_ratio >= 0.4:
                return 1.0
            elif rim_ratio >= 0.3:
                return 2.0
            elif rim_ratio >= 0.2:
                return 3.0
            elif rim_ratio >= 0.1:
                return 4.0
            else:
                # 计算边缘角度差决定更严重的分级（5-10级）
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
            return 5.0  # 异常处理

    # 计算视盘边缘角度（用于DDLS评分）
    def _calc_rim_angles(self, disc_mask, contour):
        mask = np.zeros_like(disc_mask)
        cv2.drawContours(mask, [contour], -1, 255, 1)  # 绘制轮廓
        yx = np.column_stack(np.where(mask.T))  # 获取轮廓点坐标
        # 计算各点相对于中心点的角度（0-360度）
        angles = np.arctan2(yx[:, 1] - contour[0][0][0], yx[:, 0] - contour[0][0][1]) * 180 / np.pi
        return angles % 360

    # 拟合PCA模型（需在训练前调用）
    def fit_pca(self, features):
        features = np.vstack(features)  # 将特征列表转为矩阵
        features = self.imputer.fit_transform(features)  # 填充缺失值
        self.scaler.fit(features)  # 拟合标准化器
        self.pca.fit(self.scaler.transform(features))  # 拟合PCA模型
        self.is_fitted = True

    # 特征变换：标准化 -> PCA降维 -> 计算GRI（青光眼风险指数）
    def transform_features(self, features):
        features = self.imputer.transform([features])  # 填充缺失值
        pca_feat = self.pca.transform(self.scaler.transform(features))[0]  # 降维
        # 计算GRI（公式来自文献）
        gri = 6.8375 - 1.1325 * pca_feat[0] + 1.65 * pca_feat[1] + 2.7225 * pca_feat[2] + 0.675 * pca_feat[3] + 0.665 * \
              pca_feat[4]
        return np.concatenate([pca_feat, [gri]])  # 6维特征（5 PCA + GRI）


# 青光眼分类模型：基于临床特征的二元分类器
class GlaucomaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义网络结构：6输入 -> 64 -> 32 -> 1输出
        self.fc = nn.Sequential(
            nn.Linear(6, 64),
            nn.BatchNorm1d(64),  # 批归一化
            nn.ReLU(),  # 激活函数
            nn.Dropout(0.3),  # 随机失活
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出概率
        )
        # 注册GRI的正态分布参数（固定值，不参与训练）
        self.register_buffer('gri_normal', torch.tensor([8.68, 1.67]))  # 正常GRI分布（均值, 标准差）
        self.register_buffer('gri_abnormal', torch.tensor([4.84, 2.08]))  # 异常GRI分布

    def forward(self, x):
        # 提取GRI特征（第6维）并计算正态分布概率
        gri = x[:, 5].unsqueeze(1)  # 保持维度（batch_size, 1）
        normal_p = torch.exp(-0.5 * ((gri - self.gri_normal[0]) / self.gri_normal[1]) ** 2)
        abnormal_p = torch.exp(-0.5 * ((gri - self.gri_abnormal[0]) / self.gri_abnormal[1]) ** 2)
        gri_pred = (abnormal_p > normal_p).float()  # 比较概率得到初步预测
        return self.fc(x), gri_pred  # 返回网络预测和GRI预测


# 主函数：训练分类模型
def main():
    # 初始化数据集（路径需替换为实际掩码目录）
    dataset = GlaucomaDataset('C:/Users/SaintCHEN/Desktop/FutureEyes/dataset/Refuge/Train/Disc_Cup_Masks/All')
    extractor = MedicalFeatureExtractor()  # 特征提取器

    # 提取所有样本的特征和标签
    all_features = []
    labels = []
    for mask, label in tqdm(DataLoader(dataset, batch_size=1)):
        features = extractor.extract(mask)
        all_features.append(features)
        labels.append(label)

    # 拟合PCA模型并进行特征变换
    extractor.fit_pca(all_features)
    processed = np.array([extractor.transform_features(f) for f in all_features])

    # 转换为Tensor并上传到设备
    X = torch.from_numpy(processed).float().to(device)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1).to(device)

    # 划分训练集和测试集（80%训练，20%测试）
    X_train, X_test, y_train, y_test = train_test_split(
        X.cpu().numpy(), y.cpu().numpy(),
        test_size=0.2, stratify=labels, random_state=42
    )
    # 重新转换为Tensor（保持设备一致）
    X_train = torch.from_numpy(X_train).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    X_test = torch.from_numpy(X_test).float().to(device)
    y_test = torch.from_numpy(y_test).float().to(device)

    # 初始化模型、损失函数和优化器
    model = GlaucomaClassifier().to(device)
    criterion = nn.BCELoss()  # 二元交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 训练循环
    best_acc = 0.0
    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        outputs, _ = model(X_train)  # 前向传播
        loss = criterion(outputs, y_train)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 参数更新

        # 验证阶段
        model.eval()
        with torch.no_grad():
            preds, gri_pred = model(X_test)
            acc = ((preds > 0.5).float() == y_test).float().mean()  # 计算准确率
            if acc > best_acc:  # 保存最佳模型
                best_acc = acc
                torch.save(model.state_dict(),
                           'C:/Users/SaintCHEN/Desktop/FutureEyes/models/best_glaucoma_detector.pth')

        # 每10轮打印训练信息
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1:03d} | Loss: {loss.item():.4f} | Acc: {acc.item():.4f}')

    print(f'Best Test Accuracy: {best_acc.item():.4f}')

    # 保存预处理模型（标准化器、缺失值填充器、PCA模型）
    joblib.dump(extractor.scaler, 'C:/Users/SaintCHEN/Desktop/FutureEyes/models/scaler.pkl')
    joblib.dump(extractor.imputer, 'C:/Users/SaintCHEN/Desktop/FutureEyes/models/imputer.pkl')
    joblib.dump(extractor.pca, 'C:/Users/SaintCHEN/Desktop/FutureEyes/models/pca.pkl')


if __name__ == '__main__':
    main()