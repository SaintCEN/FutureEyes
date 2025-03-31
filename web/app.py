from flask import Flask, request, jsonify
import torch
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image
import io
from torchvision import transforms
from werkzeug.utils import secure_filename
import os
from torch import nn
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 允许所有跨域请求

# 配置
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 疾病映射
DISEASE_MAP = {
    'D': '糖尿病性视网膜病变',
    'G': '青光眼',
    'N': '正常',
    'C': '白内障',
    'A': '黄斑病变',
    'H': '高血压视网膜病变',
    'M': '病理性近视',
    'O': '其他眼底疾病'
}

# 初始化设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 辅助函数
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# 模型定义和加载
class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.left_model = models.resnet50(pretrained=False)
        self.right_model = models.resnet50(pretrained=False)
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


def load_model(model_class, model_path):
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


try:
    model_D = load_model(ResNet50, 'models/best_model_D.pth')
    model_G = load_model(ResNet50, 'models/best_model_G.pth')

    # 综合模型定义
    class EfficientNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # 使用与训练代码相同的预训练权重配置
            self.base = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
            self.base.classifier = torch.nn.Identity()  # 移除最后的分类层

            # 必须与solve_rest.py中的结构完全一致
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(1536 * 2, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128),  # 新增的隐藏层
                torch.nn.ReLU(),
                torch.nn.Linear(128, 6),  # 输出层
                torch.nn.Sigmoid()  # 添加Sigmoid激活
            )

        def forward(self, x_left, x_right):
            feat_left = self.base(x_left)
            feat_right = self.base(x_right)
            combined = torch.cat([feat_left, feat_right], dim=1)
            return self.fc(combined)

    model_all = load_model(EfficientNet, 'models/best_model_all.pth')
except Exception as e:
    print(f"模型加载失败: {str(e)}")
    exit(1)

# 图像预处理
def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0: return img
        img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
        return np.stack([img1, img2, img3], axis=-1)


def load_ben_color(image, sigmaX=10):
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (300, 300))
    return cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = np.array(image)
        image = load_ben_color(image)
        image = transform(image).unsqueeze(0)
        return image.to(device)
    except Exception as e:
        raise ValueError(f"图像预处理失败: {str(e)}")


# API端点
@app.route('/api/predict', methods=['POST'])
def predict():
    # 检查文件上传
    if 'left' not in request.files or 'right' not in request.files:
        return jsonify({'error': '需要上传左右眼图像'}), 400

    left_file = request.files['left']
    right_file = request.files['right']

    # 验证文件
    if not (allowed_file(left_file.filename) and allowed_file(right_file.filename)):
        return jsonify({'error': '只支持PNG/JPG/JPEG格式'}), 400

    try:
        # 保存文件（可选，用于调试）
        left_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(left_file.filename))
        right_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(right_file.filename))
        left_file.save(left_path)
        right_file.save(right_path)

        # 预处理
        left_tensor = preprocess_image(open(left_path, 'rb').read())
        right_tensor = preprocess_image(open(right_path, 'rb').read())

        # 预测逻辑
        results = set()

        # D模型预测
        with torch.no_grad():
            d_left = torch.sigmoid(model_D(left_tensor, left_tensor)).item()
            d_right = torch.sigmoid(model_D(right_tensor, right_tensor)).item()
        if d_left > 0.5 or d_right > 0.5:
            results.add('D')

        # G模型预测
        with torch.no_grad():
            g_left = torch.sigmoid(model_G(left_tensor, left_tensor)).item()
            g_right = torch.sigmoid(model_G(right_tensor, right_tensor)).item()
        if g_left > 0.5 or g_right > 0.5:
            results.add('G')

        # 综合模型预测
        with torch.no_grad():
            outputs = model_all(left_tensor, right_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()[0]
            other_diseases = ['N', 'C', 'A', 'H', 'M', 'O']
            for i, code in enumerate(other_diseases):
                if probs[i] > 0.7:
                    results.add(code)

        # 结果处理
        if 'N' in results and len(results) > 1:
            results.remove('N')
        final_results = ['N'] if not results else list(results)

        return jsonify({
            'status': 'success',
            'results': [DISEASE_MAP[code] for code in final_results],
            'probabilities': {
                'diabetes_left': d_left,
                'diabetes_right': d_right,
                'glaucoma_left': g_left,
                'glaucoma_right': g_right,
                'other_diseases': dict(
                    zip(['normal', 'cataract', 'amd', 'hypertension', 'myopia', 'other'], probs.tolist()))
            }
        })

    except Exception as e:
        app.logger.error(f'预测错误: {str(e)}')
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# 健康检查端点
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'models_loaded': bool(model_D and model_G and model_all)
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)