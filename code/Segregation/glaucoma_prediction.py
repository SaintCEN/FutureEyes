import torch
import cv2
import os
import joblib
import numpy as np
import pandas as pd
from torch import nn
from torchvision import transforms
from tqdm import tqdm
import glob
from glaucoma_classification import MedicalFeatureExtractor
from segregation_train_glaucoma import LWBNA_UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        gri = x[:, 5].unsqueeze(1)
        normal_p = torch.exp(-0.5 * ((gri - self.gri_normal[0]) / self.gri_normal[1]) ** 2)
        abnormal_p = torch.exp(-0.5 * ((gri - self.gri_abnormal[0]) / self.gri_abnormal[1]) ** 2)
        gri_pred = (abnormal_p > normal_p).float()
        return self.fc(x), gri_pred

def load_models():
    segmentation_model = LWBNA_UNet().to(device)
    segmentation_model.load_state_dict(torch.load("best_LWBNA_unet_G.pth", map_location=device))
    segmentation_model.eval()
    classifier = GlaucomaClassifier().to(device)
    classifier.load_state_dict(torch.load("best_glaucoma_detector.pth", map_location=device))
    classifier.eval()
    extractor = MedicalFeatureExtractor()
    extractor.scaler = joblib.load('scaler.pkl')
    extractor.imputer = joblib.load('imputer.pkl')
    extractor.pca = joblib.load('pca.pkl')
    extractor.is_fitted = True
    return segmentation_model, classifier, extractor

def predict_image(image_path, segmentation_model, classifier, extractor):
    img = cv2.imread(image_path)
    if img is None:
        return 0.0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (320, 320))
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    with torch.no_grad():
        mask_pred = segmentation_model(img_tensor)
        mask_pred = torch.argmax(mask_pred, dim=1).squeeze(0)
    mask_grayscale = torch.zeros_like(mask_pred, dtype=torch.uint8)
    mask_grayscale[mask_pred == 0] = 25
    mask_grayscale[mask_pred == 1] = 125
    mask_grayscale[mask_pred == 2] = 225
    mask_input = mask_grayscale.unsqueeze(0).unsqueeze(0).float() / 255.0
    features = extractor.extract(mask_input)
    processed_features = extractor.transform_features(features)
    feature_tensor = torch.tensor(processed_features, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs, _ = classifier(feature_tensor)
        return outputs.item()

def predict_subjects(image_dir, segmentation_model, classifier, extractor):
    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
    subject_probs = {}
    for img_path in tqdm(image_paths, desc="Processing images"):
        base_name = os.path.basename(img_path)
        parts = base_name.split('_')
        if len(parts) < 2:
            continue
        subject_id = parts[0]
        eye_type = parts[1].split('.')[0]
        prob = predict_image(img_path, segmentation_model, classifier, extractor)
        if subject_id not in subject_probs:
            subject_probs[subject_id] = {'left': 0.0, 'right': 0.0}
        subject_probs[subject_id][eye_type] = prob
    return subject_probs

def save_predictions(image_dir, csv_path, segmentation_model, classifier, extractor):
    df = pd.read_csv(csv_path)
    subject_probs = predict_subjects(image_dir, segmentation_model, classifier, extractor)
    df['G'] = df['ID'].apply(
        lambda pid: max(
            subject_probs.get(str(pid), {}).get('left', 0.0),
            subject_probs.get(str(pid), {}).get('right', 0.0)
        )
    )
    df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    models = load_models()
    save_predictions('C:/Users/SaintCHEN/Desktop/FutureEyes/dataset/Test_All', 'C:/Users/SaintCHEN/Desktop/FutureEyes/outputs/Saint_ODIR.csv', *models)