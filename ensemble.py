import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import pickle as pkl
import sys
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F

import torch.nn as nn
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed
import torchvision.models as models
from timm.optim import Lookahead
from torch.optim.lr_scheduler import OneCycleLR
from itertools import product
import time
from transformers import ViTForImageClassification, ViTImageProcessor
import math

def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark  = False

    try:
        import cupy
        cupy.random.seed(seed)
    except:
        pass

    os.environ['PYTHONHASHSEED'] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

seed_everything()

df_path = '/home/munz/school/big_data/data/train.csv'
df = pd.read_csv(df_path)

df = df[df['chain'] != 0]

df_grouped = df.groupby('hotel_id').agg({'image': 'count', 'chain': 'first'}).reset_index()
df_grouped = df_grouped.sort_values(by='image', ascending=False)
top_100_hotel_ids = df_grouped['hotel_id'].values[:100]
df = df[df['hotel_id'].isin(top_100_hotel_ids)].reset_index(drop=True)

image_to_hotel_id = {}
for i, image in enumerate(df['image']):
    image_to_hotel_id[image] = df['hotel_id'][i]

hotel_id_to_label = {}
for i, hotel_id in enumerate(df['hotel_id'].unique()):
    hotel_id_to_label[hotel_id] = i

image_to_label = {}
for image, hotel_id in image_to_hotel_id.items():
    image_to_label[image] = hotel_id_to_label[hotel_id]

def load_and_resize_image(image_path, size):
    image = cv2.imread(image_path)
    
    h, w = image.shape[:2]
    scale = min(size[0] / h, size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)

    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    delta_h = size[0] - new_h
    delta_w = size[1] - new_w
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return image

def load_images(images_path, image_to_label, size=(224, 224), max_workers=4):
    images = []
    labels = []
    image_paths = []

    # Collect all image paths
    for root, _, files in tqdm(os.walk(images_path), desc="Collecting image paths"):
        for image_name in files:
            if image_name in image_to_label:
                image_path = os.path.join(root, image_name)
                image_paths.append((image_path, image_to_label[image_name]))

    # Use ThreadPoolExecutor to parallelize the image loading process
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_and_resize_image, image_path, size): label for image_path, label in image_paths}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading images"):
            image = future.result()
            if image is not None:
                images.append(image)
                labels.append(futures[future])

    return np.array(images), np.array(labels)

load = True
if load:
    with open('images.pkl', 'rb') as f:
        images = pkl.load(f)

    with open('labels.pkl', 'rb') as f:
        labels = pkl.load(f)
else:
    train_images_path = '/home/munz/school/big_data/data/train_images'
    target_size = (512, 512)
    images, labels = load_images(train_images_path, image_to_label, size=target_size, max_workers=6)
    with open('images.pkl', 'wb') as f:
        pkl.dump(images, f)

    with open('labels.pkl', 'wb') as f:
        pkl.dump(labels, f)

X_train_val, X_test, y_train_val, y_test = train_test_split(images, labels, test_size=0.15, random_state=42, stratify=labels)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1765, random_state=42, stratify=y_train_val)

class CombinedImageDataset(Dataset):
    def __init__(self, images, labels, vit_transform=None, common_transform=None):
        self.images = images
        self.labels = labels
        self.vit_transform = vit_transform
        self.common_transform = common_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        vit_image = self.vit_transform(image) if self.vit_transform else image
        common_image = self.common_transform(image) if self.common_transform else image

        return common_image, vit_image, label

feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

def transform_image(image):
    inputs = feature_extractor(images=image, return_tensors='pt')
    return inputs['pixel_values'][0]

vit_transform = transforms.Compose([
    transforms.Lambda(lambda img: transform_image(img)),
])

# Common transform for other models
common_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets and loaders
batch_size = 32

train_dataset = CombinedImageDataset(X_train, y_train, vit_transform=vit_transform, common_transform=common_transform)
val_dataset = CombinedImageDataset(X_val, y_val, vit_transform=vit_transform, common_transform=common_transform)
test_dataset = CombinedImageDataset(X_test, y_test, vit_transform=vit_transform, common_transform=common_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcFace, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = nn.functional.linear(nn.functional.normalize(input), nn.functional.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

class ArcFaceModel(nn.Module):
    def __init__(self, backbone, arcface):
        super(ArcFaceModel, self).__init__()
        self.backbone = backbone
        self.arcface = arcface

    def forward(self, x, labels):
        features = self.backbone(x)
        logits = self.arcface(features, labels)
        return logits

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = len(df['hotel_id'].unique())

vgg16_model = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
vgg16_model.classifier[6] = nn.Linear(vgg16_model.classifier[6].in_features, num_classes)

resnet101_model = models.resnet101(weights='ResNet101_Weights.IMAGENET1K_V1')
resnet101_model.fc = nn.Linear(resnet101_model.fc.in_features, num_classes)

vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=num_classes)

arcface_backbone = models.resnet101(weights='ResNet101_Weights.IMAGENET1K_V1')
in_features = arcface_backbone.fc.in_features
arcface_backbone.fc = nn.Identity()
arcface = ArcFace(in_features, num_classes)
arcface_model = ArcFaceModel(arcface_backbone, arcface)

vgg16_model.to(device)
resnet101_model.to(device)
vit_model.to(device)
arcface_model.to(device)

print('Models loaded')

vgg_state_dict_path = '/home/munz/school/big_data/models/vgg16/model_1.pth'
resnet_state_dict_path = '/home/munz/school/big_data/models/resnet101/model_5.pth'
vit_state_dict_path = '/home/munz/school/big_data/models/ViT/model_5.pth'
arcface_state_dict_path = '/home/munz/school/big_data/models/arcface/model_11.pth'

vgg16_model.load_state_dict(torch.load(vgg_state_dict_path))
resnet101_model.load_state_dict(torch.load(resnet_state_dict_path))
vit_model.load_state_dict(torch.load(vit_state_dict_path))
arcface_model.load_state_dict(torch.load(arcface_state_dict_path))

y_pred_top_k = []
y_pred = []
y_true = []
k = 5
ensemble = 4

with torch.no_grad():
    vgg16_model.eval()
    resnet101_model.eval()
    vit_model.eval()
    arcface_model.eval()

    for standard_data, vit_data, label in tqdm(test_loader, desc="Testing", leave=False):
        standard_data = standard_data.to(device)
        vit_data = vit_data.to(device)
        label = label.to(device)

        # Get outputs from each model
        vgg16_output = vgg16_model(standard_data)
        resnet101_output = resnet101_model(standard_data)
        vit_output = vit_model(vit_data).logits
        arcface_output = arcface_model(standard_data, label)

        # Apply softmax to outputs
        vgg16_softmax = F.softmax(vgg16_output, dim=1)
        resnet101_softmax = F.softmax(resnet101_output, dim=1)
        vit_softmax = F.softmax(vit_output, dim=1)
        arcface_softmax = F.softmax(arcface_output, dim=1)

        if ensemble == 1:
            ensemble_softmax = (vgg16_softmax + resnet101_softmax + vit_softmax + arcface_softmax) / 4
        if ensemble == 2:
            ensemble_softmax = (vgg16_softmax + resnet101_softmax + vit_softmax) / 3
        if ensemble == 3:
            ensemble_softmax = (resnet101_softmax + vit_softmax + arcface_softmax) / 3
        if ensemble == 4:
            ensemble_softmax = (resnet101_softmax + vit_softmax) / 2

        top_k_values, top_k_indices = ensemble_softmax.topk(k, dim=1)
        y_pred_top_k.extend(top_k_indices.cpu().numpy())
        y_pred.extend(ensemble_softmax.argmax(dim=1).cpu().numpy())
        y_true.extend(label.cpu().numpy())
    
def average_precision_at_5(y_true, y_pred):
    """
    Calculate the average precision at 5 for a single instance.
    """
    correct = 0
    score = 0.0
    for k in range(5):
        if k < len(y_pred) and y_pred[k] == y_true:
            correct += 1
            score += correct / (k + 1)
            break
    return score

def mean_average_precision_at_5(y_true, y_pred):
    """
    Calculate the mean average precision at 5.
    """
    return np.mean([average_precision_at_5(yt, yp) for yt, yp in zip(y_true, y_pred)])

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# only resnet101 and vit
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_true, y_pred, average='macro')
map5 = mean_average_precision_at_5(y_true, y_pred_top_k)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'Mean Average Precision at 5: {map5}')