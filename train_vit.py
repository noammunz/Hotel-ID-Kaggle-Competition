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

import torch.nn as nn
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed
import torchvision.models as models
from timm.optim import Lookahead
from torch.optim.lr_scheduler import OneCycleLR
from itertools import product
import time
from transformers import ViTForImageClassification, ViTImageProcessor

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

def log(text):
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    elapsed_timestamp = f'{minutes:02}:{seconds:02}    '
    print(f'{elapsed_timestamp}{text}')
    sys.stdout.flush()

start_time = time.time()

model_name = 'ViT'
epochs = [10, 20 , 50]
batch_sizes = [16, 32]
learning_rates = [0.001, 0.0001]

search = int(sys.argv[1])
epochs, batch_size, learning_rate = list(product(epochs, batch_sizes, learning_rates))[search]

log(f'Starting search {search}')
log(f'Model: {model_name}')
log(f'Epochs: {epochs}')
log(f'Batch_size: {batch_size}')
log(f'Learning_rate: {learning_rate}')

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

log('Loading images')
with open('/home/munz/school/big_data/images.pkl', 'rb') as f:
    images = pkl.load(f)

with open('/home/munz/school/big_data/labels.pkl', 'rb') as f:
    labels = pkl.load(f)

X_train_val, X_test, y_train_val, y_test = train_test_split(images, labels, test_size=0.15, random_state=42, stratify=labels)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1765, random_state=42, stratify=y_train_val)

augmentation = True
if augmentation:
    log('Augmenting images')
    def horizontal_flip(image):
        flipped_image = cv2.flip(image, 1)
        return flipped_image

    def rotate_image(image, angle_range=(-30, 30)):
        angle = np.random.uniform(low=angle_range[0], high=angle_range[1])
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, M, (w, h))
        return rotated_image

    def adjust_brightness(image, factor_range=(0.5, 1.5)):
        factor = np.random.uniform(low=factor_range[0], high=factor_range[1])
        adjusted_image = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        return adjusted_image
    
    augmented_images = []
    augmented_labels = []

    for i in tqdm(range(len(X_train)), desc="Augmenting images"):
        image = X_train[i]
        label = y_train[i]

        augmented_images.append(image)
        augmented_labels.append(label)

        flipped_image = horizontal_flip(image)
        augmented_images.append(flipped_image)
        augmented_labels.append(label)

        rotated_image = rotate_image(image)
        augmented_images.append(rotated_image)
        augmented_labels.append(label)

        # adjusted_image = adjust_brightness(image)
        # augmented_images.append(adjusted_image)
        # augmented_labels.append(label)

    X_train = np.array(augmented_images)
    y_train = np.array(augmented_labels)
    log(f'Augmented images: {len(X_train)}')

class ImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)

        return image, label

feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

def transform_image(image):
    inputs = feature_extractor(images=image, return_tensors='pt')
    return inputs['pixel_values'][0]

transform = transforms.Compose([
    transforms.Lambda(lambda img: transform_image(img)),
])

train_dataset = ImageDataset(X_train, y_train, transform=transform)
val_dataset = ImageDataset(X_val, y_val, transform=transform)
test_dataset = ImageDataset(X_test, y_test, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = len(df['hotel_id'].unique())
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=num_classes)
model = model.to(device)

base_optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
optimizer = Lookahead(base_optimizer, k=3)
criterion = nn.CrossEntropyLoss()
scheduler = OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(train_loader), epochs=epochs, div_factor=10, final_div_factor=1, pct_start=0.1, anneal_strategy='cos')

train_losses = []
val_losses = []
val_accuracies = []

patience = 3

log('Training model')
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data).logits
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item()

    log(f'Epoch {epoch+1} - Average loss: {running_loss / len(train_loader):.4f}')
    train_losses.append(running_loss / len(train_loader))


    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data).logits
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    val_loss = val_loss / len(val_loader)
    val_losses.append(val_loss)
    accuracy = correct / len(val_loader.dataset)
    val_accuracies.append(accuracy)

    log(f'Validation loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')
    
    # early stopping
    if epoch > 5:
        last_losses = val_losses[-patience:]
        # if last losses is an increasing sequence
        if all(x < y for x, y in zip(last_losses, last_losses[1:])):
            log('Early stopping')
            epochs_completed = epoch + 1
            break

log('Training finished, saving model')

model_directory = f'/home/munz/school/big_data/models/{model_name}'
if not os.path.exists(model_directory):
    os.makedirs(model_directory)
model_save_path = f'/home/munz/school/big_data/models/{model_name}/model_{search}.pth'
if augmentation:
    model_save_path = f'/home/munz/school/big_data/models/{model_name}/model_{search}_aug.pth'
torch.save(model.state_dict(), model_save_path)

model.eval()
y_pred_top_k = []
y_pred = []
y_true = []
k = 5

log('Validating model')
with torch.no_grad():
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        output = model(data).logits
        pred = output.topk(k, dim=1)[1]
        y_pred_top_k.extend(pred.cpu().numpy())
        y_pred.extend(output.argmax(dim=1, keepdim=True).cpu().numpy().flatten())
        y_true.extend(target.cpu().numpy())

y_pred_top_k = np.array(y_pred_top_k)

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

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_true, y_pred, average='macro')
map5 = mean_average_precision_at_5(y_true, y_pred_top_k)

log(f'Test accuracy: {accuracy:.4f}')
log(f'Test precision: {precision:.4f}')
log(f'Test recall: {recall:.4f}')
log(f'Test f1: {f1:.4f}')
log(f'Mean Average Precision at 5: {map5:.4f}')

write_epochs = f'{epochs_completed}/{epochs}' if 'epochs_completed' in locals() else f'{epochs}'
end_time_in_minutes = round((time.time() - start_time) / 60, 2)
results_df = pd.DataFrame({
    'search': [search],
    'epochs': [write_epochs],
    'batch_size': [batch_size],
    'learning_rate': [learning_rate],
    'accuracy': [accuracy],
    'precision': [precision],
    'recall': [recall],
    'f1': [f1],
    'map5': [map5],
    'time': [end_time_in_minutes],
    'train_losses': [train_losses],
    'val_losses': [val_losses],
    'val_accuracies': [val_accuracies]
})

results_directory = '/home/munz/school/big_data/results'
if not os.path.exists(results_directory):
    os.makedirs(results_directory)
results_path = f'/home/munz/school/big_data/results/results_{model_name}.csv'

if augmentation:
    results_path = f'/home/munz/school/big_data/results/results_aug.csv'
    results_df['model'] = model_name
if os.path.exists(results_path):
    results_df.to_csv(results_path, mode='a', header=False, index=False)
else:
    results_df.to_csv(results_path, index=False)

log('Finished search')