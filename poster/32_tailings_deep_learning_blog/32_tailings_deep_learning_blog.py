#!/usr/bin/env python3
"""
Python code extracted from 32_tailings_deep_learning_blog.md

This code was automatically extracted from the markdown file.
You may need to adjust imports and add necessary dependencies.
"""

# %pip install torch torchvision apache-sedona databricks-mosaic rasterio

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import rasterio
from pyspark.sql import SparkSession
from sedona.register import SedonaRegistrator
import mosaic as mos

# Initialize Spark + Sedona
spark = (SparkSession.builder
    .appName("TailingsDeepLearning")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .config("spark.kryo.registrator", "org.apache.sedona.core.serde.SedonaKryoRegistrator")
    .getOrCreate())

SedonaRegistrator.registerAll(spark)
mos.enable_mosaic(spark, dbutils)

# print("✓ Environment ready")

# ======================================================================
# Code Block 2
# ======================================================================

# Download Global Tailings Portal data
import urllib.request

URL = "https://tailing.grida.no/data/tailings-dam-locations.csv"
urllib.request.urlretrieve(URL, "/tmp/tailings_dams.csv")

# Load to Spark
tailings = spark.read.csv("/tmp/tailings_dams.csv", header=True, inferSchema=True)

# Filter to active dams with coordinates
tailings_active = tailings.filter(
    (tailings.latitude.isNotNull()) & 
    (tailings.longitude.isNotNull()) &
    (tailings.status == "Active")
)

tailings_active.write.format("delta").mode("overwrite").saveAsTable("bronze.tailings_locations")

# print(f"✓ Loaded {tailings_active.count()} active tailings dams")

# ======================================================================
# Code Block 3
# ======================================================================

def generate_synthetic_sentinel2_patch(size=256, has_change=False):
    """Generate synthetic 256256 Sentinel-2 patch (RGB + NIR)."""
    
    # Base layers
    blue = np.random.uniform(0.05, 0.15, (size, size))
    green = np.random.uniform(0.10, 0.25, (size, size))
    red = np.random.uniform(0.08, 0.20, (size, size))
    nir = np.random.uniform(0.20, 0.50, (size, size))
    
    # Add tailings pond (center region)
    y, x = np.ogrid[:size, :size]
    center_y, center_x = size // 2, size // 2
    mask_pond = ((x - center_x)**2 + (y - center_y)**2) < (size // 4)**2
    
    # Pond has low NIR (water)
    nir[mask_pond] = np.random.uniform(0.02, 0.08, mask_pond.sum())
    blue[mask_pond] += 0.05
    
    # Add embankment (right edge)
    mask_embankment = (x > size * 0.7) & (y > size * 0.3) & (y < size * 0.7)
    nir[mask_embankment] = np.random.uniform(0.15, 0.30, mask_embankment.sum())
    red[mask_embankment] += 0.08
    
    image = np.stack([blue, green, red, nir], axis=-1)  # Shape: (256, 256, 4)
    
    # Generate change mask
    change_mask = np.zeros((size, size), dtype=np.float32)
    
    if has_change:
        # Simulate seepage (dark stain below embankment)
        seepage_y = int(size * 0.65)
        seepage_x = int(size * 0.75)
        for dy in range(10):
            for dx in range(15):
                y_pos = seepage_y + dy
                x_pos = seepage_x + dx
                if 0 <= y_pos < size and 0 <= x_pos < size:
                    nir[y_pos, x_pos] *= 0.7  # Darken (moisture)
                    change_mask[y_pos, x_pos] = 1.0
    
    return image, change_mask

# Generate training dataset
n_train = 500
X_before, X_after, y_change = [], [], []

for i in range(n_train):
    has_change = (i % 5 == 0)  # 20% change rate
    
    before, _ = generate_synthetic_sentinel2_patch(has_change=False)
    after, mask = generate_synthetic_sentinel2_patch(has_change=has_change)
    
    X_before.append(before)
    X_after.append(after)
    y_change.append(mask)

X_before = np.array(X_before)  # Shape: (500, 256, 256, 4)
X_after = np.array(X_after)
y_change = np.array(y_change)  # Shape: (500, 256, 256)

# print(f"✓ Generated {n_train} training samples")
print(f"Change samples: {(y_change.sum(axis=(1,2)) > 0).sum()} ({(y_change.sum(axis=(1,2)) > 0).mean()*100:.1f}%)")

# ======================================================================
# Code Block 4
# ======================================================================

class UNet(nn.Module):
    def __init__(self, in_channels=8, out_channels=1):
        """
        U-Net for change detection.
        
        Parameters:
        -----------
        in_channels : int
            Number of input channels (8 = before + after, each 4 bands)
        out_channels : int
            Number of output channels (1 = change probability)
        """
        super(UNet, self).__init__()
        
        # Encoder (downsampling)
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        
        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)
        
        # Decoder (upsampling)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)  # 512 = 256 (upconv) + 256 (skip)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # Output layer
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        """Double convolution block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc3, 2))
        
        # Decoder with skip connections
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Output (change probability)
        return torch.sigmoid(self.out(dec1))

# Initialize model
model = UNet(in_channels=8, out_channels=1)
# print(f"✓ U-Net initialized")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ======================================================================
# Code Block 5
# ======================================================================

from torch.utils.data import Dataset, DataLoader

class ChangeDetectionDataset(Dataset):
    def __init__(self, X_before, X_after, y_change):
        self.X_before = torch.from_numpy(X_before).float().permute(0, 3, 1, 2)  # (N, C, H, W)
        self.X_after = torch.from_numpy(X_after).float().permute(0, 3, 1, 2)
        self.y_change = torch.from_numpy(y_change).float().unsqueeze(1)  # (N, 1, H, W)
        
    def __len__(self):
        return len(self.X_before)
    
    def __getitem__(self, idx):
        # Concatenate before and after
        x = torch.cat([self.X_before[idx], self.X_after[idx]], dim=0)  # (8, H, W)
        y = self.y_change[idx]  # (1, H, W)
        return x, y

# Create dataset and loader
dataset = ChangeDetectionDataset(X_before, X_after, y_change)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# print(f"✓ Data loaders ready: {train_size} train, {test_size} test")

# ======================================================================
# Code Block 6
# ======================================================================

import mlflow
import mlflow.pytorch

# Training configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.BCELoss()  # Binary Cross-Entropy for change/no-change
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

n_epochs = 20

mlflow.set_experiment("/Shared/tailings-change-detection")

with mlflow.start_run():
    mlflow.log_param("epochs", n_epochs)
    mlflow.log_param("batch_size", 8)
    mlflow.log_param("learning_rate", 1e-4)
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(test_loader)
        
        print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
        
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
        mlflow.log_metric("test_loss", avg_test_loss, step=epoch)
    
    # Save model
    mlflow.pytorch.log_model(model, "unet_model")
    # print("✓ Model saved to MLflow")

# ======================================================================
# Code Block 7
# ======================================================================

model.eval()
predictions = []
ground_truth = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        
        predictions.append(outputs.cpu().numpy())
        ground_truth.append(batch_y.numpy())

predictions = np.concatenate(predictions, axis=0)  # (100, 1, 256, 256)
ground_truth = np.concatenate(ground_truth, axis=0)

# print(f"✓ Inference complete: {predictions.shape[0]} samples")

# Compute metrics
pred_binary = (predictions > 0.5).astype(np.float32)
accuracy = (pred_binary == ground_truth).mean()
precision = (pred_binary * ground_truth).sum() / (pred_binary.sum() + 1e-9)
recall = (pred_binary * ground_truth).sum() / (ground_truth.sum() + 1e-9)
f1 = 2 * precision * recall / (precision + recall + 1e-9)

print(f"\nMetrics:")
print(f"  Accuracy: {accuracy:.3f}")
print(f"  Precision: {precision:.3f}")
print(f"  Recall: {recall:.3f}")
print(f"  F1 Score: {f1:.3f}")

# ======================================================================
# Code Block 8
# ======================================================================

# Complete tailings dam monitoring pipeline with U-Net

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import mlflow
import mlflow.pytorch

# ============================================================================
# 1. U-Net Model
# ============================================================================

class UNet(nn.Module):
    def __init__(self, in_channels=8, out_channels=1):
        super(UNet, self).__init__()
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.bottleneck = self.conv_block(256, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = self.conv_block(128, 64)
        self.out = nn.Conv2d(64, out_channels, 1)
        
    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        bottleneck = self.bottleneck(F.max_pool2d(enc3, 2))
        dec3 = self.dec3(torch.cat([self.upconv3(bottleneck), enc3], 1))
        dec2 = self.dec2(torch.cat([self.upconv2(dec3), enc2], 1))
        dec1 = self.dec1(torch.cat([self.upconv1(dec2), enc1], 1))
        return torch.sigmoid(self.out(dec1))

# ============================================================================
# 2. Synthetic Data Generation
# ============================================================================

def generate_synthetic_data(n_samples=500):
    np.random.seed(42)
    size = 256
    X_before, X_after, y_change = [], [], []
    
    for i in range(n_samples):
        has_change = (i % 5 == 0)
        before = np.random.uniform(0.1, 0.3, (size, size, 4))
        after = before + np.random.normal(0, 0.02, (size, size, 4))
        mask = np.zeros((size, size))
        
        if has_change:
            y0, x0 = size // 2, size // 2
            for dy in range(10):
                for dx in range(15):
                    if 0 <= y0+dy < size and 0 <= x0+dx < size:
                        after[y0+dy, x0+dx, :] *= 0.7
                        mask[y0+dy, x0+dx] = 1.0
        
        X_before.append(before)
        X_after.append(after)
        y_change.append(mask)
    
    return np.array(X_before), np.array(X_after), np.array(y_change)

# ============================================================================
# 3. Training
# ============================================================================

X_before, X_after, y_change = generate_synthetic_data(500)

class ChangeDataset(Dataset):
    def __init__(self, X_b, X_a, y):
        self.X_b = torch.from_numpy(X_b).float().permute(0, 3, 1, 2)
        self.X_a = torch.from_numpy(X_a).float().permute(0, 3, 1, 2)
        self.y = torch.from_numpy(y).float().unsqueeze(1)
    def __len__(self):
        return len(self.X_b)
    def __getitem__(self, idx):
        return torch.cat([self.X_b[idx], self.X_a[idx]], 0), self.y[idx]

dataset = ChangeDataset(X_before, X_after, y_change)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(20):
    model.train()
    loss_sum = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    print(f"Epoch {epoch+1}: Loss={loss_sum/len(train_loader):.4f}")

# print("✓ Training complete")

# ======================================================================
# Code Block 9
# ======================================================================

# (tailings.latitude.isNotNull()) & 
# (tailings.longitude.isNotNull()) &
(tailings.status == "Active")

# ======================================================================
# Code Block 10
# ======================================================================

blue = np.random.uniform(0.05, 0.15, (size, size))
green = np.random.uniform(0.10, 0.25, (size, size))
red = np.random.uniform(0.08, 0.20, (size, size))
nir = np.random.uniform(0.20, 0.50, (size, size))

# ======================================================================
# Code Block 11
# ======================================================================

y, x = np.ogrid[:size, :size]
center_y, center_x = size // 2, size // 2
mask_pond = ((x - center_x)**2 + (y - center_y)**2) < (size // 4)**2

# ======================================================================
# Code Block 12
# ======================================================================

nir[mask_pond] = np.random.uniform(0.02, 0.08, mask_pond.sum())
blue[mask_pond] += 0.05

# ======================================================================
# Code Block 13
# ======================================================================

mask_embankment = (x > size * 0.7) & (y > size * 0.3) & (y < size * 0.7)
nir[mask_embankment] = np.random.uniform(0.15, 0.30, mask_embankment.sum())
red[mask_embankment] += 0.08

image = np.stack([blue, green, red, nir], axis=-1)  # Shape: (256, 256, 4)

# ======================================================================
# Code Block 14
# ======================================================================

change_mask = np.zeros((size, size), dtype=np.float32)

if has_change:
    pass

# ======================================================================
# Code Block 15
# ======================================================================

seepage_y = int(size * 0.65)
seepage_x = int(size * 0.75)
for dy in range(10):
        for dx in range(15):
            y_pos = seepage_y + dy
            x_pos = seepage_x + dx
            if 0 <= y_pos < size and 0 <= x_pos < size:
                nir[y_pos, x_pos] *= 0.7  # Darken (moisture)
                change_mask[y_pos, x_pos] = 1.0

return image, change_mask

# ======================================================================
# Code Block 16
# ======================================================================

has_change = (i % 5 == 0)  # 20% change rate

before, _ = generate_synthetic_sentinel2_patch(has_change=False)
after, mask = generate_synthetic_sentinel2_patch(has_change=has_change)

X_before.append(before)
X_after.append(after)
y_change.append(mask)

# ======================================================================
# Code Block 17
# ======================================================================

def __init__(self, in_channels=8, out_channels=1):
    """
    U-Net for change detection.
    
    Parameters:
    -----------
    in_channels : int
        Number of input channels (8 = before + after, each 4 bands)
    out_channels : int
        Number of output channels (1 = change probability)
    """
    super(UNet, self).__init__()

# ======================================================================
# Code Block 18
# ======================================================================

self.enc1 = self.conv_block(in_channels, 64)
self.enc2 = self.conv_block(64, 128)
self.enc3 = self.conv_block(128, 256)

# ======================================================================
# Code Block 19
# ======================================================================

self.bottleneck = self.conv_block(256, 512)

# ======================================================================
# Code Block 20
# ======================================================================

self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
self.dec3 = self.conv_block(512, 256)  # 512 = 256 (upconv) + 256 (skip)
    
self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
self.dec2 = self.conv_block(256, 128)
    
self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
self.dec1 = self.conv_block(128, 64)

# ======================================================================
# Code Block 21
# ======================================================================

self.out = nn.Conv2d(64, out_channels, kernel_size=1)
    
def conv_block(self, in_channels, out_channels):
    """Double convolution block."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )
    
def forward(self, x):
    pass

# ======================================================================
# Code Block 22
# ======================================================================

enc1 = self.enc1(x)
enc2 = self.enc2(F.max_pool2d(enc1, 2))
enc3 = self.enc3(F.max_pool2d(enc2, 2))

# ======================================================================
# Code Block 23
# ======================================================================

bottleneck = self.bottleneck(F.max_pool2d(enc3, 2))

# ======================================================================
# Code Block 24
# ======================================================================

dec3 = self.upconv3(bottleneck)
dec3 = torch.cat([dec3, enc3], dim=1)
dec3 = self.dec3(dec3)
    
dec2 = self.upconv2(dec3)
dec2 = torch.cat([dec2, enc2], dim=1)
dec2 = self.dec2(dec2)
    
dec1 = self.upconv1(dec2)
dec1 = torch.cat([dec1, enc1], dim=1)
dec1 = self.dec1(dec1)

# ======================================================================
# Code Block 25
# ======================================================================

return torch.sigmoid(self.out(dec1))

# ======================================================================
# Code Block 26
# ======================================================================

def __init__(self, X_before, X_after, y_change):
    self.X_before = torch.from_numpy(X_before).float().permute(0, 3, 1, 2)  # (N, C, H, W)
    self.X_after = torch.from_numpy(X_after).float().permute(0, 3, 1, 2)
    self.y_change = torch.from_numpy(y_change).float().unsqueeze(1)  # (N, 1, H, W)
    
def __len__(self):
    return len(self.X_before)

def __getitem__(self, idx):
    pass

# ======================================================================
# Code Block 27
# ======================================================================

x = torch.cat([self.X_before[idx], self.X_after[idx]], dim=0)  # (8, H, W)
y = self.y_change[idx]  # (1, H, W)
return x, y

# ======================================================================
# Code Block 28
# ======================================================================

mlflow.log_param("epochs", n_epochs)
mlflow.log_param("batch_size", 8)
mlflow.log_param("learning_rate", 1e-4)

for epoch in range(n_epochs):
    model.train()
    train_loss = 0.0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

# ======================================================================
# Code Block 29
# ======================================================================

outputs = model(batch_x)
loss = criterion(outputs, batch_y)

# ======================================================================
# Code Block 30
# ======================================================================

optimizer.zero_grad()
loss.backward()
optimizer.step()
        
train_loss += loss.item()
    
avg_train_loss = train_loss / len(train_loader)

# ======================================================================
# Code Block 31
# ======================================================================

model.eval()
test_loss = 0.0
with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()
    
avg_test_loss = test_loss / len(test_loader)
    
print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
    
mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
mlflow.log_metric("test_loss", avg_test_loss, step=epoch)

# ======================================================================
# Code Block 32
# ======================================================================

mlflow.pytorch.log_model(model, "unet_model")
# print("✓ Model saved to MLflow")

# ======================================================================
# Code Block 33
# ======================================================================

for batch_x, batch_y in test_loader:
    batch_x = batch_x.to(device)
    outputs = model(batch_x)
    
    predictions.append(outputs.cpu().numpy())
    ground_truth.append(batch_y.numpy())

# ======================================================================
# Code Block 34
# ======================================================================

def __init__(self, in_channels=8, out_channels=1):
    super(UNet, self).__init__()
    self.enc1 = self.conv_block(in_channels, 64)
    self.enc2 = self.conv_block(64, 128)
    self.enc3 = self.conv_block(128, 256)
    self.bottleneck = self.conv_block(256, 512)
    self.upconv3 = nn.ConvTranspose2d(512, 256, 2, 2)
    self.dec3 = self.conv_block(512, 256)
    self.upconv2 = nn.ConvTranspose2d(256, 128, 2, 2)
    self.dec2 = self.conv_block(256, 128)
    self.upconv1 = nn.ConvTranspose2d(128, 64, 2, 2)
    self.dec1 = self.conv_block(128, 64)
    self.out = nn.Conv2d(64, out_channels, 1)
    
def conv_block(self, in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.ReLU(inplace=True)
    )
    
def forward(self, x):
    enc1 = self.enc1(x)
    enc2 = self.enc2(F.max_pool2d(enc1, 2))
    enc3 = self.enc3(F.max_pool2d(enc2, 2))
    bottleneck = self.bottleneck(F.max_pool2d(enc3, 2))
    dec3 = self.dec3(torch.cat([self.upconv3(bottleneck), enc3], 1))
    dec2 = self.dec2(torch.cat([self.upconv2(dec3), enc2], 1))
    dec1 = self.dec1(torch.cat([self.upconv1(dec2), enc1], 1))
    return torch.sigmoid(self.out(dec1))

# ======================================================================
# Code Block 35
# ======================================================================

np.random.seed(42)
size = 256
X_before, X_after, y_change = [], [], []

for i in range(n_samples):
    has_change = (i % 5 == 0)
    before = np.random.uniform(0.1, 0.3, (size, size, 4))
    after = before + np.random.normal(0, 0.02, (size, size, 4))
    mask = np.zeros((size, size))
    
    if has_change:
        y0, x0 = size // 2, size // 2
        for dy in range(10):
            for dx in range(15):
                if 0 <= y0+dy < size and 0 <= x0+dx < size:
                    after[y0+dy, x0+dx, :] *= 0.7
                    mask[y0+dy, x0+dx] = 1.0
    
    X_before.append(before)
    X_after.append(after)
    y_change.append(mask)

return np.array(X_before), np.array(X_after), np.array(y_change)

# ======================================================================
# Code Block 36
# ======================================================================

def __init__(self, X_b, X_a, y):
    self.X_b = torch.from_numpy(X_b).float().permute(0, 3, 1, 2)
    self.X_a = torch.from_numpy(X_a).float().permute(0, 3, 1, 2)
    self.y = torch.from_numpy(y).float().unsqueeze(1)
def __len__(self):
    return len(self.X_b)
def __getitem__(self, idx):
    return torch.cat([self.X_b[idx], self.X_a[idx]], 0), self.y[idx]

# ======================================================================
# Code Block 37
# ======================================================================

model.train()
loss_sum = 0.0
for batch_x, batch_y in train_loader:
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    outputs = model(batch_x)
    loss = criterion(outputs, batch_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_sum += loss.item()
print(f"Epoch {epoch+1}: Loss={loss_sum/len(train_loader):.4f}")

# ======================================================================
# Code Block 38
# ======================================================================

# \$15M response = 478× ROI.

# ======================================================================
# Code Block 39
# ======================================================================

# = 26,645 images in \<8 hours.
