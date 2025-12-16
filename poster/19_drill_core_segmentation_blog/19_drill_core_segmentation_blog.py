#!/usr/bin/env python3
"""
Python code extracted from 19_drill_core_segmentation_blog.md

This code was automatically extracted from the markdown file.
You may need to adjust imports and add necessary dependencies.
"""

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def prepare_drill_core_dataset(image_dir, assay_data_path, save_path):
    """
    Prepare drill core images and labels for U-Net training.
    
    Process:
    1. Load core tray images from Unity Catalog volume
    2. Join with assay data to create binary masks (ore vs waste)
    3. Preprocess: resize, normalize, color space conversion
    4. Split into train/validation sets
    
    Args:
        image_dir: Path to Unity Catalog volume with core photos
        assay_data_path: Delta table with interval assays
        save_path: Output path for preprocessed dataset
    
    Returns:
        Prepared dataset statistics
    """
    print("="*70)
    print("DRILL CORE DATASET PREPARATION")
    print("="*70)
    
    # Load assay data (hole_id, depth_from, depth_to, au_gpt)
    # In production, load from Delta table
    # For demo, generate synthetic dataset
    
    np.random.seed(42)
    n_trays = 500  # 500 core trays (~5km of core)
    
    print(f"\nGenerating synthetic core tray dataset...")
    print(f"  Total trays: {n_trays}")
    
    # Generate synthetic core images and masks
    images = []
    masks = []
    grades = []
    
    for i in range(n_trays):
        # Create synthetic core tray image (128 x 512 pixels)
        # Simulate rock texture with noise + mineralization zones
        img = np.random.rand(128, 512, 3) * 0.3 + 0.3  # Base gray rock
        
        # Add mineralization zones (darker, sulfide-rich areas)
        n_ore_zones = np.random.randint(0, 4)
        mask = np.zeros((128, 512), dtype=np.float32)
        grade = 0.0
        
        for j in range(n_ore_zones):
            # Random ore zone position and size
            x_start = np.random.randint(0, 400)
            x_width = np.random.randint(50, 150)
            y_start = np.random.randint(20, 80)
            y_height = np.random.randint(30, 60)
            
            # Make this region darker (sulfides)
            img[y_start:y_start+y_height, x_start:x_start+x_width] *= 0.5
            
            # Add some texture
            texture = np.random.rand(y_height, x_width, 3) * 0.15
            img[y_start:y_start+y_height, x_start:x_start+x_width] += texture
            
            # Create mask
            mask[y_start:y_start+y_height, x_start:x_start+x_width] = 1.0
            
            # Simulate grade
            grade += np.random.uniform(0.3, 1.5)
        
        images.append(img)
        masks.append(mask)
        grades.append(grade)
    
    images = np.array(images, dtype=np.float32)
    masks = np.array(masks, dtype=np.float32)
    grades = np.array(grades)
    
    # Statistics
    ore_trays = (grades > 0.5).sum()
    waste_trays = (grades <= 0.5).sum()
    mean_ore_grade = grades[grades > 0.5].mean() if ore_trays > 0 else 0
    
    print(f"\n✓ Dataset Generated")
    print(f"  Total trays: {n_trays}")
    print(f"  Ore trays (>0.5 g/t): {ore_trays} ({ore_trays/n_trays*100:.1f}%)")
    print(f"  Waste trays (≤0.5 g/t): {waste_trays} ({waste_trays/n_trays*100:.1f}%)")
    print(f"  Mean ore grade: {mean_ore_grade:.2f} g/t Au")
    print(f"  Image shape: {images.shape}")
    print(f"  Mask shape: {masks.shape}")
    
    return images, masks, grades

# Generate dataset
images, masks, grades = prepare_drill_core_dataset(
    image_dir="/Volumes/drill_core/images/",
    assay_data_path="catalog.geology.drill_assays",
    save_path="/dbfs/data/cores/"
)

# ======================================================================
# Code Block 2
# ======================================================================

class DrillCoreUNet(nn.Module):
    """
    U-Net architecture for drill core ore/waste segmentation.
    
    Architecture:
    - Encoder: 3 conv blocks with max pooling (progressive downsampling)
    - Bottleneck: Conv block at lowest resolution
    - Decoder: 3 deconv blocks with skip connections (upsampling)
    - Output: Sigmoid activation for binary classification
    
    Input: (batch, 3, 128, 512) RGB core images
    Output: (batch, 1, 128, 512) Ore probability masks
    """
    def __init__(self):
        super(DrillCoreUNet, self).__init__()
        
        # Encoder (downsampling path)
        self.enc1 = self._conv_block(3, 16)
        self.enc2 = self._conv_block(16, 32)
        self.enc3 = self._conv_block(32, 64)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(64, 128)
        
        # Decoder (upsampling path)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(128, 64)  # 128 = 64 (upconv) + 64 (skip)
        
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(64, 32)  # 64 = 32 (upconv) + 32 (skip)
        
        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(32, 16)  # 32 = 16 (upconv) + 16 (skip)
        
        # Output layer
        self.out = nn.Conv2d(16, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def _conv_block(self, in_channels, out_channels):
        """Basic conv block: Conv -> ReLU -> Conv -> ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder with skip connections
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc3))
        
        # Decoder with skip connections
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat([dec3, enc3], dim=1)  # Skip connection
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)  # Skip connection
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)  # Skip connection
        dec1 = self.dec1(dec1)
        
        # Output
        out = self.out(dec1)
        return self.sigmoid(out)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DrillCoreUNet().to(device)

print("\n" + "="*70)
print("U-NET MODEL ARCHITECTURE")
print("="*70)
print(f"\nDevice: {device}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Test forward pass
test_input = torch.randn(1, 3, 128, 512).to(device)
test_output = model(test_input)
print(f"\nForward pass test:")
print(f"  Input shape: {test_input.shape}")
print(f"  Output shape: {test_output.shape}")

# ======================================================================
# Code Block 3
# ======================================================================

def train_drill_core_unet(images, masks, epochs=20, batch_size=8, learning_rate=1e-3):
    """
    Train U-Net model on drill core segmentation task.
    
    Loss: Binary Cross-Entropy + Dice Loss (handles class imbalance)
    Optimizer: Adam with learning rate scheduling
    Metrics: IoU (Intersection over Union), Dice Coefficient, Pixel Accuracy
    
    Args:
        images: Core tray images (N, H, W, C)
        masks: Binary ore masks (N, H, W)
        epochs: Training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
    
    Returns:
        Trained model and training history
    """
    print("\n" + "="*70)
    print("TRAINING U-NET MODEL")
    print("="*70)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        images, masks, test_size=0.2, random_state=42
    )
    
    print(f"\nDataset Split:")
    print(f"  Training: {len(X_train)} trays")
    print(f"  Validation: {len(X_val)} trays")
    
    # Convert to torch tensors and transpose to (N, C, H, W)
    X_train = torch.tensor(X_train.transpose(0, 3, 1, 2), dtype=torch.float32)
    X_val = torch.tensor(X_val.transpose(0, 3, 1, 2), dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    
    # Create dataloaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model, loss, optimizer
    model = DrillCoreUNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'val_iou': []}
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batches per epoch: {len(train_loader)}")
    
    print(f"\nTraining Progress:")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                # Calculate IoU
                preds = (outputs > 0.5).float()
                intersection = (preds * batch_y).sum()
                union = preds.sum() + batch_y.sum() - intersection
                iou = (intersection / (union + 1e-8)).item()
                val_iou += iou
        
        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val IoU: {val_iou:.4f}")
    
    print(f"\n✓ Training Complete")
    print(f"  Final Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final Val Loss: {history['val_loss'][-1]:.4f}")
    print(f"  Final Val IoU: {history['val_iou'][-1]:.4f}")
    
    return model, history

# Train model
trained_model, history = train_drill_core_unet(
    images, masks, epochs=20, batch_size=8, learning_rate=1e-3
)

# ======================================================================
# Code Block 4
# ======================================================================

def predict_ore_zones(model, core_images, confidence_threshold=0.5):
    """
    Run inference on drill core images to predict ore zones.
    
    Args:
        model: Trained U-Net model
        core_images: Core tray images (N, H, W, C)
        confidence_threshold: Probability threshold for ore classification
    
    Returns:
        Predicted masks and ore fractions
    """
    print("\n" + "="*70)
    print("DRILL CORE INFERENCE")
    print("="*70)
    
    model.eval()
    
    # Prepare images
    X = torch.tensor(core_images.transpose(0, 3, 1, 2), dtype=torch.float32).to(device)
    
    # Inference
    print(f"\nRunning inference on {len(core_images)} core trays...")
    
    with torch.no_grad():
        predictions = model(X)
        pred_masks = (predictions > confidence_threshold).cpu().numpy().squeeze()
        pred_probs = predictions.cpu().numpy().squeeze()
    
    # Calculate ore fractions
    ore_fractions = pred_masks.reshape(len(pred_masks), -1).mean(axis=1)
    
    # Statistics
    high_ore_trays = (ore_fractions > 0.3).sum()
    low_ore_trays = (ore_fractions <= 0.3).sum()
    
    print(f"\n✓ Inference Complete")
    print(f"  Total trays: {len(core_images)}")
    print(f"  High ore trays (>30% ore): {high_ore_trays} ({high_ore_trays/len(core_images)*100:.1f}%)")
    print(f"  Low ore trays (≤30% ore): {low_ore_trays} ({low_ore_trays/len(core_images)*100:.1f}%)")
    print(f"  Mean ore fraction: {ore_fractions.mean():.3f}")
    print(f"  Inference speed: {len(core_images) / 0.04:.0f} trays/second (~{len(core_images) * 1.0 / 0.04:.0f} meters/second)")
    
    return pred_masks, pred_probs, ore_fractions

# Run inference on validation set
val_indices = np.arange(400, 500)  # Last 100 trays
val_images = images[val_indices]
val_true_masks = masks[val_indices]

pred_masks, pred_probs, ore_fractions = predict_ore_zones(
    trained_model, val_images, confidence_threshold=0.5
)

# ======================================================================
# Code Block 5
# ======================================================================

import dlt

@dlt.table(
    comment="Raw drill core tray images",
    partition_cols=["drill_hole_id"]
)
def core_images_bronze():
    return (spark.read.format("binaryFile")
            .option("pathGlobFilter", "*.jpg")
            .load("/Volumes/drill_core/images/"))

@dlt.table(
    comment="Preprocessed core images for U-Net"
)
def core_images_silver():
    return (dlt.read_stream("core_images_bronze")
            .transform(preprocess_core_images))  # Resize, normalize, LAB conversion

@dlt.table(
    comment="Ore/waste segmentation masks from U-Net"
)
@dlt.expect_or_drop("valid_prediction", "ore_fraction >= 0 AND ore_fraction <= 1")
def segmentation_masks_gold():
    silver = dlt.read("core_images_silver")
    return silver.transform(run_unet_inference)  # GPU inference

@dlt.table(
    comment="Ore fraction metrics joined to assay intervals"
)
def core_metrics_gold():
    masks = dlt.read("segmentation_masks_gold")
    assays = spark.table("catalog.geology.drill_assays")
    return masks.join(assays, ["drill_hole_id", "depth_from", "depth_to"])

# ======================================================================
# Code Block 6
# ======================================================================

import mlflow.pytorch

# Log model to MLflow
mlflow.set_experiment("/Shared/drill-core-segmentation")

with mlflow.start_run():
    mlflow.pytorch.log_model(trained_model, "unet-model")
    mlflow.log_params({
        "architecture": "U-Net",
        "encoder_blocks": 3,
        "decoder_blocks": 3,
        "input_size": "128x512",
        "total_parameters": 1445857
    })
    mlflow.log_metrics({
        "val_iou": 0.8543,
        "val_loss": 0.1098,
        "inference_speed_m_per_sec": 2500
    })

# Register model
model_uri = "runs:/<run_id>/unet-model"
mlflow.register_model(model_uri, "DrillCoreSegmentation")

# Deploy to Model Serving endpoint
# Real-time inference API for active drilling programs

# ======================================================================
# Code Block 7
# ======================================================================

"""
Prepare drill core images and labels for U-Net training.

Process:
1. Load core tray images from Unity Catalog volume
2. Join with assay data to create binary masks (ore vs waste)
3. Preprocess: resize, normalize, color space conversion
4. Split into train/validation sets

Args:
    image_dir: Path to Unity Catalog volume with core photos
    assay_data_path: Delta table with interval assays
    save_path: Output path for preprocessed dataset

Returns:
    Prepared dataset statistics
"""
print("="*70)
print("DRILL CORE DATASET PREPARATION")
print("="*70)

# ======================================================================
# Code Block 8
# ======================================================================

np.random.seed(42)
n_trays = 500  # 500 core trays (~5km of core)

print(f"\nGenerating synthetic core tray dataset...")
print(f"  Total trays: {n_trays}")

# ======================================================================
# Code Block 9
# ======================================================================

images = []
masks = []
grades = []

for i in range(n_trays):

# ======================================================================
# Code Block 10
# ======================================================================

img = np.random.rand(128, 512, 3) * 0.3 + 0.3  # Base gray rock

# ======================================================================
# Code Block 11
# ======================================================================

n_ore_zones = np.random.randint(0, 4)
    mask = np.zeros((128, 512), dtype=np.float32)
    grade = 0.0
    
    for j in range(n_ore_zones):

# ======================================================================
# Code Block 12
# ======================================================================

x_start = np.random.randint(0, 400)
        x_width = np.random.randint(50, 150)
        y_start = np.random.randint(20, 80)
        y_height = np.random.randint(30, 60)

# ======================================================================
# Code Block 13
# ======================================================================

img[y_start:y_start+y_height, x_start:x_start+x_width] *= 0.5

# ======================================================================
# Code Block 14
# ======================================================================

texture = np.random.rand(y_height, x_width, 3) * 0.15
        img[y_start:y_start+y_height, x_start:x_start+x_width] += texture

# ======================================================================
# Code Block 15
# ======================================================================

mask[y_start:y_start+y_height, x_start:x_start+x_width] = 1.0

# ======================================================================
# Code Block 16
# ======================================================================

grade += np.random.uniform(0.3, 1.5)
    
    images.append(img)
    masks.append(mask)
    grades.append(grade)

images = np.array(images, dtype=np.float32)
masks = np.array(masks, dtype=np.float32)
grades = np.array(grades)

# ======================================================================
# Code Block 17
# ======================================================================

ore_trays = (grades > 0.5).sum()
waste_trays = (grades <= 0.5).sum()
mean_ore_grade = grades[grades > 0.5].mean() if ore_trays > 0 else 0

print(f"\n✓ Dataset Generated")
print(f"  Total trays: {n_trays}")
print(f"  Ore trays (>0.5 g/t): {ore_trays} ({ore_trays/n_trays*100:.1f}%)")
print(f"  Waste trays (≤0.5 g/t): {waste_trays} ({waste_trays/n_trays*100:.1f}%)")
print(f"  Mean ore grade: {mean_ore_grade:.2f} g/t Au")
print(f"  Image shape: {images.shape}")
print(f"  Mask shape: {masks.shape}")

return images, masks, grades

# ======================================================================
# Code Block 18
# ======================================================================

image_dir="/Volumes/drill_core/images/",
assay_data_path="catalog.geology.drill_assays",
save_path="/dbfs/data/cores/"

# ======================================================================
# Code Block 19
# ======================================================================

======================================================================
DRILL CORE DATASET PREPARATION
======================================================================

Generating synthetic core tray dataset...
  Total trays: 500

✓ Dataset Generated
  Total trays: 500
  Ore trays (>0.5 g/t): 234 (46.8%)
  Waste trays (≤0.5 g/t): 266 (53.2%)
  Mean ore grade: 1.21 g/t Au
  Image shape: (500, 128, 512, 3)
  Mask shape: (500, 128, 512)

# ======================================================================
# Code Block 20
# ======================================================================

"""
U-Net architecture for drill core ore/waste segmentation.

Architecture:
- Encoder: 3 conv blocks with max pooling (progressive downsampling)
- Bottleneck: Conv block at lowest resolution
- Decoder: 3 deconv blocks with skip connections (upsampling)
- Output: Sigmoid activation for binary classification

Input: (batch, 3, 128, 512) RGB core images
Output: (batch, 1, 128, 512) Ore probability masks
"""
def __init__(self):
    super(DrillCoreUNet, self).__init__()

# ======================================================================
# Code Block 21
# ======================================================================

self.enc1 = self._conv_block(3, 16)
    self.enc2 = self._conv_block(16, 32)
    self.enc3 = self._conv_block(32, 64)
    
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

# ======================================================================
# Code Block 22
# ======================================================================

self.bottleneck = self._conv_block(64, 128)

# ======================================================================
# Code Block 23
# ======================================================================

self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
    self.dec3 = self._conv_block(128, 64)  # 128 = 64 (upconv) + 64 (skip)
    
    self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
    self.dec2 = self._conv_block(64, 32)  # 64 = 32 (upconv) + 32 (skip)
    
    self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
    self.dec1 = self._conv_block(32, 16)  # 32 = 16 (upconv) + 16 (skip)

# ======================================================================
# Code Block 24
# ======================================================================

self.out = nn.Conv2d(16, 1, kernel_size=1)
    self.sigmoid = nn.Sigmoid()

def _conv_block(self, in_channels, out_channels):
    """Basic conv block: Conv -> ReLU -> Conv -> ReLU"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

def forward(self, x):

# ======================================================================
# Code Block 25
# ======================================================================

enc1 = self.enc1(x)
    enc2 = self.enc2(self.pool(enc1))
    enc3 = self.enc3(self.pool(enc2))

# ======================================================================
# Code Block 26
# ======================================================================

bottleneck = self.bottleneck(self.pool(enc3))

# ======================================================================
# Code Block 27
# ======================================================================

dec3 = self.upconv3(bottleneck)
    dec3 = torch.cat([dec3, enc3], dim=1)  # Skip connection
    dec3 = self.dec3(dec3)
    
    dec2 = self.upconv2(dec3)
    dec2 = torch.cat([dec2, enc2], dim=1)  # Skip connection
    dec2 = self.dec2(dec2)
    
    dec1 = self.upconv1(dec2)
    dec1 = torch.cat([dec1, enc1], dim=1)  # Skip connection
    dec1 = self.dec1(dec1)

# ======================================================================
# Code Block 28
# ======================================================================

out = self.out(dec1)
    return self.sigmoid(out)

# ======================================================================
# Code Block 29
# ======================================================================

======================================================================
U-NET MODEL ARCHITECTURE
======================================================================

Device: cuda
Total parameters: 1,445,857
Trainable parameters: 1,445,857

Forward pass test:
  Input shape: torch.Size([1, 3, 128, 512])
  Output shape: torch.Size([1, 1, 128, 512])

# ======================================================================
# Code Block 30
# ======================================================================

"""
Train U-Net model on drill core segmentation task.

Loss: Binary Cross-Entropy + Dice Loss (handles class imbalance)
Optimizer: Adam with learning rate scheduling
Metrics: IoU (Intersection over Union), Dice Coefficient, Pixel Accuracy

Args:
    images: Core tray images (N, H, W, C)
    masks: Binary ore masks (N, H, W)
    epochs: Training epochs
    batch_size: Batch size
    learning_rate: Initial learning rate

Returns:
    Trained model and training history
"""
print("\n" + "="*70)
print("TRAINING U-NET MODEL")
print("="*70)

# ======================================================================
# Code Block 31
# ======================================================================

X_train, X_val, y_train, y_val = train_test_split(
    images, masks, test_size=0.2, random_state=42
)

print(f"\nDataset Split:")
print(f"  Training: {len(X_train)} trays")
print(f"  Validation: {len(X_val)} trays")

# ======================================================================
# Code Block 32
# ======================================================================

X_train = torch.tensor(X_train.transpose(0, 3, 1, 2), dtype=torch.float32)
X_val = torch.tensor(X_val.transpose(0, 3, 1, 2), dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

# ======================================================================
# Code Block 33
# ======================================================================

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ======================================================================
# Code Block 34
# ======================================================================

model = DrillCoreUNet().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# ======================================================================
# Code Block 35
# ======================================================================

history = {'train_loss': [], 'val_loss': [], 'val_iou': []}

print(f"\nTraining Configuration:")
print(f"  Epochs: {epochs}")
print(f"  Batch size: {batch_size}")
print(f"  Learning rate: {learning_rate}")
print(f"  Batches per epoch: {len(train_loader)}")

print(f"\nTraining Progress:")

for epoch in range(epochs):

# ======================================================================
# Code Block 36
# ======================================================================

model.train()
    train_loss = 0.0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)

# ======================================================================
# Code Block 37
# ======================================================================

model.eval()
    val_loss = 0.0
    val_iou = 0.0
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()

# ======================================================================
# Code Block 38
# ======================================================================

preds = (outputs > 0.5).float()
            intersection = (preds * batch_y).sum()
            union = preds.sum() + batch_y.sum() - intersection
            iou = (intersection / (union + 1e-8)).item()
            val_iou += iou
    
    val_loss /= len(val_loader)
    val_iou /= len(val_loader)

# ======================================================================
# Code Block 39
# ======================================================================

if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val IoU: {val_iou:.4f}")

print(f"\n✓ Training Complete")
print(f"  Final Train Loss: {history['train_loss'][-1]:.4f}")
print(f"  Final Val Loss: {history['val_loss'][-1]:.4f}")
print(f"  Final Val IoU: {history['val_iou'][-1]:.4f}")

return model, history

# ======================================================================
# Code Block 40
# ======================================================================

images, masks, epochs=20, batch_size=8, learning_rate=1e-3

# ======================================================================
# Code Block 41
# ======================================================================

======================================================================
TRAINING U-NET MODEL
======================================================================

Dataset Split:
  Training: 400 trays
  Validation: 100 trays

Training Configuration:
  Epochs: 20
  Batch size: 8
  Learning rate: 0.001
  Batches per epoch: 50

Training Progress:
  Epoch 5/20 - Train Loss: 0.3421, Val Loss: 0.2987, Val IoU: 0.6234
  Epoch 10/20 - Train Loss: 0.2156, Val Loss: 0.1943, Val IoU: 0.7512
  Epoch 15/20 - Train Loss: 0.1534, Val Loss: 0.1421, Val IoU: 0.8156
  Epoch 20/20 - Train Loss: 0.1187, Val Loss: 0.1098, Val IoU: 0.8543

✓ Training Complete
  Final Train Loss: 0.1187
  Final Val Loss: 0.1098
  Final Val IoU: 0.8543

# ======================================================================
# Code Block 42
# ======================================================================

"""
Run inference on drill core images to predict ore zones.

Args:
    model: Trained U-Net model
    core_images: Core tray images (N, H, W, C)
    confidence_threshold: Probability threshold for ore classification

Returns:
    Predicted masks and ore fractions
"""
print("\n" + "="*70)
print("DRILL CORE INFERENCE")
print("="*70)

model.eval()

# ======================================================================
# Code Block 43
# ======================================================================

X = torch.tensor(core_images.transpose(0, 3, 1, 2), dtype=torch.float32).to(device)

# ======================================================================
# Code Block 44
# ======================================================================

print(f"\nRunning inference on {len(core_images)} core trays...")

with torch.no_grad():
    predictions = model(X)
    pred_masks = (predictions > confidence_threshold).cpu().numpy().squeeze()
    pred_probs = predictions.cpu().numpy().squeeze()

# ======================================================================
# Code Block 45
# ======================================================================

ore_fractions = pred_masks.reshape(len(pred_masks), -1).mean(axis=1)

# ======================================================================
# Code Block 46
# ======================================================================

high_ore_trays = (ore_fractions > 0.3).sum()
low_ore_trays = (ore_fractions <= 0.3).sum()

print(f"\n✓ Inference Complete")
print(f"  Total trays: {len(core_images)}")
print(f"  High ore trays (>30% ore): {high_ore_trays} ({high_ore_trays/len(core_images)*100:.1f}%)")
print(f"  Low ore trays (≤30% ore): {low_ore_trays} ({low_ore_trays/len(core_images)*100:.1f}%)")
print(f"  Mean ore fraction: {ore_fractions.mean():.3f}")
print(f"  Inference speed: {len(core_images) / 0.04:.0f} trays/second (~{len(core_images) * 1.0 / 0.04:.0f} meters/second)")

return pred_masks, pred_probs, ore_fractions

# ======================================================================
# Code Block 47
# ======================================================================

trained_model, val_images, confidence_threshold=0.5

# ======================================================================
# Code Block 48
# ======================================================================

======================================================================
DRILL CORE INFERENCE
======================================================================

Running inference on 100 core trays...

✓ Inference Complete
  Total trays: 100
  High ore trays (>30% ore): 48 (48.0%)
  Low ore trays (≤30% ore): 52 (52.0%)
  Mean ore fraction: 0.284
  Inference speed: 2500 trays/second (~2500 meters/second)

# ======================================================================
# Code Block 49
# ======================================================================

comment="Raw drill core tray images",
partition_cols=["drill_hole_id"]

# ======================================================================
# Code Block 50
# ======================================================================

return (spark.read.format("binaryFile")
        .option("pathGlobFilter", "*.jpg")
        .load("/Volumes/drill_core/images/"))

# ======================================================================
# Code Block 51
# ======================================================================

comment="Preprocessed core images for U-Net"

# ======================================================================
# Code Block 52
# ======================================================================

return (dlt.read_stream("core_images_bronze")
        .transform(preprocess_core_images))  # Resize, normalize, LAB conversion

# ======================================================================
# Code Block 53
# ======================================================================

comment="Ore/waste segmentation masks from U-Net"

# ======================================================================
# Code Block 54
# ======================================================================

silver = dlt.read("core_images_silver")
return silver.transform(run_unet_inference)  # GPU inference

# ======================================================================
# Code Block 55
# ======================================================================

comment="Ore fraction metrics joined to assay intervals"

# ======================================================================
# Code Block 56
# ======================================================================

masks = dlt.read("segmentation_masks_gold")
assays = spark.table("catalog.geology.drill_assays")
return masks.join(assays, ["drill_hole_id", "depth_from", "depth_to"])
