#!/usr/bin/env python3
"""
Python code extracted from 33_solar_defect_detection_blog.md

This code was automatically extracted from the markdown file.
You may need to adjust imports and add necessary dependencies.
"""

def extract_module_patch(image, corners, margin=0.1):
    """Extract individual solar module from thermal image."""
    # Get bounding box from polygon corners
    xs = [c['x'] for c in corners]
    ys = [c['y'] for c in corners]
    
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    # Add margin for context
    width = x_max - x_min
    height = y_max - y_min
    
    x_min = max(0, x_min - margin * width)
    x_max = min(image.width, x_max + margin * width)
    y_min = max(0, y_min - margin * height)
    y_max = min(image.height, y_max + margin * height)
    
    return image.crop((x_min, y_min, x_max, y_max))

# ======================================================================
# Code Block 2
# ======================================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# ======================================================================
# Code Block 3
# ======================================================================

for epoch in range(num_epochs):
    # Training
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, device
    )
    
    # Validation
    val_loss, val_acc, preds, labels = validate(
        model, val_loader, criterion, device
    )
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')

# ======================================================================
# Code Block 4
# ======================================================================

# DATASET STATISTICS
# =====================================
# Total Images: 137
# Total Solar Modules: 5,469
#  Defective Modules: 90 (1.6%)
#  Healthy Modules: 5,379 (98.4%)
# Images with Defects: 40/137 (29.2%)
# Average Modules per Image: 39.9

# ======================================================================
# Code Block 5
# ======================================================================

# FINAL RESULTS
# =====================================
# Best Validation Accuracy: 97.4%
# Precision (Defective): 95.2%
# Recall (Defective): 93.8%
# F1-Score: 94.5%

# ======================================================================
# Code Block 6
# ======================================================================

xs = [c['x'] for c in corners]
ys = [c['y'] for c in corners]

x_min, x_max = min(xs), max(xs)
y_min, y_max = min(ys), max(ys)

# ======================================================================
# Code Block 7
# ======================================================================

width = x_max - x_min
height = y_max - y_min

x_min = max(0, x_min - margin * width)
x_max = min(image.width, x_max + margin * width)
y_min = max(0, y_min - margin * height)
y_max = min(image.height, y_max + margin * height)

return image.crop((x_min, y_min, x_max, y_max))

# ======================================================================
# Code Block 8
# ======================================================================

transforms.Resize((224, 224)),
transforms.RandomHorizontalFlip(),
transforms.RandomVerticalFlip(),
transforms.RandomRotation(10),
transforms.ColorJitter(brightness=0.2, contrast=0.2),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])

# ======================================================================
# Code Block 9
# ======================================================================

train_loss, train_acc = train_epoch(
    model, train_loader, criterion, optimizer, device
)

# ======================================================================
# Code Block 10
# ======================================================================

val_loss, val_acc, preds, labels = validate(
    model, val_loader, criterion, device
)

# ======================================================================
# Code Block 11
# ======================================================================

if val_acc > best_val_acc:
    best_val_acc = val_acc
    torch.save(model.state_dict(), 'best_model.pth')
