# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm
import os

# Import your model from the other file
from model import FacePADModel

# --- 1. HYPERPARAMETERS & CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = 'data/'
MODEL_SAVE_PATH = 'saved_models/'
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 20
BACKBONE = 'resnet50' # or 'resnet18'
WEIGHT_DECAY = 1e-4

# --- 2. DATA LOADING & TRANSFORMATION ---
# Use strong augmentation for training, simple for validation
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Use ImageFolder to load data
train_dataset = ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transforms)
val_dataset = ImageFolder(os.path.join(DATA_DIR, 'val'), transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

print(f"Found {len(train_dataset)} training images and {len(val_dataset)} validation images.")
print(f"Classes: {train_dataset.classes}") # Should be ['attack', 'real'] or similar

# --- 3. MODEL, OPTIMIZER, LOSS FUNCTION ---
model = FacePADModel(backbone_name=BACKBONE, pretrained=True).to(DEVICE)

# Optional: Implement two-stage fine-tuning
# model.freeze_backbone() # Freeze for the first few epochs if dataset is small

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss() # This is standard for classification
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3, verbose=True)

# --- 4. TRAINING & VALIDATION LOOP ---
best_val_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    # --- Training Phase ---
    model.train()
    train_loss = 0.0
    train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]")
    for inputs, labels in train_loop:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_loop.set_postfix(loss=loss.item())

    # --- Validation Phase ---
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]")
    with torch.no_grad():
        for inputs, labels in val_loop:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # --- Log and Save ---
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    
    print(f"Epoch {epoch+1} Summary: "
          f"Train Loss: {avg_train_loss:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, "
          f"Val Accuracy: {val_accuracy:.2f}%")

    scheduler.step(avg_val_loss) # Update LR based on validation loss

    if avg_val_loss < best_val_loss:
        print(f"Validation loss decreased ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving model...")
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, 'best_model.pth'))

print("Training finished!")