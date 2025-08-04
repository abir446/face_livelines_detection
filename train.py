# train.py (Corrected for BatchNorm with last batch)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm
import os

# Import your custom model from model.py
from model import FacePADModel

# --- 1. CONFIGURATION & HYPERPARAMETERS ---
DATA_DIR = 'data/real_and_fake_face/'  # <<< MAKE SURE THIS PATH IS CORRECT!
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = 'saved_models/'
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 20
BACKBONE = 'resnet50'
WEIGHT_DECAY = 1e-4
VALIDATION_SPLIT = 0.2

def main():
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    pin_memory = True if DEVICE.type == 'cuda' else False

    # --- 2. DATA LOADING & TRANSFORMATION ---
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

    print(f"Loading data from: {DATA_DIR}")
    full_dataset = ImageFolder(DATA_DIR, transform=val_transforms)

    total_size = len(full_dataset)
    val_size = int(total_size * VALIDATION_SPLIT)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_subset.dataset.transform = train_transforms
    val_subset.dataset.transform = val_transforms

    # <<< CHANGE APPLIED HERE: Added drop_last=True to the train_loader
    # This prevents batches of size 1, which would crash the BatchNorm layer.
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=pin_memory, drop_last=True)
    
    # It's not necessary for the validation loader, but doesn't hurt.
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=pin_memory)

    print("\n--- Data Summary ---")
    print(f"Total samples: {total_size}")
    print(f"Training samples: {len(train_subset)}")
    print(f"Validation samples: {len(val_subset)}")
    print(f"Classes found: {full_dataset.classes}\n")

    # --- 3. MODEL, OPTIMIZER, LOSS FUNCTION ---
    print("--- Model Setup ---")
    print(f"Using device: {DEVICE}")
    print(f"Using backbone: {BACKBONE}")
    model = FacePADModel(backbone_name=BACKBONE, pretrained=True).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # --- 4. TRAINING & VALIDATION LOOP ---
    best_val_loss = float('inf')
    print("\n--- Starting Training ---")
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]", leave=False)
        for inputs, labels in train_loop:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]", leave=False)
        with torch.no_grad():
            for inputs, labels in val_loop:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total

        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} Summary: "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Accuracy: {val_accuracy:.2f}%")
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            print(f"Validation loss decreased ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving model...")
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, 'best_model.pth'))
        print("-" * 50)

    print("--- Training finished! ---")
    print(f"Best model saved to {os.path.join(MODEL_SAVE_PATH, 'best_model.pth')}")

if __name__ == '__main__':
    main()