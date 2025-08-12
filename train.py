# enhanced_train.py - Improved Face PAD Training Script

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
import numpy as np
import random
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import your custom model
from model import FacePADModel

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.ce_loss = nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing)
    
    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class MixUpDataset:
    """MixUp augmentation wrapper"""
    def __init__(self, dataset, alpha=0.2):
        self.dataset = dataset
        self.alpha = alpha
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def mixup_data(self, x, y, alpha=0.2):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

class MetricsTracker:
    """Track and compute various metrics"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.targets = []
        self.probabilities = []
    
    def update(self, preds, targets, probs):
        self.predictions.extend(preds.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        self.probabilities.extend(probs.cpu().numpy())
    
    def compute_metrics(self):
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        probabilities = np.array(self.probabilities)
        
        # Basic metrics
        accuracy = (predictions == targets).mean()
        
        # AUC-ROC
        try:
            auc = roc_auc_score(targets, probabilities[:, 1])
        except:
            auc = 0.5
        
        # PAD-specific metrics (APCER, BPCER, EER)
        real_mask = targets == 1
        fake_mask = targets == 0
        
        # APCER: Attack Presentation Classification Error Rate
        fake_predictions = predictions[fake_mask]
        apcer = (fake_predictions == 1).mean() if len(fake_predictions) > 0 else 0
        
        # BPCER: Bona fide Presentation Classification Error Rate  
        real_predictions = predictions[real_mask]
        bpcer = (real_predictions == 0).mean() if len(real_predictions) > 0 else 0
        
        # Equal Error Rate (simplified approximation)
        eer = (apcer + bpcer) / 2
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'apcer': apcer,
            'bpcer': bpcer,
            'eer': eer
        }

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_advanced_transforms():
    """Enhanced data augmentation pipeline"""
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomRotation(15),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms

def create_optimizer_scheduler(model, train_loader_len):
    """Create optimizer with warmup and cosine annealing"""
    # Filter parameters that require gradients
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = optim.AdamW(
        trainable_params,
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    # Warmup + Cosine Annealing
    warmup_steps = CONFIG['warmup_epochs'] * train_loader_len
    total_steps = CONFIG['num_epochs'] * train_loader_len
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp loss computation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_epoch(model, train_loader, optimizer, criterion, scaler, scheduler, device, use_mixup=True):
    """Enhanced training epoch with MixUp and mixed precision"""
    model.train()
    running_loss = 0.0
    mixup_dataset = MixUpDataset(None, alpha=CONFIG['mixup_alpha'])
    
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Apply MixUp
        if use_mixup and random.random() < CONFIG['mixup_prob']:
            inputs, targets_a, targets_b, lam = mixup_dataset.mixup_data(inputs, targets, CONFIG['mixup_alpha'])
            
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        running_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]})
    
    return running_loss / len(train_loader)

def validate_epoch(model, val_loader, criterion, device):
    """Enhanced validation with comprehensive metrics"""
    model.eval()
    running_loss = 0.0
    metrics_tracker = MetricsTracker()
    
    progress_bar = tqdm(val_loader, desc="Validation", leave=False)
    
    with torch.no_grad():
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            metrics_tracker.update(predictions, targets, probabilities)
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = running_loss / len(val_loader)
    metrics = metrics_tracker.compute_metrics()
    
    return avg_loss, metrics

def plot_training_history(history):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    
    # Accuracy
    axes[0, 1].plot(history['val_accuracy'], label='Val Accuracy')
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].legend()
    
    # AUC
    axes[1, 0].plot(history['val_auc'], label='Val AUC')
    axes[1, 0].set_title('Validation AUC-ROC')
    axes[1, 0].legend()
    
    # EER
    axes[1, 1].plot(history['val_eer'], label='Val EER')
    axes[1, 1].set_title('Validation Equal Error Rate')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['model_save_path'], 'training_history.png'))
    plt.close()

# Configuration
CONFIG = {
    'data_dir': 'data/real_and_fake_face/',
    'model_save_path': 'saved_models/',
    'learning_rate': 1e-3,
    'batch_size': 32,
    'num_epochs': 20,
    'backbone': 'resnet50',
    'weight_decay': 1e-4,
    'validation_split': 0.2,
    'warmup_epochs': 2,
    'mixup_alpha': 0.2,
    'mixup_prob': 0.5,
    'early_stopping_patience': 7,
    'focal_loss_gamma': 2.0,
    'focal_loss_alpha': 1.0,
    'label_smoothing': 0.1,
    'seed': 42
}

def main():
    # Setup
    set_seed(CONFIG['seed'])
    os.makedirs(CONFIG['model_save_path'], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data loading
    train_transforms, val_transforms = get_advanced_transforms()
    
    print(f"Loading data from: {CONFIG['data_dir']}")
    full_dataset = ImageFolder(CONFIG['data_dir'], transform=val_transforms)
    
    total_size = len(full_dataset)
    val_size = int(total_size * CONFIG['validation_split'])
    train_size = total_size - val_size
    
    generator = torch.Generator().manual_seed(CONFIG['seed'])
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    # Apply transforms
    train_subset.dataset.transform = train_transforms
    val_subset.dataset.transform = val_transforms
    
    # Data loaders
    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(
        train_subset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=4, 
        pin_memory=pin_memory, 
        drop_last=True
    )
    val_loader = DataLoader(
        val_subset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=4, 
        pin_memory=pin_memory
    )
    
    print(f"\nDataset Summary:")
    print(f"Total samples: {total_size}")
    print(f"Training samples: {len(train_subset)}")
    print(f"Validation samples: {len(val_subset)}")
    print(f"Classes: {full_dataset.classes}")
    
    # Model setup
    model = FacePADModel(backbone_name=CONFIG['backbone'], pretrained=True).to(device)
    model.freeze_backbone()
    
    # Advanced loss function
    criterion = FocalLoss(
        alpha=CONFIG['focal_loss_alpha'],
        gamma=CONFIG['focal_loss_gamma'],
        label_smoothing=CONFIG['label_smoothing']
    )
    
    # Optimizer and scheduler
    optimizer, scheduler = create_optimizer_scheduler(model, len(train_loader))
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Early stopping
    early_stopping = EarlyStopping(patience=CONFIG['early_stopping_patience'])
    
    # Training history
    history = defaultdict(list)
    best_val_loss = float('inf')
    
    print(f"\nStarting Enhanced Training...")
    print(f"Epochs: {CONFIG['num_epochs']}")
    print(f"Using MixUp: α={CONFIG['mixup_alpha']}, p={CONFIG['mixup_prob']}")
    print(f"Using Focal Loss: γ={CONFIG['focal_loss_gamma']}, α={CONFIG['focal_loss_alpha']}")
    print("-" * 60)
    
    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, scheduler, device)
        
        # Validation
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_auc'].append(val_metrics['auc'])
        history['val_apcer'].append(val_metrics['apcer'])
        history['val_bpcer'].append(val_metrics['bpcer'])
        history['val_eer'].append(val_metrics['eer'])
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"AUC: {val_metrics['auc']:.4f} | EER: {val_metrics['eer']:.4f}")
        print(f"APCER: {val_metrics['apcer']:.4f} | BPCER: {val_metrics['bpcer']:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'config': CONFIG
            }, os.path.join(CONFIG['model_save_path'], 'best_model.pth'))
            print("✓ New best model saved!")
        
        # Early stopping check
        if early_stopping(val_loss, model):
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        print("-" * 60)
    
    # Plot training history
    plot_training_history(history)
    
    print("\n🎉 Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {CONFIG['model_save_path']}")
    print(f"Training history plot saved to: {os.path.join(CONFIG['model_save_path'], 'training_history.png')}")

if __name__ == '__main__':
    main()