import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ================= CONFIG =================
TRAIN_DIR = 'model_data/train'
VAL_DIR = 'model_data/val'
TEST_DIR = 'model_data/test'

NUM_CLASSES = 3
INPUT_SIZE = 240                 
BATCH_SIZE = 8
EPOCHS = 50
INITIAL_LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.4
FOCAL_GAMMA = 1.0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ==========================================

# ---------- Focal Loss ----------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ---------- Model ----------
def get_densenet(num_classes=3, dropout_rate=0.4):
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(dropout_rate * 0.7),
        nn.Linear(512, num_classes)
    )
    return model

# ---------- DenseNet Unfreezing Management ----------
def set_backbone_trainable(model, stage='none'):
    """
    Stage options:
      - 'none': Backbone frozen, only classifier trainable
      - 'stage1': denseblock4 + norm5 trainable
      - 'stage2': denseblock3 + transition3 + denseblock4 + norm5 trainable
      - 'all': Entire backbone trainable
    """
    # Freeze all features initially
    for param in model.features.parameters():
        param.requires_grad = False
        
    if stage == 'none':
        pass
    elif stage == 'stage1':
        for name, child in model.features.named_children():
            if name in ['denseblock4', 'norm5']:
                for param in child.parameters():
                    param.requires_grad = True
    elif stage == 'stage2':
        for name, child in model.features.named_children():
            if name in ['denseblock3', 'transition3', 'denseblock4', 'norm5']:
                for param in child.parameters():
                    param.requires_grad = True
    elif stage == 'all':
        for param in model.features.parameters():
            param.requires_grad = True

def build_optimizer(model, head_lr, backbone_lr_ratio=0.1, weight_decay=1e-4):
    """
    Applies lower learning rate to pretrained backbone layers and higher to the head.
    """
    backbone_params = [p for n, p in model.features.named_parameters() if p.requires_grad]
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]

    param_groups = []
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': head_lr * backbone_lr_ratio})
    if head_params:
        param_groups.append({'params': head_params, 'lr': head_lr})

    return optim.AdamW(param_groups, weight_decay=weight_decay)

# ---------- Transforms ----------
def get_transforms(input_size):
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

# ---------- Train/Validate Step ----------
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    for inputs, labels in tqdm(loader, desc="Train", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    return running_loss / len(loader), accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average='weighted')

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Val", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    val_f1 = f1_score(all_labels, all_preds, average='weighted')
    val_acc = accuracy_score(all_labels, all_preds)
    class_f1 = f1_score(all_labels, all_preds, average=None)
    conf = confusion_matrix(all_labels, all_preds)
    return running_loss / len(loader), val_acc, val_f1, class_f1, conf, np.array(all_probs), np.array(all_labels)

# ---------- Main Loop ----------
def main():
    print(f"Using device: {DEVICE}")

    train_tf, val_tf = get_transforms(INPUT_SIZE)

    train_dataset = ImageFolder(TRAIN_DIR, transform=train_tf)
    val_dataset = ImageFolder(VAL_DIR, transform=val_tf)
    test_dataset = ImageFolder(TEST_DIR, transform=val_tf)

    print("Classes:", train_dataset.classes)

    targets = np.array(train_dataset.targets)
    class_counts = np.bincount(targets, minlength=NUM_CLASSES)
    print("Class counts in training set:", class_counts)
    
    # Balance mini-batches with WeightedRandomSampler
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[targets]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize model with frozen backbone
    model = get_densenet(num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE).to(DEVICE)
    set_backbone_trainable(model, stage='none')

    # alpha=None because sampler handles class imbalance
    criterion = FocalLoss(alpha=None, gamma=FOCAL_GAMMA)
    optimizer = build_optimizer(model, head_lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val_f1 = 0.0
    best_epoch = -1

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")

        # Progressive unfreezing schedule with differential learning rates
        if epoch == 10:
            print(">> Unfreezing DenseBlock 4 & Norm5 (Stage 1)...")
            set_backbone_trainable(model, stage='stage1')
            optimizer = build_optimizer(model, head_lr=INITIAL_LR * 0.5, backbone_lr_ratio=0.1, weight_decay=WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        elif epoch == 20:
            print(">> Unfreezing DenseBlock 3 & 4 (Stage 2)...")
            set_backbone_trainable(model, stage='stage2')
            optimizer = build_optimizer(model, head_lr=INITIAL_LR * 0.2, backbone_lr_ratio=0.1, weight_decay=WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        elif epoch == 30:
            print(">> Unfreezing Entire Backbone...")
            set_backbone_trainable(model, stage='all')
            optimizer = build_optimizer(model, head_lr=INITIAL_LR * 0.05, backbone_lr_ratio=0.1, weight_decay=WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE)
        val_loss, val_acc, val_f1, class_f1, conf, _, _ = validate(model, val_loader, criterion, DEVICE)

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Weighted F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Weighted F1: {val_f1:.4f}")
        print(f"Val Class-wise F1: {np.round(class_f1, 4)}")

        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
            }, 'best_densenet_model.pth')
            print(f">> New best model saved (Val F1: {val_f1:.4f})")

    print(f"\nTraining Complete. Best Val F1: {best_val_f1:.4f} at epoch {best_epoch}")

    # Test Evaluation
    print("\nRunning Evaluation on Test Set...")
    checkpoint = torch.load('best_densenet_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc, test_f1, test_class_f1, test_conf, _, _ = validate(model, test_loader, criterion, DEVICE)
    
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Weighted F1: {test_f1:.4f}")
    print(f"Test Class-wise F1: {np.round(test_class_f1, 4)}")
    print(f"Test Confusion Matrix:\n{test_conf}")

if __name__ == "__main__":
    main()
