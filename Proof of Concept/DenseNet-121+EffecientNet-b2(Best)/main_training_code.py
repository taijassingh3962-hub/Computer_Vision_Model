import os
import sys
import json
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
from tqdm import tqdm
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

CONFIG = {
    # Data paths
    'dataset_dir': 'model_data/all_images',
    'weights_dir': 'weights_456',

    # Model
    'num_classes': 3,
    'class_names': ['bacterial', 'normal', 'viral'],
    'input_size': 456,

    # Training
    'batch_size': 16,
    'accumulation_steps': 4,  # Simulates batch_size of 64 mathematically, Gradient Accumulation Used
    'epochs_per_stage': 10,
    'initial_lr': 3e-4,
    'weight_decay': 5e-4,
    'dropout_rate': 0.3,
    'early_stop_patience': 4,
    'grad_clip_max_norm': 1.0,
    
    # K-Fold Stacking
    'n_splits': 5,

    # Normalization (ImageNet)
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if DEVICE.type == 'cuda':
    # Autotuner to find the fastest convolution algorithms for your specific GPU architecture
    torch.backends.cudnn.benchmark = True 
    print(">> cuDNN Benchmark Enabled for Max GPU Utilization <<")

def get_densenet121(num_classes=3, dropout_rate=0.5):
    """DenseNet-121 with custom classification head and severe dropout."""
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

def get_efficientnet_b2(num_classes=3, dropout_rate=0.5):
    """EfficientNet-B2 with custom classification head and severe dropout."""
    model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(dropout_rate * 0.7),
        nn.Linear(512, num_classes)
    )
    return model

class MetaClassifier(nn.Module):
    """Stacking Meta-Classifier to combine raw logits from base models."""
    def __init__(self, num_classes=3):
        super().__init__()
        self.fc = nn.Linear(6, num_classes)
        
    def forward(self, x):
        return self.fc(x)

def freeze_backbone_densenet(model, stage=1):
    """Progressive unfreezing for DenseNet-121."""
    for param in model.features.parameters():
        param.requires_grad = False
    
    unfreeze_layers = []
    if stage >= 1:
        unfreeze_layers.extend(['denseblock4', 'norm5'])
    if stage >= 2:
        unfreeze_layers.extend(['denseblock3', 'transition3'])
    if stage >= 3:
        unfreeze_layers.extend(['denseblock2', 'transition2'])
        
    for name, child in model.features.named_children():
        if name in unfreeze_layers:
            for param in child.parameters():
                param.requires_grad = True

def freeze_backbone_efficientnet(model, stage=1):
    """Progressive unfreezing for EfficientNet-B2."""
    for param in model.features.parameters():
        param.requires_grad = False
        
    for i, block in enumerate(model.features):
        if stage == 1 and i >= 7:
            for param in block.parameters(): param.requires_grad = True
        elif stage == 2 and i >= 6:
            for param in block.parameters(): param.requires_grad = True
        elif stage >= 3 and i >= 5:
            for param in block.parameters(): param.requires_grad = True

def build_optimizer(model, head_lr, backbone_lr_ratio=0.1, weight_decay=5e-4):
    """Build optimizer with differential learning rates."""
    backbone_params = [p for n, p in model.features.named_parameters() if p.requires_grad]
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]
    param_groups = []
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': head_lr * backbone_lr_ratio})
    if head_params:
        param_groups.append({'params': head_params, 'lr': head_lr})
    return optim.AdamW(param_groups, weight_decay=weight_decay)

def get_train_transform(input_size):
    """Aggressive augmentations to prevent overfitting (Shoulder Destroyer)."""
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=CONFIG['mean'], std=CONFIG['std'])
    ])

def get_val_transform(input_size):
    """Validation/test transform (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CONFIG['mean'], std=CONFIG['std'])
    ])

class SubsetWithTransform(torch.utils.data.Dataset):
    """Wrapper to apply distinct transforms to subsets during K-Fold CV."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """Train for one epoch with mixed precision and gradient accumulation."""
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    acc_steps = CONFIG.get('accumulation_steps', 1)
    optimizer.zero_grad()

    for i, (inputs, labels) in enumerate(tqdm(loader, desc="    Train", leave=False)):
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # Normalize the loss because gradients are accumulated over multiple batches
            loss = loss / acc_steps

        scaler.scale(loss).backward()

        if (i + 1) % acc_steps == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG['grad_clip_max_norm'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Multiply back to get the true loss value for metric tracking
        running_loss += (loss.item() * acc_steps)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
    return epoch_loss, epoch_acc, epoch_f1

def validate(model, loader, criterion, device):
    """Validate model, returns metrics + per-sample raw logits."""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_logits = [], [], []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="    Val  ", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            running_loss += loss.item()
            
            # Save raw logits (detached from graph)
            logits = outputs.detach().cpu().numpy()
            all_logits.extend(logits)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / len(loader)
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average='weighted')
    class_f1 = f1_score(all_labels, all_preds, average=None)
    conf = confusion_matrix(all_labels, all_preds)
    return val_loss, val_acc, val_f1, class_f1, conf, np.array(all_logits), np.array(all_labels)

def train_single_model(model_name, model, freeze_fn, train_loader, val_loader,
                       criterion, device, batch_size, save_path):
    """Train a single base model with 3-stage progressive unfreezing."""
    cfg = CONFIG
    print(f"\n{'='*65}")
    print(f"  TRAINING: {model_name} (3-Stage Unfreezing)")
    print(f"  Batch Size: {batch_size} | Base LR: {cfg['initial_lr']}")
    print(f"{'='*65}")

    best_oof_logits = None
    best_oof_labels = None
    
    start_time = time.time()
    
    for stage in [1, 2, 3]:
        print(f"\n  >>> STAGE {stage} UNFREEZING <<<")
        freeze_fn(model, stage)
        
        # Lower learning rate for deeper stages
        lr = cfg['initial_lr'] if stage == 1 else cfg['initial_lr'] * (0.5 ** (stage - 1))
        optimizer = build_optimizer(model, head_lr=lr, weight_decay=cfg['weight_decay'])
        scaler = torch.amp.GradScaler('cuda')
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

        best_val_f1 = 0.0
        epochs_no_improve = 0

        for epoch in range(cfg['epochs_per_stage']):
            print(f"\n  --- {model_name} Stage {stage} - Epoch {epoch+1}/{cfg['epochs_per_stage']} ---")

            t_loss, t_acc, t_f1 = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
            v_loss, v_acc, v_f1, v_cls_f1, _, oof_logits, oof_labels = validate(model, val_loader, criterion, device)

            print(f"  Train Loss: {t_loss:.4f} | Acc: {t_acc:.4f} | F1: {t_f1:.4f}")
            print(f"  Val   Loss: {v_loss:.4f} | Acc: {v_acc:.4f} | F1: {v_f1:.4f} | Class F1: {np.round(v_cls_f1, 3)}")

            scheduler.step(v_f1)

            if v_f1 > best_val_f1:
                best_val_f1 = v_f1
                epochs_no_improve = 0
                best_oof_logits = oof_logits
                best_oof_labels = oof_labels
                torch.save({'model_state_dict': model.state_dict()}, save_path)
                print(f"  >> New best {model_name} (Stage {stage}) saved! (Val F1: {v_f1:.4f})")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= cfg['early_stop_patience']:
                    print(f"\n  >> Early stopping in Stage {stage}.")
                    break
        
        # After stage finishes, ALWAYS load the best weights for the start of the next stage
        checkpoint = torch.load(save_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])

    elapsed = time.time() - start_time
    print(f"\n  {model_name} completed all 3 stages in {elapsed:.1f}s")
    
    return model, best_val_f1, best_oof_logits, best_oof_labels

class PlainImageFolder(ImageFolder):
    """Base dataset containing only PIL images (transforms applied later)."""
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        return sample, target

def run_training():
    """Stratified 5-Fold Training to generate OOF logits and train Meta-Classifier."""
    cfg = CONFIG
    os.makedirs(cfg['weights_dir'], exist_ok=True)

    print("=" * 65)
    print("  STRATIFIED 5-FOLD ENSEMBLE TRAINING (DenseNet + EfficientNet)")
    print("=" * 65)

    full_dataset = PlainImageFolder(cfg['dataset_dir'])
    targets = np.array(full_dataset.targets)
    
    # Calculate Class Weights for Cross Entropy Loss
    class_counts = np.bincount(targets, minlength=cfg['num_classes'])
    print(f"\n  Full Dataset distribution: {dict(zip(full_dataset.classes, class_counts.tolist()))}")
    
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * cfg['num_classes']
    class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)
    print(f"  Calculated Class Weights: {np.round(class_weights, 3)}")

    # -------------------------------------------------------------
    # DYNAMIC HOLD-OUT TEST SET SPLIT (10%)
    # -------------------------------------------------------------
    cv_idx, test_idx = train_test_split(
        np.arange(len(targets)), test_size=0.10, stratify=targets, random_state=42
    )
    cv_targets = targets[cv_idx]
    
    # Save the test indices so run_test evaluates exactly the same images!
    test_idx_path = os.path.join(cfg['weights_dir'], 'test_indices.npy')
    np.save(test_idx_path, test_idx)
    
    print(f"  Dynamic Split: {len(cv_idx)} images for 5-Fold CV | {len(test_idx)} images held out for Test.")
    print(f"  Test indices safely saved to: {test_idx_path}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    train_tf = get_train_transform(cfg['input_size'])
    val_tf = get_val_transform(cfg['input_size'])

    skf = StratifiedKFold(n_splits=cfg['n_splits'], shuffle=True, random_state=42)
    
    # Arrays to store OOF logits
    all_oof_dense = np.zeros((len(cv_idx), cfg['num_classes']))
    all_oof_effnet = np.zeros((len(cv_idx), cfg['num_classes']))
    all_oof_labels = np.zeros(len(cv_idx))

    for fold, (train_fold_idx, val_fold_idx) in enumerate(skf.split(np.zeros(len(cv_targets)), cv_targets)):
        print(f"\n{'='*65}")
        print(f"  FOLD {fold + 1} / {cfg['n_splits']}")
        print(f"{'='*65}")

        # Map fold indices back to absolute dataset indices
        train_idx_abs = cv_idx[train_fold_idx]
        val_idx_abs = cv_idx[val_fold_idx]

        train_subset = SubsetWithTransform(Subset(full_dataset, train_idx_abs), transform=train_tf)
        val_subset = SubsetWithTransform(Subset(full_dataset, val_idx_abs), transform=val_tf)

        # Sampler for training data
        fold_targets = targets[train_idx_abs]
        fold_counts = np.bincount(fold_targets, minlength=cfg['num_classes'])
        fold_weights = 1.0 / np.maximum(fold_counts, 1)
        sample_weights = fold_weights[fold_targets]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(
            train_subset, batch_size=cfg['batch_size'], sampler=sampler, 
            num_workers=4, pin_memory=True, prefetch_factor=2
        )
        val_loader = DataLoader(
            val_subset, batch_size=cfg['batch_size'], shuffle=False, 
            num_workers=4, pin_memory=True, prefetch_factor=2
        )

        # Train DenseNet
        model_dense = get_densenet121(cfg['num_classes'], cfg['dropout_rate']).to(DEVICE)
        dense_path = os.path.join(cfg['weights_dir'], f'fold_{fold+1}_densenet121.pth')
        _, _, oof_dense, oof_labels = train_single_model(
            f"DenseNet-121 (Fold {fold+1})", model_dense, freeze_backbone_densenet,
            train_loader, val_loader, criterion, DEVICE, cfg['batch_size'], dense_path
        )
        
        # Train EfficientNet
        model_effnet = get_efficientnet_b2(cfg['num_classes'], cfg['dropout_rate']).to(DEVICE)
        effnet_path = os.path.join(cfg['weights_dir'], f'fold_{fold+1}_efficientnet_b2.pth')
        _, _, oof_effnet, _ = train_single_model(
            f"EfficientNet-B2 (Fold {fold+1})", model_effnet, freeze_backbone_efficientnet,
            train_loader, val_loader, criterion, DEVICE, cfg['batch_size'], effnet_path
        )

        # Save OOF predictions (using the relative index for the CV pool)
        all_oof_dense[val_fold_idx] = oof_dense
        all_oof_effnet[val_fold_idx] = oof_effnet
        all_oof_labels[val_fold_idx] = oof_labels
        
        torch.cuda.empty_cache()

    # ---- Train Meta-Classifier ----
    print(f"\n{'='*65}")
    print(f"  TRAINING META-CLASSIFIER (OOF Stacking)")
    print(f"{'='*65}")
    
    oof_features = np.concatenate([all_oof_dense, all_oof_effnet], axis=1) # [5000, 6]
    
    # Split OOF predictions into Train/Val for Meta-Classifier early stopping (prevents memorization)
    meta_train_idx, meta_val_idx = train_test_split(
        np.arange(len(all_oof_labels)), test_size=0.20, stratify=all_oof_labels, random_state=42
    )
    
    X_meta = torch.tensor(oof_features, dtype=torch.float32).to(DEVICE)
    y_meta = torch.tensor(all_oof_labels, dtype=torch.long).to(DEVICE)
    
    meta_train_loader = DataLoader(
        torch.utils.data.TensorDataset(X_meta[meta_train_idx], y_meta[meta_train_idx]), 
        batch_size=32, shuffle=True
    )
    meta_val_loader = DataLoader(
        torch.utils.data.TensorDataset(X_meta[meta_val_idx], y_meta[meta_val_idx]), 
        batch_size=32, shuffle=False
    )
    
    meta_model = MetaClassifier(cfg['num_classes']).to(DEVICE)
    meta_optimizer = optim.AdamW(meta_model.parameters(), lr=1e-3, weight_decay=1e-3)
    meta_path = os.path.join(cfg['weights_dir'], 'best_meta_classifier.pth')
    
    best_meta_loss = float('inf')
    epochs_no_improve = 0
    patience = 5
    
    for epoch in range(100):
        meta_model.train()
        for inputs, labels in meta_train_loader:
            meta_optimizer.zero_grad()
            outputs = meta_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            meta_optimizer.step()
            
        # Validation for early stopping
        meta_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in meta_val_loader:
                outputs = meta_model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
        val_loss /= len(meta_val_loader)
        
        if val_loss < best_meta_loss:
            best_meta_loss = val_loss
            epochs_no_improve = 0
            torch.save({'model_state_dict': meta_model.state_dict()}, meta_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Meta-Classifier early stopping triggered at epoch {epoch+1}")
                break
    
    print(f"  Meta-Classifier training complete. Weights saved to: {meta_path}")
    print(f"{'='*65}\n  PIPELINE COMPLETE!\n{'='*65}")

def load_trained_models():
    """Load all 5 folds of base models and the meta-classifier."""
    cfg = CONFIG
    weights_dir = cfg['weights_dir']

    dense_models = []
    effnet_models = []
    
    print("\n  Loading 5-Fold Ensemble Weights...")
    for fold in range(1, cfg['n_splits'] + 1):
        # DenseNet
        d_path = os.path.join(weights_dir, f'fold_{fold}_densenet121.pth')
        m_dense = get_densenet121(cfg['num_classes'], cfg['dropout_rate']).to(DEVICE)
        m_dense.load_state_dict(torch.load(d_path, map_location=DEVICE, weights_only=True)['model_state_dict'])
        m_dense.eval()
        dense_models.append(m_dense)
        
        # EfficientNet
        e_path = os.path.join(weights_dir, f'fold_{fold}_efficientnet_b2.pth')
        m_effnet = get_efficientnet_b2(cfg['num_classes'], cfg['dropout_rate']).to(DEVICE)
        m_effnet.load_state_dict(torch.load(e_path, map_location=DEVICE, weights_only=True)['model_state_dict'])
        m_effnet.eval()
        effnet_models.append(m_effnet)
        
    meta_path = os.path.join(weights_dir, 'best_meta_classifier.pth')
    meta_model = MetaClassifier(cfg['num_classes']).to(DEVICE)
    meta_model.load_state_dict(torch.load(meta_path, map_location=DEVICE, weights_only=True)['model_state_dict'])
    meta_model.eval()
    
    print("  Successfully loaded all base models and Meta-Classifier.")
    return dense_models, effnet_models, meta_model

def run_test(dense_models=None, effnet_models=None, meta_model=None):
    """Evaluate 5-fold averaged ensemble + meta-classifier on test set."""
    cfg = CONFIG

    if dense_models is None or effnet_models is None or meta_model is None:
        dense_models, effnet_models, meta_model = load_trained_models()

    val_tf = get_val_transform(cfg['input_size'])
    
    full_dataset = PlainImageFolder(cfg['dataset_dir'])
    targets = np.array(full_dataset.targets)
    
    # Load the exact 10% hold-out test set saved during training
    test_idx_path = os.path.join(cfg['weights_dir'], 'test_indices.npy')
    if not os.path.exists(test_idx_path):
        raise FileNotFoundError(f"Cannot find {test_idx_path}. You must run training first!")
        
    test_idx = np.load(test_idx_path)
    
    test_subset = SubsetWithTransform(Subset(full_dataset, test_idx), transform=val_tf)
    test_loader = DataLoader(test_subset, batch_size=16, shuffle=False, num_workers=2)

    class_names = full_dataset.classes
    all_preds, all_labels = [], []

    print(f"\n{'='*65}")
    print(f"  TEST SET EVALUATION (Stacking + Averaging)")
    print(f"{'='*65}")

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="  Ensemble Inference"):
            inputs = inputs.to(DEVICE)
            all_labels.extend(labels.numpy())
            
            with torch.amp.autocast('cuda'):
                # Average DenseNet Logits
                d_logits = []
                for model in dense_models:
                    d_logits.append(model(inputs))
                d_avg = sum(d_logits) / len(d_logits)
                
                # Average EfficientNet Logits
                e_logits = []
                for model in effnet_models:
                    e_logits.append(model(inputs))
                e_avg = sum(e_logits) / len(e_logits)
                
                # Concat and Meta-Classifier
                combined_logits = torch.cat([d_avg, e_avg], dim=1)
                final_outputs = meta_model(combined_logits)
            
            _, preds = torch.max(final_outputs, 1)
            all_preds.extend(preds.cpu().numpy())

    ens_acc = accuracy_score(all_labels, all_preds)
    ens_f1 = f1_score(all_labels, all_preds, average='weighted')
    ens_cls_f1 = f1_score(all_labels, all_preds, average=None)
    ens_conf = confusion_matrix(all_labels, all_preds)

    print(f"\n  Accuracy:    {ens_acc:.4f}")
    print(f"  Weighted F1: {ens_f1:.4f}")
    print(f"  Class F1:    {np.round(ens_cls_f1, 4)}")
    print(f"\n  Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    print(f"  Confusion Matrix:\n  {ens_conf}")

def predict_single_image(image, dense_models, effnet_models, meta_model, transform, class_names, device):
    """Run averaged 5-fold ensemble + meta-classifier on a single image."""
    if image.mode != 'RGB':
        image = image.convert('RGB')

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        # Average DenseNet Logits
        d_logits = []
        for model in dense_models:
            with torch.amp.autocast('cuda'):
                d_logits.append(model(input_tensor))
        d_avg = sum(d_logits) / len(d_logits)
        
        # Average EfficientNet Logits
        e_logits = []
        for model in effnet_models:
            with torch.amp.autocast('cuda'):
                e_logits.append(model(input_tensor))
        e_avg = sum(e_logits) / len(e_logits)
        
        # Meta-Classifier
        combined_logits = torch.cat([d_avg, e_avg], dim=1)
        final_outputs = meta_model(combined_logits)
        
        final_probs = torch.softmax(final_outputs, dim=1).cpu().numpy()[0]

    predicted_class = class_names[np.argmax(final_probs)]
    confidence = float(np.max(final_probs))
    result = {class_names[i]: float(final_probs[i]) for i in range(len(class_names))}

    return result, predicted_class, confidence

def main():
    parser = argparse.ArgumentParser(description='Stratified 5-Fold Stacking Ensemble Classifier')
    parser.add_argument('--mode', type=str, default='all', choices=['train', 'test', 'all'])
    args = parser.parse_args()

    if args.mode in ['train', 'all']:
        run_training()

    if args.mode in ['test', 'all']:
        run_test()

if __name__ == '__main__':
    main()
