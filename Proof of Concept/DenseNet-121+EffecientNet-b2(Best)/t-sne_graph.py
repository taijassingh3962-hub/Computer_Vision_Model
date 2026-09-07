import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib

# Use Agg backend for headless mode
matplotlib.use('Agg')

# Import our ensemble logic
from main_ensemble_model import (
    CONFIG, DEVICE, load_trained_models, get_val_transform, 
    PlainImageFolder, SubsetWithTransform
)

def extract_ensemble_features():
    cfg = CONFIG
    print("=================================================================")
    print("  ENSEMBLE t-SNE VISUALIZATION")
    print("=================================================================")
    
    # Load Models
    dense_models, effnet_models, meta_model = load_trained_models()
    
    val_tf = get_val_transform(cfg['input_size'])
    full_dataset = PlainImageFolder(cfg['dataset_dir'])
    
    # Load identical test subset used in evaluation
    test_idx_path = os.path.join(cfg['weights_dir'], 'test_indices.npy')
    if not os.path.exists(test_idx_path):
        print(f"Error: {test_idx_path} not found.")
        sys.exit(1)
        
    test_idx = np.load(test_idx_path)
    test_subset = SubsetWithTransform(Subset(full_dataset, test_idx), transform=val_tf)
    
    # Use larger batch size to speed up extraction
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False, num_workers=2)

    all_features = []
    all_labels = []

    print("\n  Extracting 6D Logits for t-SNE...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="  Inference"):
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
                
                # Concat -> 6D feature space used by Meta-Classifier
                combined_logits = torch.cat([d_avg, e_avg], dim=1)
                
            all_features.append(combined_logits.cpu().numpy())

    return np.concatenate(all_features, axis=0), np.array(all_labels), full_dataset.classes

def plot_tsne(features, labels, class_names):
    print("\n  Computing t-SNE (this may take a minute)...")
    # Initialize t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate='auto')
    embeddings = tsne.fit_transform(features)

    # Setup Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    colors = ['#E74C3C', '#2ECC71', '#3498DB']  # Bacterial (red), Normal (green), Viral (blue)
    markers = ['o', 's', '^']

    # Scatter plot for each class
    for cls_idx, cls_name in enumerate(class_names):
        mask = labels == cls_idx
        ax.scatter(
            embeddings[mask, 0], embeddings[mask, 1],
            c=colors[cls_idx], label=cls_name, marker=markers[cls_idx],
            s=40, alpha=0.7, edgecolors='white', linewidths=0.3
        )

    # Formatting
    ax.legend(fontsize=12, loc='upper right', framealpha=0.9,
              facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
    ax.set_title('t-SNE — 5-Fold Ensemble Model\n(6D Combined Logit Space on Test Set)',
                 fontsize=14, fontweight='bold', color='white', pad=15)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=11, color='#cccccc')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=11, color='#cccccc')
    ax.tick_params(colors='#888888')
    for spine in ax.spines.values():
        spine.set_color('#333333')

    # Save
    output_dir = 'visualizations'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'tsne_new_ensemble.png')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"\n  Success! t-SNE plot saved to: {save_path}")

if __name__ == '__main__':
    features, labels, class_names = extract_ensemble_features()
    plot_tsne(features, labels, class_names)
