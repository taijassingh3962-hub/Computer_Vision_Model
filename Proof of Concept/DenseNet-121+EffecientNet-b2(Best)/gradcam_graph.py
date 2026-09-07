import os
import sys
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image, Image as PILImage
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Use Agg backend for headless mode
matplotlib.use('Agg')

from main_ensemble_model import (
    CONFIG, DEVICE, load_trained_models, PlainImageFolder
)

class GradCAM:
    """Minimal Grad-CAM implementation using PyTorch hooks."""
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self.bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        input_tensor.requires_grad_(True)
        
        output = self.model(input_tensor)
        predicted_class = output.argmax(dim=1).item()
        if target_class is None:
            target_class = predicted_class

        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, predicted_class

    def remove(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()

def generate_ensemble_gradcam():
    cfg = CONFIG
    output_dir = os.path.join('visualizations', 'gradcam')
    os.makedirs(output_dir, exist_ok=True)

    print("=================================================================")
    print("  ENSEMBLE GRAD-CAM VISUALIZATION (10 Models)")
    print("=================================================================")
    
    dense_models, effnet_models, meta_model = load_trained_models()
    
    # We need the raw dataset for getting sample images
    test_dataset = PlainImageFolder(cfg['dataset_dir'])
    
    # Only pick from the holdout test indices
    test_idx_path = os.path.join(cfg['weights_dir'], 'test_indices.npy')
    if not os.path.exists(test_idx_path):
        print(f"Error: {test_idx_path} not found.")
        sys.exit(1)
        
    test_idx = np.load(test_idx_path)
    
    class_names = test_dataset.classes
    samples_per_class = 4
    
    # Collect sample indices from the test set for each class
    class_indices = {i: [] for i in range(len(class_names))}
    for idx in test_idx:
        _, label = test_dataset.samples[idx]
        if len(class_indices[label]) < samples_per_class:
            class_indices[label].append(idx)
        if all(len(v) >= samples_per_class for v in class_indices.values()):
            break

    # Inference transform
    transform = transforms.Compose([
        transforms.Resize((cfg['input_size'], cfg['input_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    display_transform = transforms.Compose([
        transforms.Resize((cfg['input_size'], cfg['input_size'])),
        transforms.ToTensor(),
    ])

    # Setup GradCAM for all 10 models
    # DenseNet-121 target: model.features.denseblock4
    # EfficientNet-B2 target: model.features[-1]
    dense_gcs = [GradCAM(m, m.features.denseblock4) for m in dense_models]
    effnet_gcs = [GradCAM(m, m.features[-1]) for m in effnet_models]

    n_classes = len(class_names)
    fig, axes = plt.subplots(n_classes * samples_per_class, 4, figsize=(20, 5 * n_classes * samples_per_class))
    fig.patch.set_facecolor('#1a1a2e')

    row = 0
    print("\n  Generating Grad-CAM for sample images...")
    
    for cls_idx, cls_name in enumerate(class_names):
        for sample_idx in class_indices[cls_idx]:
            img_path, true_label = test_dataset.samples[sample_idx]
            original_img = Image.open(img_path).convert('RGB')

            input_tensor = transform(original_img).unsqueeze(0).to(DEVICE)
            display_tensor = display_transform(original_img)
            display_img = display_tensor.permute(1, 2, 0).numpy()

            # 1. Average DenseNet CAM across 5 Folds
            dense_cams = []
            for gc in dense_gcs:
                cam, _ = gc.generate(input_tensor.clone(), target_class=true_label)
                cam_res = np.array(PILImage.fromarray((cam * 255).astype(np.uint8)).resize(
                    (cfg['input_size'], cfg['input_size']), PILImage.BILINEAR)) / 255.0
                dense_cams.append(cam_res)
            avg_dense_cam = np.mean(dense_cams, axis=0)
            if avg_dense_cam.max() > 0: avg_dense_cam = avg_dense_cam / avg_dense_cam.max()

            # 2. Average EfficientNet CAM across 5 Folds
            effnet_cams = []
            for gc in effnet_gcs:
                cam, _ = gc.generate(input_tensor.clone(), target_class=true_label)
                cam_res = np.array(PILImage.fromarray((cam * 255).astype(np.uint8)).resize(
                    (cfg['input_size'], cfg['input_size']), PILImage.BILINEAR)) / 255.0
                effnet_cams.append(cam_res)
            avg_effnet_cam = np.mean(effnet_cams, axis=0)
            if avg_effnet_cam.max() > 0: avg_effnet_cam = avg_effnet_cam / avg_effnet_cam.max()

            # 3. Complete 10-Fold Ensemble CAM
            cam_ensemble = (avg_dense_cam + avg_effnet_cam) / 2.0
            if cam_ensemble.max() > 0: cam_ensemble = cam_ensemble / cam_ensemble.max()

            # Plots
            # Column 0: Original
            ax = axes[row, 0]
            ax.imshow(display_img)
            ax.set_title(f'Original\nTrue: {cls_name}', fontsize=10, color='white', fontweight='bold')
            ax.axis('off')

            # Column 1: DenseNet-121 (5-Fold Avg)
            ax = axes[row, 1]
            heatmap = cm.jet(avg_dense_cam)[:, :, :3]
            overlay = np.clip(0.5 * display_img + 0.5 * heatmap, 0, 1)
            ax.imshow(overlay)
            ax.set_title('DenseNet-121\n(5-Fold Avg CAM)', fontsize=10, color='#2ECC71', fontweight='bold')
            ax.axis('off')

            # Column 2: EfficientNet-B2 (5-Fold Avg)
            ax = axes[row, 2]
            heatmap = cm.jet(avg_effnet_cam)[:, :, :3]
            overlay = np.clip(0.5 * display_img + 0.5 * heatmap, 0, 1)
            ax.imshow(overlay)
            ax.set_title('EfficientNet-B2\n(5-Fold Avg CAM)', fontsize=10, color='#3498DB', fontweight='bold')
            ax.axis('off')

            # Column 3: Complete Ensemble CAM
            ax = axes[row, 3]
            heatmap = cm.jet(cam_ensemble)[:, :, :3]
            overlay = np.clip(0.5 * display_img + 0.5 * heatmap, 0, 1)
            ax.imshow(overlay)
            ax.set_title('Final Ensemble\n(10-Model Avg CAM)', fontsize=10, color='#F39C12', fontweight='bold')
            ax.axis('off')

            row += 1
            print(f"    Processed {cls_name} sample")

    plt.suptitle('Grad-CAM — 5-Fold DenseNet-121 + EfficientNet-B2 Ensemble',
                 fontsize=16, fontweight='bold', color='white', y=1.01)
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'gradcam_new_ensemble.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Success! Grad-CAM plot saved to: {save_path}")

    # Cleanup hooks
    for gc in dense_gcs: gc.remove()
    for gc in effnet_gcs: gc.remove()

if __name__ == '__main__':
    generate_ensemble_gradcam()
