import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim

# Replaced ResNet with DenseNet-121
# DenseNet's dense connections perfectly preserved micro-opacities since is uses concatination connections rather then skip connection
# This is just to test weather the model will work on my GPU

class HybridCheXNetModel(nn.Module):
    def __init__(self, num_classes=3):
        super(HybridCheXNetModel, self).__init__()

        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.cnn_backbone = densenet.features
        
        # 2. Perfect 25 x 25 Natural Grid (625 Tokens)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((25, 25))
        self.projector = nn.Linear(1024, 768)
        
        # 3. Vision Transformer Brain
        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        features = self.cnn_backbone(x)           # [Batch, 1024, 25, 25]
        features = self.adaptive_pool(features)   
        features = features.flatten(2).transpose(1, 2) # [Batch, 625, 1024]
        
        tokens = self.projector(features)              # [Batch, 625, 768]
        vit_output = self.transformer(tokens)          
        vit_summary = vit_output.mean(dim=1)           
        return self.classifier(vit_summary)

def print_vram_usage(step_name):
    if torch.cuda.is_available():
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"[{step_name}] -> VRAM Reserved: {reserved:.2f} MB")

if __name__ == "__main__":
    TEST_BATCH_SIZE = 2  
    
    print(f"\n--- HYBRID DENSENET-121 STRESS TEST (Batch Size: {TEST_BATCH_SIZE}) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HybridCheXNetModel(num_classes=3).to(device)
    model.train() 

    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # 448x448 Resolution test
    dummy_xrays = torch.randn(TEST_BATCH_SIZE, 3, 900, 900).to(device)
    dummy_targets = torch.randint(0, 3, (TEST_BATCH_SIZE,)).to(device)
    
    print_vram_usage("After Data Loading")

    optimizer.zero_grad()
    predictions = model(dummy_xrays)
    
    # Simple CE loss just for VRAM math test
    loss = F.cross_entropy(predictions, dummy_targets)
    print_vram_usage("After FORWARD Pass")

    loss.backward()
    print_vram_usage("After BACKWARD Pass")

    optimizer.step()
    print_vram_usage("After OPTIMIZER Step")
    
    print("\nTEST SUCCESSFUL! 625 Tokens processed via DenseNet + ViT.")
