import torch
import torch.nn as nn
import torchvision.models as models


# Using the same code of phase-1. In this phase we will try to integrate our vision transformer

class HybridPneumoniaModel(nn.Module):
    def __init__(self):
        super(HybridPneumoniaModel, self).__init__()

        # Since we are using resnet-50
        # Using default weights for now to updating them later(and to avoid warnings)
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Extracting upto layer-3 (conv4_x) which will give the final output in 14 × 14 × 1024
        # We know that ViT-16 uses O(N^2) memory so we will use 14 × 14 which can get a max of 196 tokens which will not lead to OOM
        self.cnn_backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))

        # Linear layer to project 1024 ResNet channels to 768 ViT embedding size(Since ViT only takes 768 channels)
        self.projector = nn.Linear(1024, 768)
        
        # Transformer Encoder Engine (The Brain)
        # We are only using 4 layers of attention which is enough (Saves VRAM, prevents overfitting)
        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

    # Updating the forward pass accroding to ViT-16
    def forward(self, x):
        
        # Phase-1 output from resnet
        features = self.cnn_backbone(x)          # Shape: [2, 1024, 14, 14]
        features = self.adaptive_pool(features)  # Shape: [2, 1024, 14, 14]
        
        # Phase-2: The Bridge (Image to Sequence(Tokens))
        # 1. Flatten the 14x14 spatial dimensions into 196 tokens
        features = features.flatten(2)           # Shape: [2, 1024, 196]
        
        # 2. Swap dimensions for Transformer (Batch, Tokens, Embedding)
        features = features.transpose(1, 2)      # Shape: [2, 196, 1024]
        
        # 3. Project 1024 -> 768
        tokens = self.projector(features)        # Shape: [2, 196, 768]
        
        # The Transformer Engine
        vit_output = self.transformer(tokens)    # Shape: [2, 196, 768]
        
        return vit_output
# Test Run
if __name__ == "__main__":
    dummy_xray = torch.randn(2, 3, 224, 224) 
    model = HybridPneumoniaModel()
    
    final_features = model(dummy_xray)
    
    print("Input Shape:", dummy_xray.shape)
    print("Transformer Output Shape:", final_features.shape)
