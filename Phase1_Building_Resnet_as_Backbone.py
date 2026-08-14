import torch
import torch.nn as nn
import torchvision.models as models

class HybridPneumoniaModel(nn.Module):
    def __init__(self):
        super(HybridPneumoniaModel, self).__init__()

        # Since we are using resnet-50
        # Using default weights for now to updating them later(and to avoid warnings)
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Extracting upto layer-3 (conv4_x) which will give the final output in 14 × 14 × 1024
        # We know that ViT-16 uses O(N^2) memory so we will use can get a max of 196 tokens which will not lead to OOM
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

    def forward(self, x):
        # Extract features using cnn
        features = self.cnn_backbone(x)
        # Cap the resolution
        features = self.adaptive_pool(features)
        
        return features

# Test Run

if __name__ == "__main__":
    # We will create a dumy x-ray to test
    # (Batch_Size=2, Channels=3, Height=224, Width=224)
    dummy_xray = torch.randn(2, 3, 224, 224) 
    
    # Model initialize karo
    model = HybridPneumoniaModel()
    
    # Dummy image ko model mein pass karo
    output_features = model(dummy_xray)
    
    print("Input Shape:", dummy_xray.shape)
    print("Output Shape:", output_features.shape)
