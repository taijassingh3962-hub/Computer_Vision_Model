import torch
import torch.nn as nn
import torchvision.models as models

# After Testing that the pipeline is working we took the real dimensions of images with resolution of 1024x1024 which is the passed to resnet-50 to get 64x64 desampled output
# Which then further compressed to 30x30 to get the maximum token of 900 (sweetspot) to feed ViT-16

class HybridPneumoniaModel(nn.Module):
    def __init__(self):
        super(HybridPneumoniaModel, self).__init__()

        # Since we are using resnet-50
        # Using default weights for now to updating them later(and to avoid warnings)
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Extracting upto layer-3 (conv4_x) which will give the final output in (new: 64x64z1024)(old: 14×14×1024)
        # We know that ViT-16 uses O(N^2) memory so we will use 64 × 64 which can get a max of 900 tokens which will not lead to OOM
        self.cnn_backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3
        )
        # Compressing 64x64 to 30x30
        self.adaptive_pool = nn.AdaptiveAvgPool2d((30, 30))

        # Linear layer to project 1024 ResNet channels to 768 ViT embedding size(Since ViT only takes 768 channels)
        self.projector = nn.Linear(1024, 768)
        
        # Transformer Encoder Engine (The Brain)
        # We are only using 4 layers of attention which is enough (Saves VRAM, prevents overfitting)
        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # This will create a linear layer to give 3 probabilites
        self.classifier = nn.Linear(768, 3)

    # We will override the deafult train fumction of python
    def train(self, mode=True):
        # 1. This will switch on normal training
        super(HybridPneumoniaModel, self).train(mode)
        
        # 2. Then converting every batchnorm layers to 'eval'(Freeze) mode 
        for module in self.cnn_backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                module.weight.requires_grad = False
                module.bias.requires_grad = False

    # Updating the forward pass accroding to ViT-16
    def forward(self, x):
        
        # Phase-1 output from resnet
        features = self.cnn_backbone(x)          # Shape: [2, 1024, 30, 30]
        features = self.adaptive_pool(features)  # Shape: [2, 1024, 30, 30]
        
        # Phase-2: The Bridge (Image to Sequence(Tokens))
        # 1. Flatten the 30x30 spatial dimensions into 900 tokens
        features = features.flatten(2).transpose(1,2)           # Shape: [2, 1024, 900]-->[2, 900, 1024]
                
        # 3. Project 1024 -> 768
        tokens = self.projector(features)        # Shape: [2, 900, 768]
        
        # The Transformer Engine
        vit_output = self.transformer(tokens)    # Shape: [2, 900, 768]
        
        # Phase-3: The 3-Way Classification Head
        # 1. Global Average Pooling: We take the average across all 900 tokens (dim=1).
        # Since we don't need 900 different answers and need 1 master summary of the whole X-ray.
        vit_summary = vit_output.mean(dim=1)     # Shape: [2, 768] (That 1 is transfered in all 768 channels)
        
        # 2. Final Output: Passing the summary through a linear layer to get 3 probabilities
        # (Normal, Viral, Bacterial)
        final_prediction = self.classifier(vit_summary) # Shape: [2, 3]
        
        return final_prediction

import torch
import torch.nn as nn
import torch.nn.functional as F

# Im[plementing focal loss fumction
class FocalLoss(nn.Module):
    # Using gamma=2 which is optimal value in (1-pt)^gamma
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        # Convert list to tensor if user passed a normal list
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = alpha 

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            self.alpha = self.alpha.to(targets.device)
            
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        return focal_loss.mean()
