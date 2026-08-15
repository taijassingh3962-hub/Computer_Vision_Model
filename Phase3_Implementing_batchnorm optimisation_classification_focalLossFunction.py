import torch
import torch.nn as nn
import torchvision.models as models


# Using the same code of phase-1 and phase-2. In this phase we will include optimixation(btachnorm), classification process and focal loss function.

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
        
        # Phase-3: The 3-Way Classification Head
        # 1. Global Average Pooling: We take the average across all 196 tokens (dim=1).
        # Since we don't need 196 different answers and need 1 master summary of the whole X-ray.
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
    # Using gamma=2 which is optimal value in (1-pt)^gamma.
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

# Test run
import torch
import torch.nn as nn
import torch.optim as optim # From here we will use optimsing function like AdamW

if __name__ == "__main__":
    print("Starting the Model and Calling the Optimizer...\n")

    # 1. Instantiate the Model
    model = HybridPneumoniaModel()
    
    # 2. Instantiate the Focal Loss
    criterion = FocalLoss(gamma=2.0, alpha=[0.66, 1.0, 2.0])
    
    # 3. Instantiate the Optimizer
    
    # We are using lr=0.001 for now and increase it further for more accuracy and f1 score
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # DUMMY DATA
    dummy_xrays = torch.randn(2, 3, 224, 224) 
    dummy_targets = torch.tensor([0, 1]) # Image 1 is Normal, Image 2 is Viral

    # THE 5-STEP TRAINING LOOP (Model Improvement Loop)

    # Step 1: FORWARD PASS
    # Model sees the two x-rays and creates gusses (Logits)
    predictions = model(dummy_xrays) 
    
    # Step 2: CALCULATE LOSS
    # Compares the gussess of model from target
    loss = criterion(predictions, dummy_targets)
    print(f"Step 2: Initial Loss: {loss.item():.4f}")

    # Step 3: ZERO GRADIENTS 
    # Clears the gradient of previous learning to create gradients from new batches
    optimizer.zero_grad()
    
    # Step 4: BACKPROPAGATION (Find the exact faults)
    # This Cheaks which part of model made mistake
    loss.backward()
    print("Step 4: Backpropagation done! (Faults identified)")

    # Step 5: OPTIMIZER STEP
    # Optimizer adjects that perticular weights to get minimum loss in next learning
    optimizer.step()
    print("Step 5: Weights updated! Model has IMPROVED.\n")
