# Architecture Evolution & Research Log
**Project:** Edge-Optimized Pneumonia CDSS (Clinical Decision Support System)  
**Constraint:** Strict ≤ 6GB VRAM Environment  
**Dataset:** ~5,000 Clinical X-Rays  

This document tracks the R&D thought process, architectural pivots, and engineering decisions made to optimize the vision engine for clinical accuracy and edge-hardware deployment.

---

## Phase 1: The Baseline — Pure ResNet
* **The Approach:** Started with a standard pre-trained ResNet as a baseline feature extractor.
* **The Problem:** While ResNet is effective for general visual feature extraction, it suffered from feature washout (only allows skip connection via Addition) in deeper layers. It struggled to retain the fine-grained micro-opacities crucial for detecting early-stage pneumonia.
* **The Pivot:** Realized the model needed global context to understand relationships between different regions of the lungs.

---

## Phase 2: The Global Context — ResNet + ViT-16
* **The Approach:** Integrated a Vision Transformer (ViT) on top of the CNN backbone to capture long-range dependencies across the X-ray.
* **The Problem:** ViT-16 generated too many tokens, immediately causing CUDA Out-Of-Memory (OOM) errors under the 6GB VRAM constraint, even at a low batch size.
* **The Pivot:** A mathematical method was needed to strictly control token generation, along with a better CNN backbone capable of preserving low-level medical features.

---

## Phase 3: Hardware Math & Preservation — DenseNet-121 + ViT
* **The Approach:** Replaced ResNet with DenseNet-121. DenseNet's dense connections provided better preservation of fine-grained features (Uses Skip Connection via Concatenation). The downsampling grid was mathematically tuned to generate exactly 196 tokens, and later 900 tokens, while remaining within the VRAM constraint.
* **The Problem:** Empirical testing revealed a critical flaw: Transformers are highly data-hungry. With a limited dataset of approximately 5,000 images, the ViT component was prone to overfitting. Additionally, the ViT consumed approximately 99% of the available GPU memory, leaving insufficient resources for the planned Agentic LLM integration.
* **The Pivot:** Decided to remove the Transformer entirely to reduce overfitting risk, increase batch size, and free computational resources for the final CDSS pipeline.

---

## Phase 4: The Specialist Model — Pure DenseNet-121
* **The Approach:** Pivoted to a purely optimized DenseNet-121 with a custom 3-class classifier head (Normal, Bacterial, Viral). Added a Dropout(0.3) layer to reduce overfitting on the relatively small dataset.
* **The Engineering Win:**
  * **VRAM Monopoly:** Removing the ViT significantly reduced VRAM consumption. This allowed the input resolution to be increased to an extreme 850×850, enabling the model to retain finer clinical details.
  * **Batch Size Boost:** The reduced memory footprint allowed a larger training batch size, resulting in more stable gradients and faster convergence.
  * **Room for the LLM:** The freed VRAM creates the computational headroom required for the next stage.

---

## Phase 5: The Dual-CNN Fusion Experiment — DenseNet-121 + EfficientNet-B1
* **The Approach:** Designed a Feature Fusion architecture combining DenseNet-121 (1024 features) and EfficientNet-B1 (1280 features) into a 2304-feature vector. 
* **The Hypothesis:** Since the ViT proved too data-hungry, this dual-CNN architecture aimed to combine complementary feature representations without the massive parameter overhead. 
  * *DenseNet-121 ("The Detailer")* preserves fine-grained features.
  * *EfficientNet-B1 ("The Scaler")* provides multi-scale feature extraction.
* **The Mitigation Strategy:** To prevent OOM errors, Mixed Precision Training (AMP) and Gradient Accumulation were utilized.

---

## Phase 6: The Ultimate Fusion & Optimization Arsenal (DenseNet-121 + EfficientNet-B4)
* **The Approach:** Pushed the architecture further by replacing B1 with the substantially larger EfficientNet-B4, creating a high-capacity 2,816-dimensional fused feature representation.
* **The VRAM Battle:** To prevent severe VRAM pressure, we deployed an arsenal of training optimization strategies:
  * **AMP + Gradient Accumulation** for memory management.
  * **Progressive Unfreezing:** Stage 1 (Classifier Warm-Up) followed by Stage 2 (Deep Blocks).
  * **Differential Learning Rates:** High LR for the new classifier, 10% LR for the pre-trained backbones.
  * **WeightedRandomSampler & Focal Loss:** Handled the heavy class imbalance without requiring massive batches.

---

## Phase 7: The Production Pipeline — 5-Fold Stratified Meta-Ensemble (DenseNet-121 + EfficientNet-B2)
**Status: Final Deployed Architecture**

* **The Problem with Phase 6:** While powerful, training dual models simultaneously on a single 80/20 split was highly susceptible to "lucky splits" (overfitting to the validation set). It did not provide the rigorous cross-validation needed to guarantee clinical-grade robustness on truly unseen data. Furthermore, B4 proved to be slightly too large for rapid iteration.
* **The Pivot:** We transitioned from a single "Feature Fusion" model to a true **OOF (Out-Of-Fold) Stacking Ensemble** using a dedicated **Meta-Classifier**.
* **The Final Implementation:**
  1. **Architecture Downsize & Resolution Upscale:** Downgraded EfficientNet-B4 to **EfficientNet-B2** to free up VRAM, but utilized that free memory to permanently fix the input resolution to a highly optimal **456x456**.
  2. **Strict 5-Fold Cross Validation:** Implemented `StratifiedKFold` to train 5 distinct DenseNets and 5 distinct EfficientNets on completely distinct data slices, mathematically eliminating data-leakage and guaranteeing generalization.
  3. **The Meta-Classifier:** Instead of simple concatenation inside a single model, we trained a dedicated Meta-Classifier Neural Network. It takes the independent 3D logit predictions from the Averaged DenseNet (5 Folds) and the Averaged EfficientNet (5 Folds) and learns how to aggregate them based on their respective strengths.
* **The Engineering Win:** 
  * The final Meta-Classifier Ensemble achieved an incredibly robust **83.1% Weighted F1 Score** on a perfectly locked, completely unseen test set (20% holdout).
  * It proved capable of detecting healthy (Normal) lungs with **96.25% accuracy**.
  * By generating an averaged Grad-CAM across all 10 independent models, we completely eliminated the "Clever Hans" effect (where a single model cheats by looking at background noise), providing an incredibly focused and clinically trustworthy heatmap.
