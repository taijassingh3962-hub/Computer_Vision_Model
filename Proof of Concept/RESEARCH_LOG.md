# Architecture Evolution & Research Log

**Project:** Edge-Optimized Pneumonia CDSS (Clinical Decision Support System)
**Constraint:** Strict ≤ 6GB VRAM Environment
**Dataset:** ~5,000 Clinical X-Rays

This document tracks the R&D thought process, architectural pivots, and engineering decisions made to optimize the vision engine for clinical accuracy and edge-hardware deployment.

## Phase 1: The Baseline — Pure ResNet

**The Approach:**
Started with a standard pre-trained ResNet as a baseline feature extractor.

**The Problem:**
While ResNet is effective for general visual feature extraction, it suffered from **feature washout** (only allows skip connection via Addition) in deeper layers. It struggled to retain the fine-grained micro-opacities crucial for detecting early-stage pneumonia.

**The Pivot:**
Realized the model needed **global context** to understand relationships between different regions of the lungs.

## Phase 2: The Global Context — ResNet + ViT-16

**The Approach:**
Integrated a Vision Transformer (ViT) on top of the CNN backbone to capture long-range dependencies across the X-ray.

**The Problem:**
ViT-16 generated too many tokens, immediately causing **CUDA Out-Of-Memory (OOM)** errors under the 6GB VRAM constraint, even at a low batch size.

**The Pivot:**
A mathematical method was needed to strictly control token generation, along with a better CNN backbone capable of preserving low-level medical features.

## Phase 3: Hardware Math & Preservation — DenseNet-121 + ViT

**The Approach:**
Replaced ResNet with **DenseNet-121**. DenseNet's dense connections provided better preservation of fine-grained features(Uses Skip Connection via Concatenation). The downsampling grid was mathematically tuned to generate exactly **196 tokens**, and later **900 tokens**, while remaining within the VRAM constraint.

**The Problem:**
Empirical testing revealed a critical flaw: Transformers are highly data-hungry. With a limited dataset of approximately 5,000 images, the ViT component was prone to overfitting.

Additionally, the ViT consumed approximately **99% of the available GPU memory**, leaving insufficient resources for the planned Agentic LLM integration.

**The Pivot:**
Decided to remove the Transformer entirely to reduce overfitting risk, increase batch size, and free computational resources for the final CDSS pipeline.

## Phase 4: The Specialist Model — Pure DenseNet-121

**Status:** Current

**The Approach:**
Pivoted to a purely optimized **DenseNet-121** with a custom 3-class classifier head:

* Normal
* Bacterial
* Viral

Added a **Dropout(0.3)** layer to reduce overfitting on the relatively small dataset.

### The Engineering Win

**VRAM Monopoly:**
Removing the ViT significantly reduced VRAM consumption. This allowed the input resolution to be increased to an extreme **850×850**, enabling the model to retain finer clinical details.

**Batch Size Boost:**
The reduced memory footprint allowed a larger training batch size, resulting in more stable gradients and faster convergence.

**Room for the LLM:**
The freed VRAM creates the computational headroom required for the next stage: running an **Agentic LLM** alongside the vision engine to generate preliminary clinical reports using DenseNet's Grad-CAM outputs.

### Phase 5: The Dual-CNN Fusion Experiment — DenseNet-121 + EfficientNet-B1

**Status:** Prototype Workflow Completed 

**The Approach:**
Designing a **Feature Fusion** architecture by combining DenseNet-121 and EfficientNet-B1. The final layers extract features from both backbones:

* DenseNet-121: **1024 features**
* EfficientNet-B1: **1280 features**
* Combined feature vector: **2304 features**

These features are concatenated and passed through a custom 3-class classifier with a `Dropout(0.3)` layer.

**The Hypothesis:**
Since the ViT proved too data-hungry for the ~5,000-image dataset, this dual-CNN architecture aims to combine complementary feature representations without the parameter overhead of a large Transformer.

* **DenseNet-121 — "The Detailer":** Preserves fine-grained, low-level features and micro-opacities through its dense connectivity pattern.
* **EfficientNet-B1 — "The Scaler":** Provides efficient multi-scale feature extraction and captures broader structural patterns within the lungs.
* **Feature Fusion:** Combining both representations may provide a richer feature space while keeping the overall architecture substantially smaller than the ViT-based approach.

The combined architecture contains approximately **16M parameters**, compared with approximately **86M parameters** for the ViT configuration, potentially reducing overfitting risk on the relatively small clinical dataset.

**The Engineering Challenge:**
Running two independent CNN backbones simultaneously on high-resolution **850×850** X-rays creates substantially larger activation-memory requirements. This poses a significant challenge under the strict **6GB VRAM** constraint.

**The Mitigation Strategy:**
To prevent CUDA Out-of-Memory (OOM) errors during the fusion experiment, two memory-efficiency techniques are being introduced:

* **Mixed Precision Training (AMP):** Uses lower-precision computation where appropriate to substantially reduce activation and gradient memory consumption.
* **Gradient Accumulation:** Accumulates gradients across multiple smaller micro-batches, allowing the model to achieve a larger **effective batch size** without requiring the entire batch to reside in VRAM simultaneously.(Optional)

### Research Objective

This experiment tests whether **architectural diversity between two lightweight CNNs** can provide better clinical feature representation than a single DenseNet-121, while remaining feasible on edge hardware.

The experiment will ultimately be evaluated against the current **Pure DenseNet-121 baseline** using metrics such as:

* Weighted F1-Score
* Per-class F1-Score
* Validation loss
* Generalization gap
* VRAM consumption
* Training throughput

If the fusion model improves predictive performance without violating the 6GB VRAM constraint or introducing significant overfitting, it can become the new candidate vision engine for the downstream **Grad-CAM → Agentic LLM → Human-in-the-Loop CDSS** pipeline.

# Phase 6 — The Ultimate Fusion & Optimization Arsenal

### DenseNet-121 + EfficientNet-B4

Phase 6 represents the final evolution of the dual-CNN architecture. After proving the viability of feature-level fusion with **DenseNet-121 + EfficientNet-B1** in Phase 5, the next objective was to push the architecture further while remaining within the constraints of a **6GB VRAM consumer GPU**.

The B1 backbone was replaced with the substantially larger **EfficientNet-B4**, creating a high-capacity multi-scale feature extraction pipeline capable of learning richer structural patterns from medical images.

---

## Final Architecture

The model combines two complementary convolutional backbones:

| Backbone            | Role                                  |      Features |
| ------------------- | ------------------------------------- | ------------: |
| **DenseNet-121**    | Detail extraction                     |         1,024 |
| **EfficientNet-B4** | Multi-scale structural representation |         1,792 |
| **Fusion**          | Concatenated representation           |     **2,816** |
| **Classifier Head** | 3-class prediction                    | **3 classes** |

The extracted representations from both networks are concatenated:

```text
DenseNet-121
     │
     └── 1024 features
             │
             ├──────────────┐
             │              │
             │         Feature Fusion
             │              │
             │              ▼
             │        2816 features
             │              │
EfficientNet-B4             │
     │                      │
     └── 1792 features ─────┘
                            │
                            ▼
                     Custom Classifier
                            │
                            ▼
                     3-Class Output
```

The final fused representation contains **2,816 features**, providing the classifier with complementary information from both architectures.

---

# The Ultimate VRAM Battle

Running DenseNet-121 and EfficientNet-B4 simultaneously at high resolution on a **6GB GPU** creates a severe memory bottleneck.

A straightforward implementation can quickly result in:

```text
CUDA out of memory
```

Instead of reducing the architecture, Phase 6 focused on engineering the training pipeline around the available hardware.

The final system combined multiple optimization strategies developed throughout the previous phases.

---

# 1. VRAM Survival — AMP + Gradient Accumulation

The dual-backbone architecture required extremely careful memory management.

### Automatic Mixed Precision

Both forward passes were executed using PyTorch's automatic mixed precision:

```python
with torch.autocast(device_type="cuda"):
    dense_features = densenet(images)
    efficient_features = efficientnet(images)
```

A `GradScaler` was used to maintain numerical stability during mixed-precision training.

### Gradient Accumulation

Because the model could not support a large physical batch size on 6GB VRAM, gradient accumulation was used to simulate a larger effective batch.

Conceptually:

```text
Micro-batch 1 ──┐
Micro-batch 2 ──┤
Micro-batch 3 ──┼──► Accumulated Gradients ──► Optimizer Step
Micro-batch 4 ──┘
```

This allowed the model to benefit from a larger effective batch without requiring all samples to exist in GPU memory simultaneously.

---

# 2. Progressive Unfreezing

Immediately fine-tuning both large pre-trained backbones would be unnecessarily aggressive and potentially unstable.

The training process therefore began with both feature extractors completely frozen.

### Stage 1 — Classifier Warm-Up

```text
DenseNet-121      ❄ Frozen
EfficientNet-B4   ❄ Frozen
                     │
                     ▼
              Custom Classifier
                     │
                     ▼
                 Training
```

Only the newly initialized classification head was trained initially.

This allowed the classifier to learn how to interpret the combined 2,816-dimensional feature representation before modifying the pre-trained feature extractors.

### Stage 2 — Progressive Unfreezing

After the classification head had stabilized, deeper blocks of both networks were progressively unfrozen at predefined epochs.

```text
Epochs ───────────────────────────────────────►

Classifier      ███████████████████████████████
Deep Blocks     ░░░░░░████████████████████████
Mid Blocks      ░░░░░░░░░░░░██████████████████
Earlier Blocks  ░░░░░░░░░░░░░░░░░░████████████
```

This provided a controlled transition from feature reuse to task-specific fine-tuning.

---

# 3. 🎚️ Differential Learning Rates

The newly initialized classifier required significantly larger updates than the pre-trained backbones.

Therefore, separate learning rates were assigned to different parts of the architecture.

```text
Classifier LR
     │
     └── 100%

Backbone LR
     │
     └── 10% of classifier LR
```

The reduced backbone learning rate helped preserve useful ImageNet-pretrained representations while allowing the networks to gradually adapt to the medical imaging task.

The training strategy can be summarized as:

```text
New Classifier
      │
      │ High LR
      ▼
Fast adaptation

Pre-trained Backbones
      │
      │ Low LR
      ▼
Controlled adaptation
```

This reduced the risk of **catastrophic forgetting** during fine-tuning.

---

# 4. Class Imbalance Without Large Batches

The dataset contained an imbalance between the **Normal** class and the **Viral/Bacterial** classes.

Normally, increasing batch size can help provide more representative batches. However, the dual-CNN architecture made large batches impractical.

Instead, imbalance was addressed directly at the data-sampling and loss levels.

---

## WeightedRandomSampler

A `WeightedRandomSampler` was used at the DataLoader level to increase the probability of sampling underrepresented classes.

Instead of relying on naturally occurring class frequencies:

```text
Original distribution

Normal     ███████████████
Viral      ███████
Bacterial  █████
```

sampling encouraged a more balanced training exposure:

```text
Sampled distribution

Normal     ███████████
Viral      ███████████
Bacterial  ███████████
```

This allowed the model to see minority-class examples more frequently without increasing GPU memory consumption.

---

# 5. Focal Loss

Sampling alone was not considered sufficient.

A custom **Focal Loss** implementation was also introduced to reduce the influence of easy, already-correct predictions and place greater emphasis on difficult examples.

The loss was configured with:

```text
γ = 1.0
```

The underlying idea is:

```text
Easy / highly confident prediction
            │
            ▼
       Lower influence

Difficult prediction
            │
            ▼
       Higher influence
```

Combined with weighted sampling, this encouraged the model to pay greater attention to difficult and minority-class examples.

---

# Complete Training Strategy

The final Phase 6 pipeline therefore combined architectural fusion with a full optimization stack:

```text
                    Input Image
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
        DenseNet-121          EfficientNet-B4
              │                     │
          1024 feat.            1792 feat.
              │                     │
              └──────────┬──────────┘
                         │
                         ▼
                  Feature Fusion
                         │
                  2816 Features
                         │
                         ▼
                 Custom Classifier
                         │
                         ▼
                    3 Classes
```

Training was supported by:

```text
                 Phase 6 Training Stack
                         │
       ┌─────────────────┼─────────────────┐
       │                 │                 │
       ▼                 ▼                 ▼
      AMP        Gradient Accumulation   Sampling
       │                 │                 │
       └─────────────────┼─────────────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
      Progressive             Focal Loss
       Unfreezing                 │
              │                   │
              └─────────┬─────────┘
                        ▼
             Differential Learning Rates
                        │
                        ▼
                  Final Fine-Tuning
```

---

# Why This Architecture?

The motivation behind the fusion was not simply to increase parameter count.

The two networks were selected for complementary representation capabilities:

### DenseNet-121 — The Detailer

DenseNet's densely connected architecture encourages feature reuse and allows information from earlier layers to propagate throughout the network.

Its role in the fusion was primarily to capture **fine-grained visual details and local structural patterns**.

### EfficientNet-B4 — The Heavy Scaler

EfficientNet-B4 provides substantially greater representational capacity than the B1 backbone used in Phase 5.

Its role was to capture **richer multi-scale and higher-level structural representations**.

The hypothesis was that combining these representations would provide a more informative feature space than relying on either backbone independently.

---

# Hardware-Constrained Deep Learning

One of the primary goals of Phase 6 was demonstrating that a relatively heavy architecture could still be trained under strict consumer-hardware limitations.

### Hardware Constraint

```text
GPU VRAM: 6GB
```

### Architectural Challenge

```text
DenseNet-121
      +
EfficientNet-B4
      +
High-resolution inputs
      +
Training gradients
      +
Optimizer states
      ↓
Severe VRAM pressure
```

### Engineering Solution

```text
AMP
+
Gradient Accumulation
+
Frozen Backbones
+
Progressive Unfreezing
+
Differential Learning Rates
+
WeightedRandomSampler
+
Focal Loss
      ↓
Trainable dual-CNN pipeline
```

The result was not achieved through a single optimization trick, but through the interaction of multiple memory and training strategies.

---

# Final Verdict

Phase 6 became the **endgame of the dual-CNN experimentation series**.

Rather than continuing to increase architectural complexity through increasingly memory-intensive transformer models, the approach returned to highly optimized convolutional architectures and focused on extracting more performance from them through feature fusion and training engineering.

The final system combined:

* **DenseNet-121**
* **EfficientNet-B4**
* **2,816-dimensional fused feature representation**
* **Custom 3-class classifier**
* **Automatic Mixed Precision**
* **Gradient Accumulation**
* **Progressive Unfreezing**
* **Differential Learning Rates**
* **WeightedRandomSampler**
* **Custom Focal Loss**
* **6GB VRAM-aware training**

The key achievement of Phase 6 was therefore not simply building a larger CNN.

It was demonstrating that **careful architectural design and training-system engineering can make a computationally demanding multi-backbone medical imaging pipeline feasible even under severe consumer-GPU constraints.**

---

## Phase 6 in One Line

> **DenseNet-121 + EfficientNet-B4 + aggressive VRAM optimization + controlled fine-tuning = the final dual-CNN fusion pipeline.**
