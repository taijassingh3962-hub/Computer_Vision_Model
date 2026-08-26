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

## Research Direction

The architectural evolution ultimately moved from:

**ResNet → ResNet + ViT → DenseNet + ViT → Pure DenseNet-121 → Ensemble(DenseNet + EffecientNet)**

The key research decision was not to use the most complex architecture possible, but to identify the architecture that provides the best balance between:

* Clinical feature preservation
* Generalization on limited medical data
* VRAM efficiency
* Training stability
* High-resolution input capability
* Compatibility with the downstream Agentic AI pipeline
