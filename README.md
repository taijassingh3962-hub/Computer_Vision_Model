# 🩺 Agentic-CDSS: Edge-Optimized Clinical Decision Support System

## 📌 Overview

State-of-the-art medical AI models often require massive cloud infrastructure, making them impractical for real-world clinics with constrained hardware. Furthermore, traditional classification models lack the explainability required for clinical trust.

This project is an **Applied AI** research initiative to build an end-to-end Clinical Decision Support System (CDSS) for Pneumonia detection. It is specifically engineered to operate under strict edge-hardware constraints (≤ 6GB VRAM) while maintaining high diagnostic accuracy and integrating a **Human-in-the-Loop** Agentic LLM workflow.

## 🧠 Core Architecture (Vision Engine)

The foundation of this system is a Hybrid Vision architecture that combines local feature extraction with global contextual understanding, strictly optimized for low-VRAM limits.

* **Hardware-Aware Downsampling:** The architecture mathematically constrains the high-resolution input grid to generate an exact, optimized number of tokens for the Vision Transformer (ViT). This prevents OOM (Out of Memory) errors while preserving critical micro-opacities in X-rays.
* **Clinically Accurate Data Pipeline:** Unlike standard generic image datasets, medical X-rays have fixed anatomical orientations. The DataLoader strictly disables unrealistic augmentations (e.g., vertical flips) and relies on ColorJitter (brightness/contrast) to simulate variations across different hospital X-ray machines.
* **Handling Clinical Imbalance:** Medical datasets naturally suffer from class imbalance (e.g., Bacterial cases heavily outnumber Viral cases). Instead of standard Cross-Entropy, the model utilizes **Focal Loss** to penalize the majority class and prioritizes **Weighted F1-Score** as the primary evaluation metric.

## 🚀 Upcoming Commits & Roadmap

The ultimate goal is to transition this from a standalone Vision Model to an **Agentic AI Workflow**. The following phases are actively being developed:

### Phase 1: Spatial Reasoning & Explainability

* [ ] Integrate Grad-CAM to generate visual heatmaps of the predictions.
* [ ] Extract the spatial coordinates (quadrants) of the highest infection opacity to determine anatomical proximity (e.g., "infection localized near the cardiac region").

### Phase 2: Agentic LLM Integration

* [ ] Implement a Retrieval-Augmented Generation (RAG) pipeline using an LLM.
* [ ] The Vision Engine will pass the diagnostic prediction and spatial heatmap text data to the Agentic LLM.
* [ ] The LLM will automatically draft a "Preliminary Clinical Report" and medication protocol based on medical guidelines.

### Phase 3: Human-in-the-Loop Dashboard (Streamlit/Gradio)

* [ ] Develop a lightweight, interactive web interface for medical professionals.
* [ ] The dashboard will display the X-Ray, the Grad-CAM heatmap, and the AI-generated preliminary report.
* [ ] **Crucial Feature:** An `[APPROVE]` or `[EDIT]` workflow, ensuring the AI acts strictly as an assistant, leaving the final decision to the doctor.

## 🛠️ Tech Stack

* **Deep Learning Framework:** PyTorch
* **Architectures:** CNN Backbones, Vision Transformers (ViT)
* **Agentic AI:** LangChain / LlamaIndex *(Upcoming)*
* **Frontend UI:** Streamlit / Gradio *(Upcoming)*
