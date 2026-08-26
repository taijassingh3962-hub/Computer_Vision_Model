# Agentic-CDSS: Edge-Optimized Clinical Decision Support System

## Overview
State-of-the-art medical AI models often require massive cloud infrastructure, making them impractical for real-world clinics with constrained hardware. Furthermore, traditional classification models lack the explainability required for clinical trust.

This project is an Applied AI research initiative building an end-to-end Clinical Decision Support System (CDSS) for Pneumonia detection. It is engineered to operate under strict edge-hardware constraints (≤ 6GB VRAM) by leveraging an optimized deep learning vision ensemble, while maintaining high diagnostic accuracy and integrating a Human-in-the-Loop Agentic LLM workflow.

## Core Architecture (Vision Engine)
The foundation of this system is a Hybrid Vision architecture that combines local feature extraction with global contextual understanding, strictly optimized for low-VRAM limits.

* **Hardware-Aware Ensemble:** The architecture utilizes a probabilistically averaged ensemble of **DenseNet-121** and **EfficientNet-B4**. This achieves state-of-the-art accuracy without exceeding the 6GB VRAM training constraint. 
* **Clinically Accurate Data Pipeline:** Unlike standard generic image datasets, medical X-rays have fixed anatomical orientations. The DataLoader strictly disables unrealistic augmentations (e.g., vertical flips) and relies on ColorJitter (brightness/contrast) to simulate variations across different hospital X-ray machines.
* **Handling Clinical Imbalance:** Medical datasets naturally suffer from class imbalance (e.g., Bacterial cases heavily outnumber Viral cases). The training loop utilizes Focal Loss to penalize the majority class and prioritizes Weighted F1-Score as the primary evaluation metric.

## Completed Roadmap & Features

### ✅ Phase 1: Spatial Reasoning & Explainability
* **[x]** Integrated **Grad-CAM** to generate visual heatmaps of the predictions.
* **[x]** Extracted the spatial coordinates (quadrants) of the highest infection opacity to determine anatomical proximity (e.g., "infection localized near the cardiac region").

### ✅ Phase 2: Agentic LLM Integration (RAG)
* **[x]** Implemented a **Retrieval-Augmented Generation (RAG)** pipeline using LangChain, ChromaDB, and local vector embeddings.
* **[x]** The Vision Engine passes the diagnostic prediction and spatial heatmap text data to the Agentic LLM (**Google Gemini 3.6 Flash**).
* **[x]** The LLM automatically drafts a "Preliminary Clinical Report" and medication protocol by retrieving context strictly from local medical guidelines.

### ✅ Phase 3: Human-in-the-Loop Dashboard
* **[x]** Developed a lightweight, interactive web interface using **Streamlit**.
* **[x]** The dashboard displays the uploaded X-Ray, the Grad-CAM heatmap, and the AI-generated preliminary report side-by-side.
* **[x]** **Crucial Feature:** An `[APPROVE]` and `[EDIT]` workflow is fully implemented, ensuring the AI acts strictly as a clinical assistant and leaving the final diagnostic authority to the doctor.

## 🛠️ Tech Stack
* **Deep Learning Framework:** PyTorch, Torchvision
* **Vision Architecture:** DenseNet-121 & EfficientNet-B4 Ensemble
* **Explainable AI:** Custom PyTorch Grad-CAM Hook Implementation
* **Agentic AI & RAG:** LangChain, ChromaDB, HuggingFace Embeddings (`all-MiniLM-L6-v2`)
* **LLM Engine:** Google Gemini 3.6 Flash API
* **Frontend UI:** Streamlit
