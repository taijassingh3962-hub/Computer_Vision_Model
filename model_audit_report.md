# Explainable AI (XAI) & Model Audit Report

This document explores how our AI makes its decisions. By using t-SNE to see how the model groups images, and Grad-CAM to see exactly where the model is looking, we can understand the AI beyond just standard accuracy numbers.

---

## 1. Overall Performance Metrics

Before looking at the visuals, here is how well the final model performed on 624 completely unseen test images:

* **DenseNet-121 Accuracy:** 83.8%
* **EfficientNet-B4 Accuracy:** 82.7%
* ** Final Ensemble Accuracy:** **84.6%** (Weighted F1 Score: 84.7%)

The ensemble correctly fixed many mistakes made by the individual models, especially in telling the difference between Bacterial and Viral infections!

---

## 2. How the AI Groups Images (t-SNE Analysis)

*(t-SNE takes the thousands of features the AI sees and squashes them down to a 2D graph so we can see what the AI is thinking.)*

![t-SNE Ensemble Features](visualizations/tsne_ensemble.png)

**Key Observations:**
1. **Healthy Lungs are Easy:** The model successfully groups the 'Normal' X-Rays into their own separate area. This proves the AI has learned a very strong baseline for what a healthy chest looks like.
2. **The Medical Overlap:** You will notice that the 'Bacterial' and 'Viral' dots mix together quite a bit. This actually reflects real-life biology! Early-stage viral and bacterial pneumonia both create very similar white foggy patches in the lungs.
3. **The "Radiologist Ceiling":** Because viral and bacterial infections look nearly identical on an X-Ray, the model has hit a natural limit. Perfect 100% separation is basically impossible using *only* visual pictures.

---

## 3. Where the AI is Looking (Grad-CAM Heatmaps)

*(Grad-CAM generates a heatmap over the X-ray. Red/Yellow areas show exactly where the AI was looking to make its diagnosis.)*

![Grad-CAM Results](visualizations/gradcam/gradcam_all_classes.png)

**Key Observations:**
1. **Targeting the Infection (The Good):** In the majority of cases, the AI works exactly as intended. The "hot" red zones are strictly inside the lung cavities, meaning the AI is correctly finding the localized fluid and pus buildups.
2. **Finding Shortcuts (The "Clever Hans" Effect):** AI is famously lazy and will find shortcuts to solve problems. In some edge cases, you can see the heatmap lighting up on the patient's collarbones (clavicles) or shoulders! The model secretly learned that 'Normal' healthy patients often stand up straighter during their X-Rays, so it uses their posture to guess instead of looking at the lungs.

---

## 4. Future Improvements & Next Steps

To push this model from a cool research prototype to a clinical-grade medical tool, we should make two major architectural changes:

### A. Crop the Lungs (Segmentation)
To stop the AI from cheating and looking at shoulders or collarbones, we can add a preprocessing step (using a model like U-Net). This step would automatically crop out the background and feed *only* the raw lung tissue to our DenseNet/EfficientNet models. 

### B. Add Patient Data (Multi-Modal Fusion)
To break past our current accuracy ceiling, we need to give the AI more than just pictures. By combining our CNN's visual data with tabular clinical data—like the patient's White Blood Cell (WBC) count, C-Reactive Protein (CRP) levels, and body temperature—the model can make a much more educated and accurate diagnosis!
