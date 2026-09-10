import streamlit as st
import os
os.environ["CHROMA_TELEMETRY_OPT_OUT"] = "TRUE"
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, Image as PILImage
import numpy as np
import pandas as pd
import matplotlib.cm as cm

# Import our robust backend functions!
from main_ensemble_model import (
    CONFIG, DEVICE, load_trained_models, get_val_transform
)

# =====================================================================
#                        STREAMLIT UI CONFIG
# =====================================================================

st.set_page_config(
    page_title="Pneumonia X-Ray Classifier",
    page_icon="🫁",
    layout="centered",
    initial_sidebar_state="expanded"
)

# =====================================================================
#                        GRAD-CAM & SPATIAL
# =====================================================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self.bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        input_tensor.requires_grad_(True)
        
        output = self.model(input_tensor)
        predicted_class = output.argmax(dim=1).item()
        if target_class is None:
            target_class = predicted_class

        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        
        if cam.max() > 0:
            cam = cam / cam.max()
            
        return cam, predicted_class

    def remove(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()

def get_anatomical_location(heatmap):
    """Translates the hottest region of the heatmap into text coordinates."""
    threshold = 0.75
    y_coords, x_coords = np.where(heatmap > threshold)
    
    if len(y_coords) == 0:
        return "diffuse or unlocalized (no strong focal point)"
        
    center_x = int(np.mean(x_coords))
    center_y = int(np.mean(y_coords))
    H, W = heatmap.shape
    
    # Radiological Left = Image Right (X > 58%)
    # Radiological Right = Image Left (X < 42%)
    if center_x < int(W * 0.42):
        side = "Patient's Right"
    elif center_x > int(W * 0.58):
        side = "Patient's Left"
    else:
        side = "Central/Mediastinal"
        
    if center_y < int(H * 0.5):
        zone = "Upper Zone"
    else:
        zone = "Lower Zone"
        
    location = f"{side} {zone}"
    
    # Cardiac region check: X in [47%, 73%], Y in [52%, 88%]
    if (int(W * 0.47) <= center_x <= int(W * 0.73)) and (int(H * 0.52) <= center_y <= int(H * 0.88)):
        location += " (Cardiac / Pericardial Region)"
        
    return location


# =====================================================================
#                     CACHED RAG DATABASE
# =====================================================================
@st.cache_resource
def load_rag_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # For Streamlit deployment or testing we might want to ensure the path exists
    if not os.path.exists("./main_folder/chroma_db") and os.path.exists("chroma_db"):
        db_path = "chroma_db"
    else:
        db_path = "./main_folder/chroma_db"
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    return db

# =====================================================================
#                     CACHED MODEL LOADING
# =====================================================================

@st.cache_resource
def load_ensemble():
    try:
        dense_models, effnet_models, meta_model = load_trained_models()
        transform = get_val_transform(CONFIG['input_size'])
        class_names = CONFIG['class_names']
        return dense_models, effnet_models, meta_model, transform, class_names, DEVICE
    except Exception as e:
        return None, None, None, None, [], f"Error loading models: {e}"


# =====================================================================
#                         MAIN APP LOGIC
# =====================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔑 Agentic-CDSS Access")
api_key = st.sidebar.text_input("Google Gemini API Key", type="password")
if api_key:
    genai.configure(api_key=api_key)

st.title("🫁 Pneumonia X-Ray Classifier")
st.markdown("""
**Ensemble Model:** 5-Fold Stacking Ensemble (DenseNet-121 + EfficientNet-B2)  
Upload a chest X-ray image below to classify it as **bacterial pneumonia**, **viral pneumonia**, or **normal**.
""")

dense_models, effnet_models, meta_model, transform, class_names, device = load_ensemble()

if dense_models is None:
    st.error(f"**Error Loading Models:**\n{device}")  # the 6th return val is the error string if it fails
    st.stop()

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    st.image(image, caption="Uploaded X-Ray", use_container_width=True)
    st.markdown("---")
    
    if st.button("🔍 Run Classification & 10-Fold Grad-CAM", type="primary", use_container_width=True):
        with st.spinner("Analyzing image across 10 models and extracting spatial features..."):
            
            input_tensor = transform(image).unsqueeze(0).to(device)

            # Setup GradCAM for all 10 models
            dense_gcs = [GradCAM(m, m.features.denseblock4) for m in dense_models]
            effnet_gcs = [GradCAM(m, m.features[-1]) for m in effnet_models]

            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    # 1. Forward Pass DenseNets
                    d_logits = [m(input_tensor) for m in dense_models]
                    d_avg = sum(d_logits) / len(d_logits)
                    
                    # 2. Forward Pass EfficientNets
                    e_logits = [m(input_tensor) for m in effnet_models]
                    e_avg = sum(e_logits) / len(e_logits)
                    
                    # 3. Meta-Classifier Final Decision
                    combined_logits = torch.cat([d_avg, e_avg], dim=1)
                    final_logits = meta_model(combined_logits)
                    final_probs = torch.softmax(final_logits, dim=1).cpu().numpy()[0]
                    
            predicted_idx = np.argmax(final_probs)
            predicted_class = class_names[predicted_idx]
            confidence = float(final_probs[predicted_idx])

            # Generate the 10-fold GradCAM
            input_size = 456
            
            # DenseNet CAMs
            dense_cams = []
            for gc in dense_gcs:
                cam, _ = gc.generate(input_tensor.clone(), target_class=predicted_idx)
                cam_res = np.array(PILImage.fromarray((cam * 255).astype(np.uint8)).resize((input_size, input_size), PILImage.BILINEAR)) / 255.0
                dense_cams.append(cam_res)
            avg_dense_cam = np.mean(dense_cams, axis=0)
            if avg_dense_cam.max() > 0: avg_dense_cam /= avg_dense_cam.max()

            # EfficientNet CAMs
            effnet_cams = []
            for gc in effnet_gcs:
                cam, _ = gc.generate(input_tensor.clone(), target_class=predicted_idx)
                cam_res = np.array(PILImage.fromarray((cam * 255).astype(np.uint8)).resize((input_size, input_size), PILImage.BILINEAR)) / 255.0
                effnet_cams.append(cam_res)
            avg_effnet_cam = np.mean(effnet_cams, axis=0)
            if avg_effnet_cam.max() > 0: avg_effnet_cam /= avg_effnet_cam.max()

            # Ensemble CAM
            cam_ensemble = (avg_dense_cam + avg_effnet_cam) / 2.0
            if cam_ensemble.max() > 0: cam_ensemble /= cam_ensemble.max()
            
            anatomical_location = get_anatomical_location(cam_ensemble)

            # Display Image Overlay
            display_tensor = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
            ])(image)
            display_img = display_tensor.permute(1, 2, 0).numpy()
            
            heatmap_colors = cm.jet(cam_ensemble)[:, :, :3]
            overlay = np.clip(0.5 * display_img + 0.5 * heatmap_colors, 0, 1)
            
            # Cleanup hooks
            for gc in dense_gcs: gc.remove()
            for gc in effnet_gcs: gc.remove()

            # Individual Probabilities for DataFrame
            probs_d = torch.softmax(d_avg, dim=1).cpu().numpy()[0]
            probs_e = torch.softmax(e_avg, dim=1).cpu().numpy()[0]

            df = pd.DataFrame({
                "Class": class_names,
                "DenseNet-121 (5-Fold Avg)": [f"{p:.2%}" for p in probs_d],
                "EfficientNet-B2 (5-Fold Avg)": [f"{p:.2%}" for p in probs_e],
                "Meta-Classifier Final": [f"{p:.2%}" for p in final_probs]
            })

            # Save to session state
            st.session_state['ml_results'] = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'anatomical_location': anatomical_location,
                'overlay': overlay,
                'df': df,
                'predicted_idx': predicted_idx
            }

    # If results exist in session state, render them outside the button click!
    if 'ml_results' in st.session_state:
        res = st.session_state['ml_results']
        
        st.subheader(f"Prediction: **{res['predicted_class'].upper()}**")
        st.progress(res['confidence'], text=f"Confidence: {res['confidence']:.1%}")
        
        if res['predicted_class'] != "normal":
            st.warning(f"**Spatial Extraction:** The highest infection opacity is localized in the **{res['anatomical_location']}**.")
        else:
            st.success("**Spatial Extraction:** The lungs appear clear. Any minor activations map to standard anatomical structures.")

        st.markdown("### 10-Fold Grad-CAM Attention Heatmap")
        st.image(res['overlay'], caption="Ensemble Grad-CAM Overlay", use_container_width=True)

        with st.expander("📊 View Detailed Model Breakdown"):
            st.write("The final prediction is driven by the Meta-Classifier aggregating 10 base models:")
            def highlight_max(s):
                is_max = s == res['df']["Meta-Classifier Final"].iloc[res['predicted_idx']]
                return ['background-color: #2ECC71' if v else '' for v in is_max]
            st.dataframe(res['df'].style.apply(highlight_max, subset=['Meta-Classifier Final']), use_container_width=True)

        st.markdown("---")
        
        # Agentic Report Button
        if st.button("📝 Generate Agentic Clinical Report", type="primary", use_container_width=True):
            if not api_key:
                st.error("Please enter your Gemini API Key in the sidebar first!")
            else:
                with st.spinner("Agentic-CDSS is analyzing guidelines and drafting report..."):
                    try:
                        db = load_rag_db()
                        query = f"{res['predicted_class']} pneumonia treatment protocol and characteristics"
                        docs = db.similarity_search(query, k=2)
                        context = "\n\n".join([doc.page_content for doc in docs])
                        
                        prompt = f"""You are an expert AI clinical assistant (Agentic-CDSS).
A chest X-ray has been analyzed by a 10-model deep learning vision ensemble.
- Predicted Condition: {res['predicted_class'].upper()}
- Confidence: {res['confidence']:.1%}
- Anatomical Focal Point: {res['anatomical_location']}

Here are the strict medical guidelines from the hospital's knowledge base:
{context}

Based ON THESE GUIDELINES ONLY, write a short, professional "Preliminary Clinical Report" consisting of:
1. Imaging Findings (summarize the condition and location).
2. Recommended Protocol (what should the doctor do next according to the guidelines).

Format the output clearly using Markdown. Be concise and clinical. Do not hallucinate treatments outside the provided guidelines.
"""
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        response = model.generate_content(prompt)
                        
                        st.session_state['agent_report'] = response.text
                    except Exception as e:
                        st.error(f"Agentic Engine Error: {e}")

        # Render report if it exists
        if 'agent_report' in st.session_state:
            st.markdown("### 📝 Agentic-CDSS: Preliminary Clinical Report")
            
            if not st.session_state.get('edit_mode', False):
                st.info(st.session_state['agent_report'])
                
                col1, col2 = st.columns(2)
                if col1.button("✅ [APPROVE] Append to EHR", use_container_width=True):
                    st.success("Report successfully approved and appended to patient's Electronic Health Record (EHR).")
                
                if col2.button("✏️ [EDIT] Manual Override", use_container_width=True):
                    st.session_state['edit_mode'] = True
                    st.rerun()
            else:
                edited_report = st.text_area("Edit Clinical Report:", value=st.session_state['agent_report'], height=300)
                col3, col4 = st.columns(2)
                if col3.button("💾 Save & Approve", type="primary", use_container_width=True):
                    st.session_state['agent_report'] = edited_report
                    st.session_state['edit_mode'] = False
                    st.rerun()
                if col4.button("❌ Cancel", use_container_width=True):
                    st.session_state['edit_mode'] = False
                    st.rerun()

st.sidebar.markdown("""
### ℹ️ About the Model
This AI model takes in a 456x456 pixel chest X-ray and uses a 10-model deep learning ensemble to diagnose pneumonia.

**Architecture (Phase 7):**
- 5 Folds: DenseNet-121
- 5 Folds: EfficientNet-B2
- Meta-Classifier Aggregation
- 10-Fold Averaged Grad-CAM
""")
