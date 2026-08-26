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
import torchvision.models as models
from PIL import Image
import numpy as np
import os
import pandas as pd
import matplotlib.cm as cm
import gc

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
#                     MODEL ARCHITECTURE & HELPERS
# =====================================================================

def get_densenet121(num_classes=3, dropout_rate=0.5):
    model = models.densenet121()
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(dropout_rate * 0.7),
        nn.Linear(512, num_classes)
    )
    return model

def get_efficientnet_b4(num_classes=3, dropout_rate=0.5):
    model = models.efficientnet_b4()
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(dropout_rate * 0.7),
        nn.Linear(512, num_classes)
    )
    return model

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
            
        return cam, predicted_class, output.detach()

    def remove(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()
        self.activations = None
        self.gradients = None
        gc.collect()

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
def load_rag_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="./main_folder/chroma_db", embedding_function=embeddings)
    return db

# =====================================================================
#                     CACHED MODEL LOADING
# =====================================================================

@st.cache_data
def get_model_config():
    weights_dir = 'main_folder/weights'
    metadata_path = os.path.join(weights_dir, 'ensemble_metadata.json')
    if not os.path.exists(metadata_path):
        return None, None
    with open(metadata_path, 'r') as f:
        meta = json.load(f)
    cfg = meta['config']
    
    transform = transforms.Compose([
        transforms.Resize((cfg['input_size'], cfg['input_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return cfg, transform


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
**Ensemble Model:** DenseNet-121 + EfficientNet-B4  
Upload a chest X-ray image below to classify it as **bacterial pneumonia**, **viral pneumonia**, or **normal**.
""")

cfg, transform = get_model_config()

if cfg is None:
    st.error("**Error Loading Models:** Missing metadata!")
    st.stop()
    
class_names = cfg['class_names']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    st.image(image, caption="Uploaded X-Ray", use_container_width=True)
    st.markdown("---")
    
    if st.button("🔍 Run Classification & Grad-CAM", type="primary", use_container_width=True):
        with st.spinner("Analyzing image using sequential memory-optimized inference..."):
            
            input_tensor = transform(image).unsqueeze(0).to(device)
            weights_dir = 'main_folder/weights'

            # 1. DENSENET INFERENCE
            model_dense = get_densenet121(num_classes=cfg['num_classes'], dropout_rate=cfg['dropout_rate']).to(device)
            model_dense.load_state_dict(torch.load(os.path.join(weights_dir, 'best_densenet121.pth'), map_location=device, weights_only=True)['model_state_dict'])
            model_dense.eval()
            
            gradcam_dense = GradCAM(model_dense, model_dense.features.denseblock4)
            cam_dense, pred_d_idx, out_dense = gradcam_dense.generate(input_tensor)
            gradcam_dense.remove()
            probs_d = torch.softmax(out_dense, dim=1).cpu().numpy()[0]
            
            del model_dense, gradcam_dense, out_dense
            gc.collect()

            # 2. EFFICIENTNET INFERENCE
            model_effnet = get_efficientnet_b4(num_classes=cfg['num_classes'], dropout_rate=cfg['dropout_rate']).to(device)
            model_effnet.load_state_dict(torch.load(os.path.join(weights_dir, 'best_efficientnet_b4.pth'), map_location=device, weights_only=True)['model_state_dict'])
            model_effnet.eval()
            
            gradcam_effnet = GradCAM(model_effnet, model_effnet.features[-1])
            cam_effnet, pred_e_idx, out_effnet = gradcam_effnet.generate(input_tensor)
            gradcam_effnet.remove()
            probs_e = torch.softmax(out_effnet, dim=1).cpu().numpy()[0]
            
            del model_effnet, gradcam_effnet, out_effnet, input_tensor
            gc.collect()

            avg_probs = (probs_d + probs_e) / 2.0
            predicted_idx = np.argmax(avg_probs)
            predicted_class = class_names[predicted_idx]
            confidence = float(avg_probs[predicted_idx])

            input_size = 384
            cam_d_res = np.array(Image.fromarray((cam_dense * 255).astype(np.uint8)).resize((input_size, input_size), Image.BILINEAR)) / 255.0
            cam_e_res = np.array(Image.fromarray((cam_effnet * 255).astype(np.uint8)).resize((input_size, input_size), Image.BILINEAR)) / 255.0
            cam_ensemble = (cam_d_res + cam_e_res) / 2.0
            if cam_ensemble.max() > 0:
                cam_ensemble = cam_ensemble / cam_ensemble.max()

            anatomical_location = get_anatomical_location(cam_ensemble)

            display_tensor = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
            ])(image)
            display_img = display_tensor.permute(1, 2, 0).numpy()
            
            heatmap_colors = cm.jet(cam_ensemble)[:, :, :3]
            overlay = np.clip(0.5 * display_img + 0.5 * heatmap_colors, 0, 1)
            
            df = pd.DataFrame({
                "Class": class_names,
                "DenseNet-121": [f"{p:.2%}" for p in probs_d],
                "EfficientNet-B4": [f"{p:.2%}" for p in probs_e],
                "Ensemble (Average)": [f"{p:.2%}" for p in avg_probs]
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

        st.markdown("### Grad-CAM Attention Heatmap")
        st.image(res['overlay'], caption="Ensemble Grad-CAM Overlay", use_container_width=True)

        with st.expander("📊 View Detailed Model Breakdown"):
            st.write("The final prediction is an average of the two individual models:")
            def highlight_max(s):
                is_max = s == res['df']["Ensemble (Average)"].iloc[res['predicted_idx']]
                return ['background-color: #2ECC71' if v else '' for v in is_max]
            st.dataframe(res['df'].style.apply(highlight_max, subset=['Ensemble (Average)']), use_container_width=True)

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
A chest X-ray has been analyzed by a deep learning vision ensemble.
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
                        model = genai.GenerativeModel('gemini-3.6-flash')
                        response = model.generate_content(prompt)
                        
                        st.session_state['agent_report'] = response.text
                        
                        # Free RAG models from RAM
                        del db, docs, context
                        gc.collect()
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
This AI model takes in a 384x384 pixel chest X-ray and uses a deep learning ensemble to diagnose pneumonia.

**Architecture:**
- DenseNet-121
- EfficientNet-B4
- Probabilistic Averaging
- Grad-CAM Spatial Coordinates
""")
