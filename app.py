import streamlit as st
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from model import FacePADModel, GradCAM, overlay_heatmap  # import custom modules

# Set page config
st.set_page_config(page_title="Face Liveness Detection", layout="centered")

# Load model
@st.cache_resource
def load_model():
    model = FacePADModel()
    model.eval()
    return model

model = load_model()
target_layer = model.resnet.layer4[-1]
cam = GradCAM(model, target_layer)

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

st.title("🧠 Face Liveness Detection with Grad-CAM")
st.markdown("Upload a face image to check whether it's **Real** or **Spoof** with visual explanation.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)

    # Prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()

    label = "REAL" if pred_class == 1 else "SPOOF"
    color = "🟢" if label == "REAL" else "🔴"

    st.markdown(f"### Prediction: {color} {label}")
    st.markdown(f"**Confidence Score:** `{confidence:.2f}`")

    # Grad-CAM
    heatmap = cam.generate_cam(input_tensor)
    cam_overlay = overlay_heatmap(input_tensor[0], heatmap)

    st.markdown("### 🔍 Grad-CAM Explanation")
    st.image(cam_overlay, caption="Grad-CAM Overlay", use_column_width=True)
