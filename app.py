import streamlit as st
import torch
import torchvision.transforms as transforms
import cv2
from PIL import Image

# Import your custom modules from model.py
from model import FacePADModel, GradCAM, overlay_heatmap

# --- Configuration ---
st.set_page_config(page_title="Face Liveness Detection", layout="centered")
MODEL_PATH = "saved_models/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FIX 1: Define class names explicitly to match your training data folder order
# You can verify this order by checking the output of `train_dataset.classes` in your training script.
# It is likely ['training_fake', 'training_real']
CLASS_NAMES = ['training_fake', 'training_real']

# --- Model Loading ---
@st.cache_resource
def load_trained_model(model_path):
    """Loads the fine-tuned model from the specified path."""
    # FIX 2: Instantiate the model with the same architecture as the one you trained
    # and load the saved weights (the state_dict).
    model = FacePADModel(backbone_name='resnet50', pretrained=False)  # pretrained=False, as we are loading our own weights
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()  # Set the model to evaluation mode
    return model

model = load_trained_model(MODEL_PATH)

# FIX 3: The attribute in our model is 'backbone', not 'resnet'.
target_layer = model.backbone.layer4[-1]
grad_cam_generator = GradCAM(model, target_layer)

# --- Image Preprocessing ---
# This transform pipeline must be identical to the validation transforms used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Streamlit UI ---
st.title("🧠 Face Liveness Detection with Grad-CAM")
st.markdown("Upload a face image to check whether it's **Real** or **Spoof** with a visual explanation.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Transform the image and add a batch dimension
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # --- Prediction ---
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class_idx].item()

    # Map prediction index to a display name
    predicted_class_name = CLASS_NAMES[pred_class_idx]
    label = "REAL" if "real" in predicted_class_name.lower() else "SPOOF"
    
    color = "green" if label == "REAL" else "red"
    st.markdown(f"### Prediction: <span style='color:{color};'>{label}</span>", unsafe_allow_html=True)
    st.markdown(f"**Confidence Score:** `{confidence:.2%}`")
    st.progress(confidence)

    # --- Grad-CAM Explanation ---
    st.markdown("### 🔍 Grad-CAM Explanation")
    st.write("The red areas highlight where the model focused to make its decision.")

    # FIX 4: Our generate_cam function returns two values (heatmap, class_idx).
    # We need to unpack both.
    heatmap, _ = grad_cam_generator.generate_cam(input_tensor, class_idx=pred_class_idx)
    
    # We need the original tensor before normalization for a good visualization
    vis_tensor = transforms.ToTensor()(image)
    cam_overlay = overlay_heatmap(vis_tensor, heatmap)

    # FIX 5: Convert the color format from BGR (OpenCV's default) to RGB for correct display in Streamlit.
    cam_overlay_rgb = cv2.cvtColor(cam_overlay, cv2.COLOR_BGR2RGB)

    st.image(cam_overlay_rgb, caption="Grad-CAM Overlay", use_column_width=True)