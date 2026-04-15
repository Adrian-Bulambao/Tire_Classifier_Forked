import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from ultralytics import YOLO
import os

# =========================
# CONFIG
# =========================
BASELINE_SIZE = 224
HYPERTUNED_SIZE = 320
BASE_DIR = os.path.dirname(__file__)

# Speed optimization
tf.config.run_functions_eagerly(False)

# =========================
# LOAD MODELS (CACHED + WARMED UP)
# =========================
@st.cache_resource
def load_models():
    print("Files in directory:", os.listdir(BASE_DIR))

    baseline = tf.keras.models.load_model(
        os.path.join(BASE_DIR, "baseModel.keras")
    )

    hypertuned = tf.keras.models.load_model(
        os.path.join(BASE_DIR, "bestModel.keras")
    )

    yolo = YOLO(os.path.join(BASE_DIR, "yoloToh.pt"))

    # 🔥 warm-up (removes first-run lag)
    baseline_dummy = np.zeros((1, BASELINE_SIZE, BASELINE_SIZE, 3), dtype=np.float32)
    hypertuned_dummy = np.zeros((1, HYPERTUNED_SIZE, HYPERTUNED_SIZE, 3), dtype=np.float32)

    baseline.predict(baseline_dummy, verbose=0)
    hypertuned.predict(hypertuned_dummy, verbose=0)

    return baseline, hypertuned, yolo


baseline_model, hypertuned_model, yolo_model = load_models()

# =========================
# PREPROCESSING
# =========================
def preprocess_image(image, img_size):
    image = image.resize((img_size, img_size))
    image = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(image, axis=0)

# =========================
# CNN PREDICTION (FIXED FOR SIGMOID)
# =========================
def predict_cnn(model, image, img_size):
    processed = preprocess_image(image, img_size)
    preds = model.predict(processed, verbose=0)

    prob = float(preds[0][0])  # sigmoid output

    # IMPORTANT: correct interpretation
    if prob >= 0.5:
        label = "good"
        confidence = prob
    else:
        label = "defective"
        confidence = 1 - prob

    return label, confidence

# =========================
# YOLO PREDICTION
# =========================
def predict_yolo(model, image):
    results = model(image)

    probs = results[0].probs

    if probs is None:
        return "No prediction", 0.0, results

    class_id = int(probs.top1)
    confidence = float(probs.top1conf)

    return model.names[class_id], confidence, results

# =========================
# UI
# =========================
st.set_page_config(page_title="Tire Defect Classifier", layout="centered")

st.markdown("""
    <h1 style='text-align: center;'>🛞 Tire Defect Classifier</h1>
    <p style='text-align: center;'>Upload an image and choose a model to classify tire defects</p>
""", unsafe_allow_html=True)

model_choice = st.selectbox(
    "Choose Model",
    ["YOLOv8", "Baseline CNN", "HyperTuned CNN"]
)

uploaded_file = st.file_uploader(
    "Upload Tire Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing..."):

            if model_choice == "Baseline CNN":
                label, conf = predict_cnn(baseline_model, image, BASELINE_SIZE)

            elif model_choice == "HyperTuned CNN":
                label, conf = predict_cnn(hypertuned_model, image, HYPERTUNED_SIZE)

            else:
                label, conf, results = predict_yolo(yolo_model, image)

        # =========================
        # OUTPUT
        # =========================
        st.markdown("### 📊 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Prediction", label)

        with col2:
            st.metric("Confidence", f"{conf:.2%}")

        # YOLO visualization
        if model_choice == "YOLOv8":
            annotated = results[0].plot()
            st.image(annotated, caption="Detection Result", use_container_width=True)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>Built with Streamlit</p>",
    unsafe_allow_html=True
)