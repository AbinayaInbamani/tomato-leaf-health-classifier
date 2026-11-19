import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_output" / "tomato_leaf_health_model.joblib"
CONFIG_PATH = BASE_DIR / "model_output" / "model_config.json"

st.set_page_config(
    page_title="Tomato Leaf Health Classifier",
    page_icon="🍅",
    layout="centered"
)

st.title("🍅 Tomato Leaf Health Classifier")
st.write(
    "Upload a tomato leaf image. The frozen OpenCV + KNN model will classify it as "
    "**Healthy** or **Diseased / Unhealthy**."
)

@st.cache_resource
def load_model_and_config():
    model = joblib.load(MODEL_PATH)
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    return model, config

model, config = load_model_and_config()
IMAGE_SIZE = int(config.get("image_size", 224))
METHOD_CODE = config.get("method_code", "C")

def prepare_image(pil_image):
    image = np.array(pil_image.convert("RGB"))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    return image

def segment_leaf(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, saturation, value = cv2.split(hsv)

    mask = np.where(
        (saturation > 25) & (value > 20),
        255,
        0
    ).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        largest = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(mask)
        cv2.drawContours(clean_mask, [largest], -1, 255, -1)
        mask = clean_mask

    segmented = cv2.bitwise_and(image, image, mask=mask)
    return segmented, mask

def masked_values(channel, mask):
    values = channel[mask > 0]
    if len(values) == 0:
        return np.array([0], dtype=float)
    return values

def extract_method_A(image, mask):
    b, g, r = cv2.split(image)

    r_values = masked_values(r, mask)
    g_values = masked_values(g, mask)
    b_values = masked_values(b, mask)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    brown_red_mask = cv2.inRange(
        hsv,
        np.array([0, 40, 20]),
        np.array([30, 255, 230])
    )

    leaf_pixels = np.count_nonzero(mask)
    brown_pixels = np.count_nonzero(
        (brown_red_mask > 0) & (mask > 0)
    )

    brown_percentage = brown_pixels / leaf_pixels if leaf_pixels > 0 else 0

    return [
        np.mean(r_values),
        np.mean(g_values),
        np.mean(b_values),
        brown_percentage
    ]

def extract_method_B(image, mask):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    hp = masked_values(h, mask)
    sp = masked_values(s, mask)
    vp = masked_values(v, mask)

    green_mask = cv2.inRange(
        hsv,
        np.array([30, 30, 20]),
        np.array([95, 255, 255])
    )

    yellow_brown_mask = cv2.inRange(
        hsv,
        np.array([10, 40, 20]),
        np.array([35, 255, 255])
    )

    leaf_pixels = np.count_nonzero(mask)

    green_pixels = np.count_nonzero(
        (green_mask > 0) & (mask > 0)
    )
    yellow_brown_pixels = np.count_nonzero(
        (yellow_brown_mask > 0) & (mask > 0)
    )

    green_percentage = green_pixels / leaf_pixels if leaf_pixels > 0 else 0
    yellow_brown_percentage = (
        yellow_brown_pixels / leaf_pixels if leaf_pixels > 0 else 0
    )

    return [
        np.mean(hp),
        np.mean(sp),
        np.mean(vp),
        np.std(hp),
        np.std(sp),
        np.std(vp),
        green_percentage,
        yellow_brown_percentage
    ]

def calculate_glcm_entropy(glcm):
    probabilities = glcm[glcm > 0]
    return -np.sum(probabilities * np.log2(probabilities))

def extract_method_C(image, mask):
    hsv_features = extract_method_B(image, mask)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray_masked = np.where(
        mask > 0,
        gray,
        0
    ).astype(np.uint8)

    gray_quantized = (gray_masked // 16).astype(np.uint8)

    glcm = graycomatrix(
        gray_quantized,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=16,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(glcm, "contrast").mean()
    homogeneity = graycoprops(glcm, "homogeneity").mean()
    asm = graycoprops(glcm, "ASM").mean()
    entropy = calculate_glcm_entropy(glcm)

    return hsv_features + [
        contrast,
        homogeneity,
        entropy,
        asm
    ]

uploaded_file = st.file_uploader(
    "Upload a tomato leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)

    st.image(
        pil_image,
        caption="Uploaded tomato leaf",
        use_container_width=True
    )

    image = prepare_image(pil_image)
    segmented, mask = segment_leaf(image)

    if METHOD_CODE == "A":
        features = extract_method_A(segmented, mask)
    elif METHOD_CODE == "B":
        features = extract_method_B(segmented, mask)
    else:
        features = extract_method_C(segmented, mask)

    X = np.asarray(features, dtype=float).reshape(1, -1)

    prediction = int(model.predict(X)[0])

    confidence_text = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        predicted_probability = float(np.max(probabilities))
        confidence_text = f"{predicted_probability * 100:.1f}%"

    if prediction == 0:
        st.success("Prediction: HEALTHY")
    else:
        st.error("Prediction: DISEASED / UNHEALTHY")

    if confidence_text:
        st.write(f"Model confidence: **{confidence_text}**")

    with st.expander("Model information"):
        st.write("Selected method:", config.get("selected_method", "Unknown"))
        st.write("KNN k:", config.get("knn_k", "Unknown"))
        if "test_accuracy" in config:
            st.write(
                "PlantVillage final test accuracy:",
                f"{config['test_accuracy'] * 100:.2f}%"
            )
        if "test_f1" in config:
            st.write(
                "PlantVillage final test F1:",
                f"{config['test_f1'] * 100:.2f}%"
            )

    st.caption(
        "This is a binary screening model for tomato leaves. "
        "It predicts Healthy vs Diseased/Unhealthy and does not identify a specific disease."
    )
