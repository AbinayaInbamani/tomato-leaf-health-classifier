import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import streamlit as st

from PIL import Image

from skimage.feature import (
    graycomatrix,
    graycoprops
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Tomato Leaf Health Classifier",
    page_icon="🍅",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "tomato_leaf_health_model.joblib"

CONFIG_PATH = BASE_DIR / "model_config.json"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    .main-header {
        background: linear-gradient(90deg, #064d24, #0b662f);
        padding: 20px 30px;
        border-radius: 0px 0px 18px 18px;
        text-align: center;
        margin-bottom: 12px;
    }

    .main-title {
        color: white;
        font-size: 48px;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .main-subtitle {
        color: white;
        font-size: 24px;
        font-weight: 500;
    }

    .student-bar {
        background: linear-gradient(90deg, #e5f7e7, #d7f2db);
        border-radius: 16px;
        padding: 16px;
        text-align: center;
        font-size: 27px;
        font-weight: 700;
        margin-bottom: 18px;
        color: #111111;
    }

    .upload-box {
        background: linear-gradient(90deg, #eef7ff, #f5fbff);
        border: 2px solid #4285c5;
        border-radius: 18px;
        padding: 18px 24px;
        margin-bottom: 20px;
    }

    .upload-title {
        font-size: 28px;
        font-weight: 800;
        color: #08296b;
        margin-bottom: 4px;
    }

    .panel-title {
        background: linear-gradient(90deg, #064d24, #0a632f);
        color: white;
        padding: 12px;
        text-align: center;
        font-size: 26px;
        font-weight: 800;
        border-radius: 14px 14px 0px 0px;
        margin-bottom: 6px;
    }

    .healthy-card {
        border: 2px solid #2a9d45;
        border-radius: 18px;
        background: #effbef;
        padding: 28px;
        text-align: center;
        min-height: 260px;
    }

    .diseased-card {
        border: 2px solid #f07b24;
        border-radius: 18px;
        background: #fff2e8;
        padding: 28px;
        text-align: center;
        min-height: 260px;
    }

    .healthy-result {
        color: #087c20;
        font-size: 50px;
        font-weight: 900;
        line-height: 1.05;
    }

    .diseased-result {
        color: #df3b10;
        font-size: 46px;
        font-weight: 900;
        line-height: 1.1;
    }

    .confidence-label {
        font-size: 24px;
        font-weight: 800;
        margin-top: 20px;
    }

    .confidence-value {
        display: inline-block;
        background: #f45419;
        color: white;
        padding: 8px 18px;
        border-radius: 14px;
        font-size: 30px;
        font-weight: 900;
        margin-top: 8px;
    }

    .confidence-value-green {
        display: inline-block;
        background: #138b2c;
        color: white;
        padding: 8px 18px;
        border-radius: 14px;
        font-size: 30px;
        font-weight: 900;
        margin-top: 8px;
    }

    .method-card {
        background: #e9faea;
        border: 1px solid #8cd59a;
        border-radius: 16px;
        padding: 18px;
        margin-top: 14px;
    }

    .method-title {
        color: #075b23;
        font-size: 26px;
        font-weight: 850;
    }

    .method-text {
        color: #111111;
        font-size: 21px;
        font-weight: 700;
        margin-top: 6px;
    }

    .how-card {
        background: linear-gradient(90deg, #ddebff, #d8ecff);
        border-radius: 18px;
        padding: 24px;
        margin-top: 24px;
    }

    .how-title {
        color: #082d73;
        font-size: 32px;
        font-weight: 900;
    }

    .how-text {
        font-size: 22px;
        color: #111111;
        line-height: 1.4;
    }

    .small-note {
        text-align: center;
        color: #666666;
        font-size: 14px;
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-header">
        <div class="main-title">
            🍅 Tomato Leaf Health Classifier
        </div>

        <div class="main-subtitle">
            OpenCV + KNN-Based Plant Health Screening Tool
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# STUDENT
# ============================================================

st.markdown(
    """
    <div class="student-bar">
        👤 &nbsp; Student: Rishi Karthickeyan
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model_and_config():

    model = joblib.load(
        MODEL_PATH
    )

    with open(
        CONFIG_PATH,
        "r"
    ) as file:

        config = json.load(
            file
        )

    return model, config


try:

    model, config = load_model_and_config()

except Exception as error:

    st.error(
        "Unable to load the trained model or model configuration."
    )

    st.code(
        str(error)
    )

    st.stop()


IMAGE_SIZE = int(
    config.get(
        "image_size",
        224
    )
)

METHOD_CODE = config.get(
    "method_code",
    "C"
)


# ============================================================
# PREPARE IMAGE
# ============================================================

def prepare_image(
    pil_image
):

    image = np.array(
        pil_image.convert(
            "RGB"
        )
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    image = cv2.resize(
        image,
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    )

    return image


# ============================================================
# SEGMENT LEAF
# ============================================================

def segment_leaf(
    image
):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    _, saturation, value = cv2.split(
        hsv
    )

    mask = np.where(
        (
            saturation > 25
        )
        &
        (
            value > 20
        ),
        255,
        0
    ).astype(
        np.uint8
    )

    kernel = np.ones(
        (
            5,
            5
        ),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) > 0:

        largest = max(
            contours,
            key=cv2.contourArea
        )

        clean_mask = np.zeros_like(
            mask
        )

        cv2.drawContours(
            clean_mask,
            [
                largest
            ],
            -1,
            255,
            -1
        )

        mask = clean_mask

    segmented = cv2.bitwise_and(
        image,
        image,
        mask=mask
    )

    return segmented, mask


# ============================================================
# MASK HELPER
# ============================================================

def masked_values(
    channel,
    mask
):

    values = channel[
        mask > 0
    ]

    if len(values) == 0:

        return np.array(
            [0],
            dtype=float
        )

    return values


# ============================================================
# METHOD A
# SIMPLE COLOR
# ============================================================

def extract_method_A(
    image,
    mask
):

    b, g, r = cv2.split(
        image
    )

    r_values = masked_values(
        r,
        mask
    )

    g_values = masked_values(
        g,
        mask
    )

    b_values = masked_values(
        b,
        mask
    )

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    brown_mask = cv2.inRange(
        hsv,
        np.array(
            [
                0,
                40,
                20
            ]
        ),
        np.array(
            [
                30,
                255,
                230
            ]
        )
    )

    leaf_pixels = np.count_nonzero(
        mask
    )

    brown_pixels = np.count_nonzero(
        (
            brown_mask > 0
        )
        &
        (
            mask > 0
        )
    )

    brown_percentage = (
        brown_pixels / leaf_pixels
        if leaf_pixels > 0
        else 0
    )

    return [
        np.mean(r_values),
        np.mean(g_values),
        np.mean(b_values),
        brown_percentage
    ]


# ============================================================
# METHOD B
# HSV
# ============================================================

def extract_method_B(
    image,
    mask
):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    h, s, v = cv2.split(
        hsv
    )

    hp = masked_values(
        h,
        mask
    )

    sp = masked_values(
        s,
        mask
    )

    vp = masked_values(
        v,
        mask
    )

    green_mask = cv2.inRange(
        hsv,
        np.array(
            [
                30,
                30,
                20
            ]
        ),
        np.array(
            [
                95,
                255,
                255
            ]
        )
    )

    yellow_brown_mask = cv2.inRange(
        hsv,
        np.array(
            [
                10,
                40,
                20
            ]
        ),
        np.array(
            [
                35,
                255,
                255
            ]
        )
    )

    leaf_pixels = np.count_nonzero(
        mask
    )

    green_pixels = np.count_nonzero(
        (
            green_mask > 0
        )
        &
        (
            mask > 0
        )
    )

    yellow_pixels = np.count_nonzero(
        (
            yellow_brown_mask > 0
        )
        &
        (
            mask > 0
        )
    )

    green_percentage = (
        green_pixels / leaf_pixels
        if leaf_pixels > 0
        else 0
    )

    yellow_percentage = (
        yellow_pixels / leaf_pixels
        if leaf_pixels > 0
        else 0
    )

    return [
        np.mean(hp),
        np.mean(sp),
        np.mean(vp),

        np.std(hp),
        np.std(sp),
        np.std(vp),

        green_percentage,
        yellow_percentage
    ]


# ============================================================
# GLCM ENTROPY
# ============================================================

def calculate_entropy(
    glcm
):

    probabilities = glcm[
        glcm > 0
    ]

    return -np.sum(
        probabilities
        *
        np.log2(
            probabilities
        )
    )


# ============================================================
# METHOD C
# HSV + GLCM TEXTURE
# ============================================================

def extract_method_C(
    image,
    mask
):

    hsv_features = extract_method_B(
        image,
        mask
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = np.where(
        mask > 0,
        gray,
        0
    ).astype(
        np.uint8
    )

    gray = (
        gray // 16
    ).astype(
        np.uint8
    )

    glcm = graycomatrix(
        gray,
        distances=[
            1
        ],
        angles=[
            0,
            np.pi / 4,
            np.pi / 2,
            3 * np.pi / 4
        ],
        levels=16,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(
        glcm,
        "contrast"
    ).mean()

    homogeneity = graycoprops(
        glcm,
        "homogeneity"
    ).mean()

    asm = graycoprops(
        glcm,
        "ASM"
    ).mean()

    entropy = calculate_entropy(
        glcm
    )

    return (
        hsv_features
        +
        [
            contrast,
            homogeneity,
            entropy,
            asm
        ]
    )


# ============================================================
# UPLOAD AREA
# ============================================================

st.markdown(
    """
    <div class="upload-box">
        <div class="upload-title">
            ☁️ Upload a Tomato Leaf Image
        </div>

        <div>
            Choose a JPG, JPEG, or PNG image.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


uploaded_file = st.file_uploader(
    "Choose File",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    label_visibility="collapsed"
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    pil_image = Image.open(
        uploaded_file
    )

    image = prepare_image(
        pil_image
    )

    segmented, mask = segment_leaf(
        image
    )

    segmented_rgb = cv2.cvtColor(
        segmented,
        cv2.COLOR_BGR2RGB
    )


    # ========================================================
    # FEATURES
    # ========================================================

    if METHOD_CODE == "A":

        features = extract_method_A(
            segmented,
            mask
        )

        method_text = (
            "Simple Color Features + KNN "
            "(Method A)"
        )


    elif METHOD_CODE == "B":

        features = extract_method_B(
            segmented,
            mask
        )

        method_text = (
            "HSV Color Features + KNN "
            "(Method B)"
        )


    else:

        features = extract_method_C(
            segmented,
            mask
        )

        method_text = (
            "HSV Color + GLCM Texture + KNN "
            "(Method C)"
        )


    features = np.asarray(
        features,
        dtype=float
    ).reshape(
        1,
        -1
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    prediction = int(
        model.predict(
            features
        )[0]
    )


    confidence = None


    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = model.predict_proba(
            features
        )[0]

        confidence = float(
            np.max(
                probabilities
            )
        )


    # ========================================================
    # THREE COLUMNS
    # ========================================================

    col1, col2, col3 = st.columns(
        [
            1,
            1,
            1.35
        ],
        gap="medium"
    )


    # ========================================================
    # ORIGINAL IMAGE
    # ========================================================

    with col1:

        st.markdown(
            """
            <div class="panel-title">
                Original Image
            </div>
            """,
            unsafe_allow_html=True
        )

        st.image(
            pil_image,
            use_container_width=True
        )


    # ========================================================
    # SEGMENTED IMAGE
    # ========================================================

    with col2:

        st.markdown(
            """
            <div class="panel-title">
                OpenCV Segmentation
            </div>
            """,
            unsafe_allow_html=True
        )

        st.image(
            segmented_rgb,
            use_container_width=True
        )


    # ========================================================
    # RESULT
    # ========================================================

    with col3:

        st.markdown(
            """
            <div class="panel-title">
                📊 Leaf Health Result
            </div>
            """,
            unsafe_allow_html=True
        )


        if prediction == 0:

            confidence_html = ""

            if confidence is not None:

                confidence_html = f"""
                <div class="confidence-label">
                    Model Confidence:
                </div>

                <div class="confidence-value-green">
                    {confidence * 100:.1f}%
                </div>
                """


            st.markdown(
                f"""
                <div class="healthy-card">

                    <div class="healthy-result">
                        ✅ HEALTHY
                    </div>

                    {confidence_html}

                </div>
                """,
                unsafe_allow_html=True
            )


        else:

            confidence_html = ""

            if confidence is not None:

                confidence_html = f"""
                <div class="confidence-label">
                    Model Confidence:
                </div>

                <div class="confidence-value">
                    {confidence * 100:.1f}%
                </div>
                """


            st.markdown(
                f"""
                <div class="diseased-card">

                    <div class="diseased-result">
                        ⚠️ DISEASED /<br>
                        UNHEALTHY
                    </div>

                    {confidence_html}

                </div>
                """,
                unsafe_allow_html=True
            )


        st.markdown(
            f"""
            <div class="method-card">

                <div class="method-title">
                    ⚙️ Method Used
                </div>

                <div class="method-text">
                    {method_text}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # HOW IT WORKS
    # ========================================================

    st.markdown(
        """
        <div class="how-card">

            <table style="width:100%;">
                <tr>

                    <td style="
                        width:28%;
                        vertical-align:middle;
                        padding-right:25px;
                    ">

                        <div class="how-title">
                            🎯 How It Works
                        </div>

                    </td>

                    <td style="
                        border-left:2px solid #1865a8;
                        padding-left:30px;
                    ">

                        <div class="how-text">

                            The app uses <b>OpenCV</b> to segment
                            the leaf, then analyzes
                            <b>HSV color features</b> and
                            <b>GLCM texture features</b> with a
                            <b>K-Nearest Neighbors (KNN)</b>
                            model to predict whether the tomato
                            leaf is <b>Healthy</b> or
                            <b>Diseased / Unhealthy</b>.

                        </div>

                    </td>

                </tr>
            </table>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="small-note">
            Educational and research screening tool.
            The application does not identify a specific tomato disease.
        </div>
        """,
        unsafe_allow_html=True
    )


else:

    st.info(
        "Upload a tomato leaf image above to begin the analysis."
    )
