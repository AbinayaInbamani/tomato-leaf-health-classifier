import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import streamlit as st

from PIL import Image
from skimage.feature import graycomatrix, graycoprops


# ============================================================
# PAGE
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
# CSS
# ============================================================

st.html("""
<style>

.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

.main-header {
    background: linear-gradient(90deg,#064d24,#0b662f);
    padding: 22px;
    border-radius: 18px;
    text-align: center;
    margin-bottom: 14px;
}

.main-title {
    color: white;
    font-size: 46px;
    font-weight: 800;
}

.main-subtitle {
    color: white;
    font-size: 21px;
    margin-top: 5px;
}

.student-bar {
    background: #dff5e2;
    border-radius: 16px;
    padding: 16px;
    text-align: center;
    font-size: 25px;
    font-weight: 700;
    margin-bottom: 18px;
}

.upload-box {
    background: #eef7ff;
    border: 2px solid #4285c5;
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 12px;
}

.upload-title {
    color: #08296b;
    font-size: 27px;
    font-weight: 800;
}

.upload-text {
    font-size: 17px;
    margin-top: 7px;
}

.panel-title {
    background: #07602e;
    color: white;
    padding: 12px;
    border-radius: 14px 14px 0 0;
    text-align: center;
    font-size: 24px;
    font-weight: 800;
}

.healthy-card {
    border: 2px solid #2a9d45;
    background: #effbef;
    border-radius: 18px;
    padding: 30px;
    text-align: center;
    min-height: 250px;
}

.diseased-card {
    border: 2px solid #f07b24;
    background: #fff2e8;
    border-radius: 18px;
    padding: 30px;
    text-align: center;
    min-height: 250px;
}

.healthy-result {
    color: #087c20;
    font-size: 42px;
    font-weight: 900;
}

.diseased-result {
    color: #df3b10;
    font-size: 38px;
    font-weight: 900;
}

.agreement-label {
    font-size: 21px;
    font-weight: 800;
    margin-top: 25px;
}

.agreement-green {
    display: inline-block;
    margin-top: 10px;
    background: #138b2c;
    color: white;
    padding: 9px 18px;
    border-radius: 13px;
    font-size: 28px;
    font-weight: 900;
}

.agreement-orange {
    display: inline-block;
    margin-top: 10px;
    background: #f45419;
    color: white;
    padding: 9px 18px;
    border-radius: 13px;
    font-size: 28px;
    font-weight: 900;
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
    font-size: 23px;
    font-weight: 800;
}

.method-text {
    font-size: 18px;
    font-weight: 700;
    margin-top: 7px;
}

.how-card {
    background: #e5f1ff;
    border-radius: 18px;
    padding: 24px;
    margin-top: 24px;
}

.how-title {
    color: #082d73;
    font-size: 28px;
    font-weight: 900;
}

.how-text {
    font-size: 19px;
    line-height: 1.5;
    margin-top: 8px;
}

.note {
    text-align: center;
    font-size: 14px;
    color: #666666;
    margin-top: 18px;
}

</style>
""")


# ============================================================
# HEADER
# ============================================================

st.html("""
<div class="main-header">
    <div class="main-title">🍅 Tomato Leaf Health Classifier</div>
    <div class="main-subtitle">
        OpenCV + KNN-Based Plant Health Screening Tool
    </div>
</div>
""")


st.html("""
<div class="student-bar">
    👤 Student: Rishi Karthickeyan
</div>
""")


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
        "Unable to load the trained model."
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
# IMAGE PREPARATION
# ============================================================

def prepare_image(pil_image):

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
# SEGMENTATION
#
# MUST MATCH TRAINING
# saturation > 30
# value > 20
# ============================================================

def segment_leaf(image):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    _, saturation, value = cv2.split(
        hsv
    )

    mask = np.where(
        (
            (saturation > 30)
            &
            (value > 20)
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
            [largest],
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
            [0, 40, 20]
        ),
        np.array(
            [30, 255, 230]
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
            [30, 30, 20]
        ),
        np.array(
            [95, 255, 255]
        )
    )

    yellow_brown_mask = cv2.inRange(
        hsv,
        np.array(
            [10, 40, 20]
        ),
        np.array(
            [35, 255, 255]
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

def calculate_entropy(glcm):

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
# UPLOAD
# ============================================================

st.html("""
<div class="upload-box">

    <div class="upload-title">
        ☁️ Upload a Tomato Leaf Image
    </div>

    <div class="upload-text">
        Choose a JPG, JPEG, or PNG image.
        For best results, use a clear tomato leaf image
        with the leaf reasonably centered.
    </div>

</div>
""")


uploaded_file = st.file_uploader(
    "Choose image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    label_visibility="collapsed"
)


# ============================================================
# PREDICTION
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


    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    if METHOD_CODE == "A":

        features = extract_method_A(
            segmented,
            mask
        )

        method_text = (
            "Simple Color Features + KNN (Method A)"
        )


    elif METHOD_CODE == "B":

        features = extract_method_B(
            segmented,
            mask
        )

        method_text = (
            "HSV Color Features + KNN (Method B)"
        )


    else:

        features = extract_method_C(
            segmented,
            mask
        )

        method_text = (
            "HSV Color + GLCM Texture + KNN (Method C)"
        )


    features = np.asarray(
        features,
        dtype=float
    ).reshape(
        1,
        -1
    )


    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    prediction = int(
        model.predict(
            features
        )[0]
    )


    agreement = None


    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = model.predict_proba(
            features
        )[0]

        agreement = float(
            np.max(
                probabilities
            )
        )


    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(
        [
            1,
            1,
            1.35
        ]
    )


    with col1:

        st.html(
            '<div class="panel-title">Original Image</div>'
        )

        st.image(
            pil_image,
            use_container_width=True
        )


    with col2:

        st.html(
            '<div class="panel-title">OpenCV Segmentation</div>'
        )

        st.image(
            segmented_rgb,
            use_container_width=True
        )


    with col3:

        st.html(
            '<div class="panel-title">📊 Leaf Health Result</div>'
        )


        if prediction == 0:

            if agreement is not None:

                agreement_text = (
                    f"""
                    <div class="agreement-label">
                        KNN Neighbor Agreement
                    </div>

                    <div class="agreement-green">
                        {agreement * 100:.1f}%
                    </div>
                    """
                )

            else:

                agreement_text = ""


            st.html(
                f"""
                <div class="healthy-card">

                    <div class="healthy-result">
                        ✅ HEALTHY
                    </div>

                    {agreement_text}

                </div>
                """
            )


        else:

            if agreement is not None:

                agreement_text = (
                    f"""
                    <div class="agreement-label">
                        KNN Neighbor Agreement
                    </div>

                    <div class="agreement-orange">
                        {agreement * 100:.1f}%
                    </div>
                    """
                )

            else:

                agreement_text = ""


            st.html(
                f"""
                <div class="diseased-card">

                    <div class="diseased-result">
                        ⚠️ DISEASED / UNHEALTHY
                    </div>

                    {agreement_text}

                </div>
                """
            )


        st.html(
            f"""
            <div class="method-card">

                <div class="method-title">
                    ⚙️ Method Used
                </div>

                <div class="method-text">
                    {method_text}
                </div>

            </div>
            """
        )


    # ========================================================
    # HOW IT WORKS
    # ========================================================

    st.html("""
    <div class="how-card">

        <div class="how-title">
            🎯 How It Works
        </div>

        <div class="how-text">

            The application uses <b>OpenCV</b> to isolate the
            tomato leaf. It then measures <b>HSV color features</b>
            and <b>GLCM texture features</b>.

            These measurements are evaluated using a
            <b>K-Nearest Neighbors (KNN)</b> classifier trained
            on real-world tomato leaf images.

            The model classifies the leaf as
            <b>Healthy</b> or <b>Diseased / Unhealthy</b>.

        </div>

    </div>
    """)


    st.html("""
    <div class="note">

        Educational and research screening tool.
        The application does not diagnose a specific tomato disease.
        KNN Neighbor Agreement represents voting among nearby
        training samples and is not a calibrated probability.

    </div>
    """)


else:

    st.info(
        "Upload a tomato leaf image above to begin the analysis."
    )
