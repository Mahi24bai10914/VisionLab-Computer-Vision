import streamlit as st
import matplotlib.pyplot as plt

from utils import load_image
from preprocessing import preprocess_image
from enhancement import histogram_equalization, get_histogram
from feature_extraction import (
    detect_canny_edges,
    detect_harris_corners
)
from segmentation import (
    binary_threshold,
    otsu_threshold,
    adaptive_threshold
)
from comparison import sift_feature_matching


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="VisionLab",
    page_icon="🔬",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 28px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:

    st.header("🔬 VisionLab")

    st.write(
        "Computer Vision Image Analysis System"
    )

    st.markdown("---")

    selected_module = st.radio(
        "Select Module",
        [
            "🏠 Home",
            "🖼️ Image Preprocessing",
            "✨ Image Enhancement",
            "📈 Histogram Analysis",
            "🔍 Edge Detection",
            "⭐ Feature Extraction",
            "🧩 Image Segmentation",
            "🔗 Image Comparison"
        ]
    )

    st.markdown("---")

    st.subheader("Technologies")

    st.write("Python")
    st.write("OpenCV")
    st.write("NumPy")
    st.write("Matplotlib")
    st.write("Streamlit")

    st.markdown("---")

    st.caption(
        "Computer Vision Academic Project"
    )


# =========================================================
# MAIN TITLE
# =========================================================

st.markdown(
    '<div class="main-title">🔬 VisionLab</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Interactive Computer Vision Image Analysis System'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# HOME
# =========================================================

if selected_module == "🏠 Home":

    st.header("Welcome to VisionLab")

    st.write(
        "VisionLab is an interactive Computer Vision "
        "application that demonstrates different image "
        "processing and analysis techniques using Python "
        "and OpenCV."
    )

    st.markdown("---")

    st.subheader("Available Modules")

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            "🖼️ **Image Preprocessing**\n\n"
            "Grayscale conversion and Gaussian blur."
        )

        st.info(
            "✨ **Image Enhancement**\n\n"
            "Improve image contrast using histogram "
            "equalization."
        )

        st.info(
            "📈 **Histogram Analysis**\n\n"
            "Analyze pixel intensity distribution."
        )

        st.info(
            "🔍 **Edge Detection**\n\n"
            "Detect important edges using Canny."
        )

    with col2:

        st.info(
            "⭐ **Feature Extraction**\n\n"
            "Detect important image features using "
            "Harris Corner Detection."
        )

        st.info(
            "🧩 **Image Segmentation**\n\n"
            "Segment images using Binary, Otsu and "
            "Adaptive Thresholding."
        )

        st.info(
            "🔗 **Image Comparison**\n\n"
            "Compare two images using SIFT feature matching."
        )

    st.markdown("---")

    st.success(
        "Select a module from the sidebar to begin."
    )


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

elif selected_module == "🖼️ Image Preprocessing":

    st.markdown(
        '<div class="section-title">'
        '🖼️ Image Preprocessing'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        key="preprocessing"
    )

    if uploaded_file is not None:

        image, image_array, image_bgr = load_image(
            uploaded_file
        )

        gray, blurred = preprocess_image(
            image_bgr
        )

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Original Image")

            st.image(
                image,
                use_container_width=True
            )

        with col2:

            st.subheader("Grayscale Image")

            st.image(
                gray,
                use_container_width=True
            )

        st.markdown("---")

        st.subheader("Gaussian Blur")

        st.image(
            blurred,
            caption="Blurred Image",
            use_container_width=True
        )


# =========================================================
# IMAGE ENHANCEMENT
# =========================================================

elif selected_module == "✨ Image Enhancement":

    st.markdown(
        '<div class="section-title">'
        '✨ Image Enhancement'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        key="enhancement"
    )

    if uploaded_file is not None:

        image, image_array, image_bgr = load_image(
            uploaded_file
        )

        gray, blurred = preprocess_image(
            image_bgr
        )

        enhanced = histogram_equalization(
            gray
        )

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Original Grayscale")

            st.image(
                gray,
                use_container_width=True
            )

        with col2:

            st.subheader("Enhanced Image")

            st.image(
                enhanced,
                use_container_width=True
            )


# =========================================================
# HISTOGRAM ANALYSIS
# =========================================================

elif selected_module == "📈 Histogram Analysis":

    st.markdown(
        '<div class="section-title">'
        '📈 Histogram Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        key="histogram"
    )

    if uploaded_file is not None:

        image, image_array, image_bgr = load_image(
            uploaded_file
        )

        gray, blurred = preprocess_image(
            image_bgr
        )

        enhanced = histogram_equalization(
            gray
        )

        original_hist = get_histogram(
            gray
        )

        enhanced_hist = get_histogram(
            enhanced
        )

        st.subheader("Original Histogram")

        fig1, ax1 = plt.subplots()

        ax1.plot(original_hist)
        ax1.set_xlabel("Pixel Intensity")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Original Image Histogram")

        st.pyplot(fig1)

        st.subheader("Enhanced Histogram")

        fig2, ax2 = plt.subplots()

        ax2.plot(enhanced_hist)
        ax2.set_xlabel("Pixel Intensity")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Enhanced Image Histogram")

        st.pyplot(fig2)


# =========================================================
# EDGE DETECTION
# =========================================================

elif selected_module == "🔍 Edge Detection":

    st.markdown(
        '<div class="section-title">'
        '🔍 Edge Detection'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        key="edges"
    )

    if uploaded_file is not None:

        image, image_array, image_bgr = load_image(
            uploaded_file
        )

        gray, blurred = preprocess_image(
            image_bgr
        )

        edges = detect_canny_edges(
            blurred
        )

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Original Image")

            st.image(
                image,
                use_container_width=True
            )

        with col2:

            st.subheader("Canny Edges")

            st.image(
                edges,
                use_container_width=True
            )


# =========================================================
# FEATURE EXTRACTION
# =========================================================

elif selected_module == "⭐ Feature Extraction":

    st.markdown(
        '<div class="section-title">'
        '⭐ Feature Extraction'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        key="features"
    )

    if uploaded_file is not None:

        image, image_array, image_bgr = load_image(
            uploaded_file
        )

        gray, blurred = preprocess_image(
            image_bgr
        )

        corners = detect_harris_corners(
            gray,
            image_array
        )

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Original Image")

            st.image(
                image,
                use_container_width=True
            )

        with col2:

            st.subheader("Harris Corners")

            st.image(
                corners,
                use_container_width=True
            )


# =========================================================
# IMAGE SEGMENTATION
# =========================================================

elif selected_module == "🧩 Image Segmentation":

    st.markdown(
        '<div class="section-title">'
        '🧩 Image Segmentation'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        key="segmentation"
    )

    if uploaded_file is not None:

        image, image_array, image_bgr = load_image(
            uploaded_file
        )

        gray, blurred = preprocess_image(
            image_bgr
        )

        binary = binary_threshold(gray)
        otsu = otsu_threshold(gray)
        adaptive = adaptive_threshold(gray)

        col1, col2, col3 = st.columns(3)

        with col1:

            st.subheader("Binary")

            st.image(
                binary,
                use_container_width=True
            )

        with col2:

            st.subheader("Otsu")

            st.image(
                otsu,
                use_container_width=True
            )

        with col3:

            st.subheader("Adaptive")

            st.image(
                adaptive,
                use_container_width=True
            )


# =========================================================
# IMAGE COMPARISON
# =========================================================

elif selected_module == "🔗 Image Comparison":

    st.markdown(
        '<div class="section-title">'
        '🔗 Image Comparison'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload two images to compare their "
        "visual features using SIFT."
    )

    col1, col2 = st.columns(2)

    with col1:

        first_file = st.file_uploader(
            "Upload first image",
            type=["jpg", "jpeg", "png"],
            key="first_comparison"
        )

    with col2:

        second_file = st.file_uploader(
            "Upload second image",
            type=["jpg", "jpeg", "png"],
            key="second_comparison"
        )

    if first_file is not None and second_file is not None:

        image1, array1, _ = load_image(
            first_file
        )

        image2, array2, _ = load_image(
            second_file
        )

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("First Image")

            st.image(
                image1,
                use_container_width=True
            )

        with col2:

            st.subheader("Second Image")

            st.image(
                image2,
                use_container_width=True
            )

        result, match_count = sift_feature_matching(
            array1,
            array2
        )

        st.markdown("---")

        if result is not None:

            st.subheader(
                f"🔗 SIFT Matches: {match_count}"
            )

            st.image(
                result,
                caption="SIFT Feature Matching",
                use_container_width=True
            )

        else:

            st.warning(
                "Not enough features were detected "
                "in one or both images."
            )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "VisionLab | Interactive Computer Vision "
    "Image Analysis System"
)