
# streamlit run "/Users/asmizaffor/Downloads/6TH SEM/minor project/app.py"

# app.py
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import joblib
from PIL import Image
import os

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Function to change pages
def change_page(page_name):
    st.session_state.page = page_name

# Configuration
st.set_page_config(
    page_title="BreastAI - Cancer Detection System",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS for entire app
st.markdown("""
<style>
    /* Global baby pink background */
    .main, .stApp {
        background-color: #ffe6f2 !important;
    }
    
    /* Navigation sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffd1e0 !important;
        border-right: 2px solid #ff99c2;
    }
    
    /* Navigation title */
    [data-testid="stSidebar"] h1 {
        color: white !important;
        text-shadow: 1px 1px 2px #4d001f;
    }
    
    /* All buttons */
    .stButton>button {
        background-color: #ff66b3 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton>button:hover {
        background-color: #ff3385 !important;
        transform: scale(1.05);
    }
    
    /* Page titles */
     li, h1, h2, h3 {
        color: #4d001f !important;
    }
    
    /* Cards */
    .process-card {
        background: white !important;
        border-radius: 15px;
        padding: 25px;
        margin: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .process-card h3 {
        color: #4d001f !important;
    }
    .process-card p {
        color: #4d001f !important;  /* Matching your theme */
        opacity: 0.9;
        font-size: 1rem;
    }
    
    /* Patient Form Container */
    [data-testid="stForm"] {
        background-color: #FFC0D9 !important;
        border-radius: 15px !important;
        padding: 25px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
        border: 1px solid #ff99c2 !important;  /* Light pink border */
    }
    
    /* Form Title */
    [data-testid="stForm"] h3 {
        color: #4d001f !important;  /* Dark maroon */
    }
    
    /* Change all form labels to maroon */
    label p {
        color: #4d001f !important;
        font-weight: 500 !important;
    }
    
    /* Specific targeting for patient form */
    [data-testid="stForm"] label p {
        color: #4d001f !important;
    }
    
    .stException, .stAlert {
        color: #4d001f !important;
    }
    
    /* Malignant card styling */
    .malignant {
        background-color: #FFCCCC !important;  /* Light red background */
        border: 2px solid #4d001f !important;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
    }
    .benign {
        background-color: #FFC0D7 !important;  /* Light green */
        border: 2px solid #004d00 !important;
    }
    
    /* Image caption styling */
    .stImage > div > div {
        color: #4d001f !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
    
    
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>Navigation</h1>", unsafe_allow_html=True)
    
    if st.button("🏠 Home"):
        change_page("Home")
    if st.button("🔍 Detection"):
        change_page("Detection")
    if st.button("ℹ️ About"):
        change_page("About")

# Home Page
if st.session_state.page == 'Home':
    st.title("BREAST CANCER DETECTION SYSTEM 🎗️")
    
    # Header image
    try:
        st.image("assets/img2.jpg", use_container_width=True)
    except:
        st.image("https://www.emro.who.int/images/stories/ncds/bcam_2022_banner.jpg", 
                 use_container_width=True)
    
    
    # How It Works section
    st.subheader("📋 Simple 3-Step Process")
    
    steps = st.columns(3)
    with steps[0]:
        st.markdown("""
        <div class="process-card">
            <h3>1. Upload</h3>
            <img src="https://cdn-icons-png.flaticon.com/512/126/126477.png" width=80>
            <p>Submit your histopathology image in JPG/PNG format</p>
        </div>
        """, unsafe_allow_html=True)
    
    with steps[1]:
        st.markdown("""
        <div class="process-card">
            <h3>2. Analyze</h3>
            <img src="https://cdn-icons-png.freepik.com/512/809/809497.png" width=80>
            <p>Our AI processes the image in seconds</p>
        </div>
        """, unsafe_allow_html=True)
    
    with steps[2]:
        st.markdown("""
        <div class="process-card">
            <h3>3. Results</h3>
            <img src="https://t4.ftcdn.net/jpg/10/64/30/33/360_F_1064303343_SbaJIvWWbmlt3ZBSkbkfOKXcoIKa6OHt.jpg" width=80>
            <p>Receive detailed diagnosis report</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Call to Action 
    st.markdown("""<div style="text-align: center; padding: 2rem; background: white; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin: 2rem 0;">
    <h2 style="color: #4d001f;">Ready to Get Started?</h2>
    <p style="color: #666666; font-size: 1.1rem;">Detect potential cancer markers in your histopathology images today</p>
    </div>""", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
      if st.button("Begin Analysis Now", type="primary",use_container_width=True, help="Click to start cancer detection analysis"):
        change_page("Detection")

# DETECTION PAGE
elif st.session_state.page == 'Detection':
    st.title("Cancer Detection Portal")

    
    @st.cache_resource
    def load_model():
        return tf.keras.models.load_model("best_model.keras")

    try:
        model = load_model()
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.stop()

    with st.container():
        with st.form("patient_info"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Patient Name", placeholder="John Doe", key="name_input")
                age = st.number_input("Age", min_value=18, max_value=100, value=35, key="age_input")
            with col2:
                gender = st.selectbox("Gender", ["Female", "Male", "Other"], key="gender_select")
                upload_date = st.date_input("Scan Date", key="date_input")

            
            uploaded_file = st.file_uploader("Choose histopathology image", 
                                          type=["jpg", "png", "jpeg"],
                                          key="image_upload")

            submitted = st.form_submit_button("🚀 Submit for Diagnosis")

    if submitted:
        # Validates all inputs 
        if not all([name, age, uploaded_file]):
            st.warning("⚠️ Please fill all fields and upload an image")
            st.stop()

        try:
            # Processes image
            image = Image.open(uploaded_file).convert("RGB")
            img = np.array(image)
            img = cv2.resize(img, (260, 260))
            img_array = np.expand_dims(img, axis=0).astype(np.float32) / 255.0

            # Prediction
            with st.spinner("🔬 Analyzing tissue patterns..."):
                predictions = model.predict(img_array)
                class_idx = np.argmax(predictions)
                class_name = "Malignant" if class_idx == 1 else "Benign"
                confidence = np.max(predictions) * 100 

            # Displays results
            with st.container():
                col_img, col_result = st.columns([1, 2])

                with col_img:
                    st.image(image, 
                           caption="Uploaded Histopathology Image",
                           use_container_width=True)

                with col_result:
                    st.subheader("Diagnosis Report")
                    
                    result_theme = "malignant" if class_name == "Malignant" else "benign"
                    
                    st.markdown(f"""
                    <div class="prediction-box {result_theme}">
                        <div style="display: flex; align-items: center; gap: 15px;">
                            <h2 style="margin: 0;">{'⚠️' if result_theme == 'malignant' else '✅'}</h2>
                            <div>
                                <h3 style="margin: 0;">{class_name}</h3>
                                <p style="margin: 0; font-size: 1.2em;">Confidence: {confidence:.1f}%</p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Recommendations 
                    if class_name == "Malignant":
                        st.markdown("""
                        ### 🚨 Immediate Recommendations
                        - Schedule oncologist consultation within 48 hours
                        - Request biopsy confirmation
                        - Review treatment options
                        """)
                    else:
                        st.markdown("""
                        ### ✅ Preventive Measures
                        - Annual mammogram screening
                        - Monthly self-examinations
                        - Maintain healthy BMI
                        """)

        except Exception as e:
            st.error(f"❌ Processing error: {str(e)}")
            st.info("Supported formats: JPG, PNG, JPEG | Max size: 5MB")

# About Page
elif st.session_state.page == 'About':
    st.title("About Us")
    
    # Custom CSS injection
    st.markdown("""
    <style>
    .about-text {
        color: #800000 !important;
        font-family: 'Arial', sans-serif;
        line-height: 1.7;
        text-align: justify;
    }
    .about-text h2 {
        color: #5E0C15 !important;
        border-bottom: 2px solid #800000;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
    }
    .about-list {
        margin-left: 1.8rem;
        margin-top: 0.5rem;
    }
    .about-list li {
        margin-bottom: 0.8rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
        <div class="about-text">
        <h2>Our Mission</h2>
        
        <p>Breast cancer affects millions worldwide, with early detection being crucial for successful treatment. 
        Histopathological analysis of biopsy samples remains the diagnostic gold standard, but manual examination 
        under microscopes is time-intensive and subject to human error. This system aims to support pathologists 
        by providing AI-powered insights for faster, more consistent preliminary assessments.</p>

        <h2>Our AI-Driven Solution</h2>
        
        <p>This system utilizes advanced deep learning technology to analyze breast histopathology images and classify 
        tumors as <strong>benign</strong> or <strong>malignant</strong>. Key features include:</p>
        
        <div class="about-list">
            &#10004; <strong>Real-time predictions</strong> via intuitive web interface<br>
            &#10004; <strong>Confidence scoring</strong> for uncertain cases<br>
            &#10004; <strong>Multi-magnification support</strong> (40X-400X)<br>
            
        </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
        ## Technology Stack ⚙️
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin: 20px 0;">
            <div style="padding: 15px; background: #ffffgf; color: #4d001f; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                <h4>🧠 Deep Learning Core</h4>
                <ul>
                    <li>EfficientNetB2 Architecture</li>
                    <li>TensorFlow/Keras Implementation</li>
                    <li>Transfer Learning Optimization</li>
                </ul>
            </div>
            <div style="padding: 15px; background: #ffffgf; color: #4d001f; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                <h4>🖥️ Application Layer</h4>
                <ul>
                    <li>Streamlit Web Framework</li>
                    <li>OpenCV Image Processing</li>
                    <li>Cloud-Ready Architecture</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
        ## Performance Metrics 📊
        """)
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Overall Accuracy", "74%", "14% improvement from v1")
        with metric_col2:
            st.metric("Sensitivity", "75.2%", "True Positive Rate")
        with metric_col3:
            st.metric("Specificity", "74.3%", "True Negative Rate")
        
        st.markdown("---")
        

# CSS Styling Update
st.markdown("""
<style>
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .benign {
        background: #e8f5e9;
        border-color: #43a047;
    }
    .malignant {
        background: #ffebee;
        border-color: #e53935;
    }
    [data-testid="stForm"] {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 2rem;
    }
</style>
""", unsafe_allow_html=True)