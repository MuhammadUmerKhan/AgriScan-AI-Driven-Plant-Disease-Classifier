import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Streamlit page configuration
st.set_page_config(
    page_title="Plant Disease Classification",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            margin-top: -10px;
        }
        /* Main Title */
        .main-title {
            font-size: 2.5em;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
            margin-bottom: 20px;
        }
        /* Section Titles */
        .section-title {
            font-size: 1.8em;
            color: #C0C0C0;
            font-weight: bold;
            margin-top: 30px;
            text-align: left;
        }
        /* Tab Title Customization */
        .stTab {
            font-size: 1.4em;  /* Increase tab title font size */
            font-weight: bold;
            color: #2980B9;
        }
        /* Section Content */
        .section-content{
            text-align: center;
        }
        /* Home Page Content */
        .intro-title {
            font-size: 2.5em;
            color: #00ce39;
            font-weight: bold;
            text-align: center;
        }
        .intro-subtitle {
            font-size: 1.2em;
            color: #017721;
            text-align: center;
        }
        .content {
            font-size: 1em;
            color: #7F8C8D;
            text-align: justify;
            line-height: 1.6;
        }
        .highlight {
            # color: #068327;
            font-weight: bold;
        }
        /* Separator Line */
        .separator {
            height: 2px;
            background-color: #BDC3C7;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        /* Prediction Text Styling */
        .prediction-text {
            font-size: 2em;
            font-weight: bold;
            color: #2980B9;
            text-align: center;
            text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
        }
        /* Footer */
        .footer {
            font-size: 14px;
            color: #95A5A6;
            margin-top: 20px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)


# Title Heading (appears above tabs and remains on all pages)
st.markdown('<div class="main-title">🌱 Welcome to the Plant Disease Classification Tool 🌱</div>', unsafe_allow_html=True)

# Tab layout
tab1, tab2 = st.tabs(["🏠 Home", "📋 Find Defection on Image"])

# First Tab: Home
with tab1:
    st.markdown('<div class="section-title">👋 About Me</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            Hi! I’m <span class="highlight">Muhammad Umer Khan</span>, an aspiring AI/Data Scientist passionate about 
            <span class="highlight">🤖 Natural Language Processing (NLP)</span> and 🧠 Machine Learning. 
            Currently pursuing my Bachelor’s in Computer Science, I bring hands-on experience in developing intelligent recommendation systems, 
            performing data analysis, and building machine learning models. 🚀
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🎯 Project Overview</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            Here are some of the key projects I have worked on:
            <ul>
                <li><span class="highlight">📋 Description:</span> 
                    Potato diseases pose a significant threat to global food security, causing substantial yield losses. 
                    Accurate and timely disease detection is crucial for effective management and control. 
                    Recent advancements in image processing and deep learning have revolutionized disease 
                    classification, enabling automated and efficient solutions.<br/>
                </li>
                <li><span class="highlight">🤝 Steps to Reproduce:</span> 
                    The <a href="https://www.kaggle.com/datasets/arjuntejaswi/plant-village" target="_blank" style="color: silver; font-weight: bold;">dataset</a> was captured in a real potato farm in an uncontrolled environment using a high-resolution 
                    digital camera and smartphone. The farm is located in Holeta, Ethiopia. Dataset preparation is challenging, 
                    and this dataset could help researchers in the field of computer vision.<br/>
                </li>
                <li><span class="highlight">🔄 Data Preprocessing and Augmentation:</span> 
                    These were key steps in building the potato image classification model. Here’s an overview:
                    <ul>
                        <li><span class="highlight">Image Cleaning:</span>
                            <ul>
                                <li>Removing noise, artifacts, and unwanted objects.</li>
                            </ul>
                        </li>
                        <li><span class="highlight">Image Resizing:</span>
                            <ul>
                                <li>Converting images to a consistent size for efficient processing.</li>
                            </ul>
                        </li>
                        <li><span class="highlight">Image Normalization:</span>
                            <ul>
                                <li>Adjusting pixel values to a specific range (e.g., 0-1).</li>
                            </ul>
                        </li>
                        <li><span class="highlight">Data Augmentation:</span>
                            <ul>
                                <li>Creating new training samples by applying transformations:</li>
                                <ul>
                                    <li>Rotation 🔄</li>
                                    <li>Flipping 🔁</li>
                                    <li>Zooming 🔍</li>
                                    <li>Cropping ✂️</li>
                                </ul>
                                <li>Helps increase data diversity and prevents overfitting.</li>
                            </ul>
                        </li>
                    </ul>
                </li>
                <li><span class="highlight">🍟 Potato Image Classification:</span> 
                    Built and deployed an image classification model to identify different potato diseases using 
                    Convolutional Neural Networks (CNNs). The model was trained on a dataset of potato plant images 
                    and was deployed using Streamlit for real-time predictions. 📡<br/>
                </li>
            </ul>
        </div>
    """, unsafe_allow_html=True)


    st.markdown('<div class="section-title">💻 Technologies & Tools</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            <ul>
                <li><span class="highlight">🔤 Languages & Libraries:</span> Python, NumPy, Pandas, Matplotlib, TensorFlow, Keras, and OpenCV.</li>
                <li><span class="highlight">⚙️ Approaches:</span> Convolutional Neural Networks (CNNs), Data Augmentation, Transfer Learning, and Image Preprocessing Techniques.</li>
                <li><span class="highlight">🌐 Deployment:</span> Streamlit for building an interactive, user-friendly web-based system.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# Second Tab: Find Defection on Image
with tab2:
    MODEL = tf.keras.models.load_model("./model/desease_classifier_v3.h5")
    CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
    
    st.markdown('<div class="section-title">🌿 Plant Disease Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="content">Upload an image of a plant leaf, and the model will predict whether it is affected by Early Blight, Late Blight, or if it is Healthy 🌱.</div><br/>', unsafe_allow_html=True)

    # Layout with two columns
    col1, col2 = st.columns([1, 2])  # 1: Image section, 2: Prediction section

    with col1:
        uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=250)  # Adjust the width as needed

    with col2:
        if uploaded_file is not None:
            # Preprocess the image
            image = image.resize((256, 256))  # Adjust size as per model input
            image_batch = np.expand_dims(image, axis=0)

            # Make predictions
            predictions = MODEL.predict(image_batch)
            predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
            confidence = np.max(predictions[0])

            # Display prediction in stylish font
            st.markdown(f'<br/><br/><br/><br/><br/><br/><br/><div style="color: silver; font-size: 2em; font-weight: bold; text-align: center;">Prediction: {predicted_class} 🍃</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="color: silver; font-size: 2em; font-weight: bold; text-align: center;">Confidence: {confidence:.2f} 🔍</div>', unsafe_allow_html=True)

# Add a separator line before the footer
st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

# Footer
# Footer
st.markdown("""
    <div class="footer">
        Developed by <a href="https://portfolio-sigma-mocha-67.vercel.app" target="_blank" style="color: silver; font-weight: bold;">Muhammad Umer Khan</a>. Powered by TensorFlow and Streamlit. 🌐
    </div>
""", unsafe_allow_html=True)
