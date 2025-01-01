import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

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
st.markdown('<div class="main-title">🌱 Welcome to the Potato Plant Disease Classification Tool 🌱</div>', unsafe_allow_html=True)

# Tab layout
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Dashboard", "🥔 Potato Disease Analysis", "🍅 Tomato Disease Analysis", "🌶️ Pepper Disease Analysis"])

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
                    Plant 🌿 diseases pose a significant threat to global food security, causing substantial yield losses. 
                    Accurate and timely disease detection is crucial for effective management and control. 
                    Recent advancements in image processing and deep learning have revolutionized disease 
                    classification, enabling automated and efficient solutions.<br/>
                </li>
                <li><span class="highlight">🥔 Potato Plant Disease Detection:</span> 
                    Built disease detection 🦠 model to identify different potato diseases using Convolutional Neural Networks (CNNs) 🧠. 
                    The model was trained on a 
                    <a href="https://www.kaggle.com/datasets/arjuntejaswi/plant-village" target="_blank" style="color: silver; font-weight: bold;">dataset</a>
                    of potato plant images and is deployed for real-time predictions. 📡<br/>
                    <ul>
                        <li><span class="highlight">🤝 Steps to Reproduce:</span>
                            <ul>
                                <li>Captured in a real potato farm.</li>
                                <li>Uncontrolled environment using a high-resolution digital camera and smartphone.</li>
                                <li>Dataset aids researchers in computer vision.</li>
                            </ul>
                        </li>
                        <li><span class="highlight">🔄 Data Preprocessing and Augmentation:</span>
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
                                        </ul>
                                    </ul>
                                </li>
                            </ul>
                        </li>
                    </ul>
                </li>
                <li><span class="highlight">🍅 Tomato Plant Disease Detection:</span> 
                    Built disease detection 🦠 model to identify different tomato diseases using Transfer Learning 🧠. 
                    The model was trained on a 
                    <a href="https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf" target="_blank" style="color: silver; font-weight: bold;">dataset</a>
                    of tomato plant images and is deployed for real-time predictions. 📡<br/>
                    <ul>
                        <li><span class="highlight">🤝 Steps to Reproduce:</span>
                            <ul>
                                <li>Captured in a real tomato farm.</li>
                                <li>Uncontrolled environment using a high-resolution digital camera and smartphone.</li>
                                <li>Dataset aids researchers in computer vision.</li>
                            </ul>
                        </li>
                        <li><span class="highlight">🔄 Data Preprocessing and Augmentation:</span>
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
                            </ul>
                        </li>
                    </ul>
                </li>
                <li><span class="highlight">🌶️ Pepper Plant Disease Detection:</span> 
                    Built disease detection 🦠 model to identify different pepper diseases using Transfer Learning 🧠. 
                    The model was trained on a 
                    <a href="https://www.kaggle.com/datasets/arjuntejaswi/plant-village" target="_blank" style="color: silver; font-weight: bold;">dataset</a>
                    of pepper plant images and is deployed for real-time predictions. 📡<br/>
                    <ul>
                        <li><span class="highlight">🤝 Steps to Reproduce:</span>
                            <ul>
                                <li>Captured in a real pepper farm.</li>
                                <li>Uncontrolled environment using a high-resolution digital camera and smartphone.</li>
                                <li>Dataset aids researchers in computer vision.</li>
                            </ul>
                        </li>
                        <li><span class="highlight">🔄 Data Preprocessing and Augmentation:</span>
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
                            </ul>
                        </li>
                    </ul>
                </li>
            </ul>
        </div>
    """, unsafe_allow_html=True)


    # Future Work Section
    st.markdown('<div class="section-title">🚀 Future Work</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            While this project currently focuses on potato plant disease classification, I aim to expand its scope to cover:
            <ul>
                <li><span class="highlight">🌾 Multi-Crop Disease Detection:</span> Incorporating classification models for other crops like tomatoes, wheat, and corn.</li>
                <li><span class="highlight">🤝 Farmer-Friendly Mobile App:</span> Developing a user-friendly mobile application to enable real-time field diagnosis and recommendations for farmers.</li>
            </ul>
            These enhancements aim to provide a comprehensive tool for farmers and agricultural researchers, contributing to sustainable farming practices. 🌱
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">💻 Technologies & Tools</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
            <ul>
                <li><span class="highlight">🔤 Languages & Libraries:</span> Python, NumPy, Pandas, Matplotlib, TensorFlow, Keras, and Scikit-Learn.</li>
                <li><span class="highlight">⚙️ Approaches:</span> Convolutional Neural Networks (CNNs), Data Augmentation, Transfer Learning, and Image Preprocessing Techniques.</li>
                <li><span class="highlight">🌐 Deployment:</span> Streamlit for building an interactive, user-friendly web-based system.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# Second Tab: Find Defection on Image
with tab2:
    st.markdown('<div class="section-title">🥔 Potato Plant Disease Classification 🦠</div>', unsafe_allow_html=True)
    st.markdown('''
    <div class="content">
        Upload a clear image of a potato plant leaf 🥔, and the model will identify its health status or diagnose any potential disease as following:.
        <ul>
            <li>Healthy 🌱.</li>
            <li>Early Blight 🦠.</li>
            <li>Late Blight 🦠.</li>
        </ul>
    </div><br/>
    ''', unsafe_allow_html=True)

    # Layout with two columns
    col1, col2 = st.columns([1, 2])  # 1: Image section, 2: Prediction section

    with col1:
        uploaded_file = st.file_uploader("📸 Upload a leaf image:", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=250)  # Adjust the width as needed

    with col2:
        if uploaded_file is not None:
            # Preprocess the image
            image = image.resize((256, 256))  # Adjust size as per model input
            image_batch = np.expand_dims(image, axis=0)

            potato_classifier_model = tf.keras.models.load_model('./model/potato_desease_classifier_v3.h5')
            potato_target_labels = ["Early Blight", "Late Blight", "Healthy"]
            
            # Make predictions
            predictions = potato_classifier_model.predict(image_batch)
            predicted_class = potato_target_labels[np.argmax(predictions[0])]
            confidence = np.max(predictions[0])
            
            if predicted_class.lower() == "healthy":
                status_message = f"Your 🥔 plant is <span style='color: #4CAF50;'>{predicted_class}</span>  🌱."
            else:
                status_message = f"Disease Detected: <span style='color:  #c40000;;'>{predicted_class}</span> 🦠."

            # Display the result with dynamic color for the disease name only
            st.markdown(f'''
            <br/><br/><br/><br/><br/><br/><br/><br/><br/>
            <div style="font-size: 2em; font-weight: bold; text-align: center;">
                {status_message}
            </div>
            <div style="color: silver; font-size: 2em; font-weight: bold; text-align: center;">
                Confidence: {confidence:.2f} 🔍.
            </div>
            ''', unsafe_allow_html=True)


with tab3:   
    st.markdown('<div class="section-title">🍅 Tomato Plant Disease Classification 🦠</div>', unsafe_allow_html=True)
    st.markdown('''
    <div class="content">
        Upload a clear image of a tomato plant leaf 🍅, and the model will identify its health status or diagnose any potential disease from the following:
        <ul>
            <li>Tomato Healthy 🌱.</li>
            <li>Tomato Mosaic Virus 🦠.</li>
            <li>Target Spot 🦠.</li>
            <li>Bacterial Spot 🦠.</li>
            <li>Tomato Yellow Leaf Curl Virus 🦠.</li>
            <li>Late Blight 🦠.</li>
            <li>Leaf Mold 🦠.</li>
            <li>Early Blight 🦠.</li>
            <li>Spider Mites (Two-spotted Spider Mite) 🦠.</li>
            <li>Septoria Leaf Spot 🦠.</li>
        </ul>
    </div><br/>
    ''', unsafe_allow_html=True)

    # Layout with two columns
    col1, col2 = st.columns([1, 2])  # 1: Image section, 2: Prediction section

    with col1:
        uploaded_file = st.file_uploader("🖼️ Upload a leaf image:", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=250)  # Adjust the width as needed

    with col2:
        if uploaded_file is not None:
            # Preprocess the image
            img = load_img(uploaded_file, target_size=(256, 256))
            img_array = img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0) 

            # Load the trained model
            tomato_classifier_model = tf.keras.models.load_model('./model/tomato_desease_classifier_v1.h5')

            # Predict
            predictions = tomato_classifier_model.predict(img_array)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions[0])

            # Mapping predictions to class names
            class_names = ['Bacterial spot',
                            'Early blight',
                            'Late blight',
                            'Leaf Mold',
                            'Septoria leaf spot',
                            'Spider Mites (Two-spotted Spider Mite)',
                            'Target Spot',
                            'Yellow Leaf Curl Virus',
                            'Mosaic Virus',
                            'Healthy']
            
            predicted_class_name = class_names[predicted_class_index]

            # Assign colors based on health status
            if predicted_class_name.lower() == "healthy":
                status_message = f"Your 🍅 plant is <span style='color: #4CAF50;'>{predicted_class_name}</span>  🌱."
            else:
                status_message = f"Disease Detected: <span style='color: #c40000;'>{predicted_class_name}</span> 🦠."

            # Display the result with dynamic color for the disease name only
            st.markdown(f'''
            <br/><br/><br/><br/><br/><br/><br/><br/><br/>
            <div style="font-size: 2em; font-weight: bold; text-align: center;">
                {status_message}
            </div>
            <div style="color: silver; font-size: 2em; font-weight: bold; text-align: center;">
                Confidence: {confidence:.2f} 🔍.
            </div>
            ''', unsafe_allow_html=True)

    # Add a separator line before the footer
    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
with tab4:   
    st.markdown('<div class="section-title">🌶️ Pepper Plant Disease Classification 🦠</div>', unsafe_allow_html=True)
    st.markdown('''
    <div class="content">
        Upload a clear image of a tomato plant leaf 🌶️, and the model will identify its health status or diagnose any potential disease from the following:
        <ul>
            <li>Pepper Bell Healthy 🌱.</li>
            <li>Pepper Bell Bacterial Spot 🦠.</li>
        </ul>
    </div><br/>
    ''', unsafe_allow_html=True)

    # Layout with two columns
    col1, col2 = st.columns([1, 2])  # 1: Image section, 2: Prediction section

    with col1:
        uploaded_file = st.file_uploader("⬆️ Upload a leaf image:", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=250)  # Adjust the width as needed

    with col2:
        if uploaded_file is not None:
            # Preprocess the image
            img = load_img(uploaded_file, target_size=(256, 256))
            img_array = img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0) 

            # Load the trained model
            pepper_classifier_model = tf.keras.models.load_model('./model/pepper_desease_classifier_v1.h5')

            predictions = pepper_classifier_model.predict(img_array)
            confidence = predictions[0][0]

            # Assign colors based on health status
            threshold = 0.5
            if confidence <= threshold:
                predicted_class_name = "Bacterial Spot"
                status_message = f"Disease Detected: <span style='color: #c40000;'>{predicted_class_name}</span>  🦠."
                confidence = 1 - confidence
            else:
                predicted_class_name = "Healthy"
                status_message = f"Your 🌶️ plant is <span style='color: #4CAF50;'>{predicted_class_name}</span> 🌱."

            # Display the result with dynamic color for the disease name only
            st.markdown(f'''
            <br/><br/><br/><br/><br/><br/><br/><br/><br/>
            <div style="font-size: 2em; font-weight: bold; text-align: center;">
                {status_message}
            </div>
            <div style="color: silver; font-size: 2em; font-weight: bold; text-align: center;">
                Confidence: {confidence:.2f} 🔍.
            </div>
            ''', unsafe_allow_html=True)

    # Add a separator line before the footer
    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)
    
# Footer
st.markdown("""
    <div class="footer">
        Developed by <a href="https://portfolio-sigma-mocha-67.vercel.app" target="_blank" style="color: silver; font-weight: bold;">Muhammad Umer Khan</a>. Powered by TensorFlow and Streamlit. 🌐
    </div>
""", unsafe_allow_html=True)