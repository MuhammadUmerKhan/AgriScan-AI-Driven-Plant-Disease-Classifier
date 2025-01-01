# 🌱 Plant Disease Detection System 🌿

Welcome to the **Plant Disease Detection System**! This project leverages **Deep Learning** and **Transfer Learning** to identify plant diseases from images, helping farmers and gardeners diagnose issues quickly and effectively. 🌾  

---

## 📚 Table of Contents  
- [🔍 Overview](#-overview)  
- [🔧 Project Structure](#-project-structure)  
- [💻 Technologies Used](#-technologies-used)  
- [✔️ Current Work](#-current-work)  
- [🎯 Planned Future Enhancements](#-planned-future-enhancements)  
- [🚀 Getting Started](#-getting-started)  
- [🔄 Prerequisites](#-prerequisites)  
- [📚 Acknowledgments](#-acknowledgments)  

---  

## 📚 Description:

- **Potato Plant Disease Detection**: Potato diseases pose a significant threat to global food security, causing substantial yield losses. Accurate and timely disease detection is crucial for effective management and control. Recent advancements in image processing and deep learning have revolutionized disease classification, enabling automated and efficient solutions.

- **Tomato Plant Disease Detection**: Similar to potato plants, tomato diseases also affect crop yield significantly. This system classifies diseases like tomato mosaic virus, leaf mold, and bacterial spot, enabling early intervention for crop protection.

---

## 🔍 Overview  

This project uses a **Convolutional Neural Network (CNN)** to classify plant diseases from leaf images. The model helps in early diagnosis, potentially saving crops and improving yield.  

---  

## 🔧 Project Structure  

1. **Data Preprocessing**:  
   - Cleaned and augmented image data for better generalization.  
   - Split data into training, validation, and testing sets.
2. **Image Resizing**:
   - Converting images to (224, 224) a consistent size for efficient processing.
3. **Image Normalization**:
   - Adjusting pixel values to a specific range (e.g., 0-1).
4. **Data Augmentation**:
   - Creating new training samples by applying transformations:
        - Rotation 🖀
        - Flipping 🖁
        - Zooming 🔍
        - Cropping ✂️
5. **Model Architecture**:  
   - Built a CNN with layers optimized for image classification.  
   - Trained on labeled datasets to recognize diseases like early blight, late blight, tomato mosaic virus, and bacterial spot.  
6. **Deployment**:  
   - Integrated the trained model into a **[Streamlit](https://plant-leaf-desease-classification.streamlit.app/)** app for user-friendly interactions.  

---  

## 💻 Technologies Used  
- **🕰️ Python**: Core programming language.  
- **🎡 TensorFlow/Keras**: For building and training the CNN model.  
- **📊 Streamlit**: For deploying the application.  
- **🧮 NumPy & Pandas**: For data handling.  
- **🌀 Matplotlib & Seaborn**: For visualizing data and results.  

---  

## ✔️ Current Work  

- Trained the model to identify diseases in specific plant species (Potato and Tomato).
- Deployed a user-friendly app for uploading leaf images and predicting diseases.  
- Achieved high accuracy on the test dataset.  

---  

## 🎯 Planned Future Enhancements  

1. **🌱 Add More Diseases**:  
   - Expand the dataset to include more plant species and disease types.  

2. **⚙️ Advanced Models**:  
   - Experiment with transfer learning and pre-trained models like ResNet or EfficientNet.  

3. **📈 Improved Metrics**:  
   - Enhance precision and recall for minority disease classes.  

4. **📱 UI/UX Improvements**:  
   - Add options for uploading multiple images and providing batch predictions.  

---  

## 🚀 Getting Started  

To set up this project locally:  

1. **Clone the repository**:  
   ```bash  
   git clone https://github.com/MuhammadUmerKhan/Plant-Disease-Detection-System.git
   ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the application:
    ```bash
    streamlit run disease_classifier.py
    ```

---

## 🔄 Prerequisites  
- Python 3.x
- Required packages are listed in requirements.txt.

---

## 📚 Acknowledgments  

- **Datasets:**  
   - [PlantVillage Dataset (Potato)](https://www.kaggle.com/datasets/arjuntejaswi/plant-village) for labeled potato leaf images.
   - [Tomato Leaf Dataset](https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf) for labeled tomato leaf images.
