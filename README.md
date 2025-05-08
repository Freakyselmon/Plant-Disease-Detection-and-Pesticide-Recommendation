🌿 Plant Disease Detection and Pesticide Recommendation
A deep learning-based web app that detects plant diseases from leaf images using a Convolutional Neural Network (CNN) and provides appropriate pesticide recommendations. The system is powered by TensorFlow for disease classification and integrates LLaMA 3 8B for intelligent, context-aware pesticide recommendations based on the detected diseases.

🚀 Features
Real-time Disease Prediction: Upload a leaf image to get an instant disease prediction from multiple plant diseases or "Healthy."

Accurate Predictions: Uses a deep CNN trained on the PlantVillage dataset for high classification accuracy.

Pesticide Recommendations: Provides tailored pesticide recommendations based on the predicted disease, including dosage, application method, frequency, and the effective range.

Context-Aware Guidance: Integrated with LLaMA 3 8B for generating detailed, context-aware recommendations for precision agriculture.

User-Friendly Web Interface: Simple and intuitive UI built with Streamlit (or Flask).

Mobile/Tablet-Ready: Optimized for responsive design and easy to use across different devices.

📁 Dataset
Source: PlantVillage Dataset on Kaggle

Structure:

PlantVillage/
├── Apple___Scab/
├── Tomato___Leaf_Curl/
├── ...
└── Healthy/
Ensure that the dataset is organized into folders, one per class.

🧠 Model Overview
Architecture: CNN with 3 convolutional layers followed by dense layers for classification.

Input Image Size: 224x224x3 (resized images for uniformity).

Training:

Image augmentation using ImageDataGenerator to improve robustness.

Trained to classify 38 different plant diseases plus "Healthy."

Option for Transfer Learning: Optionally, MobileNetV2 or EfficientNet can be used for better accuracy and faster inference.

Model Output: The model predicts the disease class with a confidence score.

Integration with LLaMA 3 8B: Using this language model to provide relevant pesticide recommendations based on detected diseases.

🔧 Installation

# Clone the repository
git clone https://github.com/yourusername/plant-disease-detector.git
cd plant-disease-detector

# Install dependencies
pip install -r requirements.txt
🏋️‍♂️ Training the Model
Adjust the DATA_DIR in the script to point to the dataset.

Train the model with:

python train_model.py
The model is saved in model/plant_disease_model.h5.

Class mappings are saved in model/class_indices.json.

🖼️ Inference / Prediction
To make a prediction using the trained model, you can use the predict_disease function:

from app.model_utils import predict_disease
from tensorflow.keras.preprocessing import image
import numpy as np

img = image.load_img("test_leaf.jpg", target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

result = predict_disease(img_array)
print("Prediction:", result)
💊 Pesticide Recommendation
The model is paired with a CSV file (pesticides.csv) that maps each disease to:

Recommended pesticide

Dosage (e.g., 2 ml/litre)

Application method (e.g., foliar spray)

Frequency (e.g., every 7 days)

Effective Against (e.g., specific symptoms)

Example pesticides.csv snippet:

Plant Disease,Pesticide,Dosage,Application Method,Frequency,Effective Against
Tomato Leaf Curl,XYZ Pesticide,2 ml/litre,Foliar spray,Every 7 days,Leaf curl; wilting
Rice Blast,ABC Fungicide,3 ml/litre,Foliar spray,Every 10 days,Fungal infection; leaf spot
...
The predict_disease function works seamlessly with the CSV file to provide pesticide recommendations based on the detected plant disease.

🧪 Evaluation
Validation Accuracy: The model achieves [insert accuracy %] on validation data.

Loss and Accuracy Plot: Graphs are generated to track model performance during training.

Classification Report: Generated using sklearn.metrics for detailed evaluation.

Misclassified Images: Optionally, misclassified images can be visualized to analyze performance.

📊 Accuracy Plot
The training process includes a plot that visualizes training and validation accuracy over epochs to help assess model performance.

🛠️ Tech Stack
Languages: Python 3.10+

Machine Learning: TensorFlow / Keras

Libraries: NumPy, Matplotlib, OpenCV, scikit-learn

Web Framework: Flask or Streamlit (optional)

Data Visualization: Matplotlib, Plotly

Pesticide Recommendation: LLaMA 3 8B for NLP-backed recommendations

Others: Pandas, Pillow

💡 Future Improvements
Integration of Transfer Learning Models: MobileNetV2 or EfficientNet for improved model accuracy and faster predictions.

Real-time Camera Prediction: Allow users to use their mobile/web camera to capture leaf images for disease prediction.

Expand to More Crops: Extend the system to handle a wider range of crops and diseases.

Deployment: Deploy as a mobile or web app using platforms like Heroku, AWS, or Google Cloud for wider accessibility.

🙌 Acknowledgements
PlantVillage Dataset: PlantVillage dataset used for training.

TensorFlow Team: For developing the powerful deep learning framework.

LLaMA 3 8B: Used to generate intelligent, context-aware pesticide recommendations.

Open-Source Contributors: Thanks to all contributors for their valuable input.

📄 License
This project is licensed under the MIT License.

Optional Additions:
requirements.txt: Ensure you include a requirements.txt file with all the necessary Python packages:

tensorflow==2.x
numpy==1.21.2
pandas==1.3.3
scikit-learn==0.24.2
opencv-python==4.5.3.56
matplotlib==3.4.3
Pillow==8.3.1
streamlit==0.88.0
llama==3.8
Sample Images: Provide sample prediction images to demonstrate how the tool works.

Deployment Instructions: If deploying as a web or mobile app, provide step-by-step instructions for hosting on platforms like Heroku, AWS, or Google Cloud.

# Plant-Disease-Detection-and-Pesticide-Recommendation
# Plant-Disease-Detection-and-Pesticide-Recommendation
