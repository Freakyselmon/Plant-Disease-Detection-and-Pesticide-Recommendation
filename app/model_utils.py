# This module contains functions to load the model and make predictions.
# app/model_utils.py
import tensorflow as tf
import numpy as np
import pandas as pd

# Load the trained model
model = tf.keras.models.load_model("model/plant_disease_model.h5")

# Path to your CSV file containing disease and pesticide data
csv_path = "/Users/shabbirshaikh/Documents/crop_disease_ai/data/disease_pesticide.csv"

# Load the CSV data
disease_data = pd.read_csv(csv_path)

# Check the actual column names in the CSV to avoid KeyError
print("CSV Columns:", disease_data.columns)

# Assuming 'Plant Disease' is the correct column name, adjust it if necessary
if 'Plant Disease' in disease_data.columns:
    class_names = disease_data['Plant Disease'].unique().tolist()
else:
    print("Error: 'Plant Disease' column not found. Please check your CSV file.")
    class_names = []

def predict_disease(image_array):
    """
    Predicts the disease class from a preprocessed image array.
    
    Args:
        image_array (np.array): A single image with shape (1, height, width, 3), scaled between 0 and 1.
        
    Returns:
        str: Predicted class name or error message if prediction fails.
    """
    # Ensure image is a batch (add an extra dimension for batch size)
    if len(image_array.shape) == 3:
        image_array = np.expand_dims(image_array, axis=0)
    
    try:
        # Make prediction
        prediction = model.predict(image_array)
        class_index = np.argmax(prediction[0])
        confidence = prediction[0][class_index]
        
        # Set a confidence threshold for "Healthy" classification
        if class_names and class_names[class_index] == 'Healthy' and confidence < 0.8:
            return "Uncertain: Possibly Healthy"
        
        return class_names[class_index] if class_names else "Error: No class names found"
    
    except Exception as e:
        return f"Error during prediction: {e}"


