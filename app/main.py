# This is the main entry point for the application.
from app.preprocess import preprocess_image
from app.model_utils import predict_disease
from app.recommender import get_recommendation


def process_crop_image(image_path):
    image = preprocess_image(image_path)
    disease = predict_disease(image)
    pesticide, dosage = get_recommendation(disease)
    response = f"Detected: {disease}. Use {pesticide} at {dosage}."
    return response

if __name__ == "__main__":
    print(process_crop_image("test_images/test1.jpg"))
