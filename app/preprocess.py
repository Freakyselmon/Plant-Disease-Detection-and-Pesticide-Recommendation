from PIL import Image
import numpy as np

def preprocess_image(image_file):
    image = Image.open(image_file).resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image
