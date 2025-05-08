# This script trains a CNN model to classify plant diseases using the PlantVillage dataset.
# This script assumes you have the PlantVillage dataset organized in a specific directory structure.
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# --- CONFIG ---
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = "/Users/shabbirshaikh/Documents/archive (5)/PlantVillage"  # change if your dataset folder is named differently
MODEL_PATH = "model/plant_disease_model.h5"

# --- DATA PREPROCESSING ---
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# --- MODEL ARCHITECTURE ---
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*IMAGE_SIZE, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- TRAINING ---
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# --- EVALUATION ---
val_gen.reset()
preds = model.predict(val_gen)
y_pred = preds.argmax(axis=1)
y_true = val_gen.classes
class_labels = list(val_gen.class_indices.keys())

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_labels))

# --- SAVE MODEL ---
os.makedirs("model", exist_ok=True)
model.save(MODEL_PATH)
print(f"\n✅ Model saved at: {MODEL_PATH}")

# --- PLOT ACCURACY ---
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()
