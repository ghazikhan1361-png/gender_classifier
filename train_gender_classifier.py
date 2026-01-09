import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# --- CONFIGURATION [cite: 68-74] ---
DATA_DIR = 'data/utkface'
IMG_SIZE = 64
BATCH_SIZE = 32
RANDOM_SEED = 42

# --- PART A: DATA PREPROCESSING [cite: 57] ---

def load_dataset(data_dir, img_size=64, max_samples=None):
    """Loads images and extracts gender labels from filenames."""
    images = []
    labels = []
    image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
    
    if max_samples:
        image_files = image_files[:max_samples]
    
    print(f"Loading {len(image_files)} images...")
    
    for filename in image_files:
        try:
            # Format: [age]_[gender]_[race]_[date].jpg
            parts = filename.split('_')
            if len(parts) < 3: continue
            
            gender = int(parts[1]) # 0=Male, 1=Female [cite: 79]
            
            img_path = os.path.join(data_dir, filename)
            img = cv2.imread(img_path)
            if img is None: continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            
            images.append(img)
            labels.append(gender)
        except Exception:
            continue
            
    return np.array(images), np.array(labels)

# 1. Load Data
# Set max_samples=5000 for quick testing, set to None for full training [cite: 120]
X, y = load_dataset(DATA_DIR, img_size=IMG_SIZE, max_samples=None) 

# 2. Normalize Pixel Values [cite: 146]
X = X.astype('float32') / 255.0

# 3. Split Data [cite: 152-159]
# Split test set (10%)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=RANDOM_SEED, stratify=y)
# Split train/val (20% of remaining)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=RANDOM_SEED, stratify=y_temp)

print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

# --- PART B: MODEL BUILDING [cite: 175] ---

# Flatten data for Fully Connected Network [cite: 185-187]
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)
input_dim = X_train_flat.shape[1]

# Build the specific architecture requested in the Lab PDF [cite: 196-211]
model = models.Sequential([
    layers.Dense(1024, activation='relu', input_dim=input_dim),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    layers.Dense(1, activation='sigmoid')
])

# Compile [cite: 214-219]
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall')]
)

model.summary()

# Callbacks [cite: 228-247]
os.makedirs('models', exist_ok=True)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
    ModelCheckpoint('models/gender_classifier_model.h5', monitor='val_accuracy', save_best_only=True)
]

# Train [cite: 250-256]
# CHANGED: epochs set to 5 for faster training
history = model.fit(
    X_train_flat, y_train,
    batch_size=BATCH_SIZE,
    epochs=5, 
    validation_data=(X_val_flat, y_val),
    callbacks=callbacks
)

# --- EVALUATION [cite: 260] ---

# Evaluate on Test Set
results = model.evaluate(X_test_flat, y_test)
print(f"Test Loss: {results[0]:.4f}, Accuracy: {results[1]:.4f}")

# Plot History [cite: 308-326]
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.savefig('training_history.png')

# --- SAVE METADATA [cite: 377-396] ---
metadata = {
    'model_name': 'Gender Classifier Fully Connected',
    'input_shape': [IMG_SIZE, IMG_SIZE, 3],
    'flattened_features': input_dim,
    'output_classes': ['Male', 'Female'],
    'test_accuracy': results[1]
}
with open('models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)

print("Training complete. Model and metadata saved.")