import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

label_map = {
    "cat": 0,
    "dog": 1,
    "person": 2,
    "car": 3,
    "airplane": 4,
    "train": 5,
    "bike": 6,
    "motorbike": 7,
    "horse": 8,
    "animal": 9,
    "computer": 10,
}


X = np.load("X.npy")
y = np.load("y.npy")

print("Veriler yüklendi.")
print("X shape:", X.shape)
print("y shape:", y.shape)

# Veri setini eğitim ve doğrulama olarak ayır
X_train, X_val, y_train, y_val = train_test_split(X, y[:, -1], test_size=0.2, random_state=42)

# Kategorik hale getir
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=len(label_map))
y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=len(label_map))

# Basit CNN modeli
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(label_map), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Modeli eğit
history = model.fit(X_train, y_train_cat,
                    epochs=5,
                    batch_size=32,
                    validation_data=(X_val, y_val_cat))

# Eğitim ve doğrulama loss & accuracy grafiklerini çiz
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Eğitim Loss')
plt.plot(history.history['val_loss'], label='Doğrulama Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Grafiği')

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Doğruluk Grafiği')

plt.show()

# Mini sonuç raporu
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
print(f"Son epoch eğitim doğruluğu: {final_train_acc:.4f}")
print(f"Son epoch doğrulama doğruluğu: {final_val_acc:.4f}")

model.save("model.h5")
print("model kaydedildi.")