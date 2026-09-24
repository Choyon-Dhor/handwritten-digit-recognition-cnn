import os
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

print("--- Training Handwritten Digit Recognition CNN Model ---")
print("Loading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize images to [0, 1]
X_train_cnn = (X_train / 255.0).reshape(-1, 28, 28, 1)
X_test_cnn = (X_test / 255.0).reshape(-1, 28, 28, 1)

# One-hot encode targets
y_train_cnn = to_categorical(y_train, 10)
y_test_cnn = to_categorical(y_test, 10)

print(f"Training samples: {X_train_cnn.shape[0]}, Test samples: {X_test_cnn.shape[0]}")

# Build CNN model as defined in notebook
model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("Training model for 5 epochs...")
model.fit(
    X_train_cnn, y_train_cnn,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

loss, acc = model.evaluate(X_test_cnn, y_test_cnn, verbose=1)
print(f"CNN Test Accuracy: {acc * 100:.2f}%")

model.save("digit_cnn_model.h5")
print("Model saved to digit_cnn_model.h5 successfully!")
