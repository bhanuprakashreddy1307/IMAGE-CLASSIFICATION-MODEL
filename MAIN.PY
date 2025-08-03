import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# 1. Simulate random image data
# Let's say we have 1000 images of 32x32 RGB (like CIFAR-10)
num_samples = 1000
image_height = 32
image_width = 32
channels = 3
num_classes = 10

# Generate random pixel values and labels
x_train = np.random.rand(num_samples, image_height, image_width, channels).astype(np.float32)
y_train = np.random.randint(0, num_classes, size=(num_samples,))

# Simulate test set
x_test = np.random.rand(200, image_height, image_width, channels).astype(np.float32)
y_test = np.random.randint(0, num_classes, size=(200,))

# 2. Define CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, channels)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# 3. Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Train the model
history = model.fit(x_train, y_train, epochs=5,
                    validation_data=(x_test, y_test),
                    batch_size=64)

# 5. Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy on Dummy Data: {test_acc:.2f}")

# 6. Plot training accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("CNN Training Accuracy on Random Data")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
