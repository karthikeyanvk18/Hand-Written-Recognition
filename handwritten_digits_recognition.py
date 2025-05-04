import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping  # ✅ Add this line


print("Welcome to the NeuralNine (c) Handwritten Digits Recognition v0.2")

# Check GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Toggle to train a new model or load an existing one
train_new_model = True

if train_new_model:
    # Load the MNIST dataset
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalize pixel values (0–255 → 0–1)
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Build the model with Dropout
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Use EarlyStopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=2)

    # Train the model
    model.fit(X_train, y_train, epochs=20, validation_split=0.1, callbacks=[early_stop])

    # Evaluate the model
    val_loss, val_acc = model.evaluate(X_test, y_test)
    print(f"Validation Loss: {val_loss}")
    print(f"Validation Accuracy: {val_acc}")

    # Save the trained model
    model.save('handwritten_digits.keras')
else:
    # Load an existing model
    model = tf.keras.models.load_model('handwritten_digits.keras')

# Predict digits from custom images
image_number = 1
while os.path.isfile(f'static/digits/digit{image_number}.png'):
    try:
        img = cv2.imread(f'static/digits/digit{image_number}.png')[:, :, 0]  # Load as grayscale
        img = cv2.resize(img, (28, 28))  # Ensure it's 28x28
        img = np.invert(np.array(img))  # Invert colors
        img = img / 255.0  # Normalize
        img = img.reshape(1, 28, 28)  # Reshape for prediction

        prediction = model.predict(img)
        predicted_digit = np.argmax(prediction)

        print(f"The number is probably a {predicted_digit}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.title(f"Predicted: {predicted_digit}")
        plt.axis('off')
        plt.show()

        image_number += 1
    except Exception as e:
        print(f"Error reading image {image_number}: {e}")
        image_number += 1
