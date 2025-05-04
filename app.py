from flask import Flask, render_template, request, session, redirect, url_for
import os
import cv2
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.datasets import mnist  # Import MNIST dataset

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Load the pre-trained model or your updated model
model = tf.keras.models.load_model('handwritten_digits.keras')

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest'
)

# Learning Rate Scheduler
def lr_schedule(epoch):
    if epoch < 10:
        return 0.001
    else:
        return 0.0001

lr_scheduler = LearningRateScheduler(lr_schedule)

# Load MNIST dataset (train/test images and labels)
def load_mnist_data():
    # Load the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Preprocess the data: Normalize images to [0, 1] and reshape for the model
    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    # Reshape to fit the model input shape (28, 28, 1)
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

    return (train_images, train_labels), (test_images, test_labels)

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = load_mnist_data()

# Train the model on the MNIST dataset if needed (You can remove this part if you already have a trained model)
# model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(test_images, test_labels))

@app.route('/')
def index():
    session['image_number'] = 1
    return redirect(url_for('analyze'))


@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    image_number = session.get('image_number', 1)
    image_path = f'static/digits/digit{image_number}.png'

    if not os.path.isfile(image_path):
        return render_template('index.html', result="No more images to show.", image_path=None, animation_class=None, done=False)

    try:
        img = cv2.imread(image_path)[:, :, 0]
        img_resized = cv2.resize(img, (28, 28))
        img_inverted = np.invert(np.array(img_resized))
        img_normalized = img_inverted / 255.0
        img_input = img_normalized.reshape(1, 28, 28, 1)

        # Predict the digit
        prediction = model.predict(img_input)
        predicted_digit = np.argmax(prediction)

        result = f"The number is probably a {predicted_digit}"
    except Exception as e:
        result = f"Error: {e}"
        image_path = None

    # Randomly choose an animation for each new page load
    animations = ['slideIn', 'scaleIn', 'slideDown']
    animation_class = random.choice(animations)

    return render_template('index.html', result=result, image_path=image_path, animation_class=animation_class, done=False)


@app.route('/upload', methods=['POST'])
def upload():
    # Ensure the upload directory exists
    upload_dir = 'static/uploads'
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    file = request.files['file']
    if file:
        # Save the uploaded file
        file_path = os.path.join(upload_dir, file.filename)
        file.save(file_path)

        # Process the uploaded image
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
        img_resized = cv2.resize(img, (28, 28))  # Resize to 28x28
        img_inverted = np.invert(np.array(img_resized))  # Invert pixel values (if needed)
        img_normalized = img_inverted / 255.0  # Normalize pixel values to range [0, 1]
        img_input = img_normalized.reshape(1, 28, 28, 1)  # Reshape for the model

        # Predict the digit
        prediction = model.predict(img_input)
        predicted_digit = np.argmax(prediction)

        result = f"The uploaded image is predicted as: {predicted_digit}"

        return render_template('index.html', result=result, image_path=file_path, done=True)

    return render_template('index.html', result="No file uploaded.", done=False)


@app.route('/next', methods=['POST'])
def next_image():
    session['image_number'] += 1
    return redirect(url_for('analyze'))


@app.route('/restart', methods=['POST'])
def restart():
    session['image_number'] = 1
    return redirect(url_for('analyze'))


if __name__ == '__main__':
    app.run(debug=True)
