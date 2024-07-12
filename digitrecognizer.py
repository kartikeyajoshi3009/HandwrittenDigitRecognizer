import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# Load the trained model
model = load_model(r'C:\Users\karti\OneDrive\Desktop\mini project\handwritten_digit_rec.h5')

# Function to preprocess the input image
def preprocess_image(img):
    img = img.convert('L')  # Convert to grayscale
    img = ImageOps.invert(img)  # Invert colors: white background to black
    img = img.resize((28, 28))  # Resize to 28x28 pixels
    img = np.array(img)  # Convert to array
    img = img / 255.0  # Normalize to [0, 1] range
    img = np.expand_dims(img, axis=0)  # Expand dimensions to match model input
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    return img

# Function to predict the digit
def predict_digit(img):
    img = preprocess_image(img)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class[0]

# Streamlit UI
st.title('Handwritten Digit Recognition')

uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    # Load and display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Predict the digit
    predicted_class = predict_digit(img)
    st.write(f'Predicted Class: {predicted_class}')
