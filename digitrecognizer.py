import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# Load the trained model
model = load_model(r'handwritten_digit_rec.h5')

# Pre-process
def preprocess_image(img):
    img = img.convert('L')
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1) 
    return img

# Predict the digit
def predict_digit(img):
    img = preprocess_image(img)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class[0]

#UI
st.title('Handwritten Digit Recognition')
uploaded_file = st.file_uploader("Choose an image", type="png")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    predicted_class = predict_digit(img)
    st.write(f'Predicted Class: {predicted_class}')
