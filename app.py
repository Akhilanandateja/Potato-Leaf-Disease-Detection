import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

# Load the saved model
model = tf.keras.models.load_model('leaf_model.keras')

# Define image preprocessing function
def preprocess_image(image):
    img = image.resize((256, 256))  # Match the input size from the notebook
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize as done in the notebook
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit app
st.title("Potato Leaf Disease Classifier")
st.write("Upload an image of a potato leaf to classify it as Healthy or Late Blight.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess and predict
    with st.spinner('Classifying...'):
        img_array = preprocess_image(image)
        prediction = model.predict(img_array)
        probability = prediction[0][0]
        
        # Interpret the prediction (assuming binary classification: 0=Healthy, 1=Late Blight)
        if probability > 0.5:
            label = "Late Blight"
            prob = probability
        else:
            label = "Healthy"
            prob = 1 - probability
        
        st.success(f"Prediction: **{label}** (Confidence: {prob:.2%})")