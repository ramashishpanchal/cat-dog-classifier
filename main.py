import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model = load_model('E:\Ashish Projects\Project 5\cat_dog_detection.keras')

st.title("🐶🐱 Cat vs Dog Classifier")

# Upload image
uploaded_file = st.file_uploader("pexels-kmerriman-20787.jpg", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    img = Image.open(uploaded_file)
    st.image(img, use_container_width=True)


    # Preprocess image
    img = img.resize((224,224))  # use your model's input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # normalize if model trained this way

    # Predict
    prediction = model.predict(img_array)
    predicted_class = "Dog 🐶" if prediction[0][0] > 0.5 else "Cat 🐱"

    st.subheader("Prediction:")
    st.success(predicted_class)
