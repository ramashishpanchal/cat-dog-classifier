import os
os.environ['STREAMLIT_CONFIG_DIR'] = os.path.expanduser('~/.streamlit')

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model

st.title("🐶🐱 Cat vs Dog Classifier")
st.text("Checking for model file...")

if os.path.exists("model.keras"):
    st.success("✅ model.keras found!")
    model = load_model("cat_dog_detection.keras")
    st.success("✅ Model loaded successfully.")
else:
    st.error("❌ model.keras not found.")


# Upload image
uploaded_file = st.file_uploader("Upload a cat or dog image", type=["jpg", "jpeg", "png"])

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
