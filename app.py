import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("model_dropout.keras")

# CIFAR-10 class names
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Preprocess uploaded image
def preprocess_image(image):
    image = image.resize((32, 32))
    image = np.array(image).astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.set_page_config(page_title="CIFAR-10 CNN Classifier", layout="centered")
st.title("🧠 CIFAR-10 Image Classifier")
st.write("Upload an image resembling one of the CIFAR-10 categories to see its predicted class and confidence.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=False)
    
    # Preprocess and predict
    processed = preprocess_image(image)
    prediction = model.predict(processed)
    predicted_index = np.argmax(prediction[0])
    confidence = prediction[0][predicted_index] * 100

    # Display results
    st.markdown(f"### 🏷️ Predicted Label: `{class_names[predicted_index]}`")
    st.markdown(f"### 📊 Confidence: `{confidence:.2f}%`")
