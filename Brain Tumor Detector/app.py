import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the trained model
model = load_model("brain_tumor_classifier.keras")

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((128, 128))  # Resize to match model input
    image = np.array(image) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    print("Processed image shape:", image.shape)  # Debugging
    return image


print("Expected model input shape:", model.input_shape)

# Streamlit UI
st.title("Brain Tumor Detection")
st.write("Upload an MRI scan to check for brain tumor.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    image = preprocess_image(image)
    prediction = model.predict(image)

    # Display result
    st.write("### Prediction:")
    if prediction[0][0] > 0.5:
        st.error("Tumor Detected ❌")
    else:
        st.success("No Tumor Detected ✅")

        
#C:\Users\lenovo>cd "C:\Users\lenovo\OneDrive\Documents\Brain Tumor Detector"

#C:\Users\lenovo\OneDrive\Documents\Brain Tumor Detector>streamlit run app.py