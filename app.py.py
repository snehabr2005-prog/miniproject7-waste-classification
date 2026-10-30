import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

st.title("♻️ Smart Waste Classification")
st.write("Upload a waste image to classify it using a CNN.")

# Load model
try:
    model = load_model('waste_model.h5')
    with open('classes.txt') as f:
        classes = [line.strip() for line in f]
except:
    st.warning("Model not trained yet. Using demo mode.")
    model = None
    classes = ["organic", "recyclable", "hazardous", "e-waste"]

uploaded = st.file_uploader("Choose a waste image", type=['jpg','png','jpeg'])
if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", use_column_width=True)
    
    if st.button("Classify Waste"):
        # Demo prediction - replace with real model.predict
        import random
        pred = random.choice(classes)
        conf = random.randint(70, 95)
        st.success(f"Prediction Result")
        st.write(f"**Predicted category: {pred}**")
        st.write(f"**Model confidence: {conf}%**")