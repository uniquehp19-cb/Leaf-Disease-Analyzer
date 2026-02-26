import streamlit as st
from PIL import Image
import numpy as np
import random
import datetime

# ----------------------
# Page Config
# ----------------------
st.set_page_config(
    page_title="Leaf Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# ----------------------
# Sidebar
# ----------------------
st.sidebar.title("🌿 Leaf Disease Detector")
st.sidebar.write("Mini Project Demo App")
st.sidebar.write("Upload a leaf image to detect disease.")
st.sidebar.markdown("---")
st.sidebar.info("This is a demo version using random predictions.")

# ----------------------
# App Title
# ----------------------
st.title("🌿 Leaf Disease Detection System")
st.write("Upload a leaf image or take a picture with your webcam.")

# ----------------------
# Image Input
# ----------------------
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg","jpeg","png"])
cam_file = st.camera_input("Or take a picture of the leaf")

input_image = uploaded_file or cam_file

# ----------------------
# Prediction Section
# ----------------------
if input_image is not None:
    img = Image.open(input_image)
    st.image(img, caption="Input Image", use_column_width=True)

    with st.spinner("Analyzing leaf..."):
        st.sleep = 1

    # Demo prediction
    classes = ["Apple Scab", "Healthy", "Leaf Blight"]
    predicted_class = random.choice(classes)
    confidence = random.uniform(80, 99)

    st.markdown(f"## 🧪 Prediction Result")
    st.success(f"**{predicted_class}**")
    st.progress(int(confidence))
    st.write(f"Confidence: **{confidence:.2f}%**")

    # Disease Info
    st.markdown("### 📋 Disease Information")

    if predicted_class == "Apple Scab":
        st.warning("Symptoms: Brown or black spots on leaves.")
        st.write("Prevention: Reduce moisture and remove infected leaves.")
    elif predicted_class == "Leaf Blight":
        st.warning("Symptoms: Yellowing and browning of leaf edges.")
        st.write("Treatment: Remove infected leaves and apply fungicide.")
    elif predicted_class == "Healthy":
        st.balloons()
        st.success("Leaf looks healthy! ✅ No disease detected.")

    # Save prediction history
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append({
        "Time": datetime.datetime.now().strftime("%H:%M:%S"),
        "Prediction": predicted_class,
        "Confidence": f"{confidence:.2f}%"
    })

    # Show history
    st.markdown("### 📜 Prediction History")
    st.table(st.session_state.history)

    # Download result
    result_text = f"""
    Leaf Disease Detection Result
    ------------------------------
    Prediction: {predicted_class}
    Confidence: {confidence:.2f}%
    Time: {datetime.datetime.now()}
    """

    st.download_button(
        label="📥 Download Result",
        data=result_text,
        file_name="leaf_result.txt",
        mime="text/plain"
    )

else:
    st.info("Please upload an image or take a photo to get a prediction.")