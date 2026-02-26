import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from io import BytesIO

st.title("🌿 Leaf Disease Detection with Disease Names & CSV Download")

# Upload multiple leaf images
uploaded_files = st.file_uploader(
    "Upload leaf images (you can select multiple)", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

if uploaded_files:
    results = []  # Store results for table and CSV

    for uploaded_file in uploaded_files:
        # Open image
        image = Image.open(uploaded_file)
        image_cv = np.array(image)
        if image_cv.shape[2] == 4:  # Handle PNG alpha channel
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGBA2RGB)

        # Resize
        image_cv = cv2.resize(image_cv, (400, 300))

        # Convert to HSV
        hsv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2HSV)

        # Define green range
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([90, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # Diseased mask (everything NOT green)
        mask_diseased = cv2.bitwise_not(mask_green)

        # Highlight diseased areas in red
        highlighted = image_cv.copy()
        highlighted[mask_diseased > 0] = [255, 0, 0]

        # Calculate green area percentage
        green_pixels = np.sum(mask_green == 255)
        total_pixels = mask_green.size
        green_ratio = green_pixels / total_pixels

        # Rule-based disease name
        if green_ratio > 0.5:
            result = "Healthy Leaf"
            disease_name = "No Disease"
        elif green_ratio > 0.3:
            result = "Diseased Leaf"
            disease_name = "Early Blight"
        else:
            result = "Diseased Leaf"
            disease_name = "Late Blight"

        # Show original + highlighted images
        st.image(image, caption=f"Original: {uploaded_file.name}", use_column_width=True)
        st.image(highlighted, caption=f"Diseased Highlighted: {uploaded_file.name}", use_column_width=True)

        # Save results
        results.append({
            "Image": uploaded_file.name,
            "Green %": f"{green_ratio*100:.2f}%",
            "Result": result,
            "Disease Name": disease_name
        })

    # Display summary table
    st.write("### 📊 Summary of All Uploaded Leaves")
    df = pd.DataFrame(results)
    st.table(df)

    # Download CSV
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv_buffer.getvalue(),
        file_name="leaf_disease_results.csv",
        mime="text/csv"
    )