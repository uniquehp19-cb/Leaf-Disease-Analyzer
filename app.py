import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import sqlite3
import hashlib
import time
from datetime import datetime
import pandas as pd
import cv2

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Leaf Disease Detector", layout="centered")

st.markdown("""
    <style>
        section[data-testid="stSidebar"] {display: none;}
        body {background-color: #f4f9f4;}
        .stButton>button {border-radius: 8px;}
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# DATABASE SETUP
# ----------------------------
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS history (
    username TEXT,
    disease TEXT,
    confidence REAL,
    date TEXT
)
""")

conn.commit()

# ----------------------------
# PASSWORD HASH
# ----------------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ----------------------------
# SESSION STATE
# ----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

# ----------------------------
# REGISTER
# ----------------------------
def register():
    st.title("Create Account")
    new_user = st.text_input("Choose Username")
    new_pass = st.text_input("Choose Password", type="password")
    if st.button("Register"):
        if new_user == "" or new_pass == "":
            st.warning("Please fill all fields")
        else:
            hashed = hash_password(new_pass)
            try:
                c.execute("INSERT INTO users VALUES (?, ?)", (new_user, hashed))
                conn.commit()
                st.success("Account created successfully! Please login.")
            except:
                st.error("Username already exists")

# ----------------------------
# LOGIN
# ----------------------------
def login():
    st.title("Leaf Disease Detector Login")
    user = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        hashed = hash_password(password)
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (user, hashed))
        result = c.fetchone()
        if result:
            st.session_state.logged_in = True
            st.session_state.username = user
            st.rerun()
        else:
            st.error("Invalid username or password")

# ----------------------------
# AUTH PAGE
# ----------------------------
if not st.session_state.logged_in:
    choice = st.radio("Select Option", ["Login", "Register"])
    if choice == "Login":
        login()
    else:
        register()
    st.stop()

# ----------------------------
# LOAD MODEL
# ----------------------------
model = tf.keras.models.load_model("leaf_model.keras")
with open("class_names.json", "r") as f:
    class_names = json.load(f)

IMG_SIZE = 128   # MUST match your training size

# ----------------------------
# MAIN UI
# ----------------------------
st.markdown("<h1 style='text-align:center;'>🌿 Leaf Disease Detector</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center;'>Logged in as <b>{st.session_state.username}</b></p>", unsafe_allow_html=True)
st.divider()

# ----------------------------
# INPUT METHOD
# ----------------------------
method = st.radio("Select Input Method", ["Upload Image", "Use Webcam"], horizontal=True)
input_image = None
if method == "Upload Image":
    uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        input_image = uploaded_file
elif method == "Use Webcam":
    camera_file = st.camera_input("Take a Photo")
    if camera_file:
        input_image = camera_file

# ----------------------------
# PREDICTION
# ----------------------------
if input_image is not None:
    image = Image.open(input_image).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, width="stretch")

    # -------- LEAF SEGMENTATION --------
    img_np = np.array(image)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    leaf_only = cv2.bitwise_and(img_np, img_np, mask=mask)

    # -------- RESIZE & NORMALIZE --------
    img = Image.fromarray(leaf_only).resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, 0)

    # -------- PREDICT --------
    with st.spinner("Analyzing Leaf..."):
        time.sleep(1)
        predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index].replace("___", " ").replace("_", " ")
    confidence = float(np.max(predictions[0])) * 100

    with col2:
        st.subheader("Prediction Result")
        st.success(predicted_class)
        st.write(f"Confidence: {confidence:.2f}%")

    # Save History
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO history VALUES (?, ?, ?, ?)",
              (st.session_state.username, predicted_class, confidence, current_time))
    conn.commit()

# ----------------------------
# HISTORY
# ----------------------------
st.divider()
st.subheader("Last 5 Predictions")
c.execute("""
SELECT disease, confidence, date 
FROM history 
WHERE username=? 
ORDER BY date DESC 
LIMIT 5
""", (st.session_state.username,))
records = c.fetchall()
if records:
    df = pd.DataFrame(records, columns=["Disease", "Confidence (%)", "Date"])
    st.dataframe(df, width="stretch")
else:
    st.write("No prediction history yet.")

# ----------------------------
# LOGOUT
# ----------------------------
st.divider()
if st.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()