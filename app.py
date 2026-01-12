

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("mnist_cnn_model.keras")

st.title("Handwritten Digit Classifier")
st.write("Upload an image of a handwritten digit (0–9)")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # 1️⃣ Load image & convert to grayscale
    image = Image.open(uploaded_file).convert("L")

    # 2️⃣ Resize to MNIST size
    image = image.resize((28, 28))

    # 3️⃣ Convert to NumPy array
    img_array = np.array(image)

    # 4️⃣ INVERT COLORS (CRITICAL for MNIST)
    img_array = np.invert(img_array)

    # 5️⃣ Normalize (0–1)
    img_array = img_array / 255.0

    # 6️⃣ Reshape for CNN
    img_array = img_array.reshape(1, 28, 28, 1)

    # 7️⃣ Predict
    prediction = model.predict(img_array)
    digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    # 8️⃣ Display images
    st.image(image, caption="Uploaded Image (Original)", width=150)
    st.image(img_array.reshape(28, 28), caption="Model Input (What the model sees)", width=150)

    # 9️⃣ Show result
    st.success(f"Predicted Digit: {digit} ({confidence:.2f}% confidence)")
