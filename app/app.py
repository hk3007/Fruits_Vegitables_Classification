import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO

# Streamlit Header
st.header('🍎🥦 Image Classification Model: Fruits & Vegetables 🥕🍍')

# Load the trained model with error handling
try:
    model = load_model(r'D:\Programming\Fruits AND Vegetables classification\Image_classify.keras')
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# Define class labels
data_cat = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 
    'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 
    'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 
    'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 
    'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 
    'turnip', 'watermelon'
]

# Image input options
st.subheader("📸 Upload an Image or Enter a URL")

# File uploader for local images
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

# Text input for URL
image_url = st.text_input("Or enter an image URL")

# Image preprocessing function
def preprocess_image(image):
    """Preprocesses an image for model prediction."""
    try:
        img_height, img_width = 180, 180  # Model's expected input size
        image = image.resize((img_width, img_height))  # Resize image
        img_arr = tf.keras.utils.img_to_array(image)  # Convert to array
        img_arr = img_arr  # Normalize pixel values
        img_bat = np.expand_dims(img_arr, axis=0)  # Expand dimensions
        return img_bat
    except Exception as e:
        st.error(f"❌ Image preprocessing failed: {e}")
        return None

# Load image from either upload or URL
image = None
if uploaded_file:
    try:
        image = Image.open(uploaded_file)
    except UnidentifiedImageError:
        st.error("❌ Uploaded file is not a valid image. Please upload a valid JPG, JPEG, or PNG.")

elif image_url:
    try:
        response = requests.get(image_url, timeout=5)
        response.raise_for_status()  # Raises an error for bad responses (4xx, 5xx)
        image = Image.open(BytesIO(response.content))
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Unable to fetch image from URL: {e}")
    except UnidentifiedImageError:
        st.error("❌ The URL does not point to a valid image file.")

# Display and classify the image if loaded
if image:
    st.image(image, caption="Uploaded Image", width=200)

    # Preprocess image
    img_bat = preprocess_image(image)
    
    if img_bat is not None:
        try:
            # Make prediction
            predict = model.predict(img_bat)
            score = tf.nn.softmax(predict[0])  # Extract softmax probabilities

            # Display prediction results
            predicted_class = np.argmax(score)
            confidence = np.max(score) * 100

            st.write(f'✅ **Detected:** {data_cat[predicted_class]}')
            st.write(f'📊 **Confidence:** {confidence:.2f}%')

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
