import os
import cv2
import tensorflow as tf
import streamlit as st
from PIL import Image, ImageOps
import time
import numpy as np
CATEGORIES = ["NORMAL", "PNEUMONIA"]

def prepare(img):
    data = []
    data = np.array(data)
    image = img
    #image sizing
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #Next, turn the image into a numpy array
    image_array = np.array(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    np.append(data,normalized_image_array)

    # run it
    prediction = model.predict(data)
    return np.argmax(prediction) 



model = tf.keras.models.load_model("CNN Pneumonia-2")
st.title('Hello this model aims at identifying pneumonia by looking at X-rays.')
uploaded_file = st.file_uploader(label="Please upload your X-Ray", type=["JPEG", "JPG", "PNG"])
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='This is your uploaded file')
        predicted_val = prepare(image)
        if predicted_val == 0:
            st.title('NO ISSUE DETECTED.')
            st.write('Lungs seem to be healthy.')
        else:
            st.title('PNEUMONIA DETECTED.')
            st.write('Please consult a medical expert.')
    
    except Exception as e:
        time.sleep(5)
        st.write('There was an error. Please try again later or refresh the page.'
                ' Make sure to enter the data accurately and put the path WITHOUT quotes')
        st.write(e)
else:
    print()
