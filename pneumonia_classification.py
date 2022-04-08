import os
import cv2
import tensorflow as tf
import streamlit as st
from PIL import Image, ImageOps
import time
import numpy as np
CATEGORIES = ["NORMAL", "PNEUMONIA"]

def prepare(img):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    st.write(image.shape)
    #Next, turn the image into a numpy array
    image_array = np.asarray(image)

    # Load the image into the array
    data[0] = image_array

    # run it
    prediction = model.predict(data)
    return np.argmax(prediction)



model = tf.keras.models.load_model("CNN Pneumonia-2")
st.title('Hello this model aims at identifying pneumonia by looking at X-rays.')
uploaded_file = st.file_uploader(label="Please upload your X-Ray", type=["JPEG", "JPG", "PNG"])
if uploaded_file is not None:
    try:
        path = st.text_input('Add the path to your image without quotes')
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
