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
    data = [np.zeros((224,224,3)), np.zeros((224,224,3)), np.zeros((10,224,3))]
    np.array(data)
# long output omitted
    new_data = np.array(data)
    new_data.shape
    new_data.dtype
    new_data[0].shape
    (224, 224, 3)
    new_data[1].shape
    (224, 224, 3)
    new_data[2].shape
    (10, 224, 3)
    image = img
    #image sizing
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #Next, turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    new_data[0] = normalized_image_array

    # run it
    prediction = model.predict(new_data)
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
