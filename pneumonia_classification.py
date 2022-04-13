import cv2
import tensorflow as tf
import streamlit as st
from PIL import Image
import time
import numpy as np

CATEGORIES = ["NORMAL", "PNEUMONIA"]


def prepare(file):
    IMG_SIZE = 64
    img_array = np.array(file)
    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return img_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

st.set_page_config(page_title='X-Classifier - Pneumonia Detector', page_icon='🩺')
model = tf.keras.models.load_model('CNN Pneumonia-2')
st.title('Hello this model aims at identifying pneumonia by looking at X-rays.')
uploaded_file = st.file_uploader(label="Please upload your X-Ray", type=["JPEG", "JPG", "PNG"])
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='This is your uploaded file')
        prediction = model.predict([prepare(image)])
        outcome = (CATEGORIES[int(prediction[0][0])])
        if outcome == 'NORMAL':
            st.write('NO ISSUE DETECTED. Lungs seem to be normal')
        elif outcome == 'PNEUMONIA':
            st.write('PNEUMONIA DETECTED. Please consult a medical expert')

    except Exception as e:
        time.sleep(10)
        st.write('There was an error. Please try again later or refresh the page.'
                 ' Make sure to enter the data accurately and put the path WITHOUT quotes')
else:
    print()
