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
st.title('Hello! This is an application that aims at identifying pneumonia by looking at X-rays. Please upload the necessary files so that we may be able to detect pneumonia accurately.')
st.set_page_config(page_title='X-Classifier - Pneumonia Detector', page_icon='❎')
info = st.checkbox("Do you want to know more about pneumonia?")
if info:
        st.write('Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus (purulent material), causing cough with phlegm or pus, fever, chills, and difficulty breathing. A variety of organisms, including bacteria, viruses and fungi, can cause pneumonia.')
        st.write("Click this link to know more: https://www.who.int/news-room/fact-sheets/detail/pneumonia")

model = tf.keras.models.load_model('CNN Pneumonia-2')

uploaded_file = st.file_uploader(label="Please upload your X-Ray", type=["JPEG", "JPG", "PNG"])
if uploaded_file is not None:
    try: 
        image = Image.open(uploaded_file)
        st.image(image, caption='This is your uploaded file')
        prediction = model.predict([prepare(image)])
        outcome = (CATEGORIES[int(prediction[0][0])])
        with st.spinner('Classifying image...'):
            time.sleep(5)
        if outcome == 'NORMAL':
            st.success('NO ISSUE DETECTED. Lungs seem to be normal')
        elif outcome == 'PNEUMONIA':
            st.warning('PNEUMONIA DETECTED. Please consult a medical expert')

    except Exception as e:
        time.sleep(10)
        st.write('There was an error. Please try again later or refresh the page.'
                 ' Make sure to enter the data accurately and put the path WITHOUT quotes')
else:
    print()



