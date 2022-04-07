import os
import cv2
import tensorflow as tf
import streamlit as st
from PIL import Image

CATEGORIES = ["NORMAL", "PNEUMONIA"]


def prepare(filepath):
    IMG_SIZE = 64
    st.write(filepath)
    img_array = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    img_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
    return img_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)



model = tf.keras.models.load_model('CNN Pneumonia')
st.title('Hello this model aims at identifying pneumonia by looking at X-rays.')
uploaded_file = st.file_uploader(label="Please upload your X-Ray", type=["JPEG", "JPG", "PNG"])
if uploaded_file is not None:
    path = st.text_input('Add the path to your image')
    image = Image.open(uploaded_file)
    st.image(image, caption='This is your uploaded file')
    st.write("filename:", uploaded_file.name)
    prediction = model.predict([prepare(path)])
    outcome = (CATEGORIES[int(prediction[0][0])])
    st.write('Detection seems to be: ')
    st.write(outcome)
else:
    print()
