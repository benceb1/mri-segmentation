import numpy as np
import streamlit as st
from tensorflow import keras
import cv2


st.title("MRI segmentation tumor recognition")

uploaded_file = st.file_uploader("Choose an img file", accept_multiple_files=False)

model = keras.models.load_model('model.h5')

if uploaded_file is not None:
  bytes_data = uploaded_file.read()

  col1, col2 = st.columns(2)
  with col1:
    st.header('Uploaded Image')
    st.image(bytes_data, caption='Uploaded Image',)

  img_array = np.fromstring(bytes_data, np.uint8)
  cv_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
  cv_img = cv2.resize(cv_img,(128,128)) 
  cv_img = cv_img/255.0
  cv_img = cv_img.astype(np.float32)

  img_np_array = np.array([cv_img])
  prediction = model.predict(img_np_array)
  with col2:
    st.header('Prediction Image')
    st.image(prediction[0], caption='Prediction Image',)
