import streamlit as st
import pickle as pkl
import numpy as np

class_list = {'0': 'Nữ', '1': 'Nam'}

st.title('Predict the gender of Vietnamese people based on full name')

input_ec = open('ec_vinames.pkl', 'rb')
encoder = pkl.load(input_ec)

input_md = open('lrc_vinames.pkl', 'rb')
model = pkl.load(input_md)

st.header('Write a Vietnamese full name')
txt = st.text_area('', '')

if txt != '':
  if st.button('Predict'):
    feature_vector = encoder.transform([txt])
    label = str((model.predict(feature_vector))[0])

    st.header('Result')
    st.text(class_list[label])
