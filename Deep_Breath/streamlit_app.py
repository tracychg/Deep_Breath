from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import datetime
import requests
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.models import load_model
import pydub
#import tensorflow_addons as tfa

#Import for streamlit
import streamlit as st

#Import for Deep learning model
import tensorflow as tf
from tensorflow.keras.models import load_model
from preprocessing import from_audiofile_to_spectrogram

#import preprocessing pipeline
from preprocessing import from_audiofile_to_spectrogram

# Application title & subtitle
'''
# Breathing abnormalities
'''


############################ Sidebar + launching #################################################

breath_abnormalities_detection_page = "Breath abnormalities detection"
disease_classification_page = "Disease classification"

app_mode = st.sidebar.selectbox(
    "Choose the app mode",
    [
        breath_abnormalities_detection_page,
        disease_classification_page
    ],
)
st.subheader(app_mode)


#load the binary sound model
def retrieve_model():
    model = load_model('model_binary_sound_classification_v1.h5')
    return model

def get_binary_sound_prediction(model,spectrogram):
    if model.predict(spectrogram) >= 0.5:
        return "This person displays abnormalities in their breath cycle."
    else:
        return "This person does not display abnormalities in their breath cycle."

def handle_uploaded_audio_file(uploaded_file):
    a = pydub.AudioSegment.from_wav(uploaded_file)

    st.write(a.sample_width)

    samples = a.get_array_of_samples()
    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples.typecode).max
    st.write(fp_arr.shape)

    return fp_arr, 22050

model = retrieve_model()


if app_mode == breath_abnormalities_detection_page:

    # Quick instructions for the user
    st.header('Welcome to our breath abnormality detection app! 🫁')
    instructions = """
        Either upload your own record or select from the sidebar to get a prerecorded file.

        The file you select or upload will be sent through the Deep Neural Network in real-time
        and the output will be displayed to the screen.
        """
    st.write(instructions)


    file_uploader = st.sidebar.file_uploader(label="", type=".wav")

    if file_uploader is not None:
        st.write(file_uploader)
        y, sr = handle_uploaded_audio_file(file_uploader)
        spectrogram = from_audiofile_to_spectrogram(y)
        st.text(model.predict(spectrogram))
        get_binary_sound_prediction(model,spectrogram)
    #result
        print(y)




# #ask user to upload a sound
# sound = st.file_uploader("Please upload the breathing sound of the patient:",type='.wav')
# if sound is not None:
#     st.audio(sound, format='audio/wav')
#     bytesdata = tf.io.read_file(sound)

#     #with open("pip.ogg", "wb") as file:
#     #file.write(bytesdata)











# today_date = st.date_input('What is the date of the day?', value=datetime.datetime(2012, 10, 6, 12, 10, 20))
# name = st.text_input("What is the name of the patient")
# gender = st.radio("What is the gender of the patient?",["male","female","else"])
# age = st.number_input("What is the age of the patient?")
# height = st.number_input("What's the height of the patient in m?")
# weight = st.number_input("What's the weight of the patient in kg")
# sound = st.file_uploader("Please upload the breathing sound of the patient:")
