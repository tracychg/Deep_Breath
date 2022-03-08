import numpy as np
import datetime
import pandas as pd
from tensorflow.keras.models import load_model
import pydub
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

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
def retrieve_model(model_selection):
    if model_selection == "binary_sound":
        model = load_model('Deep_Breath/model_binary_sound_classification_v1.h5')
    else:
        model = load_model('Deep_Breath/resp_model_v4_model17_Tracy.h5')
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



if app_mode == breath_abnormalities_detection_page:

    model = retrieve_model("binary_sound")

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



if app_mode == disease_classification_page:

    model = retrieve_model("disease_classification")

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











# today_date = st.date_input('What is the date of the day?', value=datetime.datetime(2012, 10, 6, 12, 10, 20))
# name = st.text_input("What is the name of the patient")
# gender = st.radio("What is the gender of the patient?",["male","female","else"])
# age = st.number_input("What is the age of the patient?")
# height = st.number_input("What's the height of the patient in m?")
# weight = st.number_input("What's the weight of the patient in kg")
# sound = st.file_uploader("Please upload the breathing sound of the patient:")
