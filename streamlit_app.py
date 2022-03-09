import numpy as np
import datetime
import pandas as pd
from tensorflow.keras.models import load_model
import pydub
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa
from matplotlib import pyplot as plt
import librosa.display
import io
from scipy.io import wavfile



#import preprocessing pipeline
from preprocessing import from_audiofile_to_spectrogram

disease_dict = {0:"COPD",
                1:"Healthy",
                2:"URTI",
                3:"Bronchiectasis",
                4:"Bronchiolitis",
                5:"Pneumonia"}

instructions = """
    The file you select or upload will be sent through the Deep Neural Network in real-time
    and the output will be displayed to the screen.
    """

# Application title & subtitle
'''
# AI Breath-based Disease Prediction app
'''


############################ Sidebar + launching #################################################

breath_abnormalities_detection_page = " abnormalities detection"
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
        model = load_model('saved_models/model_binary_sound_classification_v1.h5',custom_objects={"F1Score": tfa.metrics.F1Score(num_classes=2)})
    else:
        model = load_model('saved_models/resp_model_v4_model17_Tracy.h5')
    return model

def get_binary_sound_prediction(model,spectrogram):
    if model.predict(spectrogram) >= 0.5:
        return st.text("Some abnormalities were detected in this audio fragment. Please consider speaking to a specialist as soon as possible.")
    else:
        return st.text("You are a healthy human being.")

def handle_uploaded_audio_file(uploaded_file):
    a = pydub.AudioSegment.from_wav(uploaded_file)
    samples = a.get_array_of_samples()
    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples.typecode).max
    return fp_arr, 22050

def plot_wave(y, sr):
    fig, ax = plt.subplots()
    img = librosa.display.waveshow(y, sr=sr, x_axis="time", ax=ax)
    plt.title("Waveplot of the uploaded fragment",pad=25)
    plt.ylabel("Amplitude")
    return plt.gcf()

def create_audio_player(audio_data, sample_rate):
    virtualfile = io.BytesIO()
    wavfile.write(virtualfile, rate=sample_rate, data=audio_data)
    return virtualfile

def plot_transformation(y, sr, transformation_name):
    D = librosa.stft(y)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, x_axis="time", y_axis="linear", ax=ax)
    # ax.set(title=transformation_name)
    fig.colorbar(img, ax=ax, format="%+2.f dB")

    return plt.gcf()

if app_mode == breath_abnormalities_detection_page:

    model = retrieve_model("binary_sound")

    # Quick instructions for the user
    st.header('Welcome to our breath abnormality detection app! 🫁')
    st.write(instructions)

    file_uploader = st.file_uploader(label="", type=".wav")


    #if st.button('Ready for some AI magic?',help="Just click on this button - it's gonna be amazing"):

    if file_uploader is not None:
        y, sr = handle_uploaded_audio_file(file_uploader)
        st.text("Have a listen to what the model uses as an input to predict")
        st.audio(create_audio_player(y, sr))

        spectrogram = from_audiofile_to_spectrogram(y)
        spectrogram_show = np.squeeze(spectrogram, axis=-1)
        spectrogram_show = tf.transpose(spectrogram_show)

        choice = st.radio("What do you want to do?",("Convert this audio fragment to a audio wave","Convert the sound wave into a spectrogram","Convert the sound wave into a mel-spectrogram","Get our model prediction"))

        if choice == "Convert this audio fragment to a audio wave":

            st.pyplot(plot_wave(y, sr))

        elif choice == 'Convert the sound wave into a spectrogram':

            st.pyplot(plt.imshow(spectrogram_show))

        elif choice == "Convert the sound wave into a mel-spectrogram":
            st.pyplot(plot_transformation(y,sr, "melspectrogram"))

        elif choice == "Get our model prediction":
            get_binary_sound_prediction(model,spectrogram)






if app_mode == disease_classification_page:

    model = retrieve_model("disease_classification")

    # Quick instructions for the user
    st.header('Welcome to our disease classification app! 🫁')
    st.write(instructions)

    file_uploader = st.file_uploader(label="", type=".wav")

    if st.button('Ready for some AI magic?',help="Just click on this button - it's gonna be amazing"):

        if file_uploader is not None:
            y, sr = handle_uploaded_audio_file(file_uploader)
            spectrogram = from_audiofile_to_spectrogram(y)
            st.text(model.predict(spectrogram))
            res = np.argmax(model.predict(spectrogram))
            #result
            st.text("The pattern that the model detected looks most similar to the pattern of " + disease_dict.get(res))











# today_date = st.date_input('What is the date of the day?', value=datetime.datetime(2012, 10, 6, 12, 10, 20))
# name = st.text_input("What is the name of the patient")
# gender = st.radio("What is the gender of the patient?",["male","female","else"])
# age = st.number_input("What is the age of the patient?")
# height = st.number_input("What's the height of the patient in m?")
# weight = st.number_input("What's the weight of the patient in kg")
# sound = st.file_uploader("Please upload the breathing sound of the patient:")
