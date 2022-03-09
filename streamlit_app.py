import numpy as np
#import datetime
import streamlit as st
import tensorflow as tf

#import
from functions import *

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

if app_mode == breath_abnormalities_detection_page:

    model = retrieve_model("binary_sound")

    # Quick instructions for the user
    st.header('Welcome to our breath abnormality detection app! 🫁')
    st.write(instructions)

    file_uploader = st.file_uploader(label="", type=".wav")

    if file_uploader is not None:
        y, sr = handle_uploaded_audio_file(file_uploader)
        st.text("Have a listen to what the model uses as an input to predict")
        st.audio(create_audio_player(y, sr))

        spectrogram = from_audiofile_to_spectrogram(y)

        choice = st.radio("What do you want to do?",("Convert this audio fragment to a audio wave","Convert the sound wave into a spectrogram","Convert the sound wave into a mel-spectrogram","Get our model prediction"))

        if choice == "Convert this audio fragment to a audio wave":

            st.pyplot(plot_wave(y, sr))

        elif choice == 'Convert the sound wave into a spectrogram':

            st.pyplot(plot_spectrogram(y,sr))

        elif choice == "Convert the sound wave into a mel-spectrogram":
            st.pyplot(plot_melspectrogram(y,sr))

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
            pred = model.predict(spectrogram)
            result_dict = {pred[0][0]:"COPD",
                pred[0][1]:"Healthy",
                pred[0][2]:"URTI",
                pred[0][3]:"Bronchiectasis",
                pred[0][4]:"Bronchiolitis",
                pred[0][5]:"Pneumonia"}

            max = max(pred[0])
            max2 = 0
            for v in pred[0]:
                if(v>max2 and v<max):
                        max2 = v

            if max - max2 >0.2:
                st.text("The pattern that the model detected looks most similar to the pattern of " + result_dict.get(max))
            else:
                st.text("The model cannot map the detected patterns to a disease with high confidence. It is doubting between " + result_dict.get(max) + " and " + result_dict.get(max2))







# today_date = st.date_input('What is the date of the day?', value=datetime.datetime(2012, 10, 6, 12, 10, 20))
# name = st.text_input("What is the name of the patient")
# gender = st.radio("What is the gender of the patient?",["male","female","else"])
# age = st.number_input("What is the age of the patient?")
# height = st.number_input("What's the height of the patient in m?")
# weight = st.number_input("What's the weight of the patient in kg")
# sound = st.file_uploader("Please upload the breathing sound of the patient:")
