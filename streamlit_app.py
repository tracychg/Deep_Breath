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

#breath_abnormalities_detection_page = "Breath abnormalities detection"
disease_classification_page = "Disease classification"

app_mode = st.sidebar.selectbox(
    "Choose the app mode",
    [
        #breath_abnormalities_detection_page,
        disease_classification_page
    ],
)
st.subheader(app_mode)

# if app_mode == breath_abnormalities_detection_page:

#     model = retrieve_model("binary_sound")

#     # Quick instructions for the user
#     st.header('Welcome to our breath abnormality detection app! 🫁')
#     st.write(instructions)

#     file_uploader = st.file_uploader(label="", type=".wav")

#     if file_uploader is not None:
#         y, sr = handle_uploaded_audio_file(file_uploader)
#         st.text("Have a listen to what the model uses as an input to predict")
#         st.audio(create_audio_player(y, sr))

#         spectrogram = from_audiofile_to_spectrogram(y)

#         choice = st.radio("What do you want to do?",("Convert this audio fragment to a audio wave","Convert the sound wave into a spectrogram","Convert the sound wave into a mel-spectrogram","Get our model prediction"))

#         if choice == "Convert this audio fragment to a audio wave":

#             st.pyplot(plot_wave(y, sr))

#         elif choice == 'Convert the sound wave into a spectrogram':

#             st.pyplot(plot_spectrogram(y,sr))

#         elif choice == "Convert the sound wave into a mel-spectrogram":
#             st.pyplot(plot_melspectrogram(y,sr))

#         elif choice == "Get our model prediction":
#             get_binary_sound_prediction(model,spectrogram)



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
            result_dict = {pred[0][0]:'''**COPD** \n

Chronic obstructive pulmonary disease (COPD) is the name for a group of lung conditions that cause breathing difficulties.

It includes:

emphysema – damage to the air sacs in the lungs
chronic bronchitis – long-term inflammation of the airways
COPD is a common condition that mainly affects middle-aged or older adults who smoke. Many people do not realise they have it.

The breathing problems tend to get gradually worse over time and can limit your normal activities, although treatment can help keep the condition under control.'''
                           ,
                pred[0][1]:"**Healthy**",
                pred[0][2]:'''**URTI** \n

A URTI is a viral infection that can affect the nose, throat and sinuses.
                            Upper respiratory tract infections are extremely common and there are many different viruses which cause them.
                            Children are particularly susceptible to these and it is normal for
                            children under five years old to have as many as twelve URTIs in one year. The frequency of these illnesses
                            reduces as children get older – for example a child in primary school may get around six URTIs per year.''',
                pred[0][3]:'''**Bronchiectasis**

Bronchiectasis is a long-term condition where the airways of the lungs become widened, leading to a build-up of excess mucus that can make the lungs more vulnerable to infection.

The most common symptoms of bronchiectasis include:

A persistent cough that usually brings up phlegm (sputum)
shortness of breath
The severity of symptoms can vary widely. Some people have only a few symptoms that do not appear often, while others have wide-ranging daily symptoms.

The symptoms tend to get worse if you develop an infection in your lungs.''',
                pred[0][4]:'''**Bronchiolitis**

Bronchiolitis is a common lower respiratory tract infection that affects babies and young children under 2 years old.

Most cases are mild and clear up within 2 to 3 weeks without the need for treatment, although some children have severe symptoms and need hospital treatment.

The early symptoms of bronchiolitis are similar to those of a common cold, such as a runny nose and a cough.''',
                pred[0][5]:'''**Pneumonia**

Pneumonia is swelling (inflammation) of the tissue in one or both lungs. It's usually caused by a bacterial infection.
                            It can also be caused by a virus, such as coronavirus (COVID-19).'''}

            max = max(pred[0])
            max2 = 0
            for v in pred[0]:
                if(v>max2 and v<max):
                        max2 = v

            if max - max2 >0.3:
                st.markdown(f"The pattern that the model detected looks most similar to the pattern of {result_dict.get(max)}")
                st.markdown(
                f"""<a href="https://www.england.nhs.uk/ourwork/clinical-policy/respiratory-disease//">Read more</a>""", unsafe_allow_html=True,
)
            else:
                st.markdown("The model cannot map the detected patterns to a disease with high confidence. It is doubting between " + result_dict.get(max) + " and " + result_dict.get(max2))
                st.markdown(
                f"""<a href="https://www.england.nhs.uk/ourwork/clinical-policy/respiratory-disease//">Read more</a>""", unsafe_allow_html=True,
)
