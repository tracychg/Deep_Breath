import tensorflow as tf
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




def from_audiofile_to_spectrogram(waveform):
  #audio_binary = tf.io.read_file(file_path)
  #audio, _ = tf.audio.decode_wav(contents=audio_binary)
  #waveform = tf.squeeze(audio, axis=-1)
  waveform = tf.cast(waveform, dtype=tf.float32)
  spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
  spectrogram = tf.abs(spectrogram)
  spectrogram = spectrogram[tf.newaxis,..., tf.newaxis]
  return spectrogram

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

def plot_spectrogram(y, sr):
    D = librosa.stft(y)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(D, x_axis="time", y_axis="linear", ax=ax)
    ax.set_title("Convert audio file to spectrogram",pad=20)
    ax.set_ylim(0,200)
    fig.colorbar(img, ax=ax, format="%+2.f")
    return plt.gcf()


def plot_melspectrogram(y, sr):
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, x_axis="time", y_axis="linear", ax=ax)
    ax.set_title("Convert audio file to mel-spectrogram",pad=20)
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    return plt.gcf()
