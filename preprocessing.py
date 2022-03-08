import tensorflow as tf

def from_audiofile_to_spectrogram(waveform):
  #audio_binary = tf.io.read_file(file_path)
  #audio, _ = tf.audio.decode_wav(contents=audio_binary)
  #waveform = tf.squeeze(audio, axis=-1)
  waveform = tf.cast(waveform, dtype=tf.float32)
  spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
  spectrogram = tf.abs(spectrogram)
  spectrogram = spectrogram[tf.newaxis,..., tf.newaxis]
  return spectrogram
