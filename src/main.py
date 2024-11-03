import tensorflow as tf

# Downloading the mini_speech_commands dataset from the external URL
data = tf.keras.utils.get_file(
  'mini_speech_commands.zip',
  origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
  extract=True,
  cache_dir='.', cache_subdir='data')
