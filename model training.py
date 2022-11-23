import numpy as np
import tensorflow as tf
import os
from scipy.io import wavfile
import matplotlib.pyplot as plt

from wavenet_model import WaveNet_model


## Data packaging
path = "../smaller_set/audio/"

samplerate, data = wavfile.read(os.listdir(path)[0])

# audio files have been resampled in 8 bits and 16kHz frequency

audios = np.ones((100,samplerate*10))*128
for k in range(len(os.listdir())):
    filename = os.listdir()[k]
    if filename[-4:]=='.wav':
        samplerate,data = wavfile.read(os.listdir()[k])
        if len(data)<=samplerate*10:
            audios[k,:len(data)] = data
        else:
            audios[k,:]=data[:samplerate*10]

plt.figure()
plt.plot([k/samplerate for k in range(len(audios[10,:]))],audios[10,:])
plt.xlabel("time (s)")
plt.ylabel("bits (/256)")
plt.show()


nb_train_audios = int(0.8*len(os.listdir()))
train_audios = audios[:nb_train_audios]/255.0
test_audios = audios[nb_train_audios:]/255.0

## Train model

WNmodel = WaveNet_model(
    nb_layers = 10,
    nb_filters = 256,
    audio_length=samplerate*10 # length of audio files is 10s max
).model

WNmodel.summary()

sgd = tf.keras.optimizers.Adam()  #SGD(lr=0.001, momentum=0.9) 
WNmodel.compile(sgd, loss='categorical_crossentropy', metrics=['accuracy'])

