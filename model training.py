import numpy as np
import tensorflow as tf
import os
from scipy.io import wavfile
import matplotlib.pyplot as plt
from preprocessing import quantize_data, mu_law_encoding, mu_law_expansion
from modelV2 import WavenetModel as WaveNet_modelV2

## Data packaging
path = "../smaller_set/audio/"
files = [path+i for i in os.listdir(path)]
f = wave.open(files[0], "r")
params = f.getparams()
print(params) ## audio info 

samplerate, data = wavfile.read(files[0])
audio_depth = 256
# audio files have been resampled in 8 bits and 16kHz frequency
# audio_length = samplerate*10
audio_length = samplerate*1
# audio_length = 256

#audio: (nb_samples,length_audio) matrix containing all the .wav audio files
audios = np.ones((100,audio_length))*128
for k in range(len(files)):
    filename = files[k]
    if filename[-4:]=='.wav':
        samplerate,data = wavfile.read(files[k])
        if len(data)<=audio_length:
            audios[k,:len(data)] = data
        else:
            audios[k,:]= data[:audio_length]

audios = np.reshape(audios,(100,audio_length))

# audios = np.ones((100,audio_length))*128    ## preprocessing vérifié avec une audio de 16 bits, sortie [0,255]
# for k in range(len(files)):
#     filename = files[k]
#     if filename[-4:]=='.wav':
#         samplerate,data = wavfile.read(files[k])
#         if len(data)<=audio_length:
#             audios[k,:len(data)] = quantize_data(data, audio_depth)
#         else:
#             audios[k,:]= quantize_data(data[:audio_length],audio_depth)
#
# audios = np.reshape(audios,(100,audio_length))


# Plot 1 example of audio from the dataset
plt.figure()
plt.plot([k/samplerate for k in range(len(audios[10,:]))],audios[10,:])
plt.xlabel("time (s)")
plt.ylabel("bits (/256)")
plt.show()


# one hot encoding and set separation
nb_train_audios = int(0.8*len(os.listdir(path)))
train_audios = audios[:nb_train_audios]
train_audios = tf.one_hot(train_audios, depth=audio_depth)
test_audios = tf.one_hot(audios[nb_train_audios:], depth=audio_depth)

#Model initialisation
WNmodel = WaveNet_modelV2(
    nb_residual=24,
    nb_dilatation=2,
    audio_length=audio_length   # length of audio files is 10s max
).model

WNmodel.summary()

sgd = tf.keras.optimizers.Adam()  #SGD(lr=0.001, momentum=0.9) 
WNmodel.compile(sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# Save the model as a .h5 file for Neutron
WNmodel.save("WNmodel0112_InitialConv1ResidualblocOutputlayers.h5")

## --------------- Traning ---------------
num_epochs = 10
batch_size = 32


history = WNmodel.fit(x=train_audios, y=train_audios,
                    epochs=num_epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(train_audios, train_audios),
                    verbose=1)



# Generate data with the trained model

audio_predicted = WNmodel.predict(test_audios)

# Plot the first audio file generated by the network.
sample1 = audio_predicted[10,:,0]*255+128
sample2 = audio_predicted[8,:,0]*255+128

def coded2wav(coded,bit_depth):
    coded = (coded / bit_depth) * 2. - 1
    mu_gen = mu_law_expansion(coded, bit_depth)
    return mu_gen

plt.figure()
plt.plot([k/samplerate for k in range(len(coded2wav(sample1,audio_depth)))],coded2wav(sample2,audio_depth))
plt.xlabel("time (s)")
plt.ylabel("bits (/256)")
plt.show()
