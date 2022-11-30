import numpy as np
import librosa
import glob
from hparams import hparams
import os


def quantize_data(data, classes):
    mu_x = mu_law_encoding(data, classes)
    bins = np.linspace(-1, 1, classes)
    quantized = np.digitize(mu_x, bins) - 1
    return quantized


def mu_law_encoding(data, mu):
    mu_x = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)
    return mu_x


def mu_law_expansion(data, mu):
    s = np.sign(data) * (np.exp(np.abs(data) * np.log(mu + 1)) - 1) / mu
    return s    


if __name__ == "__main__":
    para = hparams()
    wavs = glob.glob(para.path_train_wavs+'/*wav')
    os.makedirs(para.path_train_coded,exist_ok=True)

    for file_wav in wavs:
        name = os.path.split(file_wav)[-1]
        audio,_ = librosa.load(file_wav,sr=para.fs,mono=True)

        quantized_data = quantize_data(audio, para.classes)
        save_name = os.path.join(para.path_train_coded,name+'.npy')
        np.save(save_name,quantized_data)



        
