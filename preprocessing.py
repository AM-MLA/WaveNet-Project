import numpy as np
# import librosa
# import glob
# from hparams import hparams
# import os


def quantize_data(data, classes):
    """ quantization function:
            [[0,mu]] => [[0,mu]]
        with:
            mu = classes-1
    """
    mu_x = mu_law_encoding(data, classes-1)
    bins = np.linspace(-1, 1, classes)
    quantized = np.digitize(mu_x, bins) - 1
    return quantized

def coded2wav(coded, bit_depth):
    """reverse mu law and scale it to a bit depth of 16 bits
        [[0,bit_depth]] => [[0,65535]]
    """
    coded = (coded / bit_depth) * 2. - 1  # standardization
    mu_gen = (mu_law_expansion(coded, bit_depth) + 1) * 2**15
    return mu_gen


def mu_law_encoding(data, mu):
    """mu law :
        [-1,1] => [-1,1]"""
    mu_x = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)
    return mu_x


def mu_law_expansion(data, mu):
    """inverse mu law :
        [-1,1] => [-1,1]"""
    s = np.sign(data) * (np.exp(np.abs(data) * np.log(mu + 1)) - 1) / mu
    return s    


# if __name__ == "__main__":
#     para = hparams()
#     wavs = glob.glob(para.path_train_wavs+'/*wav')
#     os.makedirs(para.path_train_coded,exist_ok=True)

#     for file_wav in wavs:
#         name = os.path.split(file_wav)[-1]
#         audio,_ = librosa.load(file_wav,sr=para.fs,mono=True)

#         quantized_data = quantize_data(audio, para.classes)
#         save_name = os.path.join(para.path_train_coded,name+'.npy')
#         np.save(save_name,quantized_data)



        
