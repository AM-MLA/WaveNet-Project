# Wavent Implementation for Machine Learning Project

The goal of the repository is to provide an implementation of the WaveNet vocoder based on the thesis which is available at https://arxiv.org/abs/1609.03499.

Audio samples (French discourse) are available at https://commonvoice.mozilla.org/fr/datasets.

## Table of Contents
- [Background](#background)
- [Requirement](#Requirements)
- [Repository structure](#Repository structure)
- [Installation](#Installation)
- [Getting started](# Getting started)
- [Contributors](#Contributors)
- [References](#References)


## Background
Wavenet model is a sequence generation model that can be used for speech generation modeling. In the acoustic model modeling of speech synthesis, Wavenet can learn the mapping to the sequence of sampled values directly, so it has good synthesis effect. Currently wavenet has applications in acoustic model modeling for speech synthesis, vocoder, and has great potential in the field of speech synthesis. Several use cases are possible: generation of audio resembling human speech, generation of audio from text, generation of fairly realistic musical fragments and generation of speech conditioned on a characteristic of the speaker, such as voice timbre, accent or age.


## Requirements
Python 3 

tensorflow>=2.10.0 

numpy>=1.23.4 

matplotlib>=3.6.2 

scipy>=1.9.3 


## Repository structure
This repository contains: 1) WaveNet-Project  2) Documents 3) Demo-repository.


## Installation
```
git clone https://github.com/AM-MLA/WaveNet-Project.git
pip install -e .
```

## Getting started
### Docs for command line tools

#### params.py

Dump hyperparameters to a json file.

Usage:

```
python tojson.py --hparams="parameters you want to override" <output_json_path>
```

#### preprocess.py

Usage:

```
python preprocess.py 
```

#### model training.py

Usage:

```
python train.py --hparams="parameters you want to override"
```

#### mp32wav.py
Usage:

```
python preprocess.py 
```

### Contributors

This project exists thanks to all the people who contribute. 

## References

- [Aaron van den Oord, Sander Dieleman, Heiga Zen, et al, "WaveNet: A Generative Model for Raw Audio", 	arXiv:1609.03499, Sep 2016.](https://arxiv.org/abs/1609.03499)
- [Aaron van den Oord, Yazhe Li, Igor Babuschkin, et al, "Parallel WaveNet: Fast High-Fidelity Speech Synthesis", 	arXiv:1711.10433, Nov 2017.](https://arxiv.org/abs/1711.10433)
- [Tamamori, Akira, et al. "Speaker-dependent WaveNet vocoder." Proceedings of Interspeech. 2017.](http://www.isca-speech.org/archive/Interspeech_2017/pdfs/0314.PDF)
- [Jonathan Shen, Ruoming Pang, Ron J. Weiss, et al, "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions", arXiv:1712.05884, Dec 2017.](https://arxiv.org/abs/1712.05884)
- [Wei Ping, Kainan Peng, Andrew Gibiansky, et al, "Deep Voice 3: 2000-Speaker Neural Text-to-Speech", arXiv:1710.07654, Oct. 2017.](https://arxiv.org/abs/1710.07654)
- [Tom Le Paine, Pooya Khorrami, Shiyu Chang, et al, "Fast Wavenet Generation Algorithm", arXiv:1611.09482, Nov. 2016](https://arxiv.org/abs/1611.09482)
- [Ye Jia, Yu Zhang, Ron J. Weiss, Quan Wang, Jonathan Shen, Fei Ren, Zhifeng Chen, Patrick Nguyen, Ruoming Pang, Ignacio Lopez Moreno, Yonghui Wu, et al, "Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis" , arXiv:1806.04558v4 cs.CL 2 Jan 2019](https://arxiv.org/abs/1806.04558)
