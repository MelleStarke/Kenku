# Kenku
AI Master's Thesis Project by Melle Starke for the University of Groningen, The Netherlands.

Kenku is based on the work of [Kameoka](https://www.kecl.ntt.co.jp/people/kameoka.hirokazu/index-e.html) et al., who developed a streaming, non-autoregressive speech-to-speech voice conversion model:
- [Github repository](https://github.com/kamepong/ConvS2S-VC)
- [Paper (FastConvS2S-VC)](https://arxiv.org/abs/2104.06900)

Kenku aims to leverage the field of Disentangled Reinforcement Learning to allow for tunable output speaker properties (e.g. age, gender, nationality). As opposed to voice cloning or pre-trained models.

# Installation

Run `pip install -r ./requirements.txt`, preferably as a new `venv`.

# Usage
## Data Preparation

Kenku expects the data directory to be of a certain structure, namely a directory with:
- A `melspec` folder, containing sub-folders per participant. The names of these sub-folders function as speaker IDs, and contain `.h5` files with a `spectrogram` field.
- A `transcript` folder, with the same structure as the `melspec` folder, but with `.txt` files containing the transcript of the associated Mel-spectrogram.
- A `speaker_info.csv` file, containing 4 named columns: `ID`, `AGE`, `GENDER`, and `ACCENT`. Each ID corresponds to a sub-folder in the `melspec` and `transcript`.

Here is an example of the head of a correct `speaker_info.csv` file, adapted from the [VCTK Corpus Dataset (ver. 0.92)](https://www.kaggle.com/datasets/kynthesis/vctk-corpus):
```
ID,  AGE,GENDER,ACCENT
p225,23, F,     English
p226,22, M,     English
p227,38, M,     English
p228,22, F,     English
p229,23, F,     English
p230,22, F,     English
p231,23, F,     English
p232,23, M,     English
p233,23, F,     English
p234,22, F,     Scottish
p236,23, F,     English
...
```

### Audio to Mel-spectrograms

To achieve this structure, you can run `python -m data.convert_audio src dst`, where `src` is the directory containing speaker ID sub-folders, which in turn contain `.wav` files (though other extensions can be specified), and `dst` is the `melspec` directory, as described above.

This script also allows you to:
- Calculate the normalization scaler of the spectrograms and save it
- Apply the normalization scaler (possibly from a previously saved file)
- Disable conversion entirely. For example, to apply normalization on already converted data.

For more info, run `python -m data.convert_audio --help`

### Transcripts

Transcripts must match exactly to be counted as "the same". Even a mismatch in capital letters or punctuation can be an issue. Therefore, to standardize the transcripts, you can run `python -m data.clean_transcripts dst src`.

## Training Models

We provide a singular script to train every type of model (i.e. teacher vs. student and DRL vs. non-DRL). This script can be run with `python -m train.train_model` and allows for a large amount of optional arguments, which are explained via the `--help` argument.

These arguments are separated into 3 categories: the dataset config, model config, and train config. The arguments in these configs can also be specified through `.json` files, using the arguments `--dataset-config-path`, `--model-config-path`, and `--train-config-path`. Or all together using `--config-dir`, where the files must be called `dataset-config.json`, `model-config.json`, and `train-config.json`.

These config files do not need to specify every argument, but every argument in a config file takes precedence over ones specified in the command line.

## Hyperparameter Tuning

We provide a script for easy grid search over any and all hyperparameters. This is accessed as `python -m train.hypertune setting_dir job_dir setting_index`, where `setting_dir` contains two config files (`static_settings.json` and `variable_settings.json`), `job_dir` tells the script where to save the runs, and `setting_index` is the index of the particular variable combination. This works best with a compute cluster job scheduler such as SLURM.

The script takes the Cartesian product of all settings in `variable_settings.json` and picks the combination according to the `setting_index`.