from pprint import pp as pprint

import logging
import warnings
import numpy as np
import torch
import os
import h5py
import json
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

# TODO: standardize FFT size/frame length AND hop size/frame shift

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


DATAPATH = "/home/user/Uni/m3/thesis/project/Data"

#=== Voice Cloning Tool Kit Corpus ===#
VCTK_PATH       = os.path.join(DATAPATH, "raw/VCTK-Corpus-Alternative/VCTK-Corpus")
VCTK_AUDIO_PATH = os.path.join(VCTK_PATH, "wav48")
VCTK_TEXT_PATH  = os.path.join(VCTK_PATH, "txt")
VCTK_CLASS_PATH = os.path.join(VCTK_PATH, "speaker-info.txt")

#=== Speech Accent Archive ===#
SAA_PATH       = os.path.join(DATAPATH, "raw/SAA")
SAA_AUDIO_PATH = os.path.join(SAA_PATH, "recordings/recordings")
SAA_TEXT_PATH  = os.path.join(SAA_PATH, "reading-passage.txt")
SAA_CLASS_PATH = os.path.join(SAA_PATH, "speakers_all.csv")

#=== Defaults ===#
MELSPEC_CREATION_KWARGS = {
    'trim_silence': True,
    'top_db'      : 30,
    'num_mels'    : 80,
    
    'fft_size': 1024,
    'flen'    : 1024,
    
    'hop_size': 128,
    'fshift'  : 128,
    
    'min_freq': 80,
    'fmin'    : 80,
    
    'max_freq': 7600,
    'fmax'    : 7600,
    
    'samp_rate': 16000,
    'fs'       : 16000
  }


def walk_files(root, extension):
    for path, dirs, files in os.walk(root):
        for file in files:
            if file.endswith(extension):
                yield os.path.join(path, file)
                
def scaler_path(data_name: str):
  datapath_dict = {
    'VCTK': VCTK_PATH,
    'SAA' : SAA_PATH
  }
  try:
    path = datapath_dict[data_name.upper()]
  except KeyError:
    raise KeyError(f"{data_name} is not a valid data set name. Try VCTK or SAA.")
  
  return os.path.join(path, 'melspec_scaler.pkl')

def calc_melspec_scaler(datapath):
  melspec_scaler = StandardScaler()
  
#########################
### Format Conversion ###  
#########################

def convert_data_to_melspecs(src_datapath: str, 
                             dst_datapath: str, 
                             extension: str, 
                             overwrite: bool = False, 
                             kwargs: dict = None):
    
  kwargs = MELSPEC_CREATION_KWARGS if kwargs is None else kwargs

  num_files = len(list(walk_files(src_datapath, extension)))
  
  if overwrite:
    logger.warning("Overwriting existing melspec h5 files.")
  else:
    logger.warning("Keeping existing melspec h5 files.")
  
  for src_filepath in tqdm(walk_files(src_datapath, extension), total=num_files):
    # Path starting from src_datapath. Will be same in dst_datapath.
    relative_filepath = src_filepath.replace(f"{src_datapath}/", "")
    dst_filepath = os.path.join(dst_datapath, relative_filepath).replace(extension, '.h5')
    
    if overwrite or not os.path.exists(src_filepath):
      save_audio_file_as_melspec(src_filepath, dst_filepath, kwargs=kwargs)


def calc_and_save_norm_stats(melspec_datapath: str, stat_filepath: str):
  melspec_scaler = StandardScaler()
  
  # Gather a list of all .h5 files containing mel-spectrograms
  filepaths = list(walk_files(melspec_datapath, '.h5'))

  # Compute running statistics from all mel-spectrograms
  for filepath in tqdm(filepaths):
    melspec = read_melspec(filepath)
    # Transpose to (n_frames, n_mels) to match StandardScaler's expected input shape
    melspec_scaler.partial_fit(melspec.T)
    
  logging.warning(f"SHAPE CHECK: {np.shape(melspec)} =? (N_timepoints, 80)")

  # Save the fitted scaler (which contains the mean and variance) to a pickle file
  with open(stat_filepath, mode='wb') as f:
    pickle.dump(melspec_scaler, f)
 
 
def read_melspec(filepath):
  """
  Read a mel-spectrogram from an HDF5 file.

  Args:
    filepath (str): The path to the HDF5 file containing the mel-spectrogram.

  Returns:
    np.ndarray: A mel-spectrogram array of shape (n_mels, n_frames).
  """
  with h5py.File(filepath, "r") as f:
    melspec = f["melspec"][()]  # n_mels x n_frame
  return melspec 

def save_audio_file_as_melspec(src_filepath: str, dst_filepath: str, kwargs: dict):
  """
  Extract a mel-spectrogram from a single audio file and save it to an HDF5 file.

  This function:
    - Loads an audio file from `src_filepath`.
    - Optionally trims leading/trailing silence.
    - Optionally resamples the audio to the desired sampling rate.
    - Computes the mel-spectrogram using the given parameters.
    - Saves the resulting mel-spectrogram in HDF5 format at `dst_filepath`.

  Args:
      src_filepath (str): Path to the source audio file.
      dst_filepath (str): Path where the output HDF5 file will be saved.
      kwargs (dict): Dictionary of parameters, including:
        - 'trim_silence' (bool): If True, leading and trailing silence is trimmed based on `top_db`.
        - 'top_db' (int): The dB threshold for silence trimming.
        - 'fft_size' (int): FFT size (Frame length) for the STFT and mel-spectrogram calculation.
        - 'hop_size' (int): hop size (Frame shift) between consecutive STFT frames.
        - 'min_freq' (int): Minimum frequency (in Hz) of the mel basis calculation.
        - 'max_freq' (int): Maximum frequency (in Hz) of the mel basis calculation.
        - 'num_mels' (int): Number of channels for mel-spectrogram extraction.
        - 'samp_rate' (int): Target sampling rate (in Hz) to which the input audio is resampled.

  Logs:
      Logs progress and shape of extracted feature. Logs a failure message if an exception occurs.
  """
  try:
    # warnings.filterwarnings('ignore')
    
    # Defaults found in ln. 105-112 and 23-31 in ConvS2S_VC/extract_features.py)
    trim_silence = kwargs['trim_silence']  # default True
    top_db = kwargs['top_db']  # default 30
    fft_size = kwargs['fft_size']  # default 1024
    hop_size = kwargs['hop_size']  # default 128 / 256
    min_freq = kwargs['min_freq']  # default 80 / 0
    max_freq = kwargs['max_freq']  # default 7600 / (samp_rate/2)
    num_mels = kwargs['num_mels']  # default 80
    samp_rate = kwargs['samp_rate']  # default 16000
    
    # Load the audio file
    audio, base_samp_rate = sf.read(src_filepath)
    
    # Optionally trim silence
    if trim_silence:
      # TODO: why manual frame length and shift if we can use kwargs above?
      audio, _ = librosa.effects.trim(audio, top_db=top_db, frame_length=2048, hop_length=512)

    # Resample if the base sampling rate doesn't match the target rate
    if base_samp_rate != samp_rate:
      audio = librosa.resample(audio, orig_sr=base_samp_rate, target_sr=samp_rate)
      
    # Extract raw mel-spectrogram features (n_frame x n_mels)
    melspec = logmelfilterbank(
      audio,
      samp_rate,
      fft_size=fft_size,
      hop_size=hop_size,
      fmin=min_freq,
      fmax=max_freq,
      num_mels=num_mels
    )
    # Convert to float32 and transpose to (n_mels, n_frames)
    melspec = melspec.astype(np.float32).T
    
    # Ensure output directory exists, then save melspec in HDF5 format
    if not os.path.exists(os.path.dirname(dst_filepath)):
      os.makedirs(os.path.dirname(dst_filepath), exist_ok=True)
        
    with h5py.File(dst_filepath, "w") as f:
      f.create_dataset("melspec", data=melspec)

    # Log success and shape of the extracted mel-spectrogram
    logger.info(f"Saving to {dst_filepath}... melspec shape: [{melspec.shape}].")

  except Exception as e:
    # Log failure if something goes wrong
    logger.info(f"Saving to {dst_filepath}...FAILED.")
    raise e

def audio_file_to_spectrogram(audio_filepath: str, device: torch.device, kwargs: dict, scaler: BaseEstimator = None):
  """
  Load an audio file, optionally trim silence, resample if needed, and then extract a normalized
  log-mel-spectrogram using the provided scaler for normalization.

  Args:
    audio_filepath (str): Path to the input audio file.
    scaler (BaseEstimator): Scaler for normalizing mel-spectrogram features.
    device (torch.device): The device on which the output tensor should be placed.
    kwargs (dict): Dictionary of parameters, including:
      - 'trim_silence' (bool): If True, leading and trailing silence is trimmed based on `top_db`.
      - 'top_db' (int): The dB threshold for silence trimming.
      - 'fft_size' (int): FFT size (Frame length) for the STFT and mel-spectrogram calculation.
      - 'hop_size' (int): hop size (Frame shift) between consecutive STFT frames.
      - 'min_freq' (int): Minimum frequency (in Hz) of the mel basis calculation.
      - 'max_freq' (int): Maximum frequency (in Hz) of the mel basis calculation.
      - 'num_mels' (int): Number of channels for mel-spectrogram extraction.
      - 'samp_rate' (int): Target sampling rate (in Hz) to which the input audio is resampled.

  Returns:
      torch.Tensor: Normalized mel-spectrogram of shape (1, n_mels, n_frames) on the given device.
  """
  # Defaults found in ln. 105-112 and 23-31 in ConvS2S_VC/extract_features.py)
  trim_silence = kwargs['trim_silence']  # default True
  top_db = kwargs['top_db']  # default 30
  fft_size = kwargs['fft_size']  # default 1024
  hop_size = kwargs['hop_size']  # default 128 / 256
  min_freq = kwargs['min_freq']  # default 80 / 0
  max_freq = kwargs['max_freq']  # default 7600 / (samp_rate/2)
  num_mels = kwargs['num_mels']  # default 80
  samp_rate = kwargs['samp_rate']  # default 16000
  
  # Load the audio file
  audio, base_samp_rate = sf.read(audio_filepath)
  
  # Optionally trim silence
  if trim_silence:
    # TODO: why manual frame length and shift if we can use kwargs above?
    audio, _ = librosa.effects.trim(audio, top_db=top_db, frame_length=2048, hop_length=512)

  # Resample if the base sampling rate doesn't match the target rate
  if base_samp_rate != samp_rate:
    audio = librosa.resample(audio, orig_sr=base_samp_rate, target_sr=samp_rate)
    
  # Extract raw mel-spectrogram features (n_frame x n_mels)
  melspec = logmelfilterbank(
      audio,
      samp_rate,
      fft_size=fft_size,
      hop_size=hop_size,
      fmin=min_freq,
      fmax=max_freq,
      num_mels=num_mels
  ).astype(np.float32)
  
  plt.imshow(melspec)
  plt.show()
  
  if scaler is not None:
    # Normalize mel-spectrogram (scaler expects shape (n_frame, n_mels))
    melspec = scaler.transform(melspec)

  # Transpose to (n_mels, n_frame) and add batch dimension (1, n_mels, n_frame)
  melspec = melspec.T
  melspec = torch.tensor(melspec).unsqueeze(0).to(device, dtype=torch.float)
  print(f"MEL SPEC SHAPE: {melspec.shape}")
  return melspec

def logmelfilterbank(audio,
                     sampling_rate,
                     fft_size=1024,
                     hop_size=256,
                     win_length=None,
                     window="hann",
                     num_mels=80,
                     fmin=None,
                     fmax=None,
                     eps=1e-10,
                     ):
  """Compute log-Mel filterbank feature.
  Args:
    audio (ndarray): Audio signal (T,).
    sampling_rate (int): Sampling rate.
    fft_size (int): FFT size.
    hop_size (int): Hop size.
    win_length (int): Window length. If set to None, it will be the same as fft_size.
    window (str): Window function type.
    num_mels (int): Number of mel basis.
    fmin (int): Minimum frequency in mel basis calculation.
    fmax (int): Maximum frequency in mel basis calculation.
    eps (float): Epsilon value to avoid inf in log calculation.
  Returns:
    ndarray: Log Mel filterbank feature (#frames, num_mels).
  """
  # get amplitude spectrogram
  x_stft = librosa.stft(audio, n_fft=fft_size, hop_length=hop_size,
                        win_length=win_length, window=window, pad_mode="reflect")
  spc = np.abs(x_stft).T  # (#frames, #bins)

  # get mel basis
  fmin = 0 if fmin is None else fmin
  fmax = sampling_rate / 2 if fmax is None else fmax
  mel_basis = librosa.filters.mel(sr      = sampling_rate,
                                  n_fft   = fft_size, 
                                  n_mels  = num_mels, 
                                  fmin    = fmin, 
                                  fmax    = fmax)

  return np.log10(np.maximum(eps, np.dot(spc, mel_basis.T)))


if __name__ == "__main__":
  device = torch.device("cuda:0")
  
  dst_datapath = str(os.path.join(DATAPATH, "processed/VCTK/melspec"))
  src_datapath = VCTK_AUDIO_PATH
  
  # print(src_datapath)
  # print(src_datapath == "./../../Data/raw/VCTK-Corpus-Alternative/VCTK-Corpus/wav48")
  convert_data_to_melspecs(src_datapath, dst_datapath, '.wav', overwrite=True)
  
  # ./../../Data/raw/VCTK-Corpus-Alternative/VCTK-Corpus/wav48