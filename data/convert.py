from pprint import pp as pprint

import argparse
import joblib
import pickle
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

from __init__ import *

from tqdm import tqdm

from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

# Set up logging format
fmt = '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
datafmt = '%m/%d/%Y %I:%M:%S'
logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datafmt,
                    filename=os.path.join(KENKU_PATH, "data/logs/convert.log"), filemode='a')


# TODO: standardize FFT size/frame length AND hop size/frame shift


#=== Defaults ===#


###############
### Utility ###
###############
                
def load_config(config_path: str):
  """
  Loads the config found in config_path as a dict.
  """
  config = None
  with open(config_path, 'r') as file:
    config = json.load(config_path)
    
  return config

def save_config(config: dict, config_path: str):
  """
  Saves the config at config_path as json file.
  """
  with open(config_path, 'w') as file:
    json.dump(data_config, file, indent=4)
  
def same_conf(config: dict, config_path: str):
  """
  Checks if the provided config dict and config dict found at config_path are equal.
  """
  loaded_config = load_config(config_path)
  return config == loaded_config

  
#########################
### Format Conversion ###  
#########################

# UNUSED DUE TO RECURSIVE FILE WALKING IN main() FUNCTION

# def convert_data_to_melspecs(src_datapath: str, 
#                              dst_datapath: str, 
#                              extension: str, 
#                              overwrite: bool = False, 
#                              config: dict = None):

#   num_files = len(list(walk_files(src_datapath, extension)))
  
#   if overwrite:
#     logger.warning("Overwriting existing melspec h5 files.")
#   else:
#     logger.warning("Keeping existing melspec h5 files.")
  
#   for src_filepath in tqdm(walk_files(src_datapath, extension), total=num_files):
#     # Path starting from src_datapath. Will be same in dst_datapath.
#     relative_filepath = src_filepath.replace(f"{src_datapath}/", "")
#     dst_filepath = os.path.join(dst_datapath, relative_filepath).replace(extension, '.h5')
    
#     if overwrite or not os.path.exists(src_filepath):
#       save_audio_file_as_melspec(src_filepath, dst_filepath, kwargs=kwargs)


 
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


def audio_file_to_melspec(src_filepath: str, dst_filepath: str, overwrite = True, config: dict = None):
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
  # Don't convert if set to "not overwrite" and file already exists.
  if not overwrite and os.path.exists(dst_filepath):
    logging.info(f"File {dst_filepath} already exists. Skipping.")
    return
  
  config = DEFAULT_CONFIG if config is None else config
  
  try:
    # warnings.filterwarnings('ignore')
    
    # Defaults found in ln. 105-112 and 23-31 in ConvS2S_VC/extract_features.py)
    trim_silence = config['trim_silence']  # default True
    top_db = config['top_db']  # default 30
    fft_size = config['fft_size']  # default 1024
    hop_size = config['hop_size']  # default 128 / 256
    min_freq = config['min_freq']  # default 80 / 0
    max_freq = config['max_freq']  # default 7600 / (samp_rate/2)
    num_mels = config['num_mels']  # default 80
    samp_rate = config['samp_rate']  # default 16000
    
    # Load the audio file
    audio, base_samp_rate = sf.read(src_filepath)
    
    # Optionally trim silence
    if trim_silence:
      # TODO: why manual frame length and shift if we can use config above?
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
      min_freq=min_freq,
      max_freq=max_freq,
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
    logging.info(f"Saving to {dst_filepath}... melspec shape: [{melspec.shape}].")

  except Exception as e:
    # Log failure if something goes wrong
    logging.info(f"Saving to {dst_filepath}...FAILED.")
    raise e


def logmelfilterbank(audio,
                     sampling_rate,
                     fft_size=1024,
                     hop_size=256,
                     win_length=None,
                     window="hann",
                     num_mels=80,
                     min_freq=None,
                     max_freq=None,
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
    min_freq (int): Minimum frequency in mel basis calculation.
    max_freq (int): Maximum frequency in mel basis calculation.
    eps (float): Epsilon value to avoid inf in log calculation.
  Returns:
    ndarray: Log Mel filterbank feature (#frames, num_mels).
  """
  # get amplitude spectrogram
  x_stft = librosa.stft(audio, n_fft=fft_size, hop_length=hop_size,
                        win_length=win_length, window=window, pad_mode="reflect")
  spc = np.abs(x_stft).T  # (#frames, #bins)

  # get mel basis
  min_freq = 0 if min_freq is None else min_freq
  max_freq = sampling_rate / 2 if max_freq is None else max_freq
  mel_basis = librosa.filters.mel(sr      = sampling_rate,
                                  n_fft   = fft_size, 
                                  n_mels  = num_mels, 
                                  fmin    = min_freq, 
                                  fmax    = max_freq)

  return np.log10(np.maximum(eps, np.dot(spc, mel_basis.T)))

#####################
### Normalization ###
#####################

def calc_norm_stats(melspec_datapath: str, stat_filepath: str):
  melspec_scaler = StandardScaler()
  
  # Gather a list of all .h5 files containing mel-spectrograms
  filepaths = list(walk_files(melspec_datapath, '.h5'))

  # Compute running statistics from all mel-spectrograms
  for filepath in tqdm(filepaths):
    melspec = read_melspec(filepath)
    # Transpose to (n_frames, n_mels) to match StandardScaler's expected input shape
    melspec_scaler.partial_fit(melspec.T)
    
  # TODO: This is false!!!
  logging.warning(f"SHAPE CHECK: {np.shape(melspec)} =? (N_timepoints, 80)")

  # Save the fitted scaler (which contains the mean and variance) to a pickle file
  with open(stat_filepath, mode='wb') as f:
    pickle.dump(melspec_scaler, f)
  
  return melspc_scaler
 

def apply_norm_scaler(audio_filepath: str, device: torch.device, scaler: BaseEstimator = None):
  
  if scaler is not None:
    # Normalize mel-spectrogram (scaler expects shape (n_frame, n_mels))
    melspec = scaler.transform(melspec)

  # Transpose to (n_mels, n_frame) and add batch dimension (1, n_mels, n_frame)
  melspec = melspec.T
  melspec = torch.tensor(melspec).unsqueeze(0).to(device, dtype=torch.float)
  print(f"MEL SPEC SHAPE: {melspec.shape}")
  return melspec


############
### MAIN ###
############

def main():
    """
    Main entry point for the script.

    This script:
    - Parses command-line arguments to configure mel-spectrogram extraction.
    - Recursively scans the source directory for audio files.
    - Extracts mel-spectrogram features from each file in parallel.
    - Saves the extracted features as HDF5 files in the destination directory.
    - Also saves a JSON configuration file reflecting the current parameters.

    Typical usage:
    python this_script.py --src /path/to/source --dst /path/to/destination --ext .wav
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str,
                        default=RAW_VCTK_AUDIO_PATH,
                        help='Source directory containing input audio files.')
    parser.add_argument('--dst', type=str, default=VCTK_MELSPEC_PATH,
                        help='Destination directory for extracted features.')
    parser.add_argument('--ext', type=str, default='.wav',
                        help='File extension for input audio files (e.g. .wav).')
    parser.add_argument('--conf', type=str, default=os.path.join(VCTK_PATH, 'data_config.json'),
                        help='Path to a JSON file where the data configuration will be saved.\n')
    
    parser.add_argument('--calc-norm', action="store_true", 
                        help="Calculate mean and std as a sklearn.StandardScaler object and store as a .pkl file 1 level above --dst")
    parser.add_argument('--apply-norm', action='store_true',
                        help='Apply the sklearn.StandardScaler to the melspec data.\n')
    
    parser.add_argument('--no-convert', action='store_true', 
                        help='Prevent audio files from being saved as mel-spectrograms.')
    parser.add_argument('--no-overwrite', action='store_true',
                        help="If set, don't overwrite existing melspec h5 files.\n")
   
    parser.add_argument('--num-mels', '-mel', type=int, default=80, help='Dimension of the mel-spectrogram.')
    parser.add_argument('--samp-rate', '-r', type=int, default=16000, help='Target sampling rate (Hz).')
    parser.add_argument('--fft-size', '-l', type=int, default=1024, help='Frame length (FFT size) in samples.')
    parser.add_argument('--hop-size', '-s', type=int, default=128, help='Frame shift (hop size) in samples.')
    parser.add_argument('--min-freq', type=int, default=80, help='Minimum frequency for mel filterbank (Hz).')
    parser.add_argument('--max-freq', type=int, default=7600, help='Maximum frequency for mel filterbank (Hz).')
    parser.add_argument('--trim-silence', action='store_true',
                        help='If set, trim leading and trailing silence from audio.')
    parser.add_argument('--top-db', type=int, default=30, help='Trimming threshold in dB for silence removal.')
     
    args = parser.parse_args()

    src = args.src
    dst = args.dst
    ext = args.ext
    
    convert = not args.no_convert
    
    # Torch device setting
    if torch.cuda.is_available():
      device = torch.device('cuda:0')
      torch.cuda.set_device(device)   
    else:
      device = torch.device('cpu')
    

    #=== Data Conversion ===#
    
    if convert:
      # Create a data configuration dictionary and save it as JSON
      data_config = {
          'num_mels': args.num_mels,
          'samp_rate': args.samp_rate,
          'fft_size': args.fft_size,
          'hop_size': args.hop_size,
          'min_freq': args.min_freq,
          'max_freq': args.max_freq,
          'trim_silence': args.trim_silence,
          'top_db': args.top_db
      }
      
      overwrite = not args.no_overwrite
      
      configpath = args.conf
      if not os.path.exists(os.path.dirname(configpath)):
        os.makedirs(os.path.dirname(configpath))
      
      # Warn the user if trying to keep melspec data made with a different config.
      elif not overwrite and not same_config(data_config, configpath):
        logging.warning("Keeping old melspecs despite configs not matching.\n" +\
                      f"\tOld config:\n{dict(sorted(load_config(configpath).items()))}\n\n" +\
                      f"\tNew config:\n{dict(sorted(data_config.items()))}\n")
        
      save_config(data_config, configpath)

      # Prepare a list of file processing arguments for parallel processing
      convert_args_list = [
          [
              f,
              f.replace(src, dst).replace(ext, ".h5"),
          ]
          for f in walk_files(src, ext)
      ]
      
      convert_kwargs = {
        'overwrite': overwrite,
        'config'   : data_config,
      }

      print('Extracting features...')

      # Process all files in parallel
      results = joblib.Parallel(n_jobs=16)(
          joblib.delayed(audio_file_to_melspec)(*convert_args, **convert_kwargs)
            for convert_args in tqdm(convert_args_list, total=len(convert_args_list))
      )
    
    #=== Normalization Stat Calculation ===#
    
    norm_scaler = None
    if args.calc_norm:
      norm_stat_filepath = os.path.join(os.path.dirname(dst), 'norm_stats.pkl')
      
      print("Calculating normalization statistics...")
      
      norm_scaler = calc_norm_stats(dst, norm_stat_filepath)

    #=== Normalization Stat Application ===#
    
    if args.apply_norm:
      if not args.calc_norm:
        logging.warning("'Apply normalization' was enabled despite not being calculated this run.\n" +\
                        "Be sure you're using the correct stats.")
      
      norm_stat_filepath = os.path.join(os.path.dirname(dst), 'norm_stats.pkl')
      
      # Load from disk if it hasn't already been calculated in the last step.
      if norm_scaler is None:
        if os.path.exists(norm_stat_filepath):
          with open(norm_stat_filepath, mode='rb') as f:
            norm_scaler = pickle.load(f)
            logging.info('Loaded mel-spectrogram statistics successfully.')
        else:
            logging.error(f'Stat file not found in {norm_stat_filepath}.')
      
      melspec_filepaths = list(walk_files(dst, '.h5'))
      
      results = joblib.Parallel(n_jobs=16)(
        joblib.delayed(apply_norm_scaler)(melspec_filepath, device, norm_scaler)
          for melspec_filepath in tqdm(melspec_filepaths, total=len(melspec_filepaths))
      )


if __name__ == '__main__':
    main()
