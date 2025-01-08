import numpy as np
import os
import h5py
import logging
import matplotlib.pyplot as plt

from pathlib import Path, PurePath

from typing import List, Tuple, Optional, Union

from torch.utils.data import Dataset

from __init__ import *


###############
### Logging ###
###############

logger = logging.getLogger(__name__)

# Get the full path to the directory containing the current file
current_file_dir = Path(__file__).parent.resolve()
logfile_path = os.path.join(current_file_dir, 'logs/load.log')

# Configure file handler
logfile_handler = logging.FileHandler(logfile_path, mode = 'a')
logfile_handler.setLevel(logging.DEBUG)
logger.addHandler(logfile_handler)

# Configure logging format
log_formatter = logging.Formatter(fmt     = '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                                  datefmt = '%m/%d/%Y %I:%M:%S')
logfile_handler.setFormatter(log_formatter)


def read_melspec(filepath):
  """
  Read a mel-spectrogram from an HDF5 file.

  Args:
    filepath (str): The path to the HDF5 file containing the mel-spectrogram.

  Returns:
    np.ndarray: A mel-spectrogram array of shape (n_mels, n_frames).
  """
  try:
    with h5py.File(filepath, "r") as f:
      melspec = f["melspec"][()]  # n_mels x n_frame
      
  except KeyError as e:
    logger.error(f"'melspec' object not found in {filepath}.")
    raise KeyError(e)
  
  return melspec 

def get_speaker_dirs(melspec_folder_path: str):
  """
  Returns a sorted list of paths to each speaker's mel-spectrogram folder.

  Args:
      melspec_folder_path (str): Directory containing all speaker folders.

  Returns:
      List[str]: List of paths to speaker folders.
  """
  return list(filter(os.path.isdir,
                     map(lambda d: os.path.join(melspec_folder_path, d), 
                         sorted(os.listdir(melspec_folder_path)))))

# [os.path.join(classes_path, d) for d in sorted(os.listdir(classes_path)) 
#           if os.path.isdir(os.path.join(classes_path, d))]


class MelspecDataset(Dataset):
  def __init__(self, speaker_dirs: List[str]):
    self.n_speakers = len(speaker_dirs)
    
    self.speaker_dirs = speaker_dirs
    # Use speaker folder names as speaker IDs
    self.speaker_ids  = [PurePath(d).name for d in speaker_dirs]
    self.filenames = [[os.path.join(d, t) for t in sorted(os.listdir(d))] for d in speaker_dirs]


if __name__ == "__main__":
  ds = MelspecDataset(get_speaker_dirs(VCTK_MELSPEC_PATH))
  
  print(ds.class_dirs)
  print([len(d) for d in ds.filenames])
  
  plt.bar(range(len(ds.filenames)), sorted([len(d) for d in ds.filenames]))
  plt.ylabel("N audio samples")
  plt.xlabel("speaker")
  plt.show()