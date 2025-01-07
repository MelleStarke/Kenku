import numpy as np
import os
import h5py
import logging
import matplotlib.pyplot as plt

from typing import List, Tuple, Optional, Union

from torch.utils.data import Dataset

from __init__ import *


# Set up logging format
fmt = '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
datafmt = '%m/%d/%Y %I:%M:%S'
logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datafmt,
                    filename=os.path.join(KENKU_PATH, "data/logs/load.log"), filemode='a')


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
    logging.error(f"'melspec' object not found in {filepath}.")
    raise KeyError(e)
  
  return melspec 

def get_class_dirs(classes_path: str):
  """
  Returns a sorted list of paths to each speaker class's mel-spectrogram folder.

  Args:
      classes_path (str): Directory containing all speaker folders.

  Returns:
      List[str]: List of paths to speaker folders.
  """
  return list(filter(os.path.isdir,
                     map(lambda d: os.path.join(classes_path, d), 
                         sorted(os.listdir(classes_path)))))

# [os.path.join(classes_path, d) for d in sorted(os.listdir(classes_path)) 
#           if os.path.isdir(os.path.join(classes_path, d))]


class MelspecDataset(Dataset):
  def __init__(self, class_dirs: List[str]):
    self.n_classes = len(class_dirs)
    
    self.class_dirs = class_dirs
    self.filenames = [[os.path.join(d, t) for t in sorted(os.listdir(d))] for d in class_dirs]


if __name__ == "__main__":
  ds = MelspecDataset(get_class_dirs(VCTK_MELSPEC_PATH))
  
  print(ds.class_dirs)
  print([len(d) for d in ds.filenames])
  
  plt.bar(range(len(ds.filenames)), sorted([len(d) for d in ds.filenames]))
  plt.ylabel("N audio samples")
  plt.xlabel("speaker")
  plt.show()