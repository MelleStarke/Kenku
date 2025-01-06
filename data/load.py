import numpy as np
import os
import h5py
import logging

from __init__ import *

from torch.utils.data import Dataset

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

