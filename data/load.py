import numpy as np
import os
import h5py
import logging
import matplotlib.pyplot as plt

from copy import copy
from csv import DictReader
from dataclasses import dataclass, field

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


@dataclass
class MelspecSample:
  id: str
  speaker_id: str
  melspec_path: str
  transcript: str
  age: int
  gender: str
  accent: str
  melspec: Optional[object] = field(default=None) 
  
  def preload_melspec(self):
    self.melspec = read_melspec(self.melspec_path)


class MelspecDataset():
  def __init__(self, melspec_dir: str, transcript_dir: str, speaker_properties_path: str):
    self.speaker_ids = sorted([folder_name for folder_name in os.listdir(melspec_dir)
                               if os.path.isdir(os.path.join(melspec_dir, folder_name))])
    
    #=== Load Speaker Properties ===#
    speaker_properties = {}
    
    with open(speaker_properties_path, "r") as file:
      reader = DictReader(file, delimiter=",", skipinitialspace=True)
      for row in reader:
        speaker_id = row["ID"].strip().lower()  # Normalize speaker ID to lowercase
        age = int(row["AGE"].strip()) if row["AGE"].strip().isdigit() else None
        gender = row["GENDER"].strip().lower()
        accent = row["ACCENTS"].strip().lower()
        
        if age is None:
          logger.warning(f"Age of speaker {speaker_id} in {speaker_properties_path} parsed as None. Skipping entry.")
          continue
        
        speaker_properties[speaker_id] = {
          "age": age,
          "gender": gender,
          "accent": accent
        }
    
    #=== Create Dataset Entries ===#
    
    # Dictionary linking sentences to dataset entry IDs
    self.sentence_dict = {}
    
    self.entries = []
    
    for speaker_id in self.speaker_ids:
      try:
        props = speaker_properties[speaker_id]
      except KeyError:
        logger.warning(f"Speaker {speaker_id} not found in speaker properties object. Skipping entries of this speaker.")
        continue
        
      for file_name in os.listdir(os.path.join(transcript_dir, speaker_id)):
        transcript = ""
        
        transcript_path = os.path.join(transcript_dir, speaker_id, file_name)
        with open(transcript_path) as file:
          transcript = file.read().strip()
          
        if transcript not in self.sentence_dict:
          # Init entry idx tracker of new sentence
          self.sentence_dict[transcript] = []
        # Append with index of latest entry
        self.sentence_dict[transcript].append(len(self.entries))
        
        # Create new entry
        self.entries.append(MelspecSample(file_name,
                                          speaker_id,
                                          transcript_path.replace('.txt', '.h5'),
                                          transcript,
                                          props['age'],
                                          props['gender'],
                                          props['accent']))
      

class TestDataset(MelspecDataset):
  def __init__(*args, batch_size = 32, **kwargs):
    super(TestDataset, self).__init__(*args, **kwargs)
    
    self.randgen = np.random.default_rng(42)
    self.batch_size = batch_size
    
    self.sentence_dict = {
      k: v for k, v in self.sentence_dict.items() if len(v) >= 20
    }
    
    for entry_idx in np.concatenate(list(self.sentence_dict.values())):
      self.entries[entry_idx].preload_melspec()
      
    self.iter_buffer = None
      
  def __iter__(self):
    self.iter_buffer = copy(list(self.sentence_dict.values()))
    
  def __next__(self):
    
    src_melspecs = []
    tgt_melspecs = []
    src_classes  = []
    tgt_classes  = []  
    
    for bi in range(self.batch_size):
      if len(self.iter_buffer) == 0:
        raise StopIteration()
      
      si = self.randgen.integers(len(self.iter_buffer) - 1)
      
      xi, yi = self.randgen.choice(self.iter_buffer[si])
      self.iter_buffer[si].remove(xi)
      self.iter_buffer[si].remove(yi)
      
      if len(self.iter_buffer[si]) < 2:
        del(self.iter_buffer[si])
        
      x = self.entries[xi]
      y = self.entries[yi]
      
      src_melspecs.append(x.melspec)
      tgt_melspecs.append(y.melspec)
      src_classes.append({'m': 0, 'f': 1}[x.gender])
      tgt_classes.append({'m': 0, 'f': 1}[y.gender])
  
    return np.array(src_melspecs), np.array(src_classes), np.array(tgt_melspecs), np.array(tgt_classes)


if __name__ == "__main__":
  melspec_dir = os.path.join(VCTK_MELSPEC_PATH)
  transcript_dir = os.path.join(VCTK_PATH, "transcript_standardized")
  speaker_props_path = os.path.join(VCTK_PATH, "speaker_properties.csv")
  
  ds = MelspecDataset(melspec_dir, transcript_dir, speaker_props_path)
  
  
  
  # print(ds.class_dirs)
  # print([len(d) for d in ds.filenames])
  
  plt.bar(range(len(ds.sentence_dict)), sorted([len(d) for d in ds.sentence_dict.values()]))
  plt.ylabel("N sentence samples")
  plt.xlabel("sentence")
  plt.show()