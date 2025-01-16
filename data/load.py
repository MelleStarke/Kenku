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
  _melspec: Optional[object] = field(default=None) 
  
  def preload_melspec(self):
    self._melspec = read_melspec(self.melspec_path)
    
  @ property
  def melspec(self):
    if self._melspec is not None:
      return self._melspec
    
    return read_melspec(self.melspec_path)


class ParallelMelspecDataset(Dataset):
  def __init__(self, 
               melspec_dir: str, 
               transcript_dir: str, 
               speaker_properties_path: str,
               min_samples_per_sentence: Optional[int] = 10
  ):
    
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
          logger.warning(f"Age of speaker {speaker_id} in {speaker_properties_path} parsed as None. Skipping sample.")
          continue
        
        speaker_properties[speaker_id] = {
          "age": age,
          "gender": gender,
          "accent": accent
        }
    
    #=== Create Dataset Entries ===#
    
    # Dictionary linking sentences to dataset samples
    self.sentence_dict = {}
    
    # Folder names of speakers used as speaker IDs
    speaker_ids = sorted([folder_name for folder_name in os.listdir(melspec_dir)
                          if os.path.isdir(os.path.join(melspec_dir, folder_name))])
    
    for speaker_id in speaker_ids:
      try:
        props = speaker_properties[speaker_id]
      except KeyError:
        logger.warning(f"Speaker {speaker_id} not found in speaker properties object. Skipping samples of this speaker.")
        continue
        
      for file_name in os.listdir(os.path.join(transcript_dir, speaker_id)):
        transcript = ""
        
        transcript_path = os.path.join(transcript_dir, speaker_id, file_name)
        with open(transcript_path) as file:
          transcript = file.read().strip()
          
        if transcript not in self.sentence_dict:
          # Init add new unique sentence
          self.sentence_dict[transcript] = []
        
        melspec_path = transcript_path.replace(transcript_dir, melspec_dir).replace('.txt', '.h5')
        
        # Create new sample
        sample = (MelspecSample(file_name,
                                speaker_id,
                                melspec_path,
                                transcript,
                                props['age'],
                                props['gender'],
                                props['accent']))
        
        # Append with new sample
        self.sentence_dict[transcript].append(sample)
        
    # Remove all sentences with less samples than the specified lower bound
    for sentence, samples in list(self.sentence_dict.items()):
      if len(samples) < min_samples_per_sentence:
        del(self.sentence_dict[sentence])
    
    # Restructure self.sentence_dict to have lists of sample indices as values per unique sentence.
    # And store samples in a 1D list
    self.samples = [None] * sum(map(len, self.sentence_dict.values()))
    
    sample_idx = 0
    for sentence, samples in list(self.sentence_dict.items()):
      n_samples = len(samples)
      self.samples[sample_idx : sample_idx + n_samples] = samples
      self.sentence_dict[sentence] = list(range(sample_idx, sample_idx + n_samples))
      sample_idx += n_samples
      
      if n_samples != len(self.sentence_dict[sentence]):
        raise ValueError(f"list size mismatch. Went from {n_samples} to {len(self.sentence_dict[sentence])}")
      
    if np.concatenate(list(self.sentence_dict.values())).tolist() != list(range(len(self.samples))):
      raise ValueError(f"Enry indices do not line up. Sentenc dict indices:\n{np.array(self.sentence_dict.values())}\n" +\
                       f"Actual indices:\n{list(range(len(self.samples)))}")
    
    # A list of sample index pairs to be used for the iterator implementation
    self.iter_idxs = []
    
  def init_iter_idxs(self):
    self.iter_idxs = []
    for sample_idxs in self.sentence_dict.values():
      sample_idxs = sample_idxs.copy()
      np.random.shuffle(sample_idxs)
      # Remove the head of the list if its length is odd (i.e. make even for reshape)
      sample_idxs = sample_idxs[1:] if len(sample_idxs) % 2 else sample_idxs
      
      # Reshape to list of index pairs
      self.iter_idxs += np.reshape(sample_idxs, (-1, 2)).tolist()
  
  def __iter__(self):
    self.n_iter_calls += 1
    logger.info(f"Nr. of total __iter__() calls: {self.n_iter_calls}")
    
    self.init_iter_idxs()
    return self
  
  def __getitem__(self, idx: int):
    if len(self.iter_idxs) == 0:
      logger.warning("ParallelMelspecDataset.__getitem__() called with empty iterator index list.")
      self.init_iter_idxs()
    
    src_idx, tgt_idx = self.iter_idxs[idx]
      
    src_sample = self.samples[src_idx]
    tgt_sample = self.samples[tgt_idx]
    
    return [src_sample.melspec,
            tgt_sample.melspec,
            tgt_sample.gender]
  
  def __getitems__(self, idxs: List[int]):
    if len(self.iter_idxs) == 0:
      logger.warning("ParallelMelspecDataset.__getitems__() called with empty iterator index list.")
      self.init_iter_idxs()
    
    if torch.is_tensor(idxs):
      idxs = idxs.tolist()
    
    items = []
    for idx in idxs:
      src_idx, tgt_idx = self.iter_idxs[idx]
      
      src_sample = self.samples[src_idx]
      tgt_sample = self.samples[tgt_idx]
      
      items.append([src_sample.melspec,
                    tgt_sample.melspec,
                    tgt_sample.gender])
    
    return items

    
  def __len__(self):
    if len(self.iter_idxs) == 0:
      return sum([int(floor(len(sample_list) / 2)) for sample_list in self.sentence_dict.values()])
    
    return len(self.iter_idxs)
  
  def preload_melspec(self):
    [sample.preload_melspec() for sample in self.samples]
      


if __name__ == "__main__":
  
  print(current_file_dir)
  
  # Regular data loading
  if True:
    dataset = ParallelMelspecDataset(melspec_dir = "../Data/processed/VCTK/melspec", 
                                     transcript_dir = "../Data/processed/VCTK/transcript_standardized",
                                     speaker_properties_path = "../Data/processed/VCTK/speaker_properties.csv")

    sample = dataset.__getitem__([4])
  
  # Sentence sample distribution
  if False:
    melspec_dir = os.path.join(VCTK_MELSPEC_PATH)
    transcript_dir = os.path.join(VCTK_PATH, "transcript_standardized")
    speaker_props_path = os.path.join(VCTK_PATH, "speaker_properties.csv")
    
    ds = ParallelMelspecDataset(melspec_dir, transcript_dir, speaker_props_path)

    # print(ds.class_dirs)
    # print([len(d) for d in ds.filenames])
    
    plt.bar(range(len(ds.sentence_dict)), sorted([len(d) for d in ds.sentence_dict.values()]))
    plt.ylabel("N sentence samples")
    plt.xlabel("sentence")
    plt.show()