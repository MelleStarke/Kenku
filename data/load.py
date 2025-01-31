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
  
def signal_to_sequences(signal, seq_len = 32):
  return np.split(signal, np.arange(seq_len, np.shape(signal)[-1], step=seq_len), axis=-1)

def collate_fn(batch):
  """
  Custom collate function for torch.utils.data.DataLoader.
  Takes a list of samples and turns it into batch Tensors.
  Signal length of the batch is equal to that of longest signal,
  with other signals padded and masked until that length is reached.

  Args:
      batch (List[(ndarray, ndarray, int, int)]): List of samples.

  Returns:
      (ndarray, ndarray, ndarray, ndarray): Tuple of batches of 1.) source melspecs, 2.) target melspecs, 
                                            3.) source speaker properties, 4.) target speaker properties.
  """
  batch_size = len(batch)
  n_mels = len(batch[0][0])  # nr. of frequency features in mel-spectrogram
  
  src_sig_lengths = [sample[0].shape[1] for sample in batch]
  tgt_sig_lengths = [sample[1].shape[1] for sample in batch]
  
  src_maxlen = max(src_sig_lengths)
  tgt_maxlen = max(tgt_sig_lengths)
  
  src_melspec_batch = np.zeros((batch_size, n_mels, src_maxlen), dtype=batch[0][0].dtype)
  tgt_melspec_batch = np.zeros((batch_size, n_mels, src_maxlen), dtype=batch[0][1].dtype)
  
  src_mask_batch = np.zeros_like(src_melspec_batch)
  tgt_mark_batch = np.zeros_like(tgt_melspec_batch)
  
  src_props_batch   = np.array([sample[2] for sample in batch])
  tgt_props_batch   = np.array([sample[3] for sample in batch])
  
  for i, (src_sig_len, tgt_sig_len, (src_melspec, tgt_melspec, _, _)) in enumerate(zip(src_sig_lengths, tgt_sig_lengths, batch)):
    src_melspec_batch[i, :, :src_sig_len] = src_melspec
    tgt_melspec_batch[i, :, :tgt_sig_len] = tgt_melspec
    
    src_mask_batch[i, :, :src_sig_len] = 1.0
    tgt_mask_batch[i, :, :tgt_sig_len] = 1.0

  return (src_melspec_batch, tgt_melspec_batch, 
          src_mask_batch, tgt_mark_batch, 
          src_props_batch, tgt_props_batch)
    
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
               min_samples_per_sentence: Optional[int] = 10,
               rng: Optional[np.random.Generator] = None
  ):
    self.rng = rng if rng else np.random.default_rng(42)
    
    #=== Load Speaker Properties ===#
    speaker_properties = {}
    
    with open(speaker_properties_path, "r") as file:
      reader = DictReader(file, delimiter=",", skipinitialspace=True)
      for row in reader:
        speaker_id = row["ID"].strip().lower()  # Normalize speaker ID to lowercase
        age = int(row["AGE"].strip()) if row["AGE"].strip().isdigit() else None
        gender = row["GENDER"].strip().lower()
        accent = row["ACCENT"].strip().lower()
        
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
    self.transcript_dict = {}
    
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
          
        if transcript not in self.transcript_dict:
          # Init add new unique sentence
          self.transcript_dict[transcript] = []
        
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
        self.transcript_dict[transcript].append(sample)
        
    # Remove all sentences with less samples than the specified lower bound
    for sentence, samples in list(self.transcript_dict.items()):
      if len(samples) < min_samples_per_sentence:
        del(self.transcript_dict[sentence])
    
    # Restructure self.transcript_dict to have lists of sample indices as values per unique sentence.
    # And store samples in a 1D list
    self.samples = [None] * sum(map(len, self.transcript_dict.values()))
    
    sample_idx = 0
    for sentence, samples in list(self.transcript_dict.items()):
      n_samples = len(samples)
      self.samples[sample_idx : sample_idx + n_samples] = samples
      self.transcript_dict[sentence] = list(range(sample_idx, sample_idx + n_samples))
      sample_idx += n_samples
      
      if n_samples != len(self.transcript_dict[sentence]):
        raise ValueError(f"list size mismatch. Went from {n_samples} to {len(self.transcript_dict[sentence])}")
      
    if np.concatenate(list(self.transcript_dict.values())).tolist() != list(range(len(self.samples))):
      raise ValueError(f"Enry indices do not line up. Sentenc dict indices:\n{np.array(self.transcript_dict.values())}\n" +\
                       f"Actual indices:\n{list(range(len(self.samples)))}")
  
  def __getitem__(self, idx: int):
    if isinstance(idx, slice):
      return self.__getitems__(list(range(*idx.indices(len(self)))))
    
    tgt_sample = self.samples[idx]

    src_candidate_idxs = self.transcript_dict[tgt_sample.transcript]
    src_idx = self.rng.choice(src_candidate_idxs)
    src_sample = self.samples[src_idx]
    
    return (src_sample.melspec,
            tgt_sample.melspec,
            src_sample.gender,
            tgt_sample.gender)
    
  def __getitems__(self, idxs):
    if torch.is_tensor(idxs):
      idxs = idxs.tolist()
    
    return [self[idx] for idx in idxs]
  
  # def class_value_to_id(self, property, value):
  #   assert property.lower() in ['age', 'gender', 'accent'], \
  #     "Incorrect value for arg 'property', must be one of 'id, 'age', or 'gender'."
    
  #   self.class_ids[property](value)
  
  # def class_id_to_value(self, property, id):
  #   assert property.lower() in ['age', 'gender', 'accent'], \
  #     "Incorrect value for arg 'property', must be one of 'id, 'age', or 'gender'."

  def __len__(self):
    return len(self.samples)
  
  def preload_melspecs(self):
    """
    Pre-load mel-spectrograms for every sample.
    Avoids continuous file I/O operations, but uses more RAM.
    """
    [sample.preload_melspec() for sample in self.samples]
    


    

if __name__ == "__main__":
  
  print(current_file_dir)
  
  # Regular data loading
  if False:
    dataset = ParallelMelspecDataset(melspec_dir = "../Data/processed/VCTK/melspec", 
                                     transcript_dir = "../Data/processed/VCTK/transcript_standardized",
                                     speaker_properties_path = "../Data/processed/VCTK/speaker_properties.csv")

    sample = dataset.__getitem__([4])
  
  # Sentence sample distribution
  if True:
    dataset = ParallelMelspecDataset(melspec_dir = "../Data/processed/VCTK/melspec", 
                                     transcript_dir = "../Data/processed/VCTK/transcript_standardized",
                                     speaker_properties_path = "../Data/processed/VCTK/speaker_properties.csv",
                                     min_samples_per_sentence = 10)

    # print(ds.class_dirs)
    # print([len(d) for d in ds.filenames])
    
    print(dataset[0:2])
    
    exit()
    
    plt.bar(range(len(dataset.transcript_dict)), sorted([len(d) for d in dataset.transcript_dict.values()]))
    plt.ylabel("N sentence samples")
    plt.xlabel("sentence")
    plt.title("Sentence distribution (min = 20)")
    plt.show()