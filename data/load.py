import numpy as np
import torch
import os
import h5py
import logging
import matplotlib.pyplot as plt

from copy import copy
from csv import DictReader
from dataclasses import dataclass, field

from pathlib import Path, PurePath

from typing import List, Tuple, Optional, Union, Dict

from torch.utils.data import Dataset, DataLoader

from itertools import product

from __init__ import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
      melspec = f["melspec"][()]  # n_mels x n_timesteps
      
  except KeyError as e:
    logger.error(f"'melspec' object not found in {filepath}.")
    raise KeyError(e)
  
  return melspec
  
def signal_to_sequences(signal, seq_len = 32):
  if torch.is_tensor(signal):
    return torch.split(signal, seq_len, dim=-1)
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
      (Tensor, Tensor, Tensor, Tensor, List, List): 
          Tuple of batches of 1.) source melspecs,           2.) target melspecs, 
                              3.) source masks,              4.) target masks,
                              5.) source speaker info, 6.) target speaker info.
  """
  batch_size = len(batch)
  n_mels = len(batch[0][0])  # nr. of frequency features in mel-spectrogram
  
  #=== Signal Length Equalization ===#
  src_sig_lengths = [sample[0].shape[1] for sample in batch]
  tgt_sig_lengths = [sample[1].shape[1] for sample in batch]
  
  max_sig_len = max(max(src_sig_lengths), max(tgt_sig_lengths))
  
  src_melspec_batch = torch.zeros((batch_size, n_mels, max_sig_len), dtype=torch.float32, device=device)
  tgt_melspec_batch = torch.zeros((batch_size, n_mels, max_sig_len), dtype=torch.float32, device=device)
  
  #=== Mask Real vs. Padded Frames ===#
  src_mask_batch = torch.zeros_like(src_melspec_batch[:,0,:]).unsqueeze(1)
  tgt_mask_batch = torch.zeros_like(tgt_melspec_batch[:,0,:]).unsqueeze(1)
  
  for i, (src_sig_len, tgt_sig_len, (src_melspec, tgt_melspec, _, _)) in enumerate(zip(src_sig_lengths, tgt_sig_lengths, batch)):
    src_melspec_batch[i, :, :src_sig_len] = torch.from_numpy(src_melspec)
    tgt_melspec_batch[i, :, :tgt_sig_len] = torch.from_numpy(tgt_melspec)
    
    src_mask_batch[i, :, :src_sig_len] = 1.0
    tgt_mask_batch[i, :, :tgt_sig_len] = 1.0
    
  #=== Speaker Info ===#
  src_info_batch  = [sample[2] for sample in batch]
  tgt_info_batch  = [sample[3] for sample in batch]
  
  # Turn list of tuples of age, gender, and accent into a tuple of 3 lists of age, gender, and accent
  src_age_batch, src_gender_batch, src_accent_batch = tuple(map(list, zip(*src_info_batch)))
  tgt_age_batch, tgt_gender_batch, tgt_accent_batch = tuple(map(list, zip(*tgt_info_batch)))

  # TODO: make this prettier. A map of tensor casting over a zipped info batch and dtype should work.
  src_info_batch = (torch.tensor(src_age_batch,    dtype=torch.float),
                    torch.tensor(src_gender_batch, dtype=torch.float),
                    torch.tensor(src_accent_batch, dtype=torch.int64))

  tgt_info_batch = (torch.tensor(tgt_age_batch,    dtype=torch.float),
                    torch.tensor(tgt_gender_batch, dtype=torch.float),
                    torch.tensor(tgt_accent_batch, dtype=torch.int64))

  return (src_melspec_batch, tgt_melspec_batch, 
          src_mask_batch,    tgt_mask_batch, 
          src_info_batch,    tgt_info_batch)
    
    
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
  
  
class SpeakerInfoMixin:
  def encode_age(self, age):
    lowbound, hibound = self.age_bounds
    age_range = hibound - lowbound
    return (age - lowbound) / age_range
  
  def encode_gender(self, gender):
    gender_to_int = {'m': 0,
                     'f': 1}
    return gender_to_int[gender]
  
  def encode_accent(self, accent):
    accent_idx = self.all_accents.index(accent)
    return accent_idx
  

class ParallelDatasetFactory(SpeakerInfoMixin):
  def __init__(self, 
               melspec_dir: str, 
               transcript_dir: str, 
               speaker_info_path: str,
               age_bounds = (10, 80),
               rng: Optional[np.random.Generator] = None
  ):
    self.age_bounds = age_bounds
    self.rng = rng if rng else np.random.default_rng(42)
    
    #=== Load Speaker Info ===#
    
    speaker_info  = self.read_speaker_info(speaker_info_path)
    self.all_accents = np.unique([speaker['accent'] for speaker in speaker_info.values()]).tolist()
    
    #=== Create Dataset Samples ===#
    
    self.samples = []
    self.transcript_dict = {}
    
    # Folder names of speakers used as speaker IDs
    speaker_ids = sorted([folder_name for folder_name in os.listdir(melspec_dir)
                          if os.path.isdir(os.path.join(melspec_dir, folder_name))])
    
    for speaker_id in speaker_ids:
      try:
        info = speaker_info[speaker_id]
      except KeyError:
        logger.warning(f"Speaker {speaker_id} not found in speaker info object. Skipping samples of this speaker.")
        continue
        
      for file_name in os.listdir(os.path.join(transcript_dir, speaker_id)):
        transcript = ""
        
        transcript_path = os.path.join(transcript_dir, speaker_id, file_name)
        with open(transcript_path) as file:
          transcript = file.read().strip()
        
        melspec_path = transcript_path.replace(transcript_dir, melspec_dir).replace('.txt', '.h5')

        sample = MelspecSample(file_name,
                               speaker_id,
                               melspec_path,
                               transcript,
                               info['age'],
                               info['gender'],
                               info['accent'])

        self.samples.append(sample)
        
        if transcript not in self.transcript_dict:
          self.transcript_dict[transcript] = []
        
        latest_sample_idx = len(self.samples) - 1
        self.transcript_dict[transcript].append(latest_sample_idx)
        
    if sorted(np.concatenate(list(self.transcript_dict.values())).tolist()) != sorted(list(range(len(self.samples)))):
      raise ValueError(f"Enry indices do not line up. Sentence dict indices:\n{np.concatenate(list(self.transcript_dict.values()))}\n" +\
                       f"Actual indices:\n{list(range(len(self.samples)))}")
    
  def read_speaker_info(self, speaker_info_path):
    speaker_info = {}
    
    with open(speaker_info_path, "r") as file:
      reader = DictReader(file, delimiter=",", skipinitialspace=True)
      for row in reader:
        speaker_id = row["ID"].strip().lower()  # Normalize speaker ID to lowercase
        age = int(row["AGE"].strip()) if row["AGE"].strip().isdigit() else None
        gender = row["GENDER"].strip().lower()
        accent = row["ACCENT"].strip().lower()
        
        if age is None:
          logger.warning(f"Age of speaker {speaker_id} in {speaker_info_path} parsed as None. Skipping sample.")
          continue
        
        speaker_info[speaker_id] = {
          "age": age,
          "gender": gender,
          "accent": accent
        }
        
    return speaker_info
  
  def train_test_split(self, min_transcript_samples: int = 3, train_set_threshold: int = 10, sample_pairing = 'product'):
    if isinstance(sample_pairing, str):
      sample_pairing = (sample_pairing, sample_pairing)
    
    for sp in sample_pairing:
      assert sp.lower() in ['random', 'rand', 'product', 'prod'], \
        f"'{sample_pairing}' not a valid sample pairing mode. Choose 'random' or 'product'."
    
    train_transcript_dict = {}
    test_transcript_dict = {}
    
    for transcript, sample_idxs in self.transcript_dict.items():
      if len(sample_idxs) < min_transcript_samples:
        continue
      
      if len(sample_idxs) < train_set_threshold:
        train_transcript_dict[transcript] = sample_idxs
        
      else:
        test_transcript_dict[transcript] = sample_idxs
        
    train_set = ParallelMelspecDataset(self.samples, 
                                       train_transcript_dict,
                                       sample_pairing = sample_pairing[0],
                                       age_bounds = self.age_bounds,
                                       all_accents = self.all_accents,
                                       rng = self.rng)
    
    test_set  = ParallelMelspecDataset(self.samples, 
                                       test_transcript_dict,
                                       sample_pairing = sample_pairing[1],
                                       age_bounds = self.age_bounds,
                                       all_accents = self.all_accents,
                                       rng = self.rng)
    
    return train_set, test_set
  
  def get_dataset(self, min_transcript_samples = 10, sample_pairing = 'product'):
    assert sample_pairing.lower() in ['random', 'rand', 'product', 'prod'], \
      f"'{sample_pairing}' not a valid sample pairing mode. Choose 'random' or 'product'."
      
    new_transcript_dict = {}
    
    for transcript, sample_idxs in self.transcript_dict.items():
      if len(sample_idxs) >= min_transcript_samples:
        new_transcript_dict[transcript] = sample_idxs
    
    dataset = ParallelMelspecDataset(self.samples,
                                     new_transcript_dict, 
                                     sample_pairing = sample_pairing, 
                                     age_bounds = self.age_bounds,
                                     all_accents = self.all_accents,
                                     rng = self.rng)
    return dataset
        

class ParallelMelspecDataset(Dataset, SpeakerInfoMixin):
  def __init__(self, 
               samples: List[MelspecSample],
               transcript_dict: Dict[str, List[int]],
               age_bounds = (10, 80),
               all_accents = None,
               sample_pairing = 'random',
               rng: Optional[np.random.Generator] = None
  ):
    assert sample_pairing.lower() in ['random', 'rand', 'product', 'prod'], \
      f"'{sample_pairing}' not a valid sample pairing mode. Choose 'random' or 'product'."
    
    if accents is None:
      logger.error("List of accents isn't initialized. Accent encoding won't work.")
    
    self.age_bounds = age_bounds
    self.all_accents = all_accents
    self.rng = rng if rng else np.random.default_rng(42)
    self.sample_pairing = sample_pairing.lower()
    self._cartprod_sample_pairs = None
    
    self.samples = samples
    self.transcript_dict = transcript_dict
  
  @property
  def cartprod_sample_pairs(self):
    if self._cartprod_sample_pairs is None:
      self._cartprod_sample_pairs = np.concatenate([list(product(idxs, idxs)) 
                                                   for idxs in self.transcript_dict.values()])
      
    return self._cartprod_sample_pairs
  
  def __getitem__(self, idx: int):
    if isinstance(idx, slice):
      return self.__getitems__(list(range(*idx.indices(len(self)))))
    
    if self.sample_pairing in ['prod', 'product']:
      src_idx, tgt_idx = self.cartprod_sample_pairs[idx]
      src_sample, tgt_sample = self.samples[src_idx], self.samples[tgt_idx]
      
    else:
      tgt_sample = self.samples[idx]

      src_candidate_idxs = self.transcript_dict[tgt_sample.transcript]
      src_idx = self.rng.choice(src_candidate_idxs)
      src_sample = self.samples[src_idx]
      
    src_info = (self.encode_age(src_sample.age),
                self.encode_gender(src_sample.gender),
                self.encode_accent(src_sample.accent))
    tgt_info = (self.encode_age(tgt_sample.age),
                self.encode_gender(tgt_sample.gender),
                self.encode_accent(tgt_sample.accent))
    
    return (src_sample.melspec,
            tgt_sample.melspec,
            src_info,
            tgt_info)
    
  def __getitems__(self, idxs):
    if torch.is_tensor(idxs):
      idxs = idxs.tolist()
    
    return [self[idx] for idx in idxs]
  
  def __len__(self):
    if self.sample_pairing in ['product', 'prod']:
      return sum(map(lambda idx_list: len(idx_list) ** 2, self.transcript_dict.values()))
    
    return sum(map(len, self.transcript_dict.values()))
  
  def preload_melspecs(self):
    """
    Pre-load mel-spectrograms for every sample.
    Avoids continuous file I/O operations, but uses more RAM.
    """
    [sample.preload_melspec() for sample in self.samples]
    
  def train_test_split(self, proportion=0.2, sample_pairing = 'product'):
    assert sample_pairing.lower() in ['random', 'rand', 'product', 'prod'], \
      f"'{sample_pairing}' not a valid entry pairing mode. Choose 'random' or 'product'."
    
    train_set = copy(self)
    train_set.transcript_dict = {}
    
    test_set = copy(self)
    test_set.transcript_dict = {}
    test_set.sample_pairing = sample_pairing.lower()
    
    # "Uncache" the previously calculated entry pairs
    self._cartprod_sample_pairs = None
    
    for transcript, entry_idxs in self.transcript_dict.items():
      n_train_samples = max(2, int(np.round(len(entry_idxs) * proportion)))
      test_idxs, train_idxs = np.split(self.rng.permutation(entry_idxs), [n_train_samples])

      train_set.transcript_dict[transcript] = sorted(train_idxs)
      test_set.transcript_dict[transcript] = sorted(test_idxs)
      
    return train_set, test_set
    

if __name__ == "__main__":
  
  mode = ['basic',
          'validation',
          'distribution']

  factory = ParallelDatasetFactory(melspec_dir = "../Data/processed/VCTK/melspec", 
                                    transcript_dir = "../Data/processed/VCTK/transcript_standardized",
                                    speaker_info_path = "../Data/processed/VCTK/speaker_info.csv")
  
  dataset = factory.get_dataset()
  
  loader = DataLoader(dataset, batch_size = 8, shuffle = True, collate_fn = collate_fn)
    
  # Regular data loading
  if mode == 'basic':
    pass

  #=== Test/Validation Split ===#
  if mode == 'validation':
    # Test split
    print(len(dataset))
    train_set, test_set = dataset.train_test_split(sample_pairing='rand')
    print(len(test_set), len(train_set))
    print(len(test_set) + len(train_set))
    
    train_idxs = np.concatenate(list(train_set.transcript_dict.values()))
    test_idxs  = np.concatenate(list(test_set.transcript_dict.values()))
    print(set(train_idxs) & set(test_idxs))
    print(len(set(train_idxs) | set(test_idxs)))
  
  # Sentence sample distribution
  if mode == 'distribution':
    dataset = ParallelMelspecDataset(melspec_dir = "../Data/processed/VCTK/melspec", 
                                      transcript_dir = "../Data/processed/VCTK/transcript_standardized",
                                      speaker_info_path = "../Data/processed/VCTK/speaker_info.csv",
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