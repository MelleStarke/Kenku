import numpy as np
import torch
import os
import h5py
import logging
import joblib
import matplotlib.pyplot as plt

from copy import copy
from csv import DictReader
from dataclasses import dataclass, field

from pathlib import Path

from typing import List, Optional, Dict, Callable

from torch.utils.data import Dataset, DataLoader

from itertools import product

from network import stack_frames


###############
### Logging ###
###############

logger = logging.getLogger(__name__)

# Get the full path to the directory containing the current file
current_file_dir = Path(__file__).parent.resolve()
logfile_path = os.path.join(current_file_dir, 'logs/load.log')
os.makedirs(os.path.dirname(logfile_path), exist_ok=True)

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
          Tuple of batches of 1.) source melspecs,     2.) target melspecs, 
                              3.) source masks,        4.) target masks,
                              5.) source speaker info, 6.) target speaker info.
  """
  
  #=== Batching ===#
  batch_size = len(batch)
  n_mels = len(batch[0][0])  # nr. of frequency features in mel-spectrogram
  
  #=== Signal Length Equalization ===#
  src_sig_lengths = [sample[0].shape[1] for sample in batch]
  tgt_sig_lengths = [sample[1].shape[1] for sample in batch]
  
  max_sig_len = max(max(src_sig_lengths), max(tgt_sig_lengths))
  
  src_melspec_batch = torch.zeros((batch_size, n_mels, max_sig_len), dtype=torch.float32)
  tgt_melspec_batch = torch.zeros((batch_size, n_mels, max_sig_len), dtype=torch.float32)
  
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
    
def augment_collate_fn(augment_fn = None):
  """
  Wraps a collate function to include data augmentation.
  Args:
    augment_fn (Callable or None): A function that takes in a source and target mel-spectrogram
                                    and returns augmented versions of them. If None, no augmentation
                                    is applied.
  """
  assert augment_fn is None or isinstance(augment_fn, Callable), 'augment_fn must be a callable or None.'
  
  if augment_fn is None:
    # Use unaugmented collate function
    return collate_fn
    
  def augmented_collate_fn(batch):
    batch = [(*augment_fn(src_mel, tgt_mel), src_info, tgt_info)
            for src_mel, tgt_mel, src_info, tgt_info in batch]
    return collate_fn(batch)
  
  return augmented_collate_fn
    
@dataclass
class MelspecSample:
  """
  Data class representing a mel-spectrogram sample and its associated metadata.
  """
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
  """
  Mixin class providing methods to encode speaker information.
  """
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
               dataset_dir: str = '../Data/processed/VCTK/',
               age_bounds = (10, 40),
               rng: Optional[np.random.Generator] = None
  ):
    """
    Factory class to create ParallelMelspecDataset instances from a dataset directory.
    Args:
      dataset_dir (str): Path to the dataset directory containing 'melspec', 'transcript', and 'speaker_info.csv'.
      age_bounds (tuple): Tuple specifying the minimum and maximum ages for scaling.
      rng (np.random.Generator or None): Random number generator for reproducibility. If None, a default generator is created.
    """
    self.age_bounds = age_bounds
    self.rng = rng if rng else np.random.default_rng(42)
    
    self.samples = []
    self.transcript_dict = {}
    
    melspec_dir       = os.path.join(dataset_dir, 'melspec')
    transcript_dir    = os.path.join(dataset_dir, 'transcript')
    speaker_info_path = os.path.join(dataset_dir, 'speaker_info.csv')
    
    #=== Load Speaker Info ===#
    
    speaker_info  = self.read_speaker_info(speaker_info_path)
    self.all_accents = np.unique([speaker['accent'] for speaker in speaker_info.values()]).tolist()
    
    #=== Create Dataset Samples ===#
    
    # Folder names of speakers used as speaker IDs
    speaker_ids = sorted([folder_name for folder_name in os.listdir(melspec_dir)
                          if   os.path.isdir(os.path.join(melspec_dir, folder_name))
                           and folder_name in speaker_info.keys()])
    
    # List of args passed to joblib.Parallel
    args = []
    
    for speaker_id in speaker_ids:
      files = os.listdir(os.path.join(melspec_dir, speaker_id))
      file_paths = [file_path for file_name in files if os.path.isfile(file_path:=os.path.join(melspec_dir, speaker_id, file_name))]
      args += [(file_path, speaker_id) for file_path in file_paths]
    
    def load_sample(file_path, speaker_id):
      try:
        info = speaker_info[speaker_id]
      except KeyError:
        logger.warning(f"Speaker {speaker_id} not found in speaker info object. Skipping samples of this speaker.")
        return None

      transcript = None
      
      transcript_path = file_path.replace(melspec_dir, transcript_dir).replace('.h5', '.txt')
      with open(transcript_path) as file:
        transcript = file.read().strip()
        
      if transcript is None:
        logger.warning(f'Error reading transcript file {transcript_path}. Skipping sample.')
        return None
      elif transcript == "":
        logger.warning(f'Transcript from file {transcript_path} was read but is empty. Skipping sample.')
        return None

      file_name = os.path.basename(file_path)
      sample = MelspecSample(id=file_name,
                             speaker_id=speaker_id,
                             melspec_path=file_path,
                             transcript=transcript,
                             age=info['age'],
                             gender=info['gender'],
                             accent=info['accent'])

      return sample
        
    samples = joblib.Parallel(n_jobs=8)(
      joblib.delayed(load_sample)(file_path, speaker_id)
        for file_path, speaker_id in args)
    
    old_len = len(samples)
    self.samples = [sample for sample in samples if sample is not None]
    new_len = len(self.samples)
    logger.info(f"Removed {old_len - new_len} None samples from dataset of length {old_len}.")
    
    # Re-organise samples into a dictionary with transcripts as keys and sample indices as values
    for si, sample in enumerate(self.samples):
      transcript = sample.transcript
      if transcript not in self.transcript_dict:
        self.transcript_dict[transcript] = []

      self.transcript_dict[transcript].append(si)
        
    # Sanity check to assert that every sample is represented in the transcript dict
    if sorted(np.concatenate(list(self.transcript_dict.values())).tolist()) != sorted(list(range(len(self.samples)))):
      raise ValueError(f"Enry indices do not line up. Sentence dict indices:\n{np.concatenate(list(self.transcript_dict.values()))}\n" 
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
  
  def train_test_split(self, min_transcript_samples: int = 3, 
                             train_set_threshold: int    = 10, 
                             sample_pairing: str         = 'product', 
                             downsample: bool            = True):
      
    if isinstance(sample_pairing, str):
      sample_pairing = (sample_pairing, sample_pairing)
    
    for sp in sample_pairing:
      assert sp.lower() in ['random', 'rand', 'product', 'prod'], \
        f"'{sample_pairing}' not a valid sample pairing mode. Choose 'random' or 'product'."
    
    train_transcript_dict = {}
    test_transcript_dict = {}
    
    for transcript, sample_idxs in self.transcript_dict.items():
      n_samples = len(sample_idxs)
      if n_samples < min_transcript_samples:
        continue
      
      elif n_samples < train_set_threshold:
        test_transcript_dict[transcript]  = sample_idxs
        
      else:
        train_transcript_dict[transcript] = sample_idxs
        
    # Downsampling
    if downsample:
      for sample_idxs in [train_transcript_dict.values(), test_transcript_dict.values()]:
        # Find the lowest amount of samples across the transcripts.
        least_samples = min([len(idx_list) for idx_list in sample_idxs])
        for idx_list in sample_idxs:
          # Delete the tail of the sample idx list after shuffling.
          # This downsamples the data such that every transcript has an equal amount of samples.
          self.rng.shuffle(idx_list)
          del(idx_list[least_samples:])
      
      # TODO: remove this assert if it's never come up.
      # Assert out of the above for loop to ensure downsampling wasn't done on a copied list but on the original.
      for sample_idxs in [train_transcript_dict.values(), test_transcript_dict.values()]:
        length_head = len(list(sample_idxs)[0])
        assert all([len(idx_list) == length_head for idx_list in sample_idxs]), (
          f"Downsampling failed. Expected all sample idx lists to have the same length " 
          f"but got lengths {list(map(len, sample_idxs))}."
        )
        
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
               age_bounds = (10, 40),
               all_accents = None,
               sample_pairing = 'random',
               rng: Optional[np.random.Generator] = None
  ):
    """
    PyTorch Dataset for parallel mel-spectrogram samples paired by transcript.
    Args:
      samples (List[MelspecSample]): List of mel-spectrogram samples.
      transcript_dict (Dict[str, List[int]]): Dictionary mapping transcripts to lists of sample indices.
      age_bounds (tuple): Tuple specifying the minimum and maximum ages for scaling.
      all_accents (List[str] or None): List of all accents in the dataset for encoding. If None, accent encoding will not work.
      sample_pairing (str): Method for pairing samples. Choose 'random' or 'product'.
      rng (np.random.Generator or None): Random number generator for reproducibility. If None, a default generator is created.
    """
    assert sample_pairing.lower() in ['random', 'rand', 'product', 'prod'], \
      f"'{sample_pairing}' not a valid sample pairing mode. Choose 'random' or 'product'."
    
    if all_accents is None:
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
      
    elif self.sample_pairing in ['rand', 'random']:
      sample_idxs = np.concatenate(list(self.transcript_dict.values()))
      tgt_sample = self.samples[sample_idxs[idx]]

      src_candidate_idxs = self.transcript_dict[tgt_sample.transcript]
      src_idx = self.rng.choice(src_candidate_idxs)
      src_sample = self.samples[src_idx]
      
    else:
      raise ValueError(f"'{self.sample_pairing}' not a valid sample pairing mode. Choose 'random' or 'product'.")
      
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
  
  def frame_stacked_collate_fn(self, stack_factor = 4):
    
    def inner(batch):
      src_mel, tgt_mel, src_mask, tgt_mask, src_info, tgt_info = collate_fn(batch)
      
      src_mel = stack_frames(src_mel, stack_factor)
      tgt_mel = stack_frames(tgt_mel, stack_factor)
      
      src_mask = src_mask[:,:,::stack_factor]
      tgt_mask = tgt_mask[:,:,::stack_factor]
    
    return inner

if __name__ == "__main__":
  
  mode = ['basic',
          'test set len',
          'sample idx lengths',
          'validation',
          'distribution'][4]

  factory = ParallelDatasetFactory(dataset_dir = "../Data/processed/VCTK")
  
  dataset = factory.get_dataset()
  
  loader = DataLoader(dataset, batch_size = 8, shuffle = True, collate_fn = collate_fn)
  
  # Regular data loading
  if mode == 'sample idx lengths':
    train_set, test_set = factory.train_test_split(min_transcript_samples=5)
    print(f"train set sample idxs lengths: {list(map(len, train_set.transcript_dict.values()))}")
    print(f"test set sample idxs lengths:  {list(map(len, test_set.transcript_dict.values()))}")
    
  
  if mode == 'test set len':
    
    for mini in range(3, 11):
      train_set, test_set = factory.train_test_split(mini, sample_pairing=('prod', 'rand'))
      print((f"{mini}:\n  TRAIN: {len(train_set)} | {len(train_set.transcript_dict)}\n" 
                     f"  TEST:  {len(test_set)} | {len(test_set.transcript_dict)}" 
                     f"  prop: {len(test_set) / len(train_set):.4}"))
    

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
    factory = ParallelDatasetFactory(dataset_dir = "../Data/processed/VCTK")
    dataset = factory.get_dataset(min_transcript_samples=0)
    # print(ds.class_dirs)
    # print([len(d) for d in ds.filenames])
    
    print(f"min: {min(sorted([len(d) for d in dataset.transcript_dict.values()]))} | max: {max(sorted([len(d) for d in dataset.transcript_dict.values()]))}")
    print(f'n sentences: {len(sorted([len(d) for d in dataset.transcript_dict.values()]))}')
    print(f'n samples: {sum(sorted([len(d) for d in dataset.transcript_dict.values()]))}')
    plt.bar(range(len(dataset.transcript_dict)), sorted([len(d) for d in dataset.transcript_dict.values()]))
    plt.ylabel("N sentence samples")
    plt.xlabel("sentence")
    plt.title("Sentence distribution (min = 20)")
    plt.show()