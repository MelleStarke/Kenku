import numpy as np
import torch

import os
import logging
from pathlib import Path

from torch import nn, Tensor

from abc import ABC, abstractmethod

from data.load import ParallelDatasetFactory, ParallelMelspecDataset, collate_fn


###############
### Logging ###
###############

logger = logging.getLogger(__name__)

# Get the full path to the directory containing the current file
current_file_dir = Path(__file__).parent.resolve()
logfile_path = os.path.join(current_file_dir, 'logs/augment.log')
os.makedirs(os.path.dirname(logfile_path), exist_ok=True)

# Configure file handler
logfile_handler = logging.FileHandler(logfile_path, mode = 'a')
logfile_handler.setLevel(logging.DEBUG)
logger.addHandler(logfile_handler)

# Configure logging format
log_formatter = logging.Formatter(fmt     = '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                                  datefmt = '%m/%d/%Y %I:%M:%S')
logfile_handler.setFormatter(log_formatter)


###############
### Utility ###
###############

rng = np.random.default_rng()

def set_seed(seed: int):
  rng = np.random.default_rng(seed)


class MelspecTransform(ABC):
    
  @abstractmethod
  def __call__(self, melspec: Tensor):
    raise NotImplementedError()
  
class RandomTimeWarp(MelspecTransform):
  """
  Applies a non-linear monotonic time warp
  by compositing several sine functions on top of each other.
  """
  def __init__(self, 
               min_sines:  int = 5,
               max_sines:  int = 7,
               min_freq: float = 1.,
               max_freq: float = 5.,
               min_amp:  float = 0.5,
               max_amp:  float = 2.,
               min_mag:  float = 0.,
               max_mag:  float = 1.
  ):
    self.min_sines = min_sines
    self.max_sines = max_sines
    self.min_freq = min_freq
    self.max_freq = max_freq
    self.min_amp = min_amp
    self.max_amp = max_amp
    self.min_mag = min_mag
    self.max_mag = max_mag
    
  def compose_random_sines(self, n_frames):
    signal = np.arange(n_frames)
    n_sines = rng.integers(self.min_sines, self.max_sines + 1)
    frequency = rng.uniform(self.min_freq, self.max_freq, n_sines) / n_frames * 2 * np.pi
    amplitude = rng.uniform(self.min_amp, self.max_amp, n_sines)
    phase = rng.uniform(0, 2 * np.pi, n_sines)
    
    sine_waves = [
      amplitude[i] * np.sin(frequency[i] * signal + phase[i])
      for i in range(n_sines)
    ]
    composite = np.sum(sine_waves, axis=0)
    
    # Normalize between 0 and 1
    y_range = composite.max() - composite.min()
    composite /= y_range
    composite -= composite.min()
    
    # Scale such that the new min and max become 0 + (mag/2) and 1 - (mag/2) respectively.
    # After accumulating, this effectively scales the derivative, serving as time warping magnitude control.
    magnitude = rng.uniform(self.min_mag, self.max_mag)
    composite = composite * magnitude + (1 - magnitude) / 2
    y_range = composite.max() - composite.min()
    return composite
  
  def create_warp_matrix(self, warped_idxs: np.ndarray):
    n_frames = len(warped_idxs)
    
    warp_matrix = np.zeros((n_frames, n_frames))
    
    for frame_idx, warped_idx in enumerate(warped_idxs):
      left_idx  = int(np.floor(warped_idx))
      right_idx = min(int(np.ceil(warped_idx)), n_frames - 1)
      
      # Calculate proportional weight
      if left_idx == right_idx:
        warp_matrix[frame_idx, left_idx] = 1.
      
      else:
        right_weight = warped_idx - left_idx
        left_weight  = 1. - right_weight
        warp_matrix[frame_idx, left_idx]  = left_weight
        warp_matrix[frame_idx, right_idx] = right_weight
        
    return warp_matrix
  
  def apply_time_warp(self, melspec, warp_matrix=None, ax=None):
    n_frames = melspec.shape[-1]
    
    if warp_matrix is None:
      composite = self.compose_random_sines(n_frames)
      triu = np.triu(np.ones((n_frames, n_frames)))
      
      warped_idxs = composite @ triu
      warped_idxs *= (n_frames - 1) / warped_idxs[-1]
      
      if ax is not None:
        ax.plot(np.arange(n_frames), warped_idxs)
      
      warp_matrix = self.create_warp_matrix(warped_idxs)
      
    warped_melspec = melspec @ warp_matrix.T
    return warped_melspec
  
  def __call__(self, melspec: Tensor, ax=None):
    return self.apply_time_warp(melspec, ax=ax)
    
  
  
class RandomClip(MelspecTransform):
  def __init__(self, min_length: int = 0, max_clip: int = 1):
    if isinstance(max_clip, float):
      assert 0.0 < max_clip < 0.5, "max_clip is used on both sides of the spectrogram. " \
      f"Therefore it should be in the range (0.0, 0.5). Got: max_clip = {max_clip}"
      
    else:
      assert min_length > max_clip * 2, "max_clip is used on both sides of the spectrogram. " \
        f"Therefore min_length should be larger than max_clip * 2. Got: min_length = {min_lenght} | max_clip = {max_clip}"
      
    self.min_length = min_length
    self.max_clip   = max_clip
    
  def __call__(self, melspec: Tensor):
    n_frames = melspec.shape[-1]
    
    max_clip = self.max_clip
    if isinstance(max_clip, float):
      max_clip = int(n_frames * self.max_clip)
    
    n_clippable_frames = n_frames - self.min_length
    n_clippable_frames = min(n_clippable_frames, max_clip * 2)
    
    if n_clippable_frames < 1:
      return melspec
    
    n_clip_left = int(np.floor(n_clippable_frames / 2))
    n_clip_right = int(np.ceil(n_clippable_frames / 2))
    
    lbound = int(rng.integers(n_clip_left  + 1))
    rbound = int(n_frames - rng.integers(n_clip_right + 1))
    
    clipped_mel = melspec[..., lbound:rbound]
    return clipped_mel
    
if __name__ == '__main__':
  import matplotlib.pyplot as plt
  # rc = RandomClip(10, 0.45)
  
  rtw = RandomTimeWarp() 
  # sig = rtw(None)
  
  # plt.plot(np.arange(128), sig)    
  # plt.show()
  
  # triu = np.triu(np.ones((128, 128), dtype=float))
  # acc = np.matmul(sig, triu)
  # acc = acc / acc[-1] * 128
  
  # plt.plot(np.arange(128), acc)
  # plt.show()
  
  dataset = ParallelDatasetFactory().get_dataset()
  dataloader = torch.utils.data.DataLoader(dataset, 10, True, collate_fn=collate_fn)

  src_mels, _, src_masks, _, _, _ = next(iter(dataloader))
  
  for i in range(10):
    mask    = src_masks[i].squeeze()
    mask_idxs = torch.arange(1, len(mask) + 1)
    mask_idxs = (mask_idxs * mask)
    max_mask_idx = int(mask_idxs.max())
    src_mel = src_mels[i,...,:max_mask_idx]
    n_frames = src_mel.shape[-1]
    
    fig, axes = plt.subplots(3, 1)
    axes[0].set_xlim(0, n_frames - 1)
    axes[0].imshow(src_mel, aspect='auto')
    
    axes[1].set_xlim(0, n_frames - 1)
    clipped_mel = rtw(src_mel, ax=axes[2])
    axes[1].imshow(clipped_mel, aspect='auto')
    
    plt.show()