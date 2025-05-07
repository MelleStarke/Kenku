import numpy as np
import torch

import os
import logging
from pathlib import Path

from torch import nn, Tensor

from scipy.spatial.distance import cdist

from abc import ABC, abstractmethod


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


########################
### Random Generator ###
########################

rng = np.random.default_rng()

def set_seed(seed: int):
  rng = np.random.default_rng(seed)


####################
### Time Warping ###
####################

def compute_dtw(spec1, spec2):
    """
    Compute DTW alignment between two spectrograms
    
    Args:
        spec1: First spectrogram (T1 x F)
        spec2: Second spectrogram (T2 x F)
        
    Returns:
        D: Accumulated cost matrix
        P: Optimal path as list of (i, j) tuples
    """
    # Convert to numpy for processing if tensors
    if torch.is_tensor(spec1):
        spec1 = spec1.detach().cpu().numpy()
    if torch.is_tensor(spec2):
        spec2 = spec2.detach().cpu().numpy()
    
    # Compute pairwise distances between frames
    C = cdist(spec1.T, spec2.T, metric='euclidean')
    
    # Initialize accumulated cost matrix
    n, m = C.shape
    D = np.zeros((n, m))
    D[0, 0] = C[0, 0]
    
    # Fill first column and row
    for i in range(1, n):
        D[i, 0] = D[i-1, 0] + C[i, 0]
    for j in range(1, m):
        D[0, j] = D[0, j-1] + C[0, j]
    
    # Fill the rest of the matrix
    for i in range(1, n):
        for j in range(1, m):
            D[i, j] = C[i, j] + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
    
    # Traceback to find the optimal path
    i, j = n-1, m-1
    path = [(i, j)]
    
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            argmin = np.argmin([D[i-1, j], D[i, j-1], D[i-1, j-1]])
            if argmin == 0:
                i -= 1
            elif argmin == 1:
                j -= 1
            else:
                i -= 1
                j -= 1
        path.append((i, j))
    
    # Reverse path to start from (0,0)
    path.reverse()
    
    return D, path
  
def compose_random_sines(max_period: int,
                         min_sines:  int = 2,
                         max_sines:  int = 5,
                         min_freq: float = 1.,
                         max_freq: float = 5.,
                         min_amp:  float = 0.5,
                         max_amp:  float = 2.,
                         min_mag:  float = 0.1,
                         max_mag:  float = 0.9):
  n_sines = rng.integers(min_sines, max_sines + 1)
  frequency = rng.uniform(min_freq, max_freq, n_sines) / max_period * 2 * np.pi
  amplitude = rng.uniform(min_amp, max_amp, n_sines)
  phase = rng.uniform(0, 2 * np.pi, n_sines)
  magnitude = rng.uniform(min_mag, max_mag)
  
  def inner(frames):
    sine_waves = [
      amplitude[i] * np.sin(frequency[i] * frames + phase[i])
      for i in range(n_sines)
    ]
    composite = np.sum(sine_waves, axis=0)
    
    # Normalize between 0 and 1
    y_range = composite.max() - composite.min()
    composite /= y_range
    composite -= composite.min()
    
    # Scale such that the new min and max become 0 + (mag/2) and 1 - (mag/2) respectively.
    # After accumulating, this effectively scales the derivative, serving as time warping magnitude control.
    composite = composite * magnitude + (1 - magnitude) / 2
    y_range = composite.max() - composite.min()
    return composite

  return inner
  
def get_default_augment_fn():
  clip       = AlignedRandomClip()
  power_warp = RandomPowerWarp()
  time_warp  = RandomStretchedTimeWarp()
  
  def fn_composition(src_mel: Tensor, tgt_mel: Tensor):
    src_mel, tgt_mel = clip(src_mel, tgt_mel)
    src_mel, tgt_mel = power_warp(src_mel, tgt_mel)
    src_mel = time_warp(src_mel)
    tgt_mel = time_warp(tgt_mel)
    
    return src_mel, tgt_mel
  
  return fn_composition


class MelspecTransform(ABC):
    
  @abstractmethod
  def __call__(self, melspec: Tensor):
    raise NotImplementedError()
  

class AlignedRandomClip(MelspecTransform):
  def __init__(self, max_clip_ratio: float = 0.2):
    assert 0.0 < max_clip_ratio < 0.5, "max_clip_ratio is used on both sides of the spectrogram. " \
    f"Therefore it should be in the range (0.0, 0.5). Got: max_clip_ratio = {max_clip_ratio}"
    
    self.max_clip_ratio   = max_clip_ratio
    
  def __call__(self, src_mel: Tensor, tgt_mel: Tensor):
    # Compute DTW path
    _, path = compute_dtw(src_mel, tgt_mel)
    # Convert to numpy array for easy indexing
    path = np.array(path)
    
    # Get aligned indices from path
    src_indices = path[:,0]
    tgt_indices = path[:,1]
    
    # Determine path length and absolute max clip value
    path_length = len(path)
    max_clip = int(self.max_clip_ratio * path_length)
    
    # Try to find a correct clipping
    max_attempts = 100
    for attempt in range(max_attempts):
      # Choose random start point
      start_idx = rng.integers(max_clip + 1)
      end_idx = path_length - rng.integers(max_clip + 1) - 1
      
      # Get corresponding indices in original spectrograms
      src_start, src_end = src_indices[start_idx], src_indices[end_idx]
      tgt_start, tgt_end = tgt_indices[start_idx], tgt_indices[end_idx]
      
      # Check if the start indices are smaller than end indices
      if src_end - src_start <= 0 or tgt_end - tgt_start <= 0:
        continue  # Try again
      
      # Clip the spectrograms
      clipped_src = src_mel[..., src_start:src_end]
      clipped_tgt = tgt_mel[..., tgt_start:tgt_end]
      
      # Check if the frame dimension wasn't removed
      if len(clipped_src.shape) < 2 or len(clipped_tgt.shape) < 2:
        continue  # Try again
      
      clipped_src_ratio = clipped_src.shape[-1] / src_mel.shape[-1]
      clipped_tgt_ratio = clipped_tgt.shape[-1] / tgt_mel.shape[-1]
      
      if    clipped_src_ratio < 1 - self.max_clip_ratio * 2 \
         or clipped_src_ratio < 1 - self.max_clip_ratio * 2:
           continue  # Try again
      
       
      return clipped_src, clipped_tgt
    
    # Failsafe. Return unmodified spectrograms if no valid clipping was found.
    else:
      logger.warning(f"Not able to find a correct clipping after {max_attempts} attempts.")
      return src_mel, tgt_mel


class RandomStretchedTimeWarp(MelspecTransform):
  """
  Applies a non-linear monotonic time warp
  by compositing several sine functions on top of each other.
  """
  def __init__(self, 
               min_sines:     int = 2,
               max_sines:     int = 5,
               min_freq:    float = 1.,
               max_freq:    float = 5.,
               min_amp:     float = 0.5,
               max_amp:     float = 2.,
               min_mag:     float = 0.1,
               max_mag:     float = 0.9,
               min_stretch: float = 0.7,
               max_stretch: float = 1.3,
  ):
    self.min_sines = min_sines
    self.max_sines = max_sines
    self.min_freq = min_freq
    self.max_freq = max_freq
    self.min_amp = min_amp
    self.max_amp = max_amp
    self.min_mag = min_mag
    self.max_mag = max_mag
    self.min_stretch = min_stretch
    self.max_stretch = max_stretch
  
  def create_warp_matrix(self, warped_idxs: np.ndarray, n_src_frames=None):
    n_tgt_frames = len(warped_idxs)
    n_src_frames = n_tgt_frames if n_src_frames is None else n_src_frames
    
    warp_matrix = np.zeros((n_tgt_frames, n_src_frames)) 
    
    for frame_idx, warped_idx in enumerate(warped_idxs):
      left_idx  = int(np.floor(warped_idx))
      right_idx = min(int(np.ceil(warped_idx)), n_src_frames - 1)
      
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
    n_src_frames = melspec.shape[-1]
    n_tgt_frames = int(rng.uniform(self.min_stretch, self.max_stretch) * n_src_frames)
    
    if warp_matrix is None:
      composite_fn = compose_random_sines(max_period=n_tgt_frames,
                                          min_sines=self.min_sines,
                                          max_sines=self.max_sines,
                                          min_freq=self.min_freq,
                                          max_freq=self.max_freq,
                                          min_amp=self.min_amp,
                                          max_amp=self.max_amp,
                                          min_mag=self.min_mag,
                                          max_mag=self.max_mag)
      composite = composite_fn(np.arange(n_tgt_frames))
        
      # Upper triangular ones matrix for accumulating sine composite
      triu = np.triu(np.ones((n_tgt_frames, n_tgt_frames)))
      # Accumulate sines and scale to n_src_frames
      warped_idxs = composite @ triu
      # Scale such that the highest (i.e. last) value is equal to n_src_frames - 1
      warped_idxs *= (n_src_frames - 1) / warped_idxs[-1]
      
      if ax is not None:
        ax.plot(np.arange(n_tgt_frames), warped_idxs)
      
      warp_matrix = self.create_warp_matrix(warped_idxs, n_src_frames=n_src_frames)
      
    # Warp spectrogram through matrix multiplication  
    warped_melspec = melspec @ warp_matrix.T
    return warped_melspec
  
  def __call__(self, melspec: Tensor, ax=None):
    return self.apply_time_warp(melspec, ax=ax)

class RandomPowerWarp:
  """
  Applies a non-monotonic power warping transformation to mel-spectrograms.
  This transforms the power values within the spectrogram according to a 
  composite of random sinusoids, creating a non-linear mapping.
  """
  def __init__(self, 
               min_sines:     int = 2,
               max_sines:     int = 5,
               min_freq:    float = 0.5,
               max_freq:    float = 4.,
               min_amp:     float = 0.5,
               max_amp:     float = 2.,
               min_mag:     float = 0.1,
               max_mag:     float = 0.5,
               power_range_scale: float = 0.15
  ):
    """
    Initialize the power warping transformation.
    
    Args:
        min_sines: Minimum number of sine waves to compose
        max_sines: Maximum number of sine waves to compose
        min_freq: Minimum frequency of sine waves
        max_freq: Maximum frequency of sine waves
        min_amp: Minimum amplitude of sine waves
        max_amp: Maximum amplitude of sine waves
        min_mag: Minimum magnitude of warping
        max_mag: Maximum magnitude of warping
        identity_weight: Weight of the identity function (0-1)
                        Higher values make warping more subtle
        rng: Random number generator. If None, np.random is used.
    """
    self.min_sines = min_sines
    self.max_sines = max_sines
    self.min_freq = min_freq
    self.max_freq = max_freq
    self.min_amp = min_amp
    self.max_amp = max_amp
    self.min_mag = min_mag
    self.max_mag = max_mag
    self.power_range_scale = power_range_scale
    
    # Store the current warping function for visualization
    self.current_warping = None
      
  def make_power_warp_fn(self):
    """
    Generate a non-monotonic warping function for power values.
    
    Args:
        n_points: Number of points to sample
        
    Returns:
        A function that maps values in [0,1] to warped values
    """
    # Generate the composite sine wave
    composite_fn = compose_random_sines(
      max_period=1.0,
      min_sines=self.min_sines,
      max_sines=self.max_sines,
      min_freq=self.min_freq,
      max_freq=self.max_freq,
      min_amp=self.min_amp,
      max_amp=self.max_amp,
      min_mag=self.min_mag,
      max_mag=self.max_mag
    )
    
    def map_fn(power_vals: np.ndarray):
      sine_composite = composite_fn(power_vals)
      magnitude = sine_composite.max() - sine_composite.min()
      identity_weight = 1 - magnitude

      warped = identity_weight * power_vals + (1 - identity_weight) * sine_composite
      
      # Scale the warping function between [0,1]
      warped = (warped - warped.min()) / (warped.max() - warped.min())
      
      # Store for visualization
      x = np.linspace(0, 1, 200)
      y = identity_weight * x + (1 - identity_weight) * composite_fn(x)
      y = (y - y.min()) / (y.max() - y.min())
      self.current_warping = (np.linspace(0, 1, 200), y)
      
      return warped
        
    return map_fn
  
  def __call__(self, src_melspec: Tensor, tgt_melspec: Tensor, visualize=False):
    """
    Apply power warping to a mel-spectrogram.
    
    Args:
        melspec: Mel-spectrogram of shape (channels, frames) or (frames)
        visualize: If True, return a tuple (warped_melspec, warping_function)
        
    Returns:
        Warped mel-spectrogram with the same shape as input,
        and optionally the warping function
    """
    # Generate the warping function
    warp_fn = self.make_power_warp_fn()
    
    # Determine the range of the melspec
    min_val = min(src_melspec.min(), tgt_melspec.min())
    max_val = max(src_melspec.max(), tgt_melspec.max())
    val_diff = max_val - min_val
    
    if val_diff == 0:  # Handle constant spectrograms
      return src_melspec, tgt_melspec
    
    # Apply random in-/decrease of min_val and max_val proportional to self.power_range_scale
    scale = self.power_range_scale
    min_val += rng.uniform(-val_diff * scale, val_diff * scale)
    max_val += rng.uniform(-val_diff * scale, val_diff * scale)
  
    # Normalize
    src_norm_spec = (src_melspec - min_val) / val_diff
    tgt_norm_spec = (tgt_melspec - min_val) / val_diff
    # Apply warping
    src_warp_spec = warp_fn(src_norm_spec)
    tgt_warp_spec = warp_fn(tgt_norm_spec)
    # Rescale
    src_warp_spec = src_warp_spec * val_diff + min_val
    tgt_warp_spec = tgt_warp_spec * val_diff + min_val
    
    if visualize:
      return src_warp_spec, tgt_warp_spec, self.current_warping
    
    return src_warp_spec, tgt_warp_spec
  
  def plot_warping_function(self):
    """
    Plot the current warping function.
    """
    if self.current_warping is None:
        # Generate a warping function first
        _ = self.make_power_warp_fn()
        
    x, warped = self.current_warping
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, warped, 'b-', linewidth=2, label='Power warping function')
    plt.plot(x, x, 'k--', alpha=0.5, label='Identity')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Input power (normalized)')
    plt.ylabel('Output power (normalized)')
    plt.title('Power Warping Function')
    plt.legend()
    plt.tight_layout()
    plt.show()
  
if __name__ == '__main__':
  import matplotlib.pyplot as plt
  from data.load import ParallelDatasetFactory, ParallelMelspecDataset, collate_fn
  
  mode = [
    'clip',
    'warp',
    'power'
  ][2]
  # rc = RandomClip(10, 0.45)
  
  rtw = RandomStretchedTimeWarp() 
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

  src_mels, tgt_mels, src_masks, tgt_masks, _, _ = next(iter(dataloader))
  
  if mode == 'clip':
    from network import append_zero_frame, prepend_zero_frame
    arc = AlignedRandomClip()
    
    for i in range(10):
      # src stuff
      mask    = src_masks[i].squeeze()
      mask_idxs = torch.arange(1, len(mask) + 1)
      mask_idxs = (mask_idxs * mask)
      max_mask_idx = int(mask_idxs.max())
      src_mel = src_mels[i,...,:max_mask_idx]
      n_src_frames = src_mel.shape[-1]
      
      # tgt stuff
      mask    = tgt_masks[i].squeeze()
      mask_idxs = torch.arange(1, len(mask) + 1)
      mask_idxs = (mask_idxs * mask)
      max_mask_idx = int(mask_idxs.max())
      tgt_mel = tgt_mels[i,...,:max_mask_idx]
      n_tgt_frames = tgt_mel.shape[-1]
      
      src_clip, tgt_clip, src_start, tgt_start = arc(src_mel, tgt_mel)
      
      fig, axes = plt.subplots(2, 2, figsize=(12, 8))
      
      # Display original spectrograms
      for j, (name, mel) in enumerate(zip(['source', 'target'], [src_mel, tgt_mel])):
        im = axes[0,j].imshow(mel, aspect='auto', origin='lower')
        axes[0,j].set_title(f"Original {name}")
        fig.colorbar(im, ax=axes[0,j], fraction=0.046, pad=0.04)
      
      # Display clipped spectrograms with correct alignment
      for j, (name, mel, clipped_mel, start_idx) in enumerate(zip(['source', 'target'], [src_mel, tgt_mel], [src_clip, tgt_clip], [src_start, tgt_start])):
        # Create full-sized array with zeros (or NaN for clearer visualization)
        aligned_clip = np.full((clipped_mel.shape[0], mel.shape[1]), np.nan)
        
        # Calculate end index
        end_idx = start_idx + clipped_mel.shape[1]
        
        # Place the clipped content at the correct offset position
        aligned_clip[:, start_idx:end_idx] = clipped_mel
        
        # Create a masked array for better visualization (only show non-NaN values)
        masked_clip = np.ma.masked_where(np.isnan(aligned_clip), aligned_clip)
        
        # Plot with the same extent as the original to align x-axis
        im = axes[1,j].imshow(masked_clip, aspect='auto', origin='lower')
        axes[1,j].set_title(f"Clipped {name} (aligned)")
        
        # Add vertical lines to show clip boundaries
        axes[1,j].axvline(x=start_idx, color='r', linestyle='--', alpha=0.7)
        axes[1,j].axvline(x=end_idx-1, color='r', linestyle='--', alpha=0.7)
        
        # Also show clip boundaries on original for reference
        axes[0,j].axvline(x=start_idx, color='r', linestyle='--', alpha=0.7)
        axes[0,j].axvline(x=end_idx-1, color='r', linestyle='--', alpha=0.7)
        
        fig.colorbar(im, ax=axes[1,j], fraction=0.046, pad=0.04)
      
      plt.tight_layout()
      plt.show()
        
    
  if mode == 'warp':
    rtw = RandomStretchedTimeWarp(min_mag=0.5)
    for i in range(10):
      mask    = src_masks[i].squeeze()
      mask_idxs = torch.arange(1, len(mask) + 1)
      mask_idxs = (mask_idxs * mask)
      max_mask_idx = int(mask_idxs.max())
      src_mel = src_mels[i,...,:max_mask_idx]
      n_frames = src_mel.shape[-1]
      
      fig, axes = plt.subplots(3, 1)
      axes[0].imshow(src_mel, aspect='auto')
      axes[0].set_title("Source")
      
      warped_mel = rtw(src_mel, ax=axes[2])
      axes[1].imshow(warped_mel, aspect='auto')
      axes[1].set_title("Warped")
      
      max_x_val = max(src_mel.shape[-1], warped_mel.shape[-1])
      for ax in axes:
        ax.set_xlim(0, max_x_val - 1)
        
      axes[2].set_xlabel("Warped index")
      axes[2].set_ylabel("Source index")  
        
      print(f" src min/max: {src_mel.min():.4}, {src_mel.max():.4}\n" f"warp min/max: {warped_mel.min():.4}, {warped_mel.max():.4}")
      plt.show()
      
  if mode == 'power':
    rpw = RandomPowerWarp(min_mag=0.1)
    
    for i in range(10):
      # src stuff
      mask    = src_masks[i].squeeze()
      mask_idxs = torch.arange(1, len(mask) + 1)
      mask_idxs = (mask_idxs * mask)
      max_mask_idx = int(mask_idxs.max())
      src_mel = src_mels[i,...,:max_mask_idx]
      n_frames = src_mel.shape[-1]
      
      # tgt stuff
      mask    = tgt_masks[i].squeeze()
      mask_idxs = torch.arange(1, len(mask) + 1)
      mask_idxs = (mask_idxs * mask)
      max_mask_idx = int(mask_idxs.max())
      tgt_mel = tgt_mels[i,...,:max_mask_idx]
      n_tgt_frames = tgt_mel.shape[-1]
      
      # Apply warping with visualization
      src_warp, tgt_warp, (x, y) = rpw(src_mel, tgt_mel, visualize=True)
      
      vmin = min(src_mel.min(), tgt_mel.min(), src_warp.min(), tgt_warp.min())
      vmax = max(src_mel.max(), tgt_mel.max(), src_warp.max(), tgt_warp.max())
      
      fig, axes = plt.subplots(3, 2)
      
      axes[0,0].imshow(src_mel, vmin=vmin, vmax=vmax, aspect='auto')
      axes[0,0].set_title("Source")
      axes[0,1].imshow(tgt_mel, vmin=vmin, vmax=vmax, aspect='auto')
      axes[0,1].set_title("Target")
      
      axes[1,0].imshow(src_warp, vmin=vmin, vmax=vmax, aspect='auto')
      axes[1,0].set_title("Warped source")
      axes[1,1].imshow(tgt_warp, vmin=vmin, vmax=vmax, aspect='auto')
      axes[1,1].set_title("Warped target")
      
      axes[2,0].plot(x, y, 'b-', linewidth=2, label='Warping function')
      axes[2,0].plot(x, x, 'k--', alpha=0.5, label='Identity')
      axes[2,0].grid(True, alpha=0.3)
      axes[2,0].set_title(f"Power Warping Function")
      axes[2,0].legend()
    
      print(f" src min/max: {src_mel.min():.4}, {src_mel.max():.4}\t" f"warp min/max: {src_warp.min():.4}, {src_warp.max():.4}")
      print(f" tgt min/max: {tgt_mel.min():.4}, {tgt_mel.max():.4}\t" f"warp min/max: {tgt_warp.min():.4}, {tgt_warp.max():.4}\n")
      
      # plt.tight_layout()
      plt.show()