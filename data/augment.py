import numpy as np
import matplotlib.pyplot as plt
import torch

import os
import logging
from pathlib import Path

from torch import Tensor

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
                         max_mag:  float = 0.9,):
  """
  Create a composite of random sine waves for time warping.
  Args:
      max_period: Maximum period (in frames) of the sine waves
      min_sines: Minimum number of sine waves to compose
      max_sines: Maximum number of sine waves to compose
      min_freq: Minimum frequency of sine waves
      max_freq: Maximum frequency of sine waves
      min_amp: Minimum amplitude of sine waves
      max_amp: Maximum amplitude of sine waves
      min_mag: Minimum magnitude of warping
      max_mag: Maximum magnitude of warping
  """
  n_sines = rng.integers(min_sines, max_sines + 1)
  frequency = rng.uniform(min_freq, max_freq, n_sines) / max_period * 2 * np.pi
  amplitude = rng.uniform(min_amp, max_amp, n_sines)
  phase = rng.uniform(0, 2 * np.pi, n_sines)
  magnitude = rng.uniform(min_mag, max_mag)
  
  def inner(frames):
    sine_waves = [
      amplitude[i] * np.sin(frequency[i] * (frames - phase[i]))
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

def get_augment_fns(model_type: str = 'teacher'):
  """
  Pseudo-factory function to get train and test data augmentation functions based on model type.
  
  Args:
      model_type: Type of model ('teacher' or 'student')
  """
  model_type = model_type.lower()
  
  if 'teacher' in model_type:  
    train_clip      = AlignedRandomClip()
    train_time_warp = RandomStretchedTimeWarp()
    
    def train_augment_fn(src_mel: Tensor, tgt_mel: Tensor):
      src_mel, tgt_mel = train_clip(src_mel, tgt_mel)
      src_mel = train_time_warp(src_mel)
      tgt_mel = train_time_warp(tgt_mel)
      return src_mel, tgt_mel

    # Returning None as the augment_fn will result in unaugmented data in data.load.augment_collate_fn()
    return train_augment_fn, None
  
  elif 'student' in model_type:
    train_clip      = AlignedRandomClip(keep_aligned=True)
    train_time_warp = RandomStretchedTimeWarp()
    
    # Abuse AlignedRandomClip with max_clip_ratio=0.0 to get frame-aligned spectrograms without any actual clipping
    test_clip = AlignedRandomClip(max_clip_ratio=0.0, keep_aligned=True, max_output_frames=np.inf)
    
    def train_augment_fn(src_mel: Tensor, tgt_mel: Tensor):
      src_mel, tgt_mel = train_clip(src_mel, tgt_mel)
      src_mel, tgt_mel = train_time_warp([src_mel, tgt_mel])
      return src_mel, tgt_mel
    
    def test_augment_fn(src_mel: Tensor, tgt_mel: Tensor):
      src_mel, tgt_mel = test_clip(src_mel, tgt_mel)
      return src_mel, tgt_mel
    
    return train_augment_fn, test_augment_fn
  
  else:
    raise ValueError(f"Unknown model_type: {model_type}. Supported types are 'teacher' and 'student'.")


class MelspecTransform(ABC):
    
  @abstractmethod
  def __call__(self, melspec: Tensor):
    raise NotImplementedError()
  

class AlignedRandomClip(MelspecTransform):
  """
  Aligns and randomly clips two mel-spectrograms based on their DTW alignment path.
  Spectrograms can be kept aligned frame-by-frame if desired.
  """
  def __init__(self, 
               max_clip_ratio:    float = 0.7,
               max_output_frames: int   = 500,
               keep_aligned:      bool  = False):
    """
    Args: 
        max_clip_ratio: Maximum ratio of frames to clip from either spectrogram
                        (0.0 - 1.0). E.g. 0.3 means at most 30% of frames can be clipped.
        max_output_frames: Maximum number of frames in the output spectrograms
        keep_aligned: If True, the clipped spectrograms will be aligned frame-by-frame
                       according to the DTW path.
    """
    assert 0.0 <= max_clip_ratio < 1.0, f"max_clip_ratio must be within [0.0, 1.0). Got: max_clip_ratio = {max_clip_ratio}"
    
    self.max_clip_ratio    = max_clip_ratio
    self.max_output_frames = max_output_frames 
    self.keep_aligned      = keep_aligned
    
  def __call__(self, src_mel: Tensor, tgt_mel: Tensor):
    n_src_frames = src_mel.shape[-1]
    n_tgt_frames = tgt_mel.shape[-1]
    # Compute DTW path
    _, path = compute_dtw(src_mel, tgt_mel)
    # Convert to numpy array for easy indexing
    path = np.array(path)
    
    # Get aligned indices from path
    src_indices = path[:,0]
    tgt_indices = path[:,1]
    
    # Determine path length
    path_length = len(path)
    # Set min_clip such that the frames of the clipped spectrograms doesn't exceed self.max_output_frames
    # This sets an upper limit to the nr. of frames of a spectrogram, thereby making memory usage more consistent between batches.
    min_clip = max(1, path_length - self.max_output_frames)
    max_clip = max(min_clip, int(self.max_clip_ratio * path_length))
    
    # Randomly (uniform) determine how many steps in the path
    # should be clipped from either side.
    try:
      n_clipped_steps = rng.integers(min_clip, max_clip + 1)
    except ValueError as e:
      raise ValueError(f"min_clip: {min_clip}, max_clip: {max_clip}, path_length: {path_length}") from e
    
    start_idx = rng.integers(n_clipped_steps)
    end_idx = path_length - n_clipped_steps + start_idx
    
    # Get corresponding indices in original spectrograms
    src_start, src_end = src_indices[start_idx], src_indices[end_idx]
    tgt_start, tgt_end = tgt_indices[start_idx], tgt_indices[end_idx]
    
    if self.keep_aligned:
      # For aligned output, we work directly with the DTW path indices
      # The clipped spectrograms will have the same length (end_idx - start_idx)
      # and will be aligned frame-by-frame
      
      clipped_src_indices = src_indices[start_idx:end_idx]
      clipped_tgt_indices = tgt_indices[start_idx:end_idx]
      
      # Extract frames according to the DTW alignment
      # This creates aligned spectrograms with the same shape
      clipped_src = src_mel[..., clipped_src_indices]
      clipped_tgt = tgt_mel[..., clipped_tgt_indices]
      
      return clipped_src, clipped_tgt
    
    # Get the resulting ratios of clipped frames to total frames for both spectrograms
    src_clip_ratio = 1 - (src_end - src_start) / n_src_frames
    tgt_clip_ratio = 1 - (tgt_end - tgt_start) / n_tgt_frames
    
    # Since clipping the DTW path directly can result in over-clipping the spectrograms,
    # we must account for this by continuously un-clipping (i.e. expanding) the path indices
    # until we end up with spectrograms within the specified clipping ratio.
    # This may result in over-representation of the maximum clipping amount per specotrgram.
    # Edit: after rudimentary testing over 800 samples, this seems to happen at a rate of 1/10.
    while src_clip_ratio > self.max_clip_ratio or tgt_clip_ratio > self.max_clip_ratio:
      
      expandable_left = start_idx > 0
      expandable_right = end_idx < path_length - 1
      
      # Break out of the loop if the spectrogram is already fully unclipped
      if not (expandable_left or expandable_right):
        break
      
      # Flip between expanding left and right at each iteration by maling it conditional
      # on the new amount of (un)clipped frames
      n_unclipped_steps = end_idx - start_idx
      expand_left = (n_unclipped_steps) % 2 == 0
      
      # Flip the expansion direction if there are no more frames to unclip there.
      if expand_left and not expandable_left or (expand_right:=not expand_left) and not expandable_right:
        expand_left = not expand_left
        
      # Expand (i.e. un-clip) a single frame from either side
      if expand_left:
        start_idx -= 1
      else:
        end_idx += 1
      
      # Re-calculate the new src and tgt indices and the resulting ratios
      src_start, src_end = src_indices[start_idx], src_indices[end_idx]
      tgt_start, tgt_end = tgt_indices[start_idx], tgt_indices[end_idx]
      
      src_clip_ratio = 1 - (src_end - src_start) / n_src_frames
      tgt_clip_ratio = 1 - (tgt_end - tgt_start) / n_tgt_frames
  
    # Clip the spectrograms
    clipped_src = src_mel[..., src_start:src_end]
    clipped_tgt = tgt_mel[..., tgt_start:tgt_end]     
    
    return clipped_src, clipped_tgt


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
               max_frames: int = 500,
  ):
    """
    Args:
        min_sines: Minimum number of sine waves to compose
        max_sines: Maximum number of sine waves to compose
        min_freq: Minimum frequency of sine waves
        max_freq: Maximum frequency of sine waves
        min_amp: Minimum amplitude of sine waves
        max_amp: Maximum amplitude of sine waves
        min_mag: Minimum magnitude of warping
        max_mag: Maximum magnitude of warping
        min_stretch: Minimum stretch factor
        max_stretch: Maximum stretch factor
        max_frames: Maximum number of frames in the warped spectrogram
    """
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
    self.max_frames = max_frames
  
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
  
  def apply_time_warp(self, melspecs, warp_matrix=None, ax=None):
    if not isinstance(melspecs, list):
      melspecs = [melspecs]
    
    n_src_frames = melspecs[0].shape[-1]
    
    assert all([mel.shape[-1] == n_src_frames for mel in melspecs[:-1]]), \
      f"The provided spectrograms don't have the same number of frames: {list(map(lambda mel: mel.shape[-1], melspecs))}"
    
    n_warp_frames = int(rng.uniform(self.min_stretch, self.max_stretch) * n_src_frames)
    n_warp_frames = min(n_warp_frames, self.max_frames)
    
    # Make warp matrix
    if warp_matrix is None:
      composite_fn = compose_random_sines(max_period=n_warp_frames,
                                          min_sines=self.min_sines,
                                          max_sines=self.max_sines,
                                          min_freq=self.min_freq,
                                          max_freq=self.max_freq,
                                          min_amp=self.min_amp,
                                          max_amp=self.max_amp,
                                          min_mag=self.min_mag,
                                          max_mag=self.max_mag)
      composite = composite_fn(np.arange(n_warp_frames))
        
      # Upper triangular ones matrix for accumulating sine composite
      triu = np.triu(np.ones((n_warp_frames, n_warp_frames)))
      # Accumulate sines and scale to n_src_frames
      warped_idxs = composite @ triu
      # Scale such that the highest (i.e. last) value is equal to n_src_frames - 1
      warped_idxs *= (n_src_frames - 1) / warped_idxs[-1]
      
      if ax is not None:
        ax.plot(np.arange(n_warp_frames), warped_idxs)
      
      warp_matrix = self.create_warp_matrix(warped_idxs, n_src_frames=n_src_frames)
      
    # Warp spectrograms through matrix multiplication  
    warped_melspecs = [melspec @ warp_matrix.T for melspec in melspecs]
    
    # Only return as a list of melspecs if we have more than 1 melspec
    if len(warped_melspecs) == 1:
      return warped_melspecs[0]
    return warped_melspecs
  
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
  