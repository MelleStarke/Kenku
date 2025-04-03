import os
import numpy as np
import logging
import torch

from torch import Tensor, is_tensor
from torch.nn import functional as F
from typing import List, Tuple, Union, Optional
from pathlib import Path


device = 'cuda' if torch.cuda.is_available() else 'cpu'

###############
### Logging ###
###############

logger = logging.getLogger(__name__)

# Get the full path to the directory containing the current file
current_file_dir = Path(__file__).parent.resolve()
logfile_path = os.path.join(current_file_dir, 'logs/loss.log')
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

def apply_position_encoding(*mels: Union[Tensor, List[Tensor]], pos_weight = 1.0):
    """
    Construct a sine-cosine positional encoding matrix similar to the one
    used in the Transformer architecture. And apply it to the tensors in mels.

    Based on the Google tensor2tensor repository, this captures the notion
    of relative position in a sequence, enabling the model to learn position
    information.

    Args:
        mels(Union[Tensor, List[Tensor]]): singular or list of 3D tensors (batch x channels x frames)

    Returns:
        Union[Tensor, List[Tensor]]: original input mels with sine/cosine positional encoding applied
    """
    
    # Handle singular or list of tensors
    single_mel = False
    
    if is_tensor(mels):
      mels = [mels]
      single_mel = True
    
    # Ensure list of tensors with same shape
    else:
      assert isinstance(mels, list) and all([is_tensor(mel) for mel in mels]), f"Expected list of tensors but got {list(map(type, mels))}"
      shape_head = mels[0].shape
      assert all([mel.shape == shape_head for mel in mels]), f"Expected all mels to have the same shape, but got {[shape_head] + [mel.shape for mel in mels]}"
    
    
    batch_size, channels, frames = mels[0].shape
    position = np.arange(frames, dtype='f')
    num_timescales = channels // 2

    # Compute logarithmically spaced time scales
    log_timescale_increment = (
        np.log(10000.0 / 1.0) /
        (float(num_timescales) - 1)
    )
    inv_timescales = 1.0 * np.exp(
        np.arange(num_timescales).astype('f') * -log_timescale_increment
    )

    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)
    
    # Compute sine for one half and cosine for the other half of the channels
    signal = np.concatenate(
        [np.sin(scaled_time), np.cos(scaled_time)], axis=1
    )
    signal = np.expand_dims(signal, axis=0)  # shape: (1, length, n_units//2 * 2)

    # Reorder dimensions to (1, n_units, length)
    pos_encoding = np.transpose(signal, (0, 2, 1))
    
    #=== Apply Encoding to Mels ===#
    
    # Nr. of sine and cosine waves in the position encoding.
    # Ensures even number.
    n_waves = num_timescales * 2
    # Scale position encodings by square root of the nr. of mel channels. 
    pos_scale = channels ** 0.5
    
    # Apply position encoding
    for mi, mel in enumerate(mels):
      mels[mi][:,:n_waves,:] = mel[:,:n_waves,:] + pos_encoding / pos_scale * pos_weight
      
    if single_mel:
      return mels[0]
    
    return mels
  

######################
### Loss Functions ###
######################

def mse_loss(src_mel: Tensor, 
             tgt_mel: Tensor, 
             src_mask: Tensor, 
             tgt_mask: Tensor, 
             pos_weight = 1.0):
  device = src_mel.device
  dtype  = src_mel.dtype


def calc_loss(self, src_mel, tgt_mel, src_mask, tgt_mask, src_info, tgt_info,
              pos_weight = 1.0, gauss_width_da = 0.3):
  
  # TODO: Authors feed source mel into forward without appending zero frame,
  #       despite prepending target zero frame. This supposedly doesn't throw an error?
  #       For now I'll just append zero frame to source so the shapes match up.
  device = src_mel.device
  dtype  = src_mel.dtype
  
  sf = self.stack_factor
  
  # Stack frames along the mel-dimension, thereby reducing the frame-dimension.
  if sf > 1:
    src_mel  = stack_frames(src_mel, sf)
    tgt_mel  = stack_frames(tgt_mel, sf)
    
    src_mask = src_mask[:,:,::sf]
    tgt_mask = tgt_mask[:,:,::sf]
    
  tgt_mel = prepend_zero_frame(tgt_mel)
  src_mel = append_zero_frame(src_mel) 
  
  batch_size, n_mels, n_frames = src_mel.shape
  
  # Construct sine and cosine functions evenly split in the mel-dimension, 
  # logarithmically distributed, and sampled at n_frames. Repeat for whole batch.
  src_pos = torch.from_numpy(position_encoding(n_frames, n_mels))\
                            .to(device, dtype)\
                            .repeat(batch_size, 1, 1)
  tgt_pos = torch.from_numpy(position_encoding(n_frames, n_mels))\
                            .to(device, dtype)\
                            .repeat(batch_size, 1, 1)

  # Nr. of sine and cosine waves in the position encoding.
  # Equal to n_mels // 2 * 2.
  n_waves = src_pos.shape[1]
  # Scale position encodings by square root of n_mels. 
  pos_scale = n_mels ** 0.5
  
  # Position encoded source spectrogram batch.
  src_mel_pe = src_mel
  src_mel_pe[:,:n_waves,:] = src_mel_pe[:,:n_waves,:] + src_pos / pos_scale * pos_weight 
  tgt_mel_pe = tgt_mel
  tgt_mel_pe[:,:n_waves,:] = tgt_mel_pe[:,:n_waves,:] + tgt_pos / pos_scale * pos_weight 
  
  # for b in range(batch_size):
  #   plt.imshow(tgt_mel_pe[b,:,:])
  #   plt.show()
    
  # exit()
  
  pred_mel_pe, pred_att = self(src_mel_pe, tgt_mel_pe, src_info, tgt_info)

  #=== Masked MSE loss ===#
  
  # Match mask shape to melspec shape.
  full_tgt_mask = tgt_mask.repeat(1, n_mels, 1)
  # Correct frame offsets, calc element-wise quadratic error, and mask error in padded frames.
  masked_elem_loss = full_tgt_mask * (pred_mel_pe[:,:,:-1] - tgt_mel_pe[:,:,1:]) ** 2
  # Calculate mean over only the mel-dimension.
  masked_mel_dim_loss = torch.mean(masked_elem_loss, 1)
  # Calculate mean over only non-masked frames.
  masked_mse_loss = torch.sum(masked_mel_dim_loss) / torch.sum(tgt_mask)
  mse_loss = masked_mse_loss


def mae_loss(X: Tensor, Y: Tensor):
  pass

def distr_att_loss(A: Tensor):
  pass

def diag_att_loss(A: Tensor, tgt_variance = 0.3):
  pass

def ortho_att_loss(A: Tensor, tgt_variance = 0.3):
  pass