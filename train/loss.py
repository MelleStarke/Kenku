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

def mse_loss(prd_mel: Tensor, 
             tgt_mel: Tensor,
             tgt_mask: Tensor):
    
  device = src_mel.device
  dtype  = src_mel.dtype

  n_mels = prd_mel.shape[1]

  #=== Masked MSE loss ===#
  
  # Match mask shape to melspec shape.
  full_tgt_mask = tgt_mask.repeat(1, n_mels, 1)
  # Correct frame offsets, calc element-wise quadratic error, and mask error in padded frames.
  masked_elem_loss = full_tgt_mask * (pred_mel_pe[:,:,:-1] - tgt_mel_pe[:,:,1:]) ** 2
  # Calculate mean over only the mel-dimension.
  masked_mel_dim_loss = torch.mean(masked_elem_loss, 1)
  # Calculate mean over only non-masked frames.
  masked_mse_loss = torch.sum(masked_mel_dim_loss) / torch.sum(tgt_mask)
  
  return masked_mse_loss


def mae_loss(X: Tensor, Y: Tensor):
  pass

def auxil_att_loss(A: Tensor):
  pass

def diag_att_loss(A: Tensor, tgt_sigma = 0.3):
  masked_gauss_dist_mat = torch.zeros((batch_size, n_frames, n_frames), dtype=pred_att.dtype, device=device)
  for bi in range(batch_size):
    # Nr. of "masked on" frames. i.e. frames with mask=1
    n_src_frames_on = int(torch.sum(src_mask[bi,:,:]))
    n_tgt_frames_on = int(torch.sum(tgt_mask[bi,:,:]))
    
    src_lin_vec = torch.arange(n_frames, device=device) / n_src_frames_on
    tgt_lin_vec = torch.arange(n_frames, device=device) / n_tgt_frames_on
    
    src_vec_vstack = src_lin_vec.repeat(n_frames, 1).T
    tgt_vec_hstack = src_lin_vec.repeat(n_frames, 1)
    
    masked_gauss_dist_mat[bi,:,:] = 1. - torch.exp(-((src_vec_vstack - tgt_vec_hstack) ** 2) \
                              / (2. * gauss_width_da ** 2))
    masked_gauss_dist_mat[bi, n_src_frames_on:, n_tgt_frames_on:] = 0.
  
  diag_att_loss = torch.sum(torch.mean(pred_att * masked_gauss_dist_mat, 1)) / torch.sum(tgt_mask)
  
  pred_att_np = pred_att.detach().cpu().clone().numpy()
  
  return mse_loss, diag_att_loss, pred_att_np

def ortho_att_loss(A: Tensor, tgt_sigma = 0.3):
  pass