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


######################
### Loss Functions ###
######################

def mse_loss(prd_mel: Tensor, 
             tgt_mel: Tensor,
             tgt_mask: Tensor):

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


def mae_loss(prd_mel: Tensor, 
             tgt_mel: Tensor,
             tgt_mask: Tensor):

  n_mels = prd_mel.shape[1]

  #=== Masked MSE loss ===#
  
  # Match mask shape to melspec shape.
  full_tgt_mask = tgt_mask.repeat(1, n_mels, 1)
  # Correct frame offsets, calc element-wise quadratic error, and mask error in padded frames.
  masked_elem_loss = full_tgt_mask * (pred_mel_pe[:,:,:-1] - tgt_mel_pe[:,:,1:]) ** 2
  # Calculate mean over only the mel-dimension.
  masked_mel_dim_loss = torch.mean(masked_elem_loss, 1)
  # Calculate mean over only non-masked frames.
  masked_mae_loss = torch.sum(masked_mel_dim_loss) / torch.sum(tgt_mask)
  
  return masked_mae_loss

def auxil_att_loss(A: Tensor):
  pass

def masked_rbf_kernel_matrix(n_rows, n_cols, n_rows_on, n_cols_on, sigma):
  """
  Create a matrix using the RBF (Gaussian) kernel of shape [n_rows, n_cols].
  
  Args:
      n_rows: Number of rows in the output matrix
      n_cols: Number of columns in the output matrix
      n_rows_on: How many of the rows are masked "on" (i.e. with 1)
      n_cols_on: How many of the cols are masked "on" (i.e. with 1)
      sigma: Bandwidth parameter that controls the width of the Gaussian
      
  Returns: 
      A matrix where element [i,j] is computed using the RBF kernel
  """
  
  row_indices = torch.arange(n_rows, device=device) / (n_rows_on if n_rows_on > 1 else 1)
  col_indices = torch.arange(n_cols, device=device) / (n_cols_on if n_cols_on > 1 else 1)
  
  # Create coordinate matrices
  row_mat = row_indices.view(-1, 1).expand(-1, n_cols)
  col_mat = col_indices.view(1, -1).expand(n_rows, -1)
  
  # Calculate the RBF kernel matrix
  # 1 - exp(-((x-y)²/(2σ²)))
  masked_kernel_matrix = 1. - torch.exp(-((row_mat - col_mat) ** 2) / (2. * sigma ** 2))
  # Mask
  masked_kernel_matrix[n_rows_on:, n_cols_on:] = 0.
  
  return masked_kernel_matrix

def diag_att_loss(A: Tensor, src_mask: Tensor, tgt_mask: Tensor, tgt_sigma = 0.3):
  batch_size, n_src_frames, n_tgt_frames = A.shape
  
  target_distance_matrices = torch.zeros_like(A)
  for bi in range(batch_size):
    # Nr. of "masked on" frames. i.e. frames with mask=1
    n_src_frames_on = int(torch.sum(src_mask[bi,:,:]))
    n_tgt_frames_on = int(torch.sum(tgt_mask[bi,:,:]))
    
    target_distance_matrices[bi] = masked_rbf_kernel_matrix(n_rows=n_src_frames,
                                                            n_cols=n_tgt_frames,
                                                            n_rows_on=n_src_frames_on,
                                                            n_cols_on=n_tgt_frames_on,
                                                            sigma=tgt_sigma)
  
  # TODO: Why no division over the sum of the src frames?
  diag_att_loss = torch.sum(torch.mean(A * target_distance_matrices, 1)) / torch.sum(tgt_mask)
  
  return diag_att_loss

def ortho_att_loss(A: Tensor, src_mask: Tensor, tgt_mask: Tensor, tgt_sigma = 0.3):
  batch_size, n_src_frames, _ = A.shape
  
  target_distance_matrices = torch.zeros(batch_size, n_src_frames, n_src_frames)
  for bi in range(batch_size):
    # Nr. of "masked on" frames. i.e. frames with mask=1
    n_src_frames_on = int(torch.sum(src_mask[bi,:,:]))
    n_tgt_frames_on = int(torch.sum(tgt_mask[bi,:,:]))
    
    target_distance_matrices[bi] = masked_rbf_kernel_matrix(n_rows=n_src_frames,
                                                            n_cols=n_tgt_frames,
                                                            n_rows_on=n_src_frames_on,
                                                            n_cols_on=n_tgt_frames_on,
                                                            sigma=tgt_sigma)
  
  ortho_att_loss = torch.sum(torch.mean(A.dot(A.T) * target_distance_matrices, 1)) / torch.sum(tgt_mask)
  
  return ortho_att_loss


if __name__ == "__main__":
  import matplotlib.pyplot as plt
  
  rbf_mat = masked_rbf_kernel_matrix(20, 60, 15, 40, 0.3)
  print(rbf_mat.min(), rbf_mat.max())
  print(rbf_mat.shape)
  
  
  plt.imshow(rbf_mat)
  plt.show()