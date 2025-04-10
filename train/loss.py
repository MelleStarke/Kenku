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

def mse_loss(pred_mel: Tensor, 
             tgt_mel:  Tensor,
             tgt_mask: Tensor):
  
  # Correct frame offsets, calc element-wise quadratic error, and mask error in padded frames.
  masked_elem_loss = tgt_mask * (pred_mel - tgt_mel) ** 2
  # Calculate mean over only the mel-dimension.
  masked_mel_dim_loss = torch.mean(masked_elem_loss, 1)
  # Calculate mean over only non-masked frames.
  masked_mse_loss = torch.sum(masked_mel_dim_loss) / torch.sum(tgt_mask)
  
  return masked_mse_loss


def mae_loss(prd_mel: Tensor, 
             tgt_mel: Tensor,
             tgt_mask: Tensor):

  # Correct frame offsets, calc element-wise quadratic error, and mask error in padded frames.
  masked_elem_loss = full_tgt_mask * (pred_mel_pe - tgt_mel_pe).abs()
  # Calculate mean over only the mel-dimension.
  masked_mel_dim_loss = torch.mean(masked_elem_loss, 1)
  # Calculate mean over only non-masked frames.
  masked_mae_loss = torch.sum(masked_mel_dim_loss) / torch.sum(tgt_mask)
  
  return masked_mae_loss

def auxil_att_loss(pred_means: Tensor, pred_vars: Tensor, true_A: Tensor):
  true_mean = 

def masked_gauss_dist_matrix(n_rows, n_cols, n_rows_on, n_cols_on, sigma):
  """
  Create a distance matrix using the RBF kernel.
  Mask the bottom right rectangle of shape (n_rows - n_rows_on) x (n_cols - n_cols_on)
  
  Args:
      n_rows: Number of rows in the output matrix
      n_cols: Number of columns in the output matrix
      n_rows_on: How many of the rows are masked "on" (i.e. with 1)
      n_cols_on: How many of the cols are masked "on" (i.e. with 1)
      sigma: Bandwidth parameter that controls the width of the Gaussian
      
  Returns: 
      A matrix where element [i,j] is computed using the inverse of the RBF kernel
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
  
  # Conjunctive masking (only remove the bottom right rectangle)
  masked_kernel_matrix[n_rows_on:, n_cols_on:] = 0.
  
  # Disjunctive masking (only keep the top left rectangle)
  # masked_kernel_matrix[n_rows_on:, :] = 0.
  # masked_kernel_matrix[:, n_cols_on:] = 0.
  
  return masked_kernel_matrix

def diag_att_loss(A: Tensor, src_mask: Tensor, tgt_mask: Tensor, tgt_sigma = 0.3):
  batch_size, n_src_frames, n_tgt_frames = A.shape
  
  target_distance_matrices = torch.zeros_like(A)
  for bi in range(batch_size):
    # Nr. of "masked on" frames. i.e. frames with mask=1
    n_src_frames_on = int(torch.sum(src_mask[bi,:,:]))
    n_tgt_frames_on = int(torch.sum(tgt_mask[bi,:,:]))
    
    target_distance_matrices[bi] = masked_gauss_dist_matrix(n_rows=n_src_frames,
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
    
    target_distance_matrices[bi] = masked_gauss_dist_matrix(n_rows=n_src_frames,
                                                            n_cols=n_tgt_frames,
                                                            n_rows_on=n_src_frames_on,
                                                            n_cols_on=n_tgt_frames_on,
                                                            sigma=tgt_sigma)
  
  ortho_att_loss = torch.sum(torch.mean(A.dot(A.T) * target_distance_matrices, 1)) / torch.sum(tgt_mask)
  
  return ortho_att_loss


if __name__ == "__main__":
  import matplotlib.pyplot as plt
  from data.load import ParallelDatasetFactory, collate_fn
  from torch.utils.data import DataLoader
  from network import append_zero_frame, prepend_zero_frame
  
  mode = [
    'attention_masking'
  ][0]
  
  dataset_factory = ParallelDatasetFactory(dataset_dir = '../Data/processed/VCTK')
  
  train_set, test_set = dataset_factory.train_test_split(min_transcript_samples = 8,
                                                         train_set_threshold    = 10,
                                                         sample_pairing         = 'random',
                                                         downsample             = True)

  data_loader_kwargs = {
    'batch_size'  : 6,
    'shuffle'     : True,
    'num_workers' : 12,
    'collate_fn'  : collate_fn,
    'drop_last'   : True,
    'pin_memory'  : True
  }
  train_loader = DataLoader(train_set, **data_loader_kwargs)
  test_loader  = DataLoader(test_set,  **data_loader_kwargs)
  
  
  if mode == 'attention_masking':
    src_mel, tgt_mel, src_mask, tgt_mask, src_info, tgt_info = next(iter(test_loader))
    tgt_mel = prepend_zero_frame(tgt_mel)
    batch = src_mel, tgt_mel, src_mask, tgt_mask, src_info, tgt_info
    [print(t.shape) if is_tensor(t) else print([e.shape for e in t]) for t in batch]

    print(mse_loss(src_mel, tgt_mel[:,:,1:], tgt_mask))

    rbf_mat = masked_rbf_kernel_matrix(20, 60, 15, 40, 0.3)
    print(rbf_mat.min(), rbf_mat.max())
    print(rbf_mat.shape)


    plt.imshow(rbf_mat.detach().cpu().numpy())
    plt.show()

    N = src_mel.shape[2]
    T = tgt_mel.shape[2]

    print(f"N: {N} | T: {T}")

    W = np.zeros((6,N,T))
    for b in range(6):
      Nb = int(torch.sum(src_mask[b,:,:]))
      Tb = int(torch.sum(tgt_mask[b,:,:]))
      nN = np.arange(0,N)/Nb
      tT = np.arange(0,T)/Tb
      nN_tiled = np.tile(nN[:,np.newaxis], (1,T))
      tT_tiled = np.tile(tT[np.newaxis,:], (N,1))
      W[b,:,:] = 1.0-np.exp(-np.square(nN_tiled - tT_tiled)/(2.0*0.3**2))
      W[b,Nb:N,Tb:T] = 0.

      plt.imshow(W[b], aspect='auto')
      plt.show()