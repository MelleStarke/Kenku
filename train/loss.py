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

def auxil_att_loss(pred_means: Tensor, pred_stds: Tensor, true_A: Tensor):
  # TODO: Add masking
  batch_size, n_src_frames, n_tgt_frames = true_A.shape
  # Attention matrix is of shape N x M i.e. src_frames x tgt_frames
  N, M = n_src_frames, n_tgt_frames
  
  device = true_A.device
  dtype = true_A.dtype
  
  M_idxs = torch.arange(M, device=device, dtype=dtype).view(1,1,M)
  
  # Normalization constant
  true_row_sums = true_A.sum(dim=-1)  # shape: (batch_size, N)
  # Clip sum at small value to avoid division by 0
  true_row_sums = torch.clamp(true_row_sums, min=1e-8)
  
  # Eq. 13 in source paper. Calculates means from true attention matrix.
  true_means = (true_A * M_idxs).sum(dim=-1) / true_row_sums
  
  # Eq. 14 in source paper. Calculates variances from true attention matrix.
  true_vars = (true_A * (M_idxs - true_means[...,None]) ** 2).sum(dim=-1) / true_row_sums
  # Convert variance to std.
  true_stds = torch.sqrt(true_vars)
  
  # MAE loss between true and predicted means and stds
  mean_loss = (pred_means - true_means).abs()
  std_loss  = (pred_stds - true_stds).abs()
  
  return (mean_loss + std_loss).mean()
  

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

def ortho_att_loss(A: Tensor, src_mask: Tensor, tgt_sigma = 0.3):
  device = A.device
  batch_size, n_src_frames, _ = A.shape
  
  target_distance_matrices = torch.zeros(batch_size, n_src_frames, n_src_frames, device=device)
  for bi in range(batch_size):
    # Nr. of "masked on" frames. i.e. frames with mask=1
    n_src_frames_on = int(torch.sum(src_mask[bi,:,:]))
    
    target_distance_matrices[bi] = masked_gauss_dist_matrix(n_rows=n_src_frames,
                                                            n_cols=n_src_frames,
                                                            n_rows_on=n_src_frames_on,
                                                            n_cols_on=n_src_frames,
                                                            sigma=tgt_sigma)
  A_T = A.permute(0,2,1)
  ortho_att_loss = torch.sum(torch.mean(A.matmul(A_T) * target_distance_matrices, dim=1)) / torch.sum(tgt_mask)
  
  return ortho_att_loss


if __name__ == "__main__":
  import matplotlib.pyplot as plt
  from data.load import ParallelDatasetFactory, collate_fn
  from torch.utils.data import DataLoader
  from network import append_zero_frame, prepend_zero_frame
  
  mode = [
    'attention_masking',
    'auxil_loss'
  ][1]
  
  # dataset_factory = ParallelDatasetFactory(dataset_dir = '../Data/processed/VCTK')
  
  # train_set, test_set = dataset_factory.train_test_split(min_transcript_samples = 8,
  #                                                        train_set_threshold    = 10,
  #                                                        sample_pairing         = 'random',
  #                                                        downsample             = True)

  # data_loader_kwargs = {
  #   'batch_size'  : 6,
  #   'shuffle'     : True,
  #   'num_workers' : 12,
  #   'collate_fn'  : collate_fn,
  #   'drop_last'   : True,
  #   'pin_memory'  : True
  # }
  # train_loader = DataLoader(train_set, **data_loader_kwargs)
  # test_loader  = DataLoader(test_set,  **data_loader_kwargs)
  
  
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
      
  elif mode == 'auxil_loss':
    N = 1024
    true_A = torch.eye(N,N).view(1,N,N)
    pred_means = torch.arange(N).view(1,1,N)
    pred_stds  = torch.zeros(N).view(1,1,N)
    
    print(f"Test with ID matrix as attention matrix. Pred means: arange(N), pred stds: zeros. Expected: 0")
    print(auxil_att_loss(pred_means, pred_stds, true_A))
    
    idx_mat = torch.arange(N).view(N,1) - torch.arange(N).view(1,N)
    norm = torch.distributions.Normal(0,1)
    true_A = torch.exp(norm.log_prob(idx_mat)).view(1,N,N)
    
    pred_means = torch.arange(N).view(1,1,N)
    pred_stds  = torch.ones(N).view(1,1,N)
    
    print(f"Test with standard gaussian matrix. Pred means: arange(N), pred stds: ones. Expected: 0")
    print(auxil_att_loss(pred_means, pred_stds, true_A))