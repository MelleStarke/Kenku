import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch import Tensor
from torch.nn.utils.parametrizations import weight_norm
from typing import List, Tuple, Union, Optional

from kenku.modules import KameBlock, Attention

from data.load import ParallelMelspecDataset, ParallelDatasetFactory, collate_fn


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LoggerPlaceholder():
  def warning(self, msg):
    print(f"WARN: {msg}")
  def warn(self, msg):
    print(f"WARN: {msg}")
  def info(self, msg):
    print(f"INFO: {msg}")
  def error(self, msg):
    print(f"ERROR: {msg}")
  def debug(self, msg):
    print(f"DEBUG: {msg}")

logger = LoggerPlaceholder()
  

def prepend_zero_frame(X, n_frames=1):
  if n_frames < 1:
    logger.warn("prepend_zero_frame called with 0 or fewer frames. Returning unmodified source tensor.")
    return X
  
  batch_size, n_mels, _ = X.shape
  zero_tensor = torch.zeros((batch_size, n_mels, n_frames), dtype=X.dtype, device=X.device)
  return torch.cat((zero_tensor, X), dim=2)

def append_zero_frame(X, n_frames=1):
  if n_frames < 1:
    logger.warn("append_zero_frame called with 0 or fewer frames. Returning unmodified source tensor.")
    return X
  
  batch_size, n_mels, _ = X.shape
  zero_tensor = torch.zeros((batch_size, n_mels, n_frames), dtype=X.dtype, device=X.device)
  return torch.cat((X, zero_tensor), dim=2)

def stack_frames(X, stack_factor):
  sf = stack_factor
  batch_size, n_mels, n_frames = X.shape
  
  # How many extra empty frames we need for n_frames to be a multiple of stack_factor.
  # Such that we can reduce n_frames and multiply n_mels by this factor.
  n_frames_short = -n_frames % sf
  
  if n_frames_short > 0:
    X = append_zero_frame(X, n_frames=n_frames_short)
  
  n_frames += n_frames_short
  X = X.permute(0,2,1).reshape(batch_size, n_frames // sf, n_mels * sf).permute(0,2,1)
  return X

def unstack_frames(X, stack_factor):
  sf = stack_factor
  batch_size, n_mels, n_frames = X.shape
  
  X = X.permute(0,2,1).reshape(batch_size, n_frames * sf, n_mels // sf).permute(0,2,1)
  return X

def position_encoding(length, n_units):
    """
    Construct a sine-cosine positional encoding matrix similar to the one
    used in the Transformer architecture.

    Based on the Google tensor2tensor repository, this captures the notion
    of relative position in a sequence, enabling the model to learn position
    information.

    Args:
        length (int): Maximum sequence length.
        n_units (int): Dimensionality of the encoding (channel dimension).

    Returns:
        np.ndarray: Position encoding array of shape (1, n_units, length).
    """
    channels = n_units
    position = np.arange(length, dtype='f')
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
    position_encoding_block = np.transpose(signal, (0, 2, 1))
    return position_encoding_block


class KenkuTeacher(nn.Module):
  
  def __init__(self, 
               in_ch: int,
               conv_ch: int,
               att_ch: int,
               out_ch: int,
               embed_ch: int,
               num_accents: int,
               num_conv_layers: Optional[int] = 8,
               kernel_size: Optional[int] = 5,
               dilations: Optional[List[int]] = None,
               dropout_rate: Optional[float] = 0.1,
    ):
    super(KenkuTeacher, self).__init__()
      
    self.init_args = (in_ch, conv_ch, att_ch, out_ch, embed_ch, num_accents)
    self.init_kwargs = {
      'num_conv_layers': num_conv_layers,
      'kernel_size': kernel_size,
      'dilations': dilations,
      'dropout_rate': dropout_rate
    }
    
    self.encoder = KameBlock(
      in_ch, conv_ch, att_ch, embed_ch, num_accents, num_output_streams=2, **self.init_kwargs
    ) 
    self.pre_decoder = KameBlock(
      in_ch, conv_ch, att_ch, embed_ch, num_accents, **self.init_kwargs
    )
    self.attention = Attention()
    
    self.post_decoder = KameBlock(
      att_ch, conv_ch, out_ch, embed_ch, num_accents, **self.init_kwargs
    )
    
    self.main_loss_fn = torch.nn.MSELoss()
    
  def forward(self, X_src, X_tgt, k_src, k_tgt):
    # future_KV = torch.jit.fork(self.encoder,     (X_src, k_src))
    # future_Q  = torch.jit.fork(self.pre_decoder, (X_tgt, k_tgt))

    # K, V = torch.jit.wait(future_KV)
    # Q    = torch.jit.wait(future_Q)
    
    K, V = self.encoder(X_src, k_src)
    Q    = self.pre_decoder(X_tgt, k_tgt)
    
    R, A = self.attention(K, V, Q)
    Y = self.post_decoder(R, k_tgt)
    
    return Y, A
  
  def clear_paddings(self):
    self.encoder.clear_paddings()
    self.pre_decoder.clear_paddings()
    self.post_decoder.clear_paddings()
  
  def to_student(self, student_kwargs):
    student = KenkuStudent(*self.init_kwargs, 
                           **{**self.init_kwargs, **student_kwargs})
    
    tea_dict = self.state_dict()
    stu_dict = student.state_dict()
    
    tea_keys = list(tea_dict.keys())
    
    # Construct state dict from student state dict keys.
    # Module weights are copied from the teacher if available,
    # and copied from the student if not available.
    state_dict = dict([(sk, tea_dict[sk]) if sk in tea_keys else (sk, stu_dict[sk])
                       for sk in stu_dict.keys()])
    
    student.load_state_dict(state_dict)
    
    # Freeze copied weights.
    for name, param in student.named_parameters():
      if name in tea_keys:
        param.requires_grad = False
    
    return student
    
  
  def calc_loss(self, src_mel, tgt_mel, src_mask, tgt_mask, src_info, tgt_info,
                pos_weight = 1.0, gauss_width_da = 0.3, stack_factor = 4):
    
    # TODO: Authors feed source mel into forward without appending zero frame,
    #       despite prepending target zero frame. This supposedly doesn't throw an error?
    #       For now I'll just append zero frame to source so the shapes match up.
    device = src_mel.device
    dtype  = src_mel.dtype
    
    sf = stack_factor
    
    # Stack frames along the mel-dimension, thereby reducing the frame-dimension.
    if stack_factor > 1:
      src_mel  = stack_frames(src_mel, stack_factor)
      tgt_mel  = stack_frames(tgt_mel, stack_factor)
      
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
    
    #=== Attention Loss ===#
    
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


class KenkuStudent(nn.Module):
  pass


if __name__ == "__main__":
  from torch.utils.data import DataLoader
  
  torch.set_default_device("cuda:0")
  
  factory = ParallelDatasetFactory(dataset_dir = "../Data/processed/VCTK")
  
  dataset = factory.get_dataset(min_transcript_samples=100)
  
  loader = DataLoader(
    dataset, 
    batch_size=8,
    shuffle=True,
    num_workers=0,  # Set to 0 or os.cpu_count() depending on the environment
    drop_last=True,
    collate_fn=collate_fn,
    generator=torch.Generator(device='cuda'),
  )
  
  ch = 80
  sf = 1
  model = KenkuTeacher(ch * sf, ch * sf, ch * sf, ch * sf, 12, 11)
  
  src_mel, tgt_mel, _, _, src_info, tgt_info = next(iter(loader))
  _ = model(src_mel, tgt_mel, src_info, tgt_info)
  print(_[0].shape)
  # loss = model.calc_loss(*batch, stack_factor=sf)
  