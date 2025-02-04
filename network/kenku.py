import numpy as np
import torch
import torch.nn as nn

from pytorch_tcn import TCN
from torch import Tensor
from torch.nn.utils.parametrizations import weight_norm
from typing import List, Tuple, Union, Optional

from modules import KameBlock, Attention, position_encoding


def prepend_zero_frame(X):
  batch_size, n_mels, _ = X.shape
  zero_tensor = torch.zeros((batch_size, n_mels, 1)).to(X.device, dtype=X.dtype)
  return torch.cat((zero_tensor, X), dim=2)

def append_zero_frame(X):
  batch_size, n_mels, _ = X.shape
  zero_tensor = torch.zeros((batch_size, n_mels, 1)).to(X.device, dtype=X.dtype)
  return torch.cat((X, zero_tensor), dim=2)


class KenkuTeacher(nn.Module):
  
  def __init__(self, 
               in_ch: int,
               conv_ch: int,
               att_ch: int,
               out_ch: int,
               embed_ch: int,
               num_classes: int,
               num_conv_layers: Optional[int] = 8,
               kernel_size: Optional[int] = 5,
               dilations: Optional[List[int]] = None,
               dropout_rate: Optional[float] = 0.1,
    ):
    super(KenkuTeacher, self).__init__()
      
    self.init_args = (in_ch, conv_ch, att_ch, out_ch, embed_ch, num_classes)
    self.init_kwargs = {
      'num_conv_layers': num_conv_layers,
      'kernel_size': kernel_size,
      'dilations': dilations,
      'dropout_rate': dropout_rate
    }
    
    self.encoder = KameBlock(
      in_ch, conv_ch, att_ch, embed_ch, num_classes, num_output_streams=2, **self.init_kwargs
    ) 
    self.pre_decoder = KameBlock(
      in_ch, conv_ch, att_ch, embed_ch, num_classes, **self.init_kwargs
    )
    self.attention = Attention()
    
    self.post_decoder = KameBlock(
      att_ch, conv_ch, out_ch, embed_ch, num_classes, **self.init_kwargs
    )
    
    self.main_loss_fn = torch.nn.MSELoss()
    
  def forward(self, X_src, X_tgt, k_src, k_tgt):
    # future_KV = torch.jit.fork(self.encoder,     (X_src, k_src))
    # future_Q  = torch.jit.fork(self.pre_decoder, (X_tgt, k_tgt))

    # K, V = torch.jit.wait(future_KV)
    # Q    = torch.jit.wait(future_Q)
    
    K, V = self.encoder(X_src, k_src)
    Q    = self.pre_decoder(X_tgt, k_tgt)
    
    R = self.attention(K, V, Q)
    Y = self.post_decoder(R, k_tgt)
    
    return Y
  
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
    
  
  def calc_loss(self, src_mel, tgt_mel, src_mask, tgt_mask, src_props, tgt_props,
                pos_weight = 1.0, gauss_width_da = 0.3, reduction_factor = 3):
    device = src_mel.device
    dtype  = src_mel.dtype
    
    batch_size, n_mels, n_frames = src_mel.shape
    
    src_mel  = append_zero_frame(src_mel)
    src_mask = append_zero_frame(src_mask)
    tgt_mel  = prepend_zero_frame(tgt_mel)
    tgt_mask = prepend_zero_frame(tgt_mask)
    
    # Construct n_mels//2 sine and cosine functions in the mel-dimension, 
    # logarithmically distributed, and sampled at n_frames. Repeat for whole batch.
    src_pos = torch.from_numpy(position_encoding(n_frames + 1, n_mels))\
                              .to(device, dtype)\
                              .repeat(batch_size, 1, 1)
    tgt_pos = torch.from_numpy(position_encoding(n_frames + 1, n_mels))\
                              .to(device, dtype)\
                              .repeat(batch_size, 1, 1)

    # Nr. of sine/cosine waves in the position encoding.
    # Equal to n_mels // 2 * 2.
    n_waves = src_pos.shape[1]
    # Scale position encodings by square root of n_mels. 
    pos_scale = n_mels ** 0.5
    
    # Position encoded source spectrogram batch.
    src_mel_pe = src_mel
    src_mel_pe[:,:n_waves,:] = src_mel_pe[:,:n_waves,:] + src_pos / pos_scale * pos_weight 
    tgt_mel_pe = tgt_mel
    tgt_mel_pe[:,:n_waves,:] = tgt_mel_pe[:,:n_waves,:] + tgt_pos / pos_scale * pos_weight 
    
    pred_mel_pe = self(src_mel_pe, tgt_mel_pe, src_props, tgt_props)

    return self.main_loss_fn(pred_mel_pe, tgt_mel_pe)


class KenkuStudent(nn.Module):
  pass