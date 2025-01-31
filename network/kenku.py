import numpy as np
import torch
import torch.nn as nn

from pytorch_tcn import TCN
from torch import Tensor
from torch.nn.utils.parametrizations import weight_norm
from typing import List, Tuple, Union, Optional

from modules import KameBlock, Attention


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
      
    kame_block_kwargs = {
      'num_conv_layers': num_conv_layers,
      'kernel_size': kernel_size,
      'dilations': dilations,
      'dropout_rate': dropout_rate
    }
    
    self.encoder = KameBlock(
      in_ch, conv_ch, att_ch, embed_ch, num_classes, num_output_streams=2, **kame_block_kwargs
    ) 
    self.pre_decoder = KameBlock(
      in_ch, conv_ch, att_ch, embed_ch, num_classes, **kame_block_kwargs
    )
    self.attention = Attention()
    
    self.post_decoder = KameBlock(
      att_ch, conv_ch, out_ch, embed_ch, num_classes, **kame_block_kwargs
    )
    
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