#%%
import os
import numpy as np
import logging
import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from typing import List, Tuple, Union, Optional
from pathlib import Path


device = 'cuda' if torch.cuda.is_available() else 'cpu'

###############
### Logging ###
###############

logger = logging.getLogger(__name__)

# Get the full path to the directory containing the current file
current_file_dir = Path(__file__).parent.resolve()
logfile_path = os.path.join(current_file_dir, 'logs/modules.log')
os.makedirs(os.path.dirname(logfile_path), exist_ok=True)

# Configure file handler
logfile_handler = logging.FileHandler(logfile_path, mode = 'a')
logfile_handler.setLevel(logging.DEBUG)
logger.addHandler(logfile_handler)

# Configure logging format
log_formatter = logging.Formatter(fmt     = '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s',
                                  datefmt = '%m/%d/%Y %I:%M:%S')
logfile_handler.setFormatter(log_formatter)


def calc_padding(kernel_size, dilation, stride=1):
  """
  Calculates how much padding should be prepended on one side of a sequence.
  """
  return (kernel_size-1)*dilation + 1 - stride

def concat_embedding(emb, X):
  """
  Concatenates embedding along X dim 1 (channel dimension).
  """
  return torch.cat((emb, X), dim=1)


class KameEmbedding(nn.Module):
  def __init__(self, n_accents, n_channels):
    super(KameEmbedding, self).__init__()
    
    self.n_accents = n_accents
    self.layer = nn.Linear(n_accents + 2, n_channels).to(device)
    
  def forward(self, ages, genders, accents):
    batch_size = len(ages)
    
    assert batch_size == len(genders) == len(accents), (
      f"Batch size mismatch between age, gender, and accent vector \n" 
      f"{(len(age), len(gender), len(accent))}"
    )
     
    ages       = ages.unsqueeze(-1)
    genders    = genders.unsqueeze(-1)
    accents_oh = F.one_hot(accents, num_classes=self.n_accents).to(dtype=torch.float)
    
    encoded_speaker_info = torch.cat([ages, genders, accents_oh], dim=1).to(device)
    return self.layer(encoded_speaker_info)
    
  def __vectorize_age_pe(self, age: Union[List[int], Tensor, int]):
    """vectorizes age through position encoding.

    Args:
        age (List[int] or Tensor[int], or int): age
    """
    if not torch.is_tensor(age):
      age = torch.tensor(age, dtype=torch.float, device=device)
    
    # Ensure tensor of (batch_size, 1) even if age is a single value
    while age.ndim < 2:
      age = age.unsqueeze(-1)
        
    n_waves = self.age_dim // 2
    age_lowbound, age_hibound = self.age_bounds
    age_range = age_hibound - age_lowbound
   
    # List of multiples of pi from 0.5pi to 2pi, exponentially distributed.
    # Ex: if n_waves = 3, then wave_freqs = [0.5pi, 1pi, 2pi]
    wave_freqs = 2 ** torch.linspace(-1, 1, n_waves, dtype=torch.float) * torch.pi
    wave_freqs = wave_freqs.unsqueeze(0).to(device)
    # x-scaled and x-shifted sin/cos inputs to match the age bounds.
    sincos_inputs = wave_freqs * (age - age_lowbound) / age_range
    
    sin_age_encoding = torch.sin(sincos_inputs)
    cos_age_encoding = torch.cos(sincos_inputs)
    
    age_encoding = torch.cat([sin_age_encoding, cos_age_encoding], dim = 1)
    
    return age_encoding
    
    
class KameBlock(nn.Module):
  def __init__(self,
               in_ch: int,
               conv_ch: int,
               out_ch: int,
               embed_ch: int,
               num_accents: int,
               num_conv_layers: Optional[int] = 8,
               kernel_size: Optional[int] = 5,
               num_output_streams: Optional[int] = 1,
               signal_segment_len: int = 80,
               dilations: Optional[List[int]] = None,
               dropout_rate: Optional[float] = 0.2,
    ):
    """Class for recurring structure in FastConvS2S-VC model. Based on Kameoka et al.'s model.
       Features:
         - Linear input and output layers.
         - Several 1D, causal, dilated convolutional layers followed by GLUs, with skip connections.
         - Concatenated class embeddings at every layer (linear and conv).
         - Dynamic padding, depending on kernel size and dilation. 
           Copied from previous input if possible, such that it temporally overlaps.

    Args:
        in_ch (int): nr. of features in the input signal
        conv_ch (int): nr. of convolutional channels (uniform across conv layers)
        out_ch (int): nr. of features in the output signal
        embed_ch (int): nr. of features used to embed classes
        num_accents (int): nr. of classes
        num_layers (int, optional): nr. of convolutional layers. Defaults to 8.
        kernel_size (int, optional): kernel size. Defaults to 5.
        num_output_streams (int, optional): nr of output streams. Multiplies the out_ch by this factor.
                                            Useful for getting keys and values from a single block, for example. Defaults to 1.
        signal_segment_len (int, optional): length of the segments that the input signal is broken up in. Defaults to 80.
        dilations (List[int], optional): list of dilations in the conv layers. Defaults to [1, 3, 9, 27, 1, 3, 9, 27].
        dropout_rate (float, optional): drop-out rate. Defaults to 0.1.
    """
    super(KameBlock, self).__init__()
    self.num_output_streams = num_output_streams
    
    # Keep track of the paddings each conv layer output. To be used during the next forward call.
    self.paddings = [None] * num_conv_layers
    
    if dilations is None:
      dilations = [3**(i%4) for i in range(num_conv_layers)]
      # # TODO Temp, remove if you suddenly find this
      # dilations.reverse()
      
    else:
      assert len(dilations) == num_conv_layers,\
        f"The manually provided list of dilations has a different length ({len(dilations)}) than the nr. of layers ({num_conv_layers})."
    
    #=== Layers ===#
    self.dropout     = nn.Dropout(p=dropout_rate)
    # self.embed_layer = nn.Embedding(num_accents, embed_ch).to(device)  # Embedding of class values into class feature space.
    #                                                                    # TODO: original src code has weight norm commented out. Give a try?
    
    self.embed_layer = KameEmbedding(num_accents, embed_ch)
    
    # Use 1D Conv layer as linear layer to match up shapes better.
    self.in_layer = weight_norm(nn.Conv1d(in_channels  = in_ch + embed_ch,
                                          out_channels = conv_ch,
                                          kernel_size  = 1, 
                                          padding      = 0,
                                          device       = device))
    
    self.conv_blocks = nn.ModuleList()
    for dil in dilations:
      self.conv_blocks.append(ConvGLU(in_ch       = conv_ch + embed_ch, 
                                      out_ch      = conv_ch, 
                                      kernel_size = kernel_size, 
                                      dilation    = dil))
    
    self.out_layer = weight_norm(nn.Conv1d(in_channels  = conv_ch + embed_ch,
                                           out_channels = out_ch * num_output_streams,
                                           kernel_size  = 1, 
                                           padding      = 0,
                                           device       = device))
  
  def forward(self, X: Tensor, speaker_info: Tuple[List[int], List[str], List[str]]):
    batch_size, in_ch, timesteps = X.shape
    
    # # Take class id list, add empty dim with unsqueeze, broadcast over timesteps.
    # # Prepration for embedding layer pass and appending to input.
    # class_tensor = torch.tensor(class_id, dtype=torch.int32).unsqueeze(1).to(device).repeat(1, timesteps)
    # embedding    = self.embed_layer(class_tensor).permute(0, 2, 1)
    
    if not self.training and self.paddings[0] is not None:
      input_batch_size = len(X)
      padding_batch_size = len(self.paddings[0])
      if input_batch_size != padding_batch_size:
        logger.warning(f"Input batch size ({input_batch_size}) doesn't match padding batch size ({padding_batch_size}). Clearing paddings.")
        self.clear_paddings()
    
    age, gender, accent = speaker_info
    embedding = self.embed_layer(age, gender, accent).unsqueeze(-1).repeat(1, 1, timesteps)
    
    #=== Forward Pass ===#
    X_    = self.dropout(X)
    X_emb = concat_embedding(embedding, X_)
    X_    = self.in_layer(X_emb)
    
    # Pass through Conv GLU blocks.
    for i, layer in enumerate(self.conv_blocks):
      X_, padding = layer(X_, embedding, padding=self.paddings[i])
      
      if not self.training:
        self.paddings[i] = padding
    
    X_emb = concat_embedding(embedding, X_)
    Y = self.out_layer(X_emb)
    
    if self.num_output_streams > 1:
      # Split into multiple matrices of shape (batch_size, out_ch, timesteps).
      # For use in (K, V) split for attention module, and (Delta, Sigma, Phi) split in attention predictor
      Y_streams = torch.chunk(Y, chunks=self.num_output_streams, dim=1)
      return Y_streams  # Formatted as a tuple of tensors with equal feature/channel dims.
    
    return Y
  
  def clear_paddings(self):
    """
    Reset dynamic padding to None. Used between passing of batches.
    """
    self.paddings = [None] * len(self.conv_blocks)
  
  
class ConvGLU(nn.Module):
  """1D Convolutional GLU block. Features:
      - Speaker embedding
      - Causal padding (i.e. tail of the previous forward call's input prepended to current input)
      - Dilation
      - GLU activation (base activation function: sigmoid)
      - Skip-connection
      - Weight normalization
  """
  def __init__(self,
               in_ch: int, 
               out_ch: int, 
               kernel_size: int, 
               dilation: int): 
    super(ConvGLU, self).__init__()
      
    self.padding_len = calc_padding(kernel_size, dilation)
    self.conv = nn.Conv1d(
      in_channels  = in_ch,
      out_channels = out_ch * 2,  # Double output size for GLU (for splitting into two matrices)
      kernel_size  = kernel_size,
      dilation     = dilation,
      padding      = 0,
      device       = device
    )
    self.conv = weight_norm(self.conv)
    
  def forward(self, X: Tensor, embedding: Tensor, padding: Tensor = None):
    # TODO: maybe fiddle with doing embedding or padding first.
    # Source authors do input > embedding > padding > forw. pass > sum input and output
    X_emb = concat_embedding(embedding, X)
    
    # Pad sequence with zeros if no padding is passed.
    if padding is None:
      # Make tensor of zeros of shape (batch_size, channels) and append empty dim at the end.
      padding = torch.zeros_like(X_emb[:,:,0]).unsqueeze(-1)
      padding = padding.repeat(1, 1, self.padding_len)  # Repeat zeros over empty dim for padding.
      
    
    # Append along time dimension, prior to input X.
    X_emb_pad = torch.cat([padding, X_emb], dim=-1)
    X_        = self.conv(X_emb_pad)
    
    with torch.no_grad():  # TODO: not sure if no_grad() is useful/valid. Maybe try without.
      padding = X_emb_pad[:,:,-self.padding_len:]
      
    X_scale, X_gate = torch.chunk(X_, chunks=2, dim=1)  # Split into 2 streams for GLU
    Y = X_scale * torch.sigmoid(X_gate)  # GLU
    Y += X  # Skip-connection
    return Y, padding
  

class Attention(nn.Module):
  def forward(self, K, V, Q):
    """Manual scaled dot product attention. Since the attention matrix should be passed for loss calculation.

    Args:
        K (Tensor): Key   tensor of shape (batch, mels, frames)
        V (Tensor): Value tensor of shape (batch, mels, frames)
        Q (Tensor): Query tensor of shape (batch, mels, frames)

    Returns:
        (Tensor, Tensor): Tuple of time-warped context vector sequence R of shape (batch, mels, frames)
                          and attention matrix A of shape (batch, mels, mels).
    """
    A = nn.functional.softmax(torch.matmul(K.permute(0,2,1), Q)/np.sqrt(K.shape[1]), dim=1)
    R = torch.matmul(V,A)
    return R, A
  

class AttentionPredictor(nn.Module):
  def __init__(self,
               in_ch: int,
               conv_ch: int,
               embed_ch: int,
               num_accents: int,
               num_conv_layers: Optional[int] = 8,
               kernel_size: Optional[int] = 5,
               signal_segment_len: int = 80,
               dilations: Optional[List[int]] = None,
               dropout_rate: Optional[float] = 0.2):
      super(AttentionPredictor, self).__init__()
      
      self.pre_decoder = KameBlock(in_ch, conv_ch, 1, embed_ch, num_accents,
                                   num_conv_layers    = num_conv_layers,
                                   kernel_size        = kernel_size,
                                   num_output_streams = 3,  # One for each Gaussian parameter
                                   signal_segment_len = signal_segment_len,
                                   dilations          = dilations,
                                   dropout_rate       = dropout_rate
                                   )
      
  def forward(self, X: Tensor, speaker_info: Tuple[List[int], List[str], List[str]]):
    batch_size = len(X)
    
    # Get Gaussian parameters from pre-decoder
    mean_deltas, variances, scalars = self.pre_decoder(X, speaker_info)
    
    #=== Post-Process Gaussian Parameters ===#
    mean_deltas = torch.abs(mean_deltas)
    variances   = torch.clamp(variances, 0.001, 1.0)
    scalars     = 0.2 * torch.sigmoid(scalars) + 0.8
    
    #=== Order Means ===#
    n_frames = mean_deltas.shape[-1]
    upper_triangular_mat = torch.triu(torch.ones(n_frames, n_frames, device=device))
    means = torch.matmul(mean_deltas, upper_triangular_mat)  # Shape batch_size X out_ch(1) X n_frames



    return (mean_deltas, means, variances, scalars)
    
    

if __name__ == "__main__":
  mode = [
    'embed',
    'att pred'
  ][1]
  
  batch_size  = 16
  in_ch       = 5
  conv_ch     = 6
  out_ch      = 3
  embed_ch    = 2
  num_accents = 11
  timesteps   = 128
  
  #%%
  
  if mode == 'embed':
    
    accents = ['English', 'Scottish', 'NorthernIrish', 'Irish', 'Indian', 'Welsh', 'American', 'Canadian', 'SouthAfrican', 'Australian', 'NewZealand']
    
    age_batch = np.random.randint(10, 80, 4).tolist()
    gender_batch = np.random.choice(['m','f'], 4).tolist()
    accent_batch = np.random.choice(accents, 4).tolist()
    
    [print(batch) for batch in [age_batch, gender_batch, accent_batch]]
    
    emb = KameEmbedding(6)
    
    e = emb(age_batch, gender_batch, accent_batch)
    
    print(e)
    print(e.shape)
  
  if mode == 'att pred':
    att = AttentionPredictor(in_ch, conv_ch, embed_ch, num_accents=num_accents)
    
    X = torch.rand((batch_size, in_ch, timesteps), device=device)
    info = (torch.tensor([20] * batch_size, device=device), 
            torch.tensor([0]  * batch_size, device=device), 
            torch.tensor([0]  * batch_size, device=device)
    )
    
    #%%
    speaker_info = info
    batch_size = len(X)
    
    # Get Gaussian parameters from pre-decoder
    mean_deltas, variances, scalars = att.pre_decoder(X, speaker_info)
    
    #=== Post-Process Gaussian Parameters ===#
    mean_deltas = torch.abs(mean_deltas)
    variances   = torch.clamp(torch.abs(variances), 0.001, 1.0)
    scalars     = 0.2 * torch.sigmoid(scalars) + 0.8
    
    #=== Order Means ===#
    n_frames = mean_deltas.shape[-1]
    upper_triangular_mat = torch.triu(torch.ones(n_frames, n_frames, device=device))
    means = torch.matmul(mean_deltas, upper_triangular_mat)  # Shape batch_size X out_ch(1) X n_frames

    #%% 
    tgt_frame_idxs = torch.arange(n_frames).view(1, 1, 1, n_frames).to(device)
    unnorm_gauss_att = scalars[...,None] * torch.exp(-(tgt_frame_idxs - means[...,None])**2 / (2 * variances[...,None]**2))
    
    norm_gauss_att = unnorm_gauss_att / (unnorm_gauss_att.sum(dim=-1, keepdim=True) + 1e-6)
  # kb = KameBlock(in_ch, conv_ch, out_ch, embed_ch, num_accents)
  # X = torch.rand(batch_size, in_ch, timesteps, device=device)
  # Y = kb(X, [0] * 16)
  # print(Y)
  # print(Y.shape)
# %%
