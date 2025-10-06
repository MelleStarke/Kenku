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


class KenkuModule(nn.Module):
  """
  Base class for all Kenku modules. Provides a shared field 'inference', 
  to set and determine the current state of the module/network.
  """
  def __init__(self):
    super(KenkuModule, self).__init__()
    self.inference = False
    
  def attach_speaker_info_encoder(self, speaker_info_encoder):
    """
    Attaches the SpeakerInfoEncoder instance to all DRLKameEmbedding instances in the module tree.
    Doing this post-init is an ugly solution but necessary due to torch module initialization order.
    Args:
        speaker_info_encoder (SpeakerInfoEncoder): Speaker info prediction module to replace the embedding layer by.
    """
    for module in self.children():
      if isinstance(module, KenkuModule):
        module.attach_speaker_info_encoder(speaker_info_encoder)
    
  def train(self, *args, called_super=False, **kwargs):
    if not called_super:
      super().train(*args, **kwargs)
    self.inference = False
    
    for module in self.children():
      module.training = True
      if isinstance(module, KenkuModule):
        module.train(*args, **kwargs, called_super=True)
    
  def eval(self, *args, called_super=False, **kwargs):
    if not called_super:
      super().eval(*args, **kwargs)
      
    self.inference = False
    
    for module in self.children():
      if isinstance(module, KenkuModule):
        module.eval(*args, **kwargs, called_super=True)
  
  def infer(self, *args, called_super=False, **kwargs):
    if not called_super:
      super().eval(*args, **kwargs)
    self.inference = True
    
    for module in self.children():
      if isinstance(module, KenkuModule):
        module.infer(*args, **kwargs, called_super=True)

##################
### Embeddings ###
##################

class KameEmbedding(KenkuModule):
  def __init__(self, n_accents, n_channels, device=device):
    super(KameEmbedding, self).__init__()
    
    self.n_accents = n_accents
    self.layer = weight_norm(nn.Linear(n_accents + 2, n_channels)).to(device)
    
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
    

    
class MaskedGlobalPool(KenkuModule):
  def __init__(self, pooling_type: Optional[str] = 'mean'):
    super(MaskedGlobalPool, self).__init__()
    
    assert pooling_type in ['avg', 'max'], f"Pooling type {pooling_type} not supported. Use 'avg' or 'max'."
    self.pooling_type = pooling_type
    
  def forward(self, X: Tensor, mask: Tensor):
    if len(mask.shape) == 2:  # Ensure empty channel dim
      mask = mask[:,None,:]
      
    if mask.shape[-1] != X.shape[-1]:
      print(f"Mask shape {mask.shape} does not match input shape {X.shape}. Reshaping mask to match input.")
      batch_size, _, n_frames = X.shape
      mask = torch.ones((batch_size, 1, n_frames), device=X.device, dtype=X.dtype)
    
    if self.pooling_type.lower() == "avg":
      masked_sum = (X * mask).sum(dim=-1)  # (batch_size, channels)
      valid_lengths = mask.sum(dim=-1)    # (batch_size, 1)
      avg_pooled_frame = masked_sum / (valid_lengths + 1e-8)
      return avg_pooled_frame[:,:,None]  # Add empty frame dim
        
    elif self.pooling_type.lower() == "max":
      masked_x = X.masked_fill(~mask.bool(), -torch.inf)  # Replace masked values with -inf for max pooling
      return torch.max(masked_x, dim=-1)[0]
      
      
class SpeakerInfoEncoder(KenkuModule):
  def __init__(self,
               in_ch: int,
               conv_ch: int,
               out_ch: int,
               num_conv_layers: Optional[int] = 6,
               kernel_size: Optional[int] = 5,
               dilations: Optional[List[int]] = None,
               dropout_rate: Optional[float] = 0.2,
               pooling_type: Optional[str] = 'avg',
  ):
    super(SpeakerInfoEncoder, self).__init__()
    
    if dilations is None:
      dilations = [3**(i%3) for i in range(num_conv_layers)]  # 1,3,9,1,3,9...
    
    else:
      assert len(dilations) == num_conv_layers,\
        f"The manually provided list of dilations has a different length ({len(dilations)}) than the nr. of layers ({num_conv_layers})."
    
    #=== Layers ===#
    self.dropout     = nn.Dropout(p=dropout_rate)
    
    # Use 1D Conv layer as linear layer to match up shapes better.
    self.in_layer = weight_norm(nn.Conv1d(in_channels  = in_ch,
                                          out_channels = conv_ch,
                                          kernel_size  = 1, 
                                          padding      = 0,
                                          device       = device))
    
    self.conv_blocks = nn.ModuleList()
    for dil in dilations:
      self.conv_blocks.append(ConvGLU(in_ch       = conv_ch, 
                                      out_ch      = conv_ch, 
                                      kernel_size = kernel_size, 
                                      dilation    = dil))
      
    self.global_pool = MaskedGlobalPool(pooling_type=pooling_type)
    
    self.out_layer = weight_norm(nn.Conv1d(in_channels  = conv_ch,
                                           out_channels = out_ch,
                                           kernel_size  = 1,
                                           padding      = 0,
                                           device       = device))
        
  def forward(self, X: Tensor, mask: Tensor):
    device = X.device
    dtype = X.dtype
    
    batch_size, in_ch, n_frames = X.shape
    
    empty_embedding = torch.empty((batch_size, 0, n_frames), device=device, dtype=dtype)
    
    #=== Forward Pass ===#
    X_ = self.dropout(X)
    X_ = self.in_layer(X_)
    
    # Pass through Conv GLU blocks.
    for layer in self.conv_blocks:
      empty_embedding = torch.empty((batch_size, 0, n_frames), device=device, dtype=dtype)
      X_, _ = layer(X_, empty_embedding, padding=None)
    
    X_pool = self.global_pool(X_, mask)
    
    Y = self.out_layer(X_pool)
    return Y
  
class DRLKameEmbedding(KenkuModule):
  def __init__(self, n_accents: int, n_channels: int, device=device):
    super(DRLKameEmbedding, self).__init__()

    self.encoder = None
    self.lin_layer = weight_norm(nn.Linear(n_accents + 2, n_channels)).to(device)
  
  def attach_speaker_info_encoder(self, speaker_info_encoder: SpeakerInfoEncoder):
    assert isinstance(speaker_info_encoder, SpeakerInfoEncoder), f"Expected a SpeakerInfoEncoder instance. Got {type(speaker_info_encoder)}."
    assert speaker_info_encoder.out_layer.out_channels == self.lin_layer.in_features + 2, \
      f"SpeakerInfoEncoder output channels ({speaker_info_encoder.out_layer.out_channels}) do not match expected input channels ({self.lin_layer.in_features + 2})."
    
    object.__setattr__(self, 'encoder', speaker_info_encoder)
  
  def forward(self, sie_spectrogram: Tensor, sie_mask: Tensor):
    sie_embedding = self.encoder(sie_spectrogram, sie_mask).squeeze(-1)  # Remove empty time dim
    return self.lin_layer(sie_embedding)
    
###########################
### Convolutional Block ###    
###########################
 
class KameBlock(KenkuModule):
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
               use_drl: Optional[bool] = False,
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
      dilations = [3**(i%4) for i in range(num_conv_layers)]  # 1,3,9,27,1,3,9,27
      
    else:
      assert len(dilations) == num_conv_layers,\
        f"The manually provided list of dilations has a different length ({len(dilations)}) than the nr. of layers ({num_conv_layers})."
    
    #=== Layers ===#
    self.dropout     = nn.Dropout(p=dropout_rate)
    # self.embed_layer = nn.Embedding(num_accents, embed_ch).to(device)  # Embedding of class values into class feature space.
    #                                                                    # TODO: original src code has weight norm commented out. Give a try?
    
    if use_drl:
      self.embed_layer = DRLKameEmbedding(num_accents, embed_ch, device=device)
    else:
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
  
  def forward(self, X: Tensor, embedding_input: Union[Tuple[List[int], List[str], List[str]], Tuple[Tensor, Tensor]]):
    """
    Args:
      X (Tensor): Input tensor of shape (batch_size, in_ch, timesteps)
      embedding_input (Union[Tuple[List[int], List[str], List[str]], Tuple[Tensor, Tensor]]):
          Input for the embedding layer. Either a tuple of lists of true labels, or a batch of
          spectrograms and masks for the SpeakerInfoEncoder.
          
    Returns:
      Y (Tensor) or Tuple[Tensor]: Output tensor of shape (batch_size, out_ch, timesteps)
                                  or a tuple of tensors if num_output_streams > 1.
    """
    
    
    batch_size, in_ch, timesteps = X.shape
    
    # Automatically warn and clear dynamic paddings if the batch sizes don't line up
    # between the stored paddings and the input batch.
    if self.inference and self.paddings[0] is not None:
      input_batch_size = len(X)
      padding_batch_size = len(self.paddings[0])
      if input_batch_size != padding_batch_size:
        logger.warning(f"Input batch size ({input_batch_size}) doesn't match padding batch size ({padding_batch_size}). Clearing paddings.")
        self.clear_paddings()
    
    # === Get Embedding ===#
    # If we don't use DRL and have access to the true labels
    if isinstance(self.embed_layer, KameEmbedding):
      age, gender, accent = embedding_input
      embedding = self.embed_layer(age, gender, accent).unsqueeze(-1).repeat(1, 1, timesteps)
    
    # If we use DRL and need to infer the labels from a spectrogram
    elif isinstance(self.embed_layer, SpeakerInfoEncoder):
      sie_spectrogram, sie_mask = embedding_input
      embedding = self.embed_layer(sie_spectrogram, sie_mask).unsqueeze(-1).repeat(1, 1, timesteps)
    
    else:
      raise ValueError(f"Unexpected type of embedding layer: {type(self.embed_layer)}.")
    
    #=== Forward Pass ===#
    X_    = self.dropout(X)
    X_emb = concat_embedding(embedding, X_)
    X_    = self.in_layer(X_emb)
    
    # Pass through Conv GLU blocks.
    for i, layer in enumerate(self.conv_blocks):
      X_, padding = layer(X_, embedding, padding=self.paddings[i])
      
      if self.inference:
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
  
  
class ConvGLU(KenkuModule):
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
      padding = padding.expand(-1, -1, self.padding_len)  # Repeat zeros over empty dim for padding. TODO: expand may be more efficient, but repeat may be more reliable
      
    
    # Append along time dimension, prior to input X.
    X_emb_pad = torch.cat([padding, X_emb], dim=-1)
    X_        = self.conv(X_emb_pad)
    
    with torch.no_grad():  # TODO: not sure if no_grad() is useful/valid. Maybe try without.
      padding = X_emb_pad[:,:,-self.padding_len:]
      
    X_scale, X_gate = torch.chunk(X_, chunks=2, dim=1)  # Split into 2 streams for GLU
    Y = X_scale * torch.sigmoid(X_gate)  # GLU
    
    if Y.shape == X.shape:
      Y += X  # Skip-connection
      
    return Y, padding
  
#########################
### Attention Modules ###
#########################

class ScaledDotProductAttention(KenkuModule):
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
  

class AttentionPredictor(KenkuModule):
  def __init__(self,
               in_ch: int,
               conv_ch: int,
               embed_ch: int,
               num_accents: int,
               num_conv_layers: Optional[int] = 8,
               kernel_size: Optional[int] = 5,
               signal_segment_len: int = 80,  # TODO: Currently unused
               dilations: Optional[List[int]] = None,
               dropout_rate: Optional[float] = 0.2,
               rng: Union[torch.Generator, int] = None,
               use_drl: Optional[bool] = False,):
      super(AttentionPredictor, self).__init__()
      
      if rng is None:
        self.rng = torch.Generator(device=device)
      elif isinstance(rng, torch.Generator):
        self.rng = rng
      elif isinstance(rng, int):
        self.rng = torch.Generator(device=device).manual_seed(rng)
      else:
        raise ValueError(f"Expected passed generator to be of type None, int, or torch.Generator. Got {type(rng)}.")
      
      # 1 extra input channel for the random noise
      self.encoder = KameBlock(in_ch + 1, conv_ch, 1, embed_ch, num_accents,
                               num_conv_layers      = num_conv_layers,
                               kernel_size          = kernel_size,
                               num_output_streams   = 3,  # One for each Gaussian parameter
                               signal_segment_len   = signal_segment_len,
                               dilations            = dilations,
                               dropout_rate         = dropout_rate,
                               use_drl              = use_drl
                               )
      
      
  def forward(self, src_mels: Tensor, tgt_info: Tuple[List[int], List[str], List[str]], n_tgt_frames: int = None):
    device = src_mels.device
    dtype  = src_mels.dtype
    
    batch_size, _, n_src_frames = src_mels.shape
    n_tgt_frames = n_src_frames if n_tgt_frames is None else n_tgt_frames
    
    # Add Gaussian noise channel (mean=0, std=1) to facilitate many-to-one mapping
    gauss_noise_ch = torch.normal(0, 1, (batch_size, 1, n_src_frames), generator=self.rng, device=device, dtype=dtype)
    src_mels = torch.cat((gauss_noise_ch, src_mels), dim=1)
    
    # Get Gaussian parameters from the encoder
    mean_deltas, stds, scalars = self.encoder(src_mels, tgt_info)
    
    #=== Post-Process Gaussian Parameters ===#
    mean_deltas = torch.abs(mean_deltas)
    stds        = torch.clamp(torch.abs(stds), 0.001, 1.0)
    scalars     = 0.2 * torch.sigmoid(scalars) + 0.8
    
    #=== Order Means ===#
    upper_triangular_mat = torch.triu(torch.ones(n_src_frames, n_src_frames, device=device))
    means = torch.matmul(mean_deltas, upper_triangular_mat)  # Shape batch_size X out_ch(1) X n_frames

    tgt_frame_idxs   = torch.arange(n_tgt_frames).view(1, 1, n_tgt_frames).to(device)
    unnorm_gauss_att = scalars[...,None] * torch.exp(-(tgt_frame_idxs - means[...,None])**2 / (2 * stds[...,None]**2))
    
    norm_gauss_att = unnorm_gauss_att / (unnorm_gauss_att.sum(dim=-1, keepdim=True) + 1e-8)
    norm_gauss_att = norm_gauss_att.squeeze()  # Remove empty channel: B x 1 x N x M -> B x N x M
    
    return norm_gauss_att, means, stds
    

if __name__ == "__main__":
  import matplotlib.pyplot as plt
  
  mode = [
    'embed',
    'att pred',
    'encode embed',
    'masked pool'
  ][3]
  
  batch_size  = 4
  in_ch       = 5
  conv_ch     = 6
  out_ch      = 3
  embed_ch    = 2
  num_accents = 11
  timesteps   = 32
  
  
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
    def rand_tensor(*shape):
      return torch.rand(shape, device=device)
    
    def rand_info(batch_size):
      return (torch.rand(batch_size, device=device),
              torch.randint(2, (batch_size,), device=device),
              torch.randint(11, (batch_size,), device=device))
      
      
    def da_loss(A, gauss_width_da=0.3):
      batch_size, n_frames, m_frames = A.shape
      assert n_frames == m_frames, "Non-square att mat"
      device = 'cuda'  

      masked_gauss_dist_mat = torch.zeros((batch_size, n_frames, n_frames), dtype=A.dtype, device=device)
      for bi in range(batch_size):
        # Nr. of "masked on" frames. i.e. frames with mask=1
        n_src_frames_on = n_frames
        n_tgt_frames_on = m_frames
        
        src_lin_vec = torch.arange(n_frames, device=device) / n_src_frames_on
        tgt_lin_vec = torch.arange(n_frames, device=device) / n_tgt_frames_on
        
        src_vec_vstack = src_lin_vec.repeat(n_frames, 1).T
        tgt_vec_hstack = src_lin_vec.repeat(n_frames, 1)
        
        masked_gauss_dist_mat[bi,:,:] = 1. - torch.exp(-((src_vec_vstack - tgt_vec_hstack) ** 2) / (2. * gauss_width_da ** 2))
        masked_gauss_dist_mat[bi, n_src_frames_on:, n_tgt_frames_on:] = 0.

      # TODO: if all frames are masked on, then the formula below is larger than 
      #       torch.mean(A * masked_gauss_dist_mat) by a factor of batch_size. 
      #       Should this depend on batch_size? If not, can be fixed by ... / (n_tgt_frames_on * batch_size)
      #  nvm: This is handled by summing the mask tensor in the original DA loss calc.
      diag_att_loss = torch.sum(torch.mean(A * masked_gauss_dist_mat, 1)) / n_tgt_frames_on
    
      # print(A * masked_gauss_dist_mat)
    
      return diag_att_loss, masked_gauss_dist_mat
    
    att = AttentionPredictor(in_ch, conv_ch, embed_ch, num_accents=num_accents)
    
    X = torch.rand((batch_size, in_ch, timesteps), device=device)
    info = (torch.tensor([20] * batch_size, device=device), 
            torch.tensor([0]  * batch_size, device=device), 
            torch.tensor([0]  * batch_size, device=device)
    )
    
    A = att(X, info)
    att.train()
    n_frames = timesteps
    
    optimizer = torch.optim.Adam(att.parameters(),
                                 lr    = 5e-5,
                                 betas = (0.9, 0.999))
    #%%
    for _ in range(1024):
      X = rand_tensor(batch_size, in_ch, n_frames)
      info = rand_info(batch_size)
      A, _, _ = att(X, info)
      
      loss, tgt = da_loss(A)
      
      for b in range(batch_size):
        plt.imshow(tgt[b].detach().cpu().numpy())
        plt.show()
      
      att.zero_grad()
      loss.backward()
      optimizer.step()
      
  if mode == 'encode embed':
    # Example parameters
    batch_size = 4
    n_mels = 80
    max_frames = 200
    embedding_dim = 16
    
    # Create model
    model = SpeakerInfoEncoder(
      in_ch=n_mels,
      conv_ch=128,
      out_ch=embedding_dim,
      num_conv_layers=6,
      kernel_size=5,
      pooling_type="avg"
    )
    
    # Create dummy data with different lengths
    x = torch.randn(batch_size, n_mels, max_frames, dtype=torch.float32, device=device)
    
    # Create mask (simulate different sequence lengths)
    lengths = [150, 180, 120, 200]  # Different lengths per batch item
    mask = torch.zeros(batch_size, max_frames, dtype=torch.bool, device=device)
    for i, length in enumerate(lengths):
        mask[i, :length] = True
    
    # Forward pass
    embeddings = model(x, mask)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {embeddings.shape}")
    print(f"Embedding dimension: {embedding_dim}")
  # kb = KameBlock(in_ch, conv_ch, out_ch, embed_ch, num_accents)
  # X = torch.rand(batch_size, in_ch, timesteps, device=device)
  # Y = kb(X, [0] * 16)
  # print(Y)
  # print(Y.shape)
  
  if mode == 'masked pool':
    X = torch.ones((batch_size, in_ch, timesteps), dtype=torch.float32, device=device)
    X *= torch.arange(1, in_ch + 1, device=device)[None,:,None]  # Scale by channel index
    mask = torch.ones((batch_size, 1, timesteps), dtype=torch.float32, device=device)
    
    print(f"X: {X}")
    mgp = MaskedGlobalPool(pooling_type='avg')
    
    print(f'Full Mask: {mgp(X, mask)}')
    
    for b in range(batch_size):
      mask[b, :, 10 + 5 * b:] = 0.
      
    X[~mask.repeat(1,in_ch,1).bool()] *= 2.
    print(mask.to(dtype=torch.int32))
    # mask = torch.ones((batch_size, 1, timesteps), dtype=torch.float32, device=device)
    
    sie = SpeakerInfoEncoder(in_ch, conv_ch, out_ch)
    print(sie(X, mask))
    
    print(f'Staggered Mask: {mgp(X, mask)}')
    