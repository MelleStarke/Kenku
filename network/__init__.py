import os
import numpy as np
import logging
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch import Tensor, Generator
from torch.nn.utils.parametrizations import weight_norm
from typing import List, Tuple, Union, Optional
from pathlib import Path

from network.modules import KameBlock, ScaledDotProductAttention, AttentionPredictor

from train.loss import mse_loss, mae_loss, auxil_att_loss, diag_att_loss, ortho_att_loss


device = 'cuda' if torch.cuda.is_available() else 'cpu'

###############
### Logging ###
###############

logger = logging.getLogger(__name__)

# Get the full path to the directory containing the current file
current_file_dir = Path(__file__).parent.resolve()
logfile_path = os.path.join(current_file_dir, 'logs/network.log')
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


##############
### Models ###
##############

class KenkuModel(nn.Module):
  def __init__():
    kame_block_kwargs = {
      'num_conv_layers': num_conv_layers,
      'kernel_size': kernel_size,
      'dilations': dilations,
      'dropout_rate': dropout_rate
    }
    
    self.inference = False
    
    self.stack_factor = stack_factor
    sf = stack_factor
    
    self.src_encoder = KameBlock(
      in_ch * sf, conv_ch, att_ch, embed_ch, num_accents, num_output_streams=2, **kame_block_kwargs
    ) 
    self.tgt_encoder = KameBlock(
      in_ch * sf, conv_ch, att_ch, embed_ch, num_accents, **kame_block_kwargs
    )
    self.attention = ScaledDotProductAttention()
    
    self.decoder = KameBlock(
      att_ch, conv_ch, out_ch * sf, embed_ch, num_accents, **kame_block_kwargs
    )
    
  def train(self):
    self.inference = False
    super(KenkuModel, self).train()
    
  def eval(self):
    self.inference = False
    super(KenkuModel, self).eval()
  
  def infer(self):
    self.inference = True
    super(KenkuModel, self).eval()
    
  def encode_inputs(self, src_mel, tgt_mel, src_info, tgt_info, stack = True):
    # future_KV = torch.jit.fork(self.src_encoder,     (X_src, k_src))
    # future_Q  = torch.jit.fork(self.tgt_encoder, (X_tgt, k_tgt))

    # K, V = torch.jit.wait(future_KV)
    # Q    = torch.jit.wait(future_Q)
    
    # Automatically warn and clear dynamic paddings if the batch sizes don't line up
    # between the stored paddings and the input batch.
    if self.src_encoder.paddings[0] is not None:
      input_batch_size = len(src_mel)
      padding_batch_size = len(self.src_encoder.paddings[0])
      if input_batch_size != padding_batch_size:
        logger.warning(f"Input batch size ({input_batch_size}) doesn't match padding batch size ({padding_batch_size}). Clearing paddings.")
        self.clear_paddings()
    
    if stack:
      src_mel = stack_frames(src_mel, self.stack_factor)
      tgt_mel = stack_frames(tgt_mel, self.stack_factor)
    
    # Encoding step of forward pass
    K, V = self.src_encoder(src_mel, src_info)
    Q    = self.tgt_encoder(tgt_mel, tgt_info)
    
    return K, V, Q
  
  def clear_paddings(self):
    self.src_encoder.clear_paddings()
    self.tgt_encoder.clear_paddings()
    self.decoder.clear_paddings()
    
class KenkuTeacher(KenkuModel):
  
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
               dropout_rate: Optional[float] = 0.2,
               stack_factor: int = 4
    ):
    super(KenkuTeacher, self).__init__(
      in_ch, conv_ch, att_ch, out_ch, embed_ch, num_accents,              
      num_conv_layers = num_conv_layers,
      kernel_size     = kernel_size,
      dilations       = dilations,
      dropout_rate    = dropout_rate,
      stack_factor    = stack_factor
    )
    
    self.init_args = (in_ch, conv_ch, att_ch, out_ch, embed_ch, num_accents)
    self.init_kwargs = {**kame_block_kwargs, 'stack_factor': stack_factor}
    
  def forward(self, src_mel, tgt_mel, src_info, tgt_info, stack=True):
    K, V, Q = self.encode_inputs(src_mel, tgt_mel, src_info, tgt_info, stack=stack)
    
    R, A = self.attention(K, V, Q)
    Y = self.decoder(R, k_tgt)
    
    if stack:
      Y = unstack_frames(Y, self.stack_factor)
    
    return Y, A
  
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
                main_loss_fn = 'mse', 
                pos_weight = 1.0, 
                dal_tgt_sigma = 0.3, 
                oal_tgt_sigma = 0.3, 
                lambdas = None):
    
    # TODO: Authors feed source mel into forward without appending zero frame,
    #       despite prepending target zero frame. This supposedly doesn't throw an error?
    #       For now I'll just append zero frame to source so the shapes match up.
    
    sf = self.stack_factor
    
    # Stack frames along the mel-dimension, thereby reducing the frame-dimension.
    if sf > 1:
      src_mel  = stack_frames(src_mel, sf)
      tgt_mel  = stack_frames(tgt_mel, sf)
      
      src_mask = src_mask[:,:,::sf]
      tgt_mask = tgt_mask[:,:,::sf]
      
    tgt_mel = prepend_zero_frame(tgt_mel)
    # src_mel = append_zero_frame(src_mel)
    
    batch_size, n_mels, n_frames = src_mel.shape
    
    # Position encoding
    src_mel, tgt_mel = apply_position_encoding(src_mel, tgt_mel, pos_weight=pos_weight)
    
    # Forward pass
    pred_mel, att_matrix = self(src_mel, tgt_mel, src_info, tgt_info)
    
    # Main loss term
    main_loss_fn = {
      'mse': mse_loss,
      'mae': mae_loss
    }[main_loss_fn.lower()]
    
    main_loss = main_loss_fn(pred_mel, tgt_mel, tgt_mask)
    
    # Auxiliary attention loss
    aa_loss = auxil_att_loss(A)
    # Diagonal attention loss
    da_loss = diag_att_loss(A, tgt_sigma = dal_tgt_sigma)
    # Orthogonal attention loss
    oa_loss = ortho_att_loss(A, tgt_sigma = oal_tgt_sigma)
    
    # Combine loss terms
    if lambdas is None:
      lambdas = [1, 2000, 2000]
      
    total_loss = main_loss
    
    for lam, loss_term in zip(lambdas, [aa_loss, da_loss, oa_loss]):
      total_loss += lam * loss_term
      
    return total_loss
    

class KenkuStudent(nn.Module):
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
               dropout_rate: Optional[float] = 0.2,
               stack_factor: int = 4,
               rng: Union[torch.Generator, int] = None):
    
    super(KenkuTeacher, self).__init__(
      in_ch, conv_ch, att_ch, out_ch, embed_ch, num_accents,              
      num_conv_layers = num_conv_layers,
      kernel_size     = kernel_size,
      dilations       = dilations,
      dropout_rate    = dropout_rate,
      stack_factor    = stack_factor
    )
    
    self.attention_predictor = AttentionPredictor(
      in_ch, conv_ch, embed_ch, num_accents,
      num_conv_layers = num_conv_layers,
      kernel_size     = kernel_size,
      dilations       = dilations,
      dropout_rate    = dropout_rate,
      rng             = rng
    )
    
  def forward(self, src_mel, src_info, tgt_info, stack = True):
    pass
    
  
  def calc_loss(self, src_mel, tgt_mel, src_info, tgt_info, 
                main_loss_fn = 'mse', 
                pos_weight = 1.0, 
                dal_tgt_sigma = 0.3, 
                oal_tgt_sigma = 0.3, 
                lambdas = None):
    
    sf = self.stack_factor
    
    # Stack frames along the mel-dimension, thereby reducing the frame-dimension.
    if sf > 1:
      src_mel  = stack_frames(src_mel, sf)
      tgt_mel  = stack_frames(tgt_mel, sf)
      
      src_mask = src_mask[:,:,::sf]
      tgt_mask = tgt_mask[:,:,::sf]
      
    tgt_mel = prepend_zero_frame(tgt_mel)
    # src_mel = append_zero_frame(src_mel)
    
    batch_size, n_mels, n_frames = src_mel.shape
    
    # Position encoding
    src_mel, tgt_mel = apply_position_encoding(src_mel, tgt_mel, pos_weight=pos_weight)
    
    # Manual forward pass
    K, V, Q = self.encode_inputs(src_mel, tgt_mel, src_info, tgt_info, stack = False)
    
    # Real attention matrix from the Teacher's encoders
    _, true_A = self.attention(K, V, Q)
    
    # Time-scaled sequence according to the attention predictor
    R, pred_A, pred_means, pred_vars = self.attention_predictor(src_mel, tgt_info)
    pred_mel = self.decoder(R, tgt_info)
    
    # Main loss term
    main_loss_fn = {
      'mse': mse_loss,
      'mae': mae_loss
    }[main_loss_fn.lower()]
    
    main_loss = main_loss_fn(pred_mel, tgt_mel, tgt_mask)
    
    # Auxiliary attention loss
    aa_loss = auxil_att_loss(pred_means, pred_vars, true_A)
    # Diagonal attention loss
    da_loss = diag_att_loss(pred_A, tgt_sigma = dal_tgt_sigma)
    # Orthogonal attention loss
    oa_loss = ortho_att_loss(pred_A, tgt_sigma = oal_tgt_sigma)
    
    # Combine loss terms
    if lambdas is None:
      lambdas = [1, 2000, 2000]
      
    total_loss = main_loss
    
    for lam, loss_term in zip(lambdas, [aa_loss, da_loss, oa_loss]):
      total_loss += lam * loss_term
      
    return total_loss
    

  def clear_paddings(self):
    super(KenkuStudent, self).clear_paddings()
    self.attention_predictor.encoder.clear_paddings()

if __name__ == "__main__":
  from torch.utils.data import DataLoader
  from data.load import ParallelMelspecDataset, ParallelDatasetFactory, collate_fn
  
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
  