import os
import numpy as np
import logging
import torch

from torch import Tensor, is_tensor
from typing import List, Union, Optional
from pathlib import Path

from network.modules import (KenkuModule, 
                             KameBlock, 
                             ScaledDotProductAttention, 
                             AttentionPredictor, 
                             SpeakerInfoPredictor)

from train.loss import (mse_loss, 
                        mae_loss, 
                        auxil_att_loss, 
                        diag_att_loss, 
                        ortho_att_loss, 
                        beta_tcvae_loss_terms,
                        accent_entropy_loss)


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
      assert isinstance(mels, (list, tuple)) and all([is_tensor(mel) for mel in mels]), f"Expected list of tensors but got {type(mels)(map(type, mels))}"
      shape_head = mels[0].shape[:-1]
      assert all([mel.shape[:-1] == shape_head for mel in mels]), f"Expected all mels to have the same batch size and mel channels, but got {[shape_head] + [mel.shape for mel in mels]}"
    
    device = mels[0].device
    dtype  = mels[0].dtype
    
    batch_size, channels = mels[0].shape[:-1]
    frames = max([frame_dim := mel.shape[-1] for mel in mels])
    position = torch.arange(frames, device=device, dtype=dtype)
    num_timescales = channels // 2

    # Compute logarithmically spaced time scales
    log_timescale_increment = (
        np.log(10000.0 / 1.0) /
        (float(num_timescales) - 1)
    )
    inv_timescales = 1.0 * torch.exp(
        torch.arange(num_timescales, device=device, dtype=dtype) * -log_timescale_increment
    )

    scaled_time = position.view(frames, 1) * inv_timescales.view(1, num_timescales)
    
    # Compute sine for one half and cosine for the other half of the channels
    signal = torch.cat(
        [torch.sin(scaled_time), torch.cos(scaled_time)], dim=1
    )
    signal = signal.view(1, frames, num_timescales * 2)  # shape: (1, length, n_units//2 * 2)

    # Reorder dimensions to (1, n_units, length)
    pos_encoding = torch.permute(signal, (0, 2, 1))
    
    #=== Apply Encoding to Mels ===#
    
    # Nr. of sine and cosine waves in the position encoding.
    # Ensures even number.
    n_waves = num_timescales * 2
    # Scale position encodings by square root of the nr. of mel channels. 
    pos_scale = channels ** 0.5
    
    # Apply position encoding
    for mi, mel in enumerate(mels):
      # Correct for possibly inequal nr. of frames between mels
      max_frames = mel.shape[-1]
      mels[mi][:,:n_waves,:] = mel[:,:n_waves,:] + pos_encoding[...,:max_frames] / pos_scale * pos_weight
      
    if single_mel:
      return mels[0]
    
    return mels


##############
### Models ###
##############

class KenkuModel(KenkuModule):
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
               view_distance: int = 64,
    ):
    """
    Base Kenku model class containing the shared components of both Teacher and Student models.
    Args:
        in_ch (int): Number of input mel channels.
        conv_ch (int): Number of convolutional channels.
        att_ch (int): Number of attention channels.
        out_ch (int): Number of output mel channels.
        embed_ch (int): Number of speaker embedding channels.
        num_accents (int): Number of accent categories.
        num_conv_layers (Optional[int], optional): Number of convolutional layers in each KameBlock. Defaults to 8.
        kernel_size (Optional[int], optional): Kernel size of convolutional layers in each KameBlock. Defaults to 5.
        dilations (Optional[List[int]], optional): Dilation rates of convolutional layers in each KameBlock. 
                                                  If None, defaults to [1, 2, 4, 8, ...]. Defaults to None.
        dropout_rate (Optional[float], optional): Dropout rate used in each KameBlock. Defaults to 0.2.
        stack_factor (int, optional): Frame stacking factor. Defaults to 4.
        view_distance (int, optional): View distance for the Scaled Dot-Product Attention module. Defaults to 64.
    """
    super(KenkuModel, self).__init__()
    
    kame_block_kwargs = {
      'num_conv_layers': num_conv_layers,
      'kernel_size': kernel_size,
      'dilations': dilations,
      'dropout_rate': dropout_rate
    }
    
    self.stack_factor = stack_factor
    sf = stack_factor
    
    self.src_encoder = KameBlock(
      in_ch * sf, conv_ch, att_ch, embed_ch, num_accents, num_output_streams=2, **kame_block_kwargs
    ) 
    self.tgt_encoder = KameBlock(
      in_ch * sf, conv_ch, att_ch, embed_ch, num_accents, **kame_block_kwargs
    )
    self.attention = ScaledDotProductAttention(view_distance=view_distance)
    
    self.decoder = KameBlock(
      att_ch, conv_ch, out_ch * sf, embed_ch, num_accents, **kame_block_kwargs
    )
  
  def stack_mels(self, *mels):
    if self.stack_factor > 1:
      mels = tuple(stack_frames(mel, self.stack_factor) for mel in mels)
      if len(mels) == 1:
        mels = mels[0]
    
    if isinstance(mels, tuple) and len(mels) == 1:
      return mels[0]
    return mels
  
  def stack_masks(self, *masks):
    if self.stack_factor > 1:
      masks = tuple(mask[:,:,::self.stack_factor] for mask in masks)
      if len(masks) == 1:
        return masks[0]
      return masks
    return masks
    
  def _encode_inputs(self, src_mel, tgt_mel, src_info, tgt_info, stack = True):
    """
    Encode source and target mel-spectrograms into key, value, and query tensors.
    Args:
        src_mel (Tensor): Source mel-spectrogram tensor of shape (batch, n_mels, n_frames).
        tgt_mel (Tensor): Target mel-spectrogram tensor of shape (batch, n_mels, n_frames).
        src_info (Tensor): Source speaker properties tensor.
        tgt_info (Tensor): Target speaker properties tensor.
        stack (bool, optional): Whether to apply frame stacking. Defaults to True.
    """
    if stack:
      src_mel, tgt_mel = self.stack_mels(src_mel, tgt_mel)
    
    # Encoding step of forward pass
    K, V = self.src_encoder(src_mel, src_info)
    Q    = self.tgt_encoder(tgt_mel, tgt_info)
    
    return K, V, Q
  
  def _loss_input_preprocess(self, src_mel, tgt_mel, src_mask, tgt_mask, pos_weight=1.0):
    """
    Preprocess inputs for loss calculation by applying position encoding, frame stacking, and target zero-frame prepending.
    Args:
        src_mel (Tensor): Source mel-spectrogram tensor of shape (batch, n_mels, n_frames).
        tgt_mel (Tensor): Target mel-spectrogram tensor of shape (batch, n_mels, n_frames).
        src_mask (Tensor): Source mask tensor of shape (batch, 1, n_frames).
        tgt_mask (Tensor): Target mask tensor of shape (batch, 1, n_frames).
        pos_weight (float, optional): Weight for position encoding. Defaults to 1.0.
    """
    #=== Position Encoding ===#
    src_mel, tgt_mel = apply_position_encoding(src_mel, tgt_mel, pos_weight=pos_weight)
    
    #=== Frame Stacking ===#
    src_mel, tgt_mel = self.stack_mels(src_mel, tgt_mel)
    src_mask, tgt_mask = self.stack_masks(src_mask, tgt_mask)
    
    #=== Zero Frame Prepend ===#
    # Prepend zero frame to target as a start-of-sequence token
    tgt_mel = prepend_zero_frame(tgt_mel)
    tgt_mask = prepend_zero_frame(tgt_mask)
    
    return src_mel, tgt_mel, src_mask, tgt_mask
  
  def _calc_main_loss(self, pred_mel, tgt_mel, tgt_mask, main_loss_fn):
    try:
      main_loss_fn = {
        'mse': mse_loss,
        'mae': mae_loss
      }[main_loss_fn.lower()]
      
    except KeyError:
      raise ValueError(f"Unknown main loss function '{main_loss_fn}'. Supported: 'mse', 'mae'.")
    
    main_loss = main_loss_fn(pred_mel, tgt_mel, tgt_mask)
    
    return main_loss
  
  def _calc_att_loss(self, A, src_mask, tgt_mask, 
                     dal_tgt_sigma = 0.3,
                     oal_tgt_sigma = 0.3,
                     pred_means    = None,
                     pred_vars     = None,
                     true_A        = None):
    """
    Calculate attention losses: diagonal attention loss and orthogonal attention loss. 
    Optionally, calculate auxiliary attention loss if predicted means, variances, and true attention matrix are provided.
    Args:
        A (Tensor): Predicted attention matrix of shape (batch, n_src_frames, n_tgt_frames).
        src_mask (Tensor): Source mask tensor of shape (batch, 1, n_src_frames).
        tgt_mask (Tensor): Target mask tensor of shape (batch, 1, n_tgt_frames).
        dal_tgt_sigma (float, optional): Target sigma for diagonal attention loss. Defaults to 0.3.
        oal_tgt_sigma (float, optional): Target sigma for orthogonal attention loss. Defaults to 0.3.
        pred_means (Optional[Tensor], optional): Predicted means for auxiliary attention loss. Defaults to None.
        pred_vars (Optional[Tensor], optional): Predicted variances for auxiliary attention loss. Defaults to None.
        true_A (Optional[Tensor], optional): True attention matrix for auxiliary attention loss. Defaults to None.
    """
    # Diagonal attention loss
    da_loss = diag_att_loss(A, src_mask, tgt_mask, tgt_sigma = dal_tgt_sigma)
    # Orthogonal attention loss
    oa_loss = ortho_att_loss(A, src_mask, tgt_sigma = oal_tgt_sigma)
    
    # If the auxiliary attention loss inputs aren't passed,
    # only return diagonal and orthogonal attention losses
    if pred_means is pred_vars is true_A is None:
      return da_loss, oa_loss
    
    if any(map(lambda x: x is None, [pred_means, pred_vars, true_A])):
      raise ValueError("To calculate the auxiliary attention loss, all of pred_means, pred_vars, and true_A must be provided.")

    # Auxiliary attention loss
    aa_loss = auxil_att_loss(pred_means, pred_vars, true_A)
    
    return da_loss, oa_loss, aa_loss
    
  def clear_paddings(self):
    """
    Clear causal convolution paddings in all KameBlock modules.
    """
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
               stack_factor: int = 4,
               view_distance: int = 64
    ):
    """
    Kenku Teacher model class.
    Args:
        in_ch (int): Number of input mel channels.
        conv_ch (int): Number of convolutional channels.
        att_ch (int): Number of attention channels.
        out_ch (int): Number of output mel channels.
        embed_ch (int): Number of speaker embedding channels.
        num_accents (int): Number of accent categories.
        num_conv_layers (Optional[int], optional): Number of convolutional layers in each KameBlock. Defaults to 8.
        kernel_size (Optional[int], optional): Kernel size of convolutional layers in each KameBlock. Defaults to 5.
        dilations (Optional[List[int]], optional): Dilation rates of convolutional layers in each KameBlock. 
                                                  If None, defaults to [1, 2, 4, 8, ...]. Defaults to None.
        dropout_rate (Optional[float], optional): Dropout rate used in each KameBlock. Defaults to 0.2.
        stack_factor (int, optional): Frame stacking factor. Defaults to 4.
        view_distance (int, optional): View distance for the Scaled Dot-Product Attention module. Defaults to 64.
    """
    super(KenkuTeacher, self).__init__(
      in_ch, conv_ch, att_ch, out_ch, embed_ch, num_accents,              
      num_conv_layers      = num_conv_layers,
      kernel_size          = kernel_size,
      dilations            = dilations,
      dropout_rate         = dropout_rate,
      stack_factor         = stack_factor,
      view_distance        = view_distance
    )
    
    self._init_args = (in_ch, conv_ch, att_ch, out_ch, embed_ch, num_accents)
    self._init_kwargs = {
      'num_conv_layers'     : num_conv_layers,
      'kernel_size'         : kernel_size,
      'dilations'           : dilations,
      'dropout_rate'        : dropout_rate,
      'stack_factor'        : stack_factor,
      'view_distance'       : view_distance
    }
    
  def forward(self, src_mel, tgt_mel, src_info, tgt_info, stack=True):
    K, V, Q = self._encode_inputs(src_mel, tgt_mel, src_info, tgt_info, stack=stack)
    
    R, A = self.attention(K, V, Q)
    Y = self.decoder(R, tgt_info)
    
    if stack:
      Y = unstack_frames(Y, self.stack_factor)
    
    return Y, A
  
  def to_student(self, student_kwargs=None):
    """
    Convert this Teacher model into a Student model by transferring weights.
    Args:
        student_kwargs (dict, optional): Additional keyword arguments to pass to the KenkuStudent constructor. Defaults to None.
    """
    student_kwargs = {} if student_kwargs is None else student_kwargs
    student = KenkuStudent(*self._init_args, 
                           **{**self._init_kwargs, **student_kwargs})
    
    student.load_teacher_state_dict(self.state_dict())
    
    return student
  
  def calc_loss(self, src_mel, tgt_mel, src_mask, tgt_mask, src_info, tgt_info, 
                main_loss_fn = 'mse', 
                pos_weight = 1.0, 
                dal_tgt_sigma = 0.3, 
                oal_tgt_sigma = 0.3, 
                att_loss_weights = None,
                as_components = False,
                **kwargs):
    """
    Calculate the total loss for the Kenku Teacher model.
    
    Args:
        src_mel (Tensor): Source mel-spectrogram tensor of shape (batch, n_mels, n_frames).
        tgt_mel (Tensor): Target mel-spectrogram tensor of shape (batch, n_mels, n_frames).
        src_mask (Tensor): Source mask tensor of shape (batch, 1, n_frames).
        tgt_mask (Tensor): Target mask tensor of shape (batch, 1, n_frames).
        src_info (Tensor): Source speaker properties tensor.
        tgt_info (Tensor): Target speaker properties tensor.
        main_loss_fn (str, optional): Main loss function to use ('mse' or 'mae'). Defaults to 'mse'.
        pos_weight (float, optional): Weight for position encoding. Defaults to 1.0.
        dal_tgt_sigma (float, optional): Target sigma for diagonal attention loss. Defaults to 0.3.
        oal_tgt_sigma (float, optional): Target sigma for orthogonal attention loss. Defaults to 0.3.
        att_loss_weights (Optional[List[float]], optional): Weights for attention loss terms [da_loss_weight, oa_loss_weight]. 
                                                           Defaults to None.
        as_components (bool, optional): Whether to return loss components separately. Defaults to False.
        
    Returns:
        Union[Tensor, dict]: Total loss tensor or dictionary of loss components if as_components is True.
    """
    if att_loss_weights is not None:
      assert len(att_loss_weights) == 2, f"Incorrect amount of attention loss weights. Expected 2, got {len(att_loss_weights)}."
    
    # Preprocess inputs by applying position encoding, frame stacking, and target zero-frame prepend
    src_mel, tgt_mel, src_mask, tgt_mask = self._loss_input_preprocess(
      src_mel, tgt_mel, src_mask, tgt_mask
    )
    
    pred_mel, A = self.forward(src_mel, tgt_mel, src_info, tgt_info, stack=False)
    
    return self._calc_loss(pred_mel, tgt_mel, src_mask, tgt_mask, A,
                           main_loss_fn = main_loss_fn,
                           dal_tgt_sigma = dal_tgt_sigma,
                           oal_tgt_sigma = oal_tgt_sigma,
                           att_loss_weights = att_loss_weights,
                           as_components = as_components)
  
  def _calc_loss(self, pred_mel, tgt_mel, src_mask, tgt_mask, A,
                 main_loss_fn = 'mse', 
                 dal_tgt_sigma = 0.3, 
                 oal_tgt_sigma = 0.3, 
                 att_loss_weights = None,
                 as_components = False):
    """
    Calculate the total loss for the Kenku Teacher model.
    Separated from calc_loss to allow for reuse by child classes.
    """
    #=== Loss Terms ===#
    main_loss = self._calc_main_loss(pred_mel, tgt_mel, tgt_mask, main_loss_fn)
    da_loss, oa_loss = self._calc_att_loss(A, src_mask, tgt_mask, dal_tgt_sigma, oal_tgt_sigma)
    
    #=== Combine Loss ===#
    # Return as components if requested
    if as_components:
      return {'main loss': main_loss, 
              'da loss': da_loss, 
              'oa loss': oa_loss, 
              }, A
    
    # Else, weigh and sum loss terms
    if att_loss_weights is None:
      att_loss_weights = [2000, 2000]
      
    total_loss = main_loss
    
    for lw, loss_term in zip(att_loss_weights, [da_loss, oa_loss]):
      total_loss += lw * loss_term
      
    return total_loss, A
    

class KenkuStudent(KenkuModel):
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
               view_distance: int = 64,
               rng: Union[torch.Generator, int] = None):
    """
    Kenku Student model class.
    
    Args:
        in_ch (int): Number of input mel channels.
        conv_ch (int): Number of convolutional channels.
        att_ch (int): Number of attention channels.
        out_ch (int): Number of output mel channels.
        embed_ch (int): Number of speaker embedding channels.
        num_accents (int): Number of accent categories.
        num_conv_layers (Optional[int], optional): Number of convolutional layers in each KameBlock. Defaults to 8.
        kernel_size (Optional[int], optional): Kernel size of convolutional layers in each KameBlock. Defaults to 5.
        dilations (Optional[List[int]], optional): Dilation rates of convolutional layers in each KameBlock. 
                                                  If None, defaults to [1, 2, 4, 8, ...]. Defaults to None.
        dropout_rate (Optional[float], optional): Dropout rate used in each KameBlock. Defaults to 0.2.
        stack_factor (int, optional): Frame stacking factor. Defaults to 4.
        view_distance (int, optional): View distance for the Scaled Dot-Product Attention module. Defaults to 64.
        rng (Union[torch.Generator, int], optional): Random number generator or seed for attention predictor. Defaults to None.
    """
    super(KenkuStudent, self).__init__(
      in_ch, conv_ch, att_ch, out_ch, embed_ch, num_accents,              
      num_conv_layers      = num_conv_layers,
      kernel_size          = kernel_size,
      dilations            = dilations,
      dropout_rate         = dropout_rate,
      stack_factor         = stack_factor,
      view_distance        = view_distance
    )
    
    self.attention_predictor = AttentionPredictor(
      in_ch * stack_factor, conv_ch, embed_ch, num_accents,
      num_conv_layers      = num_conv_layers,
      kernel_size          = kernel_size,
      dilations            = dilations,
      dropout_rate         = dropout_rate,
      view_distance        = view_distance,
      rng                  = rng
    )
    
  def forward(self, src_mel, src_info, tgt_info, stack = True):   
    if isinstance(src_mel, tuple):
      print()
    if stack:
      src_mel = self.stack_mels(src_mel)
      if isinstance(src_mel, tuple):
        print()
      
    _, V = self.src_encoder(src_mel, src_info)
    pred_A, _, _ = self.attention_predictor(src_mel, tgt_info)
    
    R = V.matmul(pred_A)
    Y = self.decoder(R, tgt_info)
    
    if stack:
      Y = unstack_frames(Y, self.stack_factor)
      
    return Y, pred_A
    
  def load_teacher_state_dict(self, tea_dict: dict):
    """
    Load weights from a Kenku Teacher model state dict into this Kenku Student model.
    """
    stu_dict = self.state_dict()
    
    tea_keys = list(tea_dict.keys())
    
    # Construct state dict from student state dict keys.
    # Module weights are copied from the teacher if available,
    # and copied from the student if not available.
    state_dict = dict([(sk, tea_dict[sk]) if sk in tea_keys else (sk, stu_dict[sk])
                       for sk in stu_dict.keys()])
    
    self.load_state_dict(state_dict)
    
    # Freeze only the target encoder weights
    # Necessary for the IncrementalThawScheduler to work properly
    for param in self.tgt_encoder.parameters():
      param.requires_grad = False
  
  def calc_loss(self, src_mel, tgt_mel, src_mask, tgt_mask, src_info, tgt_info, 
                main_loss_fn = 'mse', 
                pos_weight = 1.0, 
                dal_tgt_sigma = 0.3, 
                oal_tgt_sigma = 0.3, 
                att_loss_weights = None,
                as_components = False,
                **kwargs):
    """
    Calculate the total loss for the Kenku Student model.
    
    Args:
        src_mel (Tensor): Source mel-spectrogram tensor of shape (batch, n_mels, n_frames).
        tgt_mel (Tensor): Target mel-spectrogram tensor of shape (batch, n_mels, n_frames).
        src_mask (Tensor): Source mask tensor of shape (batch, 1, n_frames).
        tgt_mask (Tensor): Target mask tensor of shape (batch, 1, n_frames).
        src_info (Tensor): Source speaker properties tensor.
        tgt_info (Tensor): Target speaker properties tensor.
        main_loss_fn (str, optional): Main loss function to use ('mse' or 'mae'). Defaults to 'mse'.
        pos_weight (float, optional): Weight for position encoding. Defaults to 1.0.
        dal_tgt_sigma (float, optional): Target sigma for diagonal attention loss. Defaults to 0.3.
        oal_tgt_sigma (float, optional): Target sigma for orthogonal attention loss. Defaults to 0.3.
        att_loss_weights (Optional[List[float]], optional): Weights for attention loss terms [da_loss_weight, oa_loss_weight, aa_loss_weight]. 
                                                           Responsible for diagonal, orthogonal, and auxiliary loss, respectively. Defaults to [2000, 2000, 1].
        as_components (bool, optional): Whether to return loss components separately. Defaults to False.
        
    Returns:
        Union[Tensor, dict]: Total loss tensor or dictionary of loss components if as_components is True.
    """
    if att_loss_weights is not None:
      assert len(att_loss_weights) == 3, f"Incorrect amount of attention loss weights. Expected 3, got {len(att_loss_weights)}."
    
    # Preprocess inputs by applying position encoding, frame stacking, and target zero-frame prepend
    src_mel, tgt_mel, src_mask, tgt_mask = self._loss_input_preprocess(
      src_mel, tgt_mel, src_mask, tgt_mask
    )
    
    #=== Forward Pass ==#
    # Cannot call self.forward() due to needing to calculate the true attention matrix
    K, V, Q = self._encode_inputs(src_mel, tgt_mel, src_info, tgt_info, stack = False)
    
    # Real attention matrix from the Teacher's encoder
    _, true_A = self.attention(K, V, Q)
    
    # Time-scaled sequence according to the attention predictor
    n_tgt_frames = tgt_mel.shape[-1]
    pred_A, pred_means, pred_vars = self.attention_predictor(src_mel, tgt_info, n_tgt_frames=n_tgt_frames)
    
    R = V.matmul(pred_A)
    pred_mel = self.decoder(R, tgt_info)
    
    return self._calc_loss(pred_mel, tgt_mel, src_mask, tgt_mask, pred_A, pred_means, pred_vars, true_A,
                           main_loss_fn = main_loss_fn,
                           dal_tgt_sigma = dal_tgt_sigma,
                           oal_tgt_sigma = oal_tgt_sigma,
                           att_loss_weights = att_loss_weights,
                           as_components = as_components)
    
  def _calc_loss(self, pred_mel, tgt_mel, src_mask, tgt_mask, pred_A, pred_means, pred_vars, true_A,
                 main_loss_fn = 'mse', 
                 dal_tgt_sigma = 0.3, 
                 oal_tgt_sigma = 0.3, 
                 att_loss_weights = None,
                 as_components = False):
    
    #=== Loss Terms ===#
    main_loss = self._calc_main_loss(pred_mel, tgt_mel, tgt_mask, main_loss_fn)
    da_loss, oa_loss, aa_loss = self._calc_att_loss(pred_A, src_mask, tgt_mask, 
                                                    dal_tgt_sigma=dal_tgt_sigma,
                                                    oal_tgt_sigma=oal_tgt_sigma,
                                                    pred_means=pred_means,
                                                    pred_vars=pred_vars,
                                                    true_A=true_A)
    
    #=== Combine Loss ===#
    # Return as components if requested
    if as_components:
      return {'main loss': main_loss, 
              'da loss': da_loss, 
              'oa loss': oa_loss,
              'aa loss': aa_loss}, pred_A
    
    # Weigh and sum loss terms
    if att_loss_weights is None:
      att_loss_weights = [1, 2000, 2000]
      
    total_loss = main_loss
    
    for lw, loss_term in zip(att_loss_weights, [da_loss, oa_loss, aa_loss]):
      total_loss += lw * loss_term
      
    return total_loss, pred_A

  def clear_paddings(self):
    super(KenkuStudent, self).clear_paddings()
    self.attention_predictor.encoder.clear_paddings()

##################
### DRL Models ###
##################

class DRLLossMixin:
  """
  Mixin class providing DRL loss calculation functionality.
  """
  def _calc_drl_loss(self, src_variational, tgt_variational, dataset_size,
                     drl_loss_weights = None, 
                     accent_entropy_weight = 1.0, 
                     as_components = False):
    """
    Calculate the DRL loss given source and target variational outputs.
    Args:
        src_variational (tuple): Tuple of (src_info, src_z, src_mu, src_log_var).
        tgt_variational (tuple): Tuple of (tgt_info, tgt_z, tgt_mu, tgt_log_var).
        dataset_size (int): Size of the dataset for loss normalization.
        drl_loss_weights (Optional[List[float]], optional): Weights for DRL loss terms [mi_loss_weight, tc_loss_weight, dw_kld_loss_weight]. 
                                                           Defaults to None.
        accent_entropy_weight (float, optional): Weight for accent entropy loss. Defaults to 1.0.
        as_components (bool, optional): Whether to return loss components separately. Defaults to False.
    """
    src_info, src_z, src_mu, src_log_var = src_variational
    tgt_info, tgt_z, tgt_mu, tgt_log_var = tgt_variational
    
    # Sum the src and target loss components individually
    mi_loss, tc_loss, dw_kld_loss = torch.stack([
      torch.stack(beta_tcvae_loss_terms(src_z, src_mu, src_log_var, dataset_size)),
      torch.stack(beta_tcvae_loss_terms(tgt_z, tgt_mu, tgt_log_var, dataset_size))
    ]).sum(dim=0) / 2
    
    # Calculate the entropy loss of the accent dimensions
    entropy_loss = (accent_entropy_loss(src_info[2:]) + accent_entropy_loss(tgt_info[2:])) / 2
    
    if drl_loss_weights is None:
      drl_loss_weights = [1.0, 1.0, 1.0]
      
    # Factor terms with alpha, beta, and gamma. Then sum
    drl_loss = torch.sum(torch.tensor(drl_loss_weights, device=mi_loss.device) * torch.stack([mi_loss, tc_loss, dw_kld_loss]))
    #=== Return Loss ===#
    
    if as_components:
      loss_components = {
        'drl loss': drl_loss,
        'mi loss': mi_loss,
        'tc loss': tc_loss,
        'dw kld loss': dw_kld_loss,
        'acc entr loss': entropy_loss
      }
      return loss_components

    return drl_loss + entropy_loss * accent_entropy_weight


class DRLKenkuTeacher(KenkuTeacher, DRLLossMixin):
  def __init__(self,
               in_ch: int,
               conv_ch: int,
               att_ch: int,
               out_ch: int,
               embed_ch: int,
               num_accents: int,
               num_conv_layers: Optional[int] = 8,
               num_si_conv_layers: Optional[int] = 6,
               kernel_size: Optional[int] = 5,
               dilations: Optional[List[int]] = None,
               dropout_rate: Optional[float] = 0.2,
               stack_factor: int = 4,
               view_distance: int = 64):
    """
    DRL Kenku Teacher model class. Uses beta-TCVAE for disentanglement of speaker properties.
    
    Args:
        in_ch (int): Number of input mel channels.
        conv_ch (int): Number of convolutional channels.
        att_ch (int): Number of attention channels.
        out_ch (int): Number of output mel channels.
        embed_ch (int): Number of speaker embedding channels.
        num_accents (int): Number of accent categories.
        num_conv_layers (Optional[int], optional): Number of convolutional layers in each KameBlock. Defaults to 8.
        num_si_conv_layers (Optional[int], optional): Number of convolutional layers in the SpeakerInfoPredictor. Defaults to 6.
        kernel_size (Optional[int], optional): Kernel size of convolutional layers in each KameBlock. Defaults to 5.
        dilations (Optional[List[int]], optional): Dilation rates of convolutional layers in each KameBlock. 
                                                  If None, defaults to [1, 2, 4, 8, ...]. Defaults to None.
        dropout_rate (Optional[float], optional): Dropout rate used in each KameBlock. Defaults to 0.2.
        stack_factor (int, optional): Frame stacking factor. Defaults to 4.
        view_distance (int, optional): View distance for the Scaled Dot-Product Attention module. Defaults to 64.
    """
    
    super(DRLKenkuTeacher, self).__init__(
      in_ch, conv_ch, att_ch, out_ch, embed_ch, num_accents,              
      num_conv_layers      = num_conv_layers,
      kernel_size          = kernel_size,
      dilations            = dilations,
      dropout_rate         = dropout_rate,
      stack_factor         = stack_factor,
      view_distance        = view_distance
    )
    
    self.speaker_info_predictor = SpeakerInfoPredictor(
      in_ch * stack_factor, conv_ch, num_accents + 2,
      num_conv_layers = num_si_conv_layers,
      kernel_size     = kernel_size,
      dilations       = dilations,
      dropout_rate    = dropout_rate
    )
    
    # Recursively initialize embed layers of all Kame blocks with DRL capability
    # This is done post-hoc to ensure the speaker_info_predictor is at root level in the PyTorch graph.
    self._init_embed_layer(use_drl = True)
    
  def forward(self, src_mel, tgt_mel, src_mask, tgt_mask, stack = True, return_variational = False):
    if stack:
      src_mel, tgt_mel = self.stack_mels(src_mel, tgt_mel)
      src_mask, tgt_mask = self.stack_masks(src_mask, tgt_mask)
      
    src_info, src_z, src_mu, src_log_var = self.speaker_info_predictor(src_mel, src_mask)
    tgt_info, tgt_z, tgt_mu, tgt_log_var = self.speaker_info_predictor(tgt_mel, tgt_mask)
    
    Y, A = super(DRLKenkuTeacher, self).forward(src_mel, tgt_mel, src_info, tgt_info, stack=False)
    
    if stack:
      Y = unstack_frames(Y, self.stack_factor)
    
    if return_variational:
      src_variational = (src_info, src_z, src_mu, src_log_var)
      tgt_variational = (tgt_info, tgt_z, tgt_mu, tgt_log_var)
      return Y, A, src_variational, tgt_variational
    
    return Y, A
    
  def calc_loss(self, src_mel, tgt_mel, src_mask, tgt_mask, dataset_size,
                main_loss_fn = 'mse', 
                pos_weight = 1.0, 
                dal_tgt_sigma = 0.3, 
                oal_tgt_sigma = 0.3, 
                att_loss_weights = None,
                drl_loss_weights = None,
                accent_entropy_weight = 1.0,
                as_components = False):
    """
    Calculate the total loss for the DRL Kenku Teacher model.
    
    Args:
        src_mel (Tensor): Source mel-spectrogram tensor of shape (batch, n_mels, n_frames).
        tgt_mel (Tensor): Target mel-spectrogram tensor of shape (batch, n_mels, n_frames).
        src_mask (Tensor): Source mask tensor of shape (batch, 1, n_frames).
        tgt_mask (Tensor): Target mask tensor of shape (batch, 1, n_frames).
        dataset_size (int): Size of the dataset for loss normalization.
        main_loss_fn (str, optional): Main loss function to use ('mse' or 'mae'). Defaults to 'mse'.
        pos_weight (float, optional): Weight for position encoding. Defaults to 1.0.
        dal_tgt_sigma (float, optional): Target sigma for diagonal attention loss. Defaults to 0.3.
        oal_tgt_sigma (float, optional): Target sigma for orthogonal attention loss. Defaults to 0.3.
        att_loss_weights (Optional[List[float]], optional): Weights for attention loss terms [da_loss_weight, oa_loss_weight]. 
                                                            Corresponding to diagonal and orthogonal loss, respectively. Defaults to None.
        drl_loss_weights (Optional[List[float]], optional): Weights for DRL loss terms [mi_loss_weight, tc_loss_weight, dw_kld_loss_weight]. 
                                                           Corresponding to mutual information, total correlation, and dimension-wise KLD loss, 
                                                           respectively. Defaults to None.
        accent_entropy_weight (float, optional): Weight for accent entropy loss. Defaults to 1.0.
        as_components (bool, optional): Whether to return loss components separately. Defaults to False.
        
    Returns:
        Union[Tensor, dict]: Total loss tensor or dictionary of loss components if as_components is True.
    """
    if att_loss_weights is not None:
      assert len(att_loss_weights) == 2, f"Incorrect amount of attention loss weights. Expected 2, got {len(att_loss_weights)}."
    
    if drl_loss_weights is not None:
      assert len(drl_loss_weights) == 3, f"Incorrect amount of DRL loss weights. Expected 3, got {len(drl_loss_weights)}."
    
    # Preprocess inputs by applying position encoding, frame stacking, and target zero-frame prepend
    src_mel, tgt_mel, src_mask, tgt_mask = self._loss_input_preprocess(
      src_mel, tgt_mel, src_mask, tgt_mask
    )
      
    #=== Forward Pass ===#
    pred_mel, A, src_variational, tgt_variational = self(src_mel, tgt_mel, src_mask, tgt_mask, stack=False, return_variational=True)
    
    #=== Main and Attention Loss Calculation ===#
    loss, A = super(DRLKenkuTeacher, self)._calc_loss(
      pred_mel, tgt_mel, src_mask, tgt_mask, A,
      main_loss_fn = main_loss_fn,
      dal_tgt_sigma = dal_tgt_sigma,
      oal_tgt_sigma = oal_tgt_sigma,
      att_loss_weights = att_loss_weights,
      as_components = as_components
    )
    
    #=== DRL Loss Calculation ===#
    drl_loss = self._calc_drl_loss(src_variational, tgt_variational, dataset_size,
                                  drl_loss_weights = drl_loss_weights,
                                  accent_entropy_weight = accent_entropy_weight,
                                  as_components = as_components)
    
    #=== Combine Loss ===#
    if as_components:
      loss_components = loss
      loss_components.update(drl_loss)
      return loss_components, A
    
    loss += drl_loss
    return loss, A
    
  def to_student(self, student_kwargs=None):
    """
    Convert this Teacher model into a Student model by transferring weights.
    Args:
        student_kwargs (dict, optional): Additional keyword arguments to pass to the DRLKenkuStudent constructor. Defaults to None.
    """
    student_kwargs = {} if student_kwargs is None else student_kwargs
    student = DRLKenkuStudent(*self._init_args, 
                              **{**self._init_kwargs, **student_kwargs})
    
    student.load_teacher_state_dict(self.state_dict())
    
    return student
  
  def clear_paddings(self):
    super(DRLKenkuTeacher, self).clear_paddings()
    self.speaker_info_predictor.clear_paddings()
    
class DRLKenkuStudent(KenkuStudent):
  def __init__(self,
               in_ch: int,
               conv_ch: int,
               att_ch: int,
               out_ch: int,
               embed_ch: int,
               num_accents: int,
               num_conv_layers: Optional[int] = 8,
               num_si_conv_layers: Optional[int] = 6,
               kernel_size: Optional[int] = 5,
               dilations: Optional[List[int]] = None,
               dropout_rate: Optional[float] = 0.2,
               stack_factor: int = 4,
               view_distance: int = 64,
               rng: Union[torch.Generator, int] = None):
    """
    DRL Kenku Student model class. Uses beta-TCVAE for disentanglement of speaker properties.
    
    Args:
        in_ch (int): Number of input mel channels.
        conv_ch (int): Number of convolutional channels.
        att_ch (int): Number of attention channels.
        out_ch (int): Number of output mel channels.
        embed_ch (int): Number of speaker embedding channels.
        num_accents (int): Number of accent categories.
        num_conv_layers (Optional[int], optional): Number of convolutional layers in each KameBlock. Defaults to 8.
        num_si_conv_layers (Optional[int], optional): Number of convolutional layers in the SpeakerInfoPredictor. Defaults to 6.
        kernel_size (Optional[int], optional): Kernel size of convolutional layers in each KameBlock. Defaults to 5.
        dilations (Optional[List[int]], optional): Dilation rates of convolutional layers in each KameBlock. 
                                                  If None, defaults to [1, 2, 4, 8, ...]. Defaults to None.
        dropout_rate (Optional[float], optional): Dropout rate used in each KameBlock. Defaults to 0.2.
        stack_factor (int, optional): Frame stacking factor. Defaults to 4.
        view_distance (int, optional): View distance for the Scaled Dot-Product Attention module. Defaults to 64.
        rng (Union[torch.Generator, int], optional): Random number generator or seed for attention predictor. Defaults to None.
    """
    super(DRLKenkuStudent, self).__init__(
      in_ch, conv_ch, att_ch, out_ch, embed_ch, num_accents,              
      num_conv_layers      = num_conv_layers,
      kernel_size          = kernel_size,
      dilations            = dilations,
      dropout_rate         = dropout_rate,
      stack_factor         = stack_factor,
      view_distance        = view_distance
    )
    
    self.speaker_info_predictor = SpeakerInfoPredictor(
      in_ch * stack_factor, conv_ch, num_accents + 2,
      num_conv_layers = num_si_conv_layers,
      kernel_size     = kernel_size,
      dilations       = dilations,
      dropout_rate    = dropout_rate
    )
    
    # Recursively initialize embed layers of all Kame blocks with DRL capability
    # This is done post-hoc to ensure the speaker_info_predictor is at root level in the PyTorch graph.
    self._init_embed_layer(use_drl = True)
    
  def forward(self, src_mel, tgt_mel, src_mask, tgt_mask, 
              stack = True,
              return_variational = False):
    if stack:
      src_mel, tgt_mel = self.stack_mels(src_mel, tgt_mel)
      src_mask, tgt_mask = self.stack_masks(src_mask, tgt_mask)
    
    src_info, src_z, src_mu, src_log_var = self.speaker_info_predictor(src_mel, src_mask)
    tgt_info, tgt_z, tgt_mu, tgt_log_var = self.speaker_info_predictor(tgt_mel, tgt_mask)
    
    Y, pred_A = super(DRLKenkuStudent, self).forward(src_mel, src_info, tgt_info, stack=False)
    
    if stack:
      Y = unstack_frames(Y, self.stack_factor)
      
    if return_variational:
      src_variational = (src_info, src_z, src_mu, src_log_var)
      tgt_variational = (tgt_info, tgt_z, tgt_mu, tgt_log_var)
      return Y, pred_A, src_variational, tgt_variational
    
    return Y, pred_A
    
  def calc_loss(self, src_mel, tgt_mel, src_mask, tgt_mask, dataset_size,
                main_loss_fn = 'mse', 
                pos_weight = 1.0, 
                dal_tgt_sigma = 0.3, 
                oal_tgt_sigma = 0.3, 
                att_loss_weights = None,
                drl_loss_weights = None,
                accent_entropy_weight = 1.0,
                as_components = False):
    """
    Calculate the total loss for the DRL Kenku Student model.
    
    Args:
        src_mel (Tensor): Source mel-spectrogram tensor of shape (batch, n_mels, n_frames).
        tgt_mel (Tensor): Target mel-spectrogram tensor of shape (batch, n_mels, n_frames).
        src_mask (Tensor): Source mask tensor of shape (batch, 1, n_frames).
        tgt_mask (Tensor): Target mask tensor of shape (batch, 1, n_frames).
        dataset_size (int): Size of the dataset for loss normalization.
        main_loss_fn (str, optional): Main loss function to use ('mse' or 'mae'). Defaults to 'mse'.
        pos_weight (float, optional): Weight for position encoding. Defaults to 1.0.
        dal_tgt_sigma (float, optional): Target sigma for diagonal attention loss. Defaults to 0.3.
        oal_tgt_sigma (float, optional): Target sigma for orthogonal attention loss. Defaults to 0.3.
        att_loss_weights (Optional[List[float]], optional): Weights for attention loss terms [da_loss_weight, oa_loss_weight]. 
                                                            Corresponding to diagonal and orthogonal loss, respectively. Defaults to None.
        drl_loss_weights (Optional[List[float]], optional): Weights for DRL loss terms [mi_loss_weight, tc_loss_weight, dw_kld_loss_weight]. 
                                                           Corresponding to mutual information, total correlation, and dimension-wise KLD loss, 
                                                           respectively. Defaults to None.
        accent_entropy_weight (float, optional): Weight for accent entropy loss. Defaults to 1.0.
        as_components (bool, optional): Whether to return loss components separately. Defaults to False.
        
    Returns:
        Union[Tensor, dict]: Total loss tensor or dictionary of loss components if as_components is True.
    """
    if att_loss_weights is not None:
      assert len(att_loss_weights) == 3, f"Incorrect amount of attention loss weights. Expected 3, got {len(att_loss_weights)}."
    
    if drl_loss_weights is not None:
      assert len(drl_loss_weights) == 3, f"Incorrect amount of DRL loss weights. Expected 3, got {len(drl_loss_weights)}."
    
    # Preprocess inputs by applying position encoding, frame stacking, and target zero-frame prepend
    src_mel, tgt_mel, src_mask, tgt_mask = self._loss_input_preprocess(
      src_mel, tgt_mel, src_mask, tgt_mask
    )
    
    
    #=== Forward Pass ===#
    # Cannot call self.forward() due to needing to calculate the true attention matrix
    src_variational = self.speaker_info_predictor(src_mel, src_mask)
    tgt_variational = self.speaker_info_predictor(tgt_mel, tgt_mask)
    src_info = src_variational[0]
    tgt_info = tgt_variational[0]
    
    K, V, Q = self._encode_inputs(src_mel, tgt_mel, src_info, tgt_info, stack = False)
    
    # Real attention matrix from the Teacher's encoder
    _, true_A = self.attention(K, V, Q)
    
    # Time-scaled sequence according to the attention predictor
    n_tgt_frames = tgt_mel.shape[-1]
    pred_A, pred_means, pred_vars = self.attention_predictor(src_mel, tgt_info, n_tgt_frames=n_tgt_frames)
    
    R = V.matmul(pred_A)
    pred_mel = self.decoder(R, tgt_info)
    
    #=== Main and Attention Loss Calculation ===#
    loss, A = super(DRLKenkuStudent, self)._calc_loss(
      pred_mel, tgt_mel, src_mask, tgt_mask, pred_A, pred_means, pred_vars, true_A,
      main_loss_fn = main_loss_fn,
      dal_tgt_sigma = dal_tgt_sigma,
      oal_tgt_sigma = oal_tgt_sigma,
      att_loss_weights = att_loss_weights,
      as_components = as_components
    )
    
    #=== DRL Loss Calculation ===#
    drl_loss = self._calc_drl_loss(src_variational, tgt_variational, dataset_size,
                                  drl_loss_weights = drl_loss_weights,
                                  accent_entropy_weight = accent_entropy_weight,
                                  as_components = as_components)
    
    #=== Combine Loss ===#
    if as_components:
      loss_components = loss
      loss_components.update(drl_loss)
      return loss_components, A
    
    loss += drl_loss
    return loss, A
  
  def clear_paddings(self):
    super(DRLKenkuTeacher, self).clear_paddings()
    self.speaker_info_predictor.encoder.clear_paddings()
