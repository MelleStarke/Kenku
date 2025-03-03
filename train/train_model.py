import argparse
import os, sys
import logging
import json
import gc

from pathlib import Path
from datetime import datetime

from typing import Union, List, Tuple, Optional


import numpy as np

import torch 
from torch import nn, Tensor, tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from tqdm import tqdm

from data.load import ParallelDatasetFactory, ParallelMelspecDataset, collate_fn
from data.util import save_config, load_config
from kenku.modules import KameBlock
from kenku.network import KenkuTeacher, stack_frames, unstack_frames, append_zero_frame


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# torch.multiprocessing.set_start_method('spawn')
# torch.set_default_device("cuda:0")

# print(f"\n\n{torch.cuda.is_available()}\n\n")

# exit(0)

###############
### Logging ###
###############

logger = logging.getLogger(__name__)

# Get the full path to the directory containing the current file
current_file_dir = Path(__file__).parent.resolve()
logfile_path = os.path.join(current_file_dir, 'logs/train_model.log')

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

class CheckpointManager:
  def __init__(self, model, optimizer, save_path, interval=100, max=20):
    self.model = model
    self.optimizer = optimizer
    self.save_path = save_path
    
    self.interval = interval
    self.max_checkpoints = max
    
    self.saved_filenames = []
    
  def inform(self, epoch: int, batch_index: int):
    if batch_index % self.interval == 0:
      filename = f'epoch{epoch}_batch{batch_index}.pt'
      
      self.save_checkpoint(filename)
      self.saved_filenames.append(filename)
      
      # Remove old checkpoint if the maximum amount is exceeded
      if len(self.saved_filenames) > self.max_checkpoints:
        self.delete_oldest_checkpoint()
      
  def save_checkpoint(self, filename):
    checkpoint = {
      'model'    : self.model.state_dict(),
      'optimizer': self.optimizer.state_dict()
    }
    checkpoint_path = os.path.join(self.save_path, filename)
    torch.save(checkpoint, checkpoint_path)
    
  def delete_oldest_checkpoint(self):
    checkpoint_path = os.path.join(self.save_path, self.saved_filenames.pop(0))
    
    if os.path.exists(checkpoint_path):
      os.remove(checkpoint_path)
    else:
      logger.error(f'No checkpoint file for deletion found in {checkpoint_path}')


class TensorboardManager:
  def __init__(model: nn.Module, 
               test_loader: DataLoader, 
               directory: str, 
               train_set_length: int,
               test_interval: int    = 100,
               max_test_batches: int = 50, 
               n_images: int         = 5, 
               device:str            = 'cpu'):
      
    self.model = model
    self.test_loader = test_loader
    self.device = device
    self.interval = test_interval
    self.max_test_batches = max_test_batches
    self.train_set_length = train_set_length
    self.writer = SummaryWriter(log_dir=directory)
    
    self.img_batch = self.make_image_batch(n_images)
    
  def make_image_batch(self, n_images):
    src_mel, tgt_mel, _, _, src_info, tgt_info = next(iter(self.test_loader))
    test_batch_size = len(src_mel)
    n_images = min(n_images, test_batch_size)
    
    src_mel  = src_mel[:n_images].to(self.device)
    tgt_mel  = tgt_mel[:n_images].to(self.device)
    src_info = tuple(k[:n_images].to(self.device) for k in src_info)
    tgt_info = tuple(k[:n_images].to(self.device) for k in tgt_info)
    
    return src_mel, tgt_mel, src_info, tgt_info
  
  def inform(epoch: int, batch_nr: int, DAL_weight: float):    
    if batch_nr % self.internal == 0:
      self.model.eval()
      
      self.model.clear_paddings()
      self.record_test_loss(epoch, batch_nr, DAL_weight)
      
      self.model.clear_paddings()
      self.record_test_melspecs(epoch, batch_nr)
      
      torch.cuda.empty_cache()
      gc.collect()
      
      self.model.train()
    
  def img_batch_to_device(self, device):
    src_mel, tgt_mel, src_info, tgt_info = self.img_batch
    
    return (src_mel.to(device),
            tgt_mel.to(device),
            tuple(k.to(device) for k in src_info),
            tuple(k.to(device) for k in tgt_info),)
  
  def record_test_loss(self, epoch_nr: int, batch_nr: int, DAL_weight: float):
    test_loss = 0.
    
    n_test_batches = min(len(self.train_loader), self.max_test_batches)
    
    for bi, batch in enumerate(self.test_loader):
      if bi == n_test_batches:
        break
      
      MSE_loss, DA_loss, A_np = self.model.calc_loss(*batch)
      loss = MSE_loss + DA_loss * DAL_weight
      test_loss += loss.detach().cpu().item()
      
    self.writer('test loss',
                test_loss / n_test_batches,
                epoch_nr * self.train_set_length + batch_nr)

  def record_test_melspecs(self, epoch_nr: int, batch_nr: int):
    model_device = self.model.encoder.in_layer.device
    
    img_batch = self.img_batch_to_device(device)
    pred_mel  = self.model(*img_batch).to(self.device)
    
    del img_batch
    
    src_mel, tgt_mel, _, _ = self.img_batch
    
    n_frames_diff = tb_pred_imgs.shape[-1] - tb_src_imgs.shape[-1]
    tb_src_imgs = append_zero_frame(tb_src_imgs, n_frames=n_frames_diff)
    tb_tgt_imgs = append_zero_frame(tb_tgt_imgs, n_frames=n_frames_diff)
    
    full_img = torch.cat([src_mel, tgt_mel, pred_mel], dim=0).unsqueeze(1)
    img_grid = make_grid(full_img, nrow=3)
    
    self.writer.add_image('test spectrograms',
                          img_grid,
                          epoch_nr * self.train_set_length + batch_nr)

def create_config_dict(args_dict: dict, keys: List[str], config_path: Optional[str] = ""):
  """Creates a config dictionary with the specified keys.
     Values found in the file at `config_path` are prioritized
     over the values passed through parsed arguments `args`.

  Args:
      args_dict (dict): Argument dictionary created from vars(argparse.ArgumentParser.parse_args()).
      keys (List[str]): Keys of the items to be included.
      config_path (Optional[str]): Path to a config file. Values in this file take precedence over the ones in `args_dict`.
  """
  config = {}
  
  if config_path:
    config = load_config(config_path)
    
  parsed_config = {k: args_dict[k] for k in keys}
  # This expression prioritizes arguments found in the config file.
  config = {**parsed_config, **config}
  
  return config


################
### Training ###
################

def train_model(model: nn.Module,
                optimizer: torch.optim.Optimizer, 
                train_loader: DataLoader, 
                test_loader: DataLoader, 
                checkpoint_manager: CheckpointManager, 
                tensorboard_writer: SummaryWriter,
                
                epochs: int             = 10,
                test_interval: int      = 100,
                DAL_weight: float       = 0.,
                DAL_weight_decay: float = None,
                max_test_batches: int   = 100):

  DAL_weight_init = DAL_weight
  if DAL_weight_decay is None:
    DAL_weight_decay = 4 / epochs
    
  train_loss_plot_interval = 100

  # Define set of src and tgt melspecs to be used as images in Tensorboard.
  # tb_img_batch will be fed into the network to visually assess performance.
  tb_img_batch = next(iter(train_loader))
  n_tb_imgs = min(3, len(tb_img_batch[0]))
  # Remove mask tensors found at indices 2 and 3
  tb_img_batch = [tb_img_batch[i] for i in [0,1,4,5]]
  # Reduce the amount of melspecs for visualization
  tb_img_batch = tuple([     batch_comp[:n_tb_imgs] if i < 2 
                        else tuple([k[:n_tb_imgs]for k in batch_comp]) 
                         for i, batch_comp in enumerate(tb_img_batch)])

  model.train()

  for epoch in range(epochs):
    print(f"===== Epoch {epoch} =====")
    # TODO: Maybe do this at the end of the epoch.
    DAL_weight = DAL_weight_init * np.exp(-epoch * DAL_weight_decay)

    running_loss = 0.

    for batch_index, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
      checkpoint_manager.inform(epoch, batch_index)

      model.clear_paddings()
      # TODO: make stack_factor a param at init
      MSE_loss, DA_loss, A_np = model.calc_loss(*batch)
      loss = MSE_loss + DA_loss * DAL_weight

      model.zero_grad()
      loss.backward()
      optimizer.step()

      running_loss += loss.item()

      # Record training loss
      if batch_index % train_loss_plot_interval == 0:
        tensorboard_writer.add_scalar('train loss',
                                      running_loss / train_loss_plot_interval,
                                      epoch * len(train_loader) + batch_index)
        running_loss = 0.
      
      # Get and record test loss
      if batch_index % test_interval == 0:
        test_loss = 0.
        n_test_batches = max_test_batches

        model.eval()
        model.clear_paddings()

        with torch.no_grad():
          for bi, batch in enumerate(test_loader):
            
            if bi == max_test_batches:
              break
            # TODO: make stack_factor a param at init
            MSE_loss, DA_loss, A_np = model.calc_loss(*batch)
            loss = MSE_loss + DA_loss * DAL_weight
            test_loss += loss.detach().cpu().item()
            
          # If train loader was fully iterated over without early stopping.
          else:
            n_test_batches = len(test_loader)

        model.clear_paddings()

        tensorboard_writer.add_scalar('test loss',
                                      test_loss / n_test_batches,
                                      epoch * len(train_loader) + batch_index)
        
        tb_pred_imgs, tb_A = model(*tb_img_batch, stack=True)
        tb_pred_imgs = tb_pred_imgs.unsqueeze(1)
        tb_src_imgs, tb_tgt_imgs, _, _, = tb_img_batch
        
        n_frames_diff = tb_pred_imgs.shape[-1] - tb_src_imgs.shape[-1]
        tb_src_imgs = append_zero_frame(tb_src_imgs, n_frames=n_frames_diff).unsqueeze(1)
        tb_tgt_imgs = append_zero_frame(tb_tgt_imgs, n_frames=n_frames_diff).unsqueeze(1)
        # tb_imgs = torch.cat([tb_src_imgs, tb_tgt_imgs, tb_pred_imgs], dim=0).unsqueeze(1)
        tb_img_grid = make_grid([*tb_src_imgs.unbind(0), *tb_tgt_imgs.unbind(0), *tb_pred_imgs.unbind(0)], nrow=3)

        tensorboard_writer.add_image('test spectrograms', tb_img_grid)
        
        # Free up VRAM
        for unneeded_var in [MSE_loss, DA_loss, A_np, loss, tb_pred_imgs, tb_src_imgs, tb_tgt_imgs]:
          del(unneeded_var)
          
        torch.cuda.empty_cache()
        gc.collect()
        model.train()


  '''
   "# losses = []\n",
    "epoch_markers = []\n",
    "\n",
    "w_da_init = _w_da_init\n",
    "\n",
    "for i in range(4):\n",
    "  w_da = w_da_init * np.exp(-i * w_da_decay)\n",
    "\n",
    "save_every = 100\n",
    "plot_every = 20\n",
    "\n",
    "imshow_batch = next(iter(loader))\n",
    "maxlen_idx = np.argmax([torch.sum(mask).item() for mask in list(imshow_batch[3])])\n",
    "src_mel, tgt_mel, _, _, src_info, tgt_info = [x[maxlen_idx] for x in imshow_batch]\n",
    "\n",
    "fig, axes = plt.subplots(3, 1, figsize=(10, 12))\n",
    "fig.suptitle(f'Train Loss | lr={lrate}; sf={sf}; w_da={_w_da_init}; bs={batch_size}')\n",
    "\n",
    "for epoch in range(epochs):\n",
    "  print(f\"===== Epoch {epoch} =====\")\n",
    "  \n",
    "  # w_da = w_da_init * np.exp(-epoch * w_da_decay)\n",
    "  \n",
    "  for bi, batch in tqdm(enumerate(loader), total=len(dataset) // batch_size):\n",
    "  # for bi, batch in enumerate(loader):\n",
    "    model.clear_paddings()\n",
    "    mse_loss, da_loss, att_np = model.calc_loss(*batch, stack_factor=sf)\n",
    "    loss = mse_loss + w_da * da_loss\n",
    "    model.clear_paddings()\n",
    "    loss_val = loss.item()\n",
    "    losses.append(loss_val)\n",
    "    \n",
    "    model.zero_grad()\n",
    "    loss.backward()\n",
    "    optimizer.step()\n",
'''

############
### MAIN ###
############

def main():

  parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
  
  parser.add_argument('--dataset-config-path', type=str, default="", metavar='STR',
                      help='Optional path to a dataset config file (.json). Used instead of parsed arguments.\n\n')
  
  parser.add_argument('--dataset-dir', type=str, default='../Data/processed/VCTK', metavar='STR',
                      help="Directory containing the 'melspec' and 'transcript' folders, and 'speaker_info.csv'.")
  parser.add_argument('--min-samples', type=int, default=8, metavar='INT',
                      help='Minimum number of samples for a transcript to be included in the train-/testset.')
  parser.add_argument('--train-set-threshold', type=int, default=10, metavar='INT',
                      help='Cut-off point for transcript samples between train and test set. ' + \
                            'All transcripts whose nr. of samples are below this threshold are part of the test set. All others of the train set.')
  parser.add_argument('--sample-pairing', nargs='?', type=str, default=['product', 'random'], metavar='STR',
                      help='How samples are paired into source and target. Choose `product` for the Cartesian product. ' + \
                            'Choose `random` to randomly pair a source sample to every target sample. ' + \
                            'You can also specify it separately for the train and test set respectively. e.g. "--sample-pairing product random".')        
  parser.add_argument('--preload-melspecs', action='store_true',
                        help='Load all mel-spectrograms in RAM to avoid continuous file I/O.\n\n\n')                
  
  parser.add_argument('--model-config-path', type=str, default="", metavar='STR',
                      help='Optional path to a model config file (.json). Used instead of parsed arguments.\n\n')
  
  parser.add_argument('--model-class', '-m', type=str, default='KenkuTeacher', metavar='STR',
                      help='Class of the model you wish to train: KenkuTeacher or KenkuStudent.')
  parser.add_argument('--in-ch', type=int, default=80, metavar='INT',
                      help='Nr. of input (i.e. frequency) channels.')
  parser.add_argument('--conv-ch', type=int, default=512, metavar='INT',
                      help='Nr. of convolutional channels.')
  parser.add_argument('--att-ch', type=int, default=512, metavar='INT',
                      help='Nr. of attention channels.')
  parser.add_argument('--out-ch', type=int, default=80, metavar='INT',
                      help='Nr. of output (i.e. frequency) channels.')
  parser.add_argument('--embed-ch', type=int, default=32, metavar='INT',
                      help='Nr. of speaker info embedding channels.')
  parser.add_argument('--num-accents', type=int, default=11, metavar='INT',
                      help='Nr. of unique accents present in the data.')
  parser.add_argument('--stack-factor', '-sf', type=int, default=4, metavar='INT',
                      help='Stacking factor used for frame stacking. Reduces signal length by the same factor.\n\n\n')
  
  
  parser.add_argument('--train-config-path', type=str, default="", metavar='STR',
                      help='Optional path to a training config file (.json). Used instead of parsed arguments.\n\n')
  
  parser.add_argument('--epochs', type=int, default=10, metavar='INT',
                      help='Nr. of epochs over the dataset.')
  parser.add_argument('--batch-size', '-bs', type=int, default=20, metavar='INT',
                      help='Batch size.')
  parser.add_argument('--learning-rate', '-lr', type=float, default=5e-5, metavar='FLOAT',
                      help='Learning rate.')
  parser.add_argument('--adam-betas', type=float, nargs=2, default=[0.9, 0.999],  metavar='FLOAT',
                      help='Betas use for the Adam optimizer.')
  parser.add_argument('--DAL-weight', '-wda', type=float, default=0., metavar='FLOAT',
                      help='Starting value of the diagonal attention loss weight.')
  parser.add_argument('--DAL-weight-decay', '-wdad', type=float, default=None, metavar='FLOAT',
                      help='Decay rate for the diagonal attention loss weight. Defaults to 4 / epochs. ' + \
                            'Decay steps are done as wda <- wda * exp(-epoch * wda_decay).')
  parser.add_argument('--test-interval', type=int, default=100, metavar='INT',
                      help='Amount of update steps between every test loss calculation.')
  parser.add_argument('--max-test-batches', type=int, default=100, metavar='INT',
                      help='Max nr. of batches to calculate the test loss over')
  parser.add_argument('--tensorboard-dir', type=str, default=None, metavar='STR',
                      help='Directory to store Tensorboard logs in. Defaults to `./tensorboard/{--model-class}`.')
  parser.add_argument('--checkpoint-dir', type=str, default=None, metavar='STR',
                      help='Directory to store checkpoints in. Defaults to `train/checkpoints/{--model-class}/{<datetime>}`. ' + \
                            'Where <datetime> is the date and time at script execution. Checkpoints include both model and optimizer params.')
  parser.add_argument('--checkpoint-interval', type=int, default=100, metavar='INT',
                      help='Amount of update steps between each checkpoint.')
  parser.add_argument('--checkpoint-max', type=int, default=20, metavar='INT',
                      help='Maximum number of checkpoints saved on disk, FIFO.')
  parser.add_argument('--from-checkpoint', type=str, default=None, metavar='STR',
                      help='Path pointing to a checkpoint file. If specified, continue training from this checkpoint.')
  
  
  #=== Merge Command Line and Config File Arguments ===#
  
  args = parser.parse_args()
  args_dict = vars(args)
  
  # Fix argument formatting
  n_args_sample_pairing = len(args_dict['sample_pairing'])
  assert 1 <= n_args_sample_pairing <= 2, \
    f'Incorrect nr. of sample pairing arguments ({n_args_sample_pairing}: {args_dict['sample_pairing']}). Expected 1 or 2.'
    
  if n_args_sample_pairing == 1:
    args_dict['sample_pairing'] = args_dict['sample_pairing'][0]
  
  # Dataset Config
  dataset_config_keys = ['dataset_dir', 'min_samples', 'train_set_threshold', 'sample_pairing',
                         'preload_melspecs']
  dataset_config = create_config_dict(args_dict, dataset_config_keys, args.dataset_config_path)
  
  # Model Config
  model_config_keys = ['model_class', 'in_ch', 'conv_ch', 'att_ch', 'out_ch', 
                       'embed_ch', 'num_accents', 'stack_factor']
  model_config = create_config_dict(args_dict, model_config_keys, args.model_config_path)
  
  # Training Config
  train_config_keys = ['epochs', 'learning_rate', 'adam_betas', 'batch_size', 'DAL_weight', 'DAL_weight_decay', 
                       'test_interval', 'max_test_batches', 'tensorboard_dir', 'checkpoint_dir', 'checkpoint_interval', 
                       'checkpoint_max', 'from_checkpoint']
  train_config = create_config_dict(args_dict, train_config_keys, args.train_config_path)
  
  #=== Load/Create Datasets ===#
  
  dataset_factory = ParallelDatasetFactory(dataset_dir = dataset_config['dataset_dir'])

  # for k, v in dataset_factory.transcript_dict.items():
  #   print(f'{len(v)}: {k}')
  
  train_set, test_set = dataset_factory.train_test_split(min_transcript_samples = dataset_config['min_samples'],
                                                         train_set_threshold    = dataset_config['train_set_threshold'],
                                                         sample_pairing         = dataset_config['sample_pairing'])
  if dataset_config['preload_melspecs']:
    train_set.preload_melspecs()
    test_set.preload_melspecs()

  data_loader_kwargs = {
    'batch_size'  : train_config['batch_size'],
    'shuffle'     : True,
    'num_workers' : 0,
    'collate_fn'  : collate_fn,
    'drop_last'   : True,
    # 'generator'   : torch.Generator(device=device)
  }
  train_loader = DataLoader(train_set, **data_loader_kwargs)
  test_loader  = DataLoader(test_set,  **data_loader_kwargs)
  print(len(test_loader))
  
  
  #=== Initialize Model ===#
  
  model_class = model_config['model_class'].lower().replace(' ', '')
  
  if model_class in ['kenkuteacher', 'teacher', 'teach', 'tea']:
    model = KenkuTeacher
  elif model_class in ['kenkustudent', 'student', 'stud', 'stu']:
    model = KenkuStudent
  else:
    raise ValueError('Incorrect model class. Use `KenkuTeacher` or `KenkuStudent`.')
  
  model_init_args = model_config.copy()
  for remove_key in ['model_class']:
    model_init_args.pop(remove_key)
    
  model = model(**model_init_args)
  
  
  #=== Initialize Optimizer ===#
  
  optimizer = torch.optim.Adam(model.parameters(),
                               lr    = train_config['learning_rate'],
                               betas = train_config['adam_betas'])
  
  
  #=== Load Checkpoint ===#
  
  if train_config['from_checkpoint']:
    checkpoint_load_path = train_config['from_checkpoint']
    checkpoint = torch.load(checkpoint_load_path, map_location=device, weights_only=True)
    
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    del(checkpoint)
    gc.collect()
  
  
  #=== Setup Checkpoint Manager ===#
  
  timestamp = timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
  checkpoint_dir = train_config['checkpoint_dir']
  
  if not checkpoint_dir:
    checkpoint_dir = os.path.join(current_file_dir, 'checkpoints', model_class, timestamp)
  
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  
  checkpoint_manager = CheckpointManager(model, 
                                         optimizer, 
                                         checkpoint_dir,
                                         interval = train_config['checkpoint_interval'],
                                         max      = train_config['checkpoint_max'])
  
  # Save config files
  save_config(dataset_config, os.path.join(checkpoint_dir, 'dataset_config.json'))
  save_config(model_config,   os.path.join(checkpoint_dir, 'model_config.json'))
  save_config(train_config,   os.path.join(checkpoint_dir, 'train_config.json'))
  
  
  #=== Setup Tensorboard Writer ===#
  
  tensorboard_dir = train_config['tensorboard_dir']
  
  if not tensorboard_dir:
    tensorboard_dir = os.path.join(current_file_dir, 'tensorboard', model_class, timestamp)
  
  if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)
  
  tensorboard_writer = SummaryWriter(tensorboard_dir)
  
  #=== Write Config Files ===#

  save_config(dataset_config, os.path.join(checkpoint_dir, 'dataset_config.json'))
  save_config(model_config,   os.path.join(checkpoint_dir, 'model_config.json'))
  save_config(train_config,   os.path.join(checkpoint_dir, 'train_config.json'))
  
  #=== Start Training ===#

  train_model(model,
              optimizer,
              train_loader,
              test_loader,
              checkpoint_manager,
              tensorboard_writer,
              epochs           = train_config['epochs'],
              test_interval    = train_config['test_interval'],
              DAL_weight       = train_config['DAL_weight'],
              DAL_weight_decay = train_config['DAL_weight_decay'],
              max_test_batches = train_config['max_test_batches']
              )

if __name__ == '__main__':
  main()
