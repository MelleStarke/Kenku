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

from tqdm import tqdm

from data.load import ParallelDatasetFactory, ParallelMelspecDataset, collate_fn
from data.util import save_config, load_config
from kenku.modules import KameBlock
from kenku.network import KenkuTeacher, stack_frames, unstack_frames


device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
                DAL_weight_decay: float = None):

  print('in train func')
  return

  DAL_weight_init = DAL_weight
  train_loss_plot_interval = 100

  # Define set of src and tgt melspecs to be used as images in Tensorboard.
  # tb_img_batch will be fed into the network to visually assess performance.
  tb_img_batch = next(iter(train_loader))
  n_tb_imgs = min(10, len(tb_img_batch[0]))
  tb_img_batch = tuple([batch_comp[:n_tb_imgs] for batch_comp in tb_img_batch])

  model.train()

  for epoch in range(epochs):
    print(f"===== Epoch {epoch} =====")
    # TODO: Maybe do this at the end of the epoch.
    DAL_weight = DAL_weight_init * np.exp(-epoch * DAL_weight_decay)

    running_loss = 0.

    for batch_index, batch in tqdm(enumerate(train_loader)):
      checkpoint_manager.inform(epoch, batch_index)

      model.clear_paddings()
      # TODO: make stack_factor a param at init
      MSE_loss, DA_loss, A_np = model.calc_loss(*batch, stack_factor=4)
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

        model.eval()
        model.clear_paddings()

        with torch.no_grad():
          for batch in test_loader:
            # TODO: make stack_factor a param at init
            MSE_loss, DA_loss, A_np = model.calc_loss(*batch, stack_factor=4)
            loss = MSE_loss + DA_loss * DAL_weight
            test_loss += loss.item()

        model.train()

        tensorboard_writer.add_scalar('test loss',
                                      test_loss / len(test_loader),
                                      epoch * len(train_loader) + batch_index)
        
        tb_pred_imgs, tb_A = model(*tb_img_batch)
        tb_src_imgs, tb_tgt_imgs, _, _, _, _, = tb_img_batch

        
        gc.collect()


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
  parser.add_argument('--min-samples', type=int, default=3, metavar='INT',
                      help='Minimum number of samples for a transcript to be included in the train-/testset.')
  parser.add_argument('--train-set-threshold', type=int, default=10, metavar='INT',
                      help='Cut-off point for transcript samples between train and test set. ' + \
                            'All transcripts whose nr. of samples are below this threshold are part of the test set. All others of the train set.')
  parser.add_argument('--sample-pairing', type=str, default='product', metavar='STR',
                      help='How samples are paired into source and target. Choose `product` for the Cartesian product. ' + \
                            'Choose `random` to randomly pair a source sample to every target sample.')        
  parser.add_argument('--preload-melspecs', action='store_true',
                        help='Load all mel-spectrograms in RAM to avoid continuous file I/O.\n\n\n')                
  
  parser.add_argument('--model-config-path', type=str, default="", metavar='STR',
                      help='Optional path to a model config file (.json). Used instead of parsed arguments.\n\n')
  
  parser.add_argument('--model-class', '-m', type=str, default='KenkuTeacher', metavar='STR',
                      help='Class of the model you wish to train: KenkuTeacher or KenkuStudent.')
  parser.add_argument('--in-ch', type=int, default=80, metavar='INT',
                      help='Nr. of input (i.e. frequency) channels.')
  parser.add_argument('--conv-ch', type=int, default=80, metavar='INT',
                      help='Nr. of convolutional channels.')
  parser.add_argument('--att-ch', type=int, default=80, metavar='INT',
                      help='Nr. of attention channels.')
  parser.add_argument('--out-ch', type=int, default=80, metavar='INT',
                      help='Nr. of output (i.e. frequency) channels.')
  parser.add_argument('--embed-ch', type=int, default=80, metavar='INT',
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
  parser.add_argument('--test-interval', type=int, default=50, metavar='INT',
                      help='Amount of update steps between every test loss calculation.')
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
                       'test_interval', 'tensorboard_dir', 'checkpoint_dir', 'checkpoint_interval', 
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
    'num_workers' : 2,
    'collate_fn'  : collate_fn,
    'drop_last'   : False,
    # 'generator'   : torch.Generator(device=device)
  }
  train_loader = DataLoader(train_set, **data_loader_kwargs)
  test_loader  = DataLoader(test_set,  **data_loader_kwargs)
  
  
  #=== Initialize Model ===#
  
  model_class = model_config['model_class'].lower().replace(' ', '')
  
  if model_class in ['kenkuteacher', 'teacher', 'teach', 'tea']:
    model = KenkuTeacher
  elif model_class in ['kenkustudent', 'student', 'stud', 'stu']:
    model = KenkuStudent
  else:
    raise ValueError('Incorrect model class. Use `KenkuTeacher` or `KenkuStudent`.')
  
  model_init_args = model_config.copy()
  for remove_key in ['model_class', 'stack_factor']:
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
              DAL_weight_decay = train_config['DAL_weight_decay']
              )

if __name__ == '__main__':
  main()
