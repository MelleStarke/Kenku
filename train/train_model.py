import argparse
import os, sys
import logging
import json

from pathlib import Path
from datetime import datetime

from typing import Union, List, Tuple, Optional


import numpy as np

import torch 
from torch import nn, Tensor
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


################
### Training ###
################

def train_model(model, model_config, train_config):
  pass


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


###########
### MAIN ###
############

def main():

  parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
  
  parser.add_argument('--dataset-config-path', '-dc', type=str, default="",
                      help='Optional path to a dataset config file (.json). Used instead of parsed arguments.\n\n')
  
  parser.add_argument('--dataset-dir', '-md', type=str, default='../Data/processed/VCTK/melspec',
                      help="Directory containing the 'melspec' and 'transcript' folders, and 'speaker_info.csv'.")
  parser.add_argument('--min-samples', '-ms', type=int, default=3,
                      help='Minimum number of samples for a transcript to be included in the train-/testset.')
  parser.add_argument('--train-set-threshold', '-tst', type=int, default=10,
                      help='Cut-off point for transcript samples between train and test set. ' + \
                            'All transcripts whose nr. of samples are below this threshold are part of the test set. All others of the train set.')
  parser.add_argument('--sample-pairing', '-sp', type=str, default='product',
                      help='How samples are paired into source and target. Choose `product` for the Cartesian product. ' + \
                            'Choose `random` to randomly pair a source sample to every target sample.\n\n\n')                             
  
  parser.add_argument('--model-config-path', '-mc', type=str, default="",
                      help='Optional path to a model config file (.json). Used instead of parsed arguments.\n\n')
  
  parser.add_argument('--model-class', '-m', type=str, default='KenkuTeacher',
                      help='Class of the model you wish to train: KenkuTeacher or KenkuStudent.')
  parser.add_argument('--in-ch', '-ich', type=int, default=80,
                      help='Nr. of input (i.e. frequency) channels.')
  parser.add_argument('--conv-ch', '-cch', type=int, default=80,
                      help='Nr. of convolutional channels.')
  parser.add_argument('--att-ch', '-ach', type=int, default=80,
                      help='Nr. of attention channels.')
  parser.add_argument('--out-ch', '-och', type=int, default=80,
                      help='Nr. of output (i.e. frequency) channels.')
  parser.add_argument('--embed-ch', '-ech', type=int, default=80,
                      help='Nr. of speaker info embedding channels.')
  parser.add_argument('--num-accents', '-na', type=int, default=11,
                      help='Nr. of unique accents present in the data.')
  parser.add_argument('--stack-factor', '-sf', type=int, default=4,
                      help='Stacking factor used for frame stacking. Reduces signal length by the same factor.\n\n\n')
  
  
  parser.add_argument('--train-config-path', '-tc', type=str, default="",
                      help='Optional path to a training config file (.json). Used instead of parsed arguments.\n\n')
  
  parser.add_argument('--epochs', '-ep', type=int, default=10,
                      help='Nr. of epochs over the dataset.')
  parser.add_argument('--learning-rate', '-lr', type=float, default=5e-5,
                      help='Learning rate.')
  parser.add_argument('--batch_size', '-bs', type=int, default=20,
                      help='Batch size.')
  parser.add_argument('--DA-weight', '-wda', type=float, default=0,
                      help='Starting value of the diagonal attention loss weight.')
  parser.add_argument('--DA-weight-decay', '-wdad', type=float, default=None,
                      help='Decay rate for the diagonal attention loss weight. Defaults to 4 / epochs. ' + \
                            'Decay steps are done as wda <- wda * exp(-epoch * wda_decay).')
  parser.add_argument('--test-interval', '-ti', type=int, default=50,
                      help='Amount of update steps between every test loss calculation.')
  parser.add_argument('--tensorboard-dir', '-tbd', type=str, default=None,
                      help='Directory to store Tensorboard logs in. Defaults to `./tensorboard/{--model-class}`.')
  parser.add_argument('--checkpoint-dir', '-cd', type=str, default=None,
                      help='Directory to store checkpoints in. Defaults to `./checkpoints/{--model-class}/{<datetime>}`. ' + \
                            'Where <datetime> is the date and time at script execution. Checkpoints include both model and optimizer params.')
  parser.add_argument('--checkpoint-interval', '-ci', type=int, default=100,
                      help='Amount of update steps between each checkpoint.')
  parser.add_argument('--checkpoint-max','-cm', type=int, default=20,
                      help='Maximum number of checkpoints saved on disk, FIFO.')
  parser.add_argument('--from-checkpoint', type=str, default=None,
                      help='Path pointing to a checkpoint file. If specified, continue training from this checkpoint.')
  
  args = parser.parse_args()
  args_dict = vars(args)
  
  dataset_config_keys = ['dataset_dir', 'min_samples', 'train_set_threshold', 'sample_pairing']
  dataset_config = create_config_dict(args_dict, dataset_config_keys, args.dataset_config_path)
  
  model_config_keys = ['model_class', 'in_ch', 'conv_ch', 'att_ch', 'out_ch', 
                       'embed_ch', 'num_accents', 'stack_factor']
  model_config = create_config_dict(args_dict, model_config_keys, args.model_config_path)
  
  train_config_keys = ['epochs', 'learning_rate', 'batch_size', 'DA_weight', 'DA_weight_decay', 
                       'test_interval', 'tensorboard_dir', 'checkpoint_dir', 'checkpoint_interval', 
                       'checkpoint_max', 'from_checkpoint']
  train_config = create_config_dict(args_dict, train_config_keys, args.train_config_path)
  
  
  #=== Load Dataset ===#
  
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
  
  #=== Load Checkpoint ===#
  
  # if train_config['from_checkpoint']:
  
  #=== Setup Checkpoint Directory ===#
  
  timestamp = timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
  
  #=== Setup Tensorboard Directory ===#
  
  #=== Setup Training Parameters ===#
  
  #=== Start Training ===#

if __name__ == '__main__':
  main()
