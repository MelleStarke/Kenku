import argparse
import os, sys
import logging

from typing import Union, List, Tuple, Optional

from pathlib import Path

import numpy as np

import torch 
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

# Local imports
sys.path.append(os.path.abspath('../data'))
sys.path.append(os.path.abspath('../kenku'))

from load import ParallelDatasetFactory, ParallelMelspecDataset, collate_fn
from modules import KameBlock
from network import KenkuTeacher, stack_frames, unstack_frames


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


###########
### MAIN ###
############

def main():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset-config', '-dc', type=str, default="",
                        help='Optional path to a dataset config file (.json). Used instead of parsed arguments.\n')
    
    parser.add_argument('--melspec-dir', '-md', type=str, default='../Data/processed/VCTK/melspec',
                        help="Directory containing the folders of each speaker's mel-spectrograms.")
    parser.add_argument('--transcript-dir', '-td', type=str, default='../Data/processed/VCTK/transcript_standardized',
                        help="Directory containing the folders of each speaker's transcripts.")
    parser.add_argument('--speaker-info-dir', '-sid', type=str, default='../Data/processed/VCTK/speaker_info.csv',
                        help="Path to the speaker_info.vsc file.")
    parser.add_argument('--min-samples', '-ms', type=int, default=3,
                        help='Minimum number of samples for a transcript to be included in the train-/testset.')
    parser.add_argument('--train-set-threshold', '-tst', type=int, default=10,
                        help='Cut-off point for transcript samples between train and test set. ' + \
                             'All transcripts whose nr. of samples are below this threshold are part of the test set. All others of the train set.')
    parser.add_argument('--sample-pairing', '-sp', type=str, default='product',
                        help='How samples are paired as source and target. Choose `product` for the Cartesian product. ' + \
                             'Choose `random` to randomly pair a source sample to every target sample.\n\n')                             
    
    parser.add_argument('--model-config', '-mc', type=str, default="",
                        help='Optional path to a model config file (.json). Used instead of parsed arguments.\n')
    
    parser.add_argument('--model-class', '-m', type=str, default='KenkuTeacher',
                        help='Class of the model you wish to train: KenkuTeacher or KenkuStudent.')
    parser.add_argument('--in-channels', '-ich', type=int, default=80,
                        help='Nr. of input (i.e. frequency) channels.')
    parser.add_argument('--conv-channels', '-cch', type=int, default=80,
                        help='Nr. of convolutional channels.')
    parser.add_argument('--att-channels', '-ach', type=int, default=80,
                        help='Nr. of attention channels.')
    parser.add_argument('--out-channels', '-och', type=int, default=80,
                        help='Nr. of output (i.e. frequency) channels.')
    parser.add_argument('--embed-channels', '-ech', type=int, default=80,
                        help='Nr. of speaker info embedding channels.')
    parser.add_argument('--num-accents', '-na', type=int, default=11,
                        help='Nr. of unique accents present in the data.')
    parser.add_argument('--stack-factor', '-sf', type=int, default=4,
                        help='Stacking factor used for frame stacking. Reduces signal length by the same factor.\n\n')
    
    
    parser.add_argument('--train-config', '-tc', type=str, default="",
                        help='Optional path to a training config file (.json). Used instead of parsed arguments.\n')
    
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
    
    args = parser.parse_args()

    # src_dir = args.src
    # dest_dir = args.dst
    # extension = args.ext
    # configpath = args.conf
    
    # convert = not args.no_convert
    
    # # Torch device setting
    # device = get_torch_device()
    

    # #=== Data Conversion ===#
    
    # if convert:
    #   # Create a data configuration dictionary and save it as JSON
    #   data_config = {
    #       'num_mels': args.num_mels,
    #       'samp_rate': args.samp_rate,
    #       'fft_size': args.fft_size,
    #       'hop_size': args.hop_size,
    #       'min_freq': args.min_freq,
    #       'max_freq': args.max_freq,
    #       'trim_silence': args.trim_silence,
    #       'top_db': args.top_db
    #   }
      
    #   overwrite = not args.no_overwrite
      
    #   if not os.path.exists(os.path.dirname(configpath)):
    #     os.makedirs(os.path.dirname(configpath))
      
    #   # Warn the user if trying to keep melspec data made with a different config.
    #   elif not overwrite and not same_config(data_config, configpath):
    #     logger.warning("Keeping old melspecs despite configs not matching.\n" +\
    #                   f"\tOld config:\n{dict(sorted(load_config(configpath).items()))}\n\n" +\
    #                   f"\tNew config:\n{dict(sorted(data_config.items()))}\n")
        
    #   save_config(data_config, configpath)


if __name__ == '__main__':
    main()
