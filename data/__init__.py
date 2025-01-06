import os
import torch
import json


DATA_PATH  = "/home/user/Uni/m3/thesis/project/Data"
KENKU_PATH = "/home/user/Uni/m3/thesis/project/Kenku"

#=== Voice Cloning Tool Kit Corpus ===#
RAW_VCTK_PATH       = os.path.join(DATA_PATH, "raw/VCTK-Corpus")
RAW_VCTK_AUDIO_PATH = os.path.join(RAW_VCTK_PATH, "wav48")

VCTK_PATH           = os.path.join(DATA_PATH, "processed/VCTK")
VCTK_MELSPEC_PATH   = os.path.join(VCTK_PATH, "melspec")

#=== Speech Accent Archive ===#_
RAW_SAA_PATH        = os.path.join(DATA_PATH, "raw/SAA")
RAW_SAA_AUDIO_PATH  = os.path.join(RAW_SAA_PATH, "recordings/recordings")

SAA_PATH            = os.path.join(DATA_PATH, "processed/SAA")
SAA_MELSPEC_PATH    = os.path.join(SAA_PATH, "melspec")


DEFAULT_CONFIG = {  # CURRENTLY BARELY USED. USING ARGPARSE DEFAULTS INSTEAD.
  'trim_silence': True,
  'top_db'      : 30,
  'num_mels'    : 80,
  
  'fft_size': 1024,
  'flen'    : 1024,
  
  'hop_size': 128,
  'fshift'  : 128,
  
  'min_freq': 80,
  'fmin'    : 80,
  
  'max_freq': 7600,
  'fmax'    : 7600,
  
  'samp_rate': 16000,
  'fs'       : 16000
}

###############
### Utility ###
###############

def get_torch_device():
  """
  Returns device of first found CUDA compatible GPU if available. CPU otherwise.
  """
  if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)   
  else:
    device = torch.device('cpu')
    
  return device
                
def load_config(config_path: str):
  """
  Loads the config found in config_path as a dict.
  """
  config = None
  with open(config_path, 'r') as file:
    config = json.load(file)
    
  return config

def save_config(config: dict, config_path: str):
  """
  Saves the config at config_path as json file.
  """
  with open(config_path, 'w') as file:
    json.dump(config, file, indent=4)
    
def merge_config(config: dict, config_path: str):
  """
  Merges the provided dict with the json file.
  """
  loaded_config = load_config(config_path)
  merged_config = {**loaded_config, **config}
  save_config(merged_config)
  return merged_config
  
def same_config(config: dict, config_path: str):
  """
  Checks if the provided config dict and config dict found at config_path are equal.
  """
  loaded_config = load_config(config_path)
  return config == loaded_config

def walk_files(root, extension):
  """
  Recursively finds and provides full paths of all files 
  with the specified extension in the provided root directory.
  """
  for path, dirs, files in os.walk(root):
    for file in files:
      if file.endswith(extension):
        yield os.path.join(path, file)
              
