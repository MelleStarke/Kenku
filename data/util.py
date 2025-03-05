import os
import json

from typing import Any

from torch import is_tensor


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
    
def merge_config(main_config: dict, sub_config: dict):
  """
  Merges the provided dict with the json file.
  Entries in main_config take precedence over those in sub_config.
  """
  merged_config = {**sub_config, **main_config}
  return merged_config
  
def walk_files(root, extension):
  """
  Recursively finds and provides full paths of all files 
  with the specified extension in the provided root directory.
  """
  for path, dirs, files in os.walk(root):
    for file in files:
      if file.endswith(extension):
        yield os.path.join(path, file)
              
def recursive_to_device(xs: Any, device):
  if is_tensor(xs):
    return xs.to(device=device)
  
  try:
    xs[0]
    xs_type = type(xs)
    return xs_type([recursive_to_device(x, device) for x in xs])
  
  except IndexError as e:
    err_msg = f"{e}\nType {type(x)} isn't indexable"
    raise IndexError(err_msg)