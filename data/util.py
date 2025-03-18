import os
import json

from typing import Any, Callable, Tuple, List

from torch import Tensor, is_tensor, DeviceObjType


def load_config(config_path: str):
  """
  Loads the config found in config_path as a dict.
  """
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

def config_to_str(config: dict, prefix=''):
  """Recursively format a config dict as a string.

  Args:
      config (dict): Possibly nested config dictionary.
      prefix (str, optional): Prefix used to correctly indent items. Defaults to ''.
  """
  lines = []
  
  for k, v in config.items():
    if isinstance(v, dict):
      lines.append(f"\n{prefix}{k}")
      sub_prefix = f'\t{prefix}'
      sub_config = config_to_str(v, prefix=sub_prefix)
      lines.append(sub_config)
      
    else:
      lines.append(f'{prefix}{k}: {v}')
  
  config_string = '\n'.join(lines)
  return config_string
  
def walk_files(root, extension):
  """
  Recursively finds and provides full paths of all files 
  with the specified extension in the provided root directory.
  """
  for path, dirs, files in os.walk(root):
    for file in files:
      if file.endswith(extension):
        yield os.path.join(path, file)
              
def recursive_to_device(xs: Union[Tensor, List, Tuple], device: DeviceObjType):
  """Recursively sends all tensors in a container to provided device. Keeps same structure.

  Args:
      xs (Union[Tensor, List, Tuple]): Tensor or (nested) container of Tensors.
      device (DeviceObjType): PyTorch device.

  Raises:
      IndexError: if xs isn't indexable.

  Returns:
      _type_: _description_
  """
  if is_tensor(xs):
    return xs.to(device=device)
  
  try:
    xs[0]
    xs_type = type(xs)
    return xs_type([recursive_to_device(x, device) for x in xs])
  
  except IndexError as e:
    raise IndexError(f"Type {type(x)} isn't indexable.") from e
  
def recursive_map(xs: Any, fn: Callable, cond = is_tensor):
  if cond(xs):
    return fn(xs)
  
  try:
    xs[0]
    xs_type = type(xs)
    return xs_type([recursive_map(x, fn, cond) for x in xs])
  
  except IndexError as e:
    err_msg = f"{e}\nType {type(xs)} isn't indexable"
    raise IndexError(err_msg)