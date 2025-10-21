import sys, os
import argparse
import subprocess

from itertools import product

from data.util import save_config, load_config

from train.train_model import DATASET_CONFIG_KEYS, MODEL_CONFIG_KEYS, TRAIN_CONFIG_KEYS


shorthand_setting_names = {
  'stack_factor': 'sf',
  'hidden_ch': 'hc',
  'learning_rate': 'lr',
  'dropout_rate': 'dr',
  'att_weight': 'aw',
  'OAL_weight_on': 'oal',
  'att_weight_decay': 'awd',
  'num_accents': 'nac',
  'min_samples': 'ms',
  'train_set_threshold': 'tst',
  'dropout_rate': 'dor',
  'main_loss': 'mlf',
  'tcvae_alpha': 'tca',
  'tcvae_beta': 'tcb',
  'tcvae_gamma': 'tcg',
  'n_thaw_layers': 'ntl',
  'ft_warmup_prop': 'ftw',
  'ft_thaw_prop': 'ftt',  
  'in_ch': 'ich',
  'conv_ch': 'cch',
  'att_ch': 'ach',
  'out_ch': 'och',
  'embed_ch': 'ech',
  'view_distance': 'avd'
}

def get_setting(config_dict, idx):
  all_setting_combos = list(product(*config_dict.values()))
  this_setting = all_setting_combos[idx - 1]
  this_setting = dict(zip(config_dict.keys(), this_setting))
  return this_setting

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Hyperparameter tuning for the model.')
  parser.add_argument('setting_dir', type=str, metavar='STR',
                      help="Directory containing `static_settings.json` and `variable_settings.json', from which to construct the configs for each run.")
  parser.add_argument('job_dir', type=str, metavar='STR',
                      help="Directory of the entire hypertuning task")
  parser.add_argument('setting_index', type=int, metavar='INT',
                      help="Index of the hyperparameter setting to run")
  parser.add_argument('--n-cores', type=int, default=None, metavar='INT',
                      help="Nr. of cores used for parallelization by the DataLoader. Defaults to os.cpu_count().")

  args = parser.parse_args()

  static_settings = load_config(os.path.join(args.setting_dir, 'static_settings.json'))
  variable_settings = load_config(os.path.join(args.setting_dir, 'variable_settings.json'))
  
  # Make sure there is no overlap between the static and variable settings
  static_keys_set = set(static_settings.keys())
  variable_keys_set = set(variable_settings.keys())
  keys_union = static_keys_set & variable_keys_set
  if len(keys_union) > 0:
    raise ValueError('There is at least one setting set as both static and variable. Make sure each setting is either static or variable' \
                    f'Setting(s) in question: {keys_union}')
    
  this_setting = get_setting(variable_settings, args.setting_index)
  
  print("Setting: ", this_setting)
  
  # Make descriptive name of this run's folder
  folder_name = ' '.join([f"{shorthand_setting_names.get(k, k)}={v}" for k, v in this_setting.items()])
  run_dir = os.path.join(args.job_dir, folder_name)
  # Make directory and save configs
  os.makedirs(run_dir, exist_ok=True)
  
  # Save the overall hypertune settings in the array-job root directory
  if args.setting_index == 1:
    save_config(static_settings, os.path.join(args.job_dir, 'static_settings.json'))
    save_config(variable_settings, os.path.join(args.job_dir, 'variable_settings.json'))

  this_setting.update(static_settings)
  
  #=== Process Hidden Channels Individually or In Tandem ===#
  sf = this_setting['stack_factor']
  if all(channel in this_setting for channel in ['conv_ch', 'att_ch']):
    for channel in ['conv_ch', 'att_ch']:
      this_setting[channel] = this_setting[channel] * sf
      
  elif 'hidden_ch' in this_setting:
    for channel in ['conv_ch', 'att_ch']:
      this_setting[channel] = this_setting['hidden_ch'] * sf
      
  else:
    raise ValueError('You need to specify the convolutional and attention channels either individually ' \
                     'as `conv_ch` and `att_ch`, or in tandem as `hidden_ch`.')
  
  #=== Process Attention Weights ===#
  if 'att_weight' in this_setting:
    for weight in ['DAL_weight', 'OAL_weight']:
      this_setting[weight] = this_setting['att_weight']
      
  else:
    raise ValueError('You need to specify the diagonal and orthogonal attention loss weights in tandem with `att_weight`.')
  
  #=== Account for View Distance and STack Factor ===#
  # By stacking frames, we essentally increase the view distance by that amount.
  # Therefore, we need to reduce the view distance by the stack factor to retain comparability.
  this_setting['view_distance'] = this_setting['view_distance'] // this_setting['stack_factor']
  
  #=== Settings Specified at Runtime ===#
  if 'n_cores' in this_setting:
    raise ValueError('`n_cores` should be specified through the shell script directly, not in the .json files.')
  this_setting['n_cores'] = args.n_cores[0] \
                            if isinstance(args.n_cores, (tuple, list)) \
                            else args.n_cores
  
  if 'run_dir' in this_setting:
    raise ValueError('`run_dir` is decided at runtime dynamically by the hypertune script.')
  this_setting['run_dir'] = run_dir
  
  #=== Construct Config Dicts ===#
  dataset_config = {k: this_setting[k] for k in DATASET_CONFIG_KEYS if k in this_setting}
  model_config   = {k: this_setting[k] for k in MODEL_CONFIG_KEYS   if k in this_setting}
  train_config   = {k: this_setting[k] for k in TRAIN_CONFIG_KEYS   if k in this_setting}
  
  # Save configs to the run directory
  for name, config in zip(['dataset', 'model', 'train'], [dataset_config, model_config, train_config]):
    config_path = os.path.join(run_dir, f"{name}_config.json")
    save_config(config, config_path)
    
  
  # Call the training script and display its output
  train_command = [
    'python', 
    '-m', 'train.train_model', 
    '--config-dir', run_dir,
  ]

  try:
    # Use subprocess.run to execute the training script and stream its output
    result = subprocess.run(
      train_command, 
      check=True,  # Raise CalledProcessError if the command returns a non-zero exit code
      stdout=sys.stdout,  # Redirect stdout to the current terminal
      stderr=sys.stderr   # Redirect stderr to the current terminal
    )
    print(f"Training script completed successfully for setting: {folder_name}")
  
  except subprocess.CalledProcessError as e:
    print(f"Training script failed for setting: {folder_name}")
    print(f"Exit code: {e.returncode}")
    sys.exit(e.returncode)

  
  
  # min_samples: 5
  # sample_pairing: 'product'
  # train_set_threshold: 10

  # stack_factor: [2, 4, 8]
  # conv_ch and att_ch (times stack_factor): [1, 12, 128] 
  # embed_ch: 16
  # dropout_rate: [0.2, 0.5]

  # epochs: 30
  # main_loss: 'mse'
  # learning_rate: [1e-5, 5e-5, 1e-6]
  # att_weight: [200, 2000]
  # OAL_weight: ['on', 'off']
  # att_weight_decay: [4, 16]
  # L2_reg_weight: 0.001
  # test_interval: 100
  # melspec_interval: 200
  # checkpoint_interval: 200
  # checkpoint_max: 2