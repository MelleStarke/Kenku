import sys, os
import argparse
import subprocess

from itertools import product

from data.util import save_config, load_config

variable_settings = {
  'stack_factor': [2, 4, 8],
  'hidden_ch': [64, 128],
  'dropout_rate': [0.2, 0.4],
  'learning_rate': [1e-5, 5e-5, 1e-6],
  'att_weight': [200, 2000],
  'OAL_weight_on': [True],
  'att_weight_decay': [0.0667, 0.0167],
}
# variable_settings = {
#   'stack_factor': [4],
#   'hidden_ch': [1, 12],
#   'dropout_rate': [0.2, 0.5],
#   'learning_rate': [5e-5],
#   'att_weight': [2000.],
#   'OAL_weight_on': [True],
#   'att_weight_decay': [16],
# }

shorthand_setting_names = {
  'stack_factor': 'sf',
  'hidden_ch': 'hc',
  'learning_rate': 'lr',
  'dropout_rate': 'dr',
  'att_weight': 'aw',
  'OAL_weight_on': 'oal',
  'att_weight_decay': 'awd',
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
  
  if args.setting_index == 1:
    save_config(static_settings, os.path.join(args.job_dir, 'static_settings.json'))
    save_config(variable_settings, os.path.join(args.job_dir, 'variable_settings.json'))
    
  this_setting = get_setting(variable_settings, args.setting_index)
  
  print("Setting: ", this_setting)
  
  # Make descriptive name of this run's folder
  folder_name = ' '.join([f"{shorthand_setting_names[k]}={v}" for k, v in this_setting.items()])
  run_dir = os.path.join(args.job_dir, folder_name)
  # Make directory and save configs
  os.makedirs(run_dir, exist_ok=True)

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
  
  #=== Settings Specified at Runtime ===#
  if 'n_cores' in this_setting:
    raise ValueError('`n_cores` should be specified through the shell script directly, not in the .json files.')
  this_setting['n_cores'] = args.n_cores,
  
  if 'run_dir' in this_setting:
    raise ValueError('`run_dir` is decided at runtime dynamically by the hypertune script.')
  this_setting['run_dir'] = run_dir
  
  #=== Construct Config Dicts ===#
  dataset_config = {k: this_setting[k] for k in [
    'dataset_dir',
    'n_cores',
    'min_samples',
    'train_set_threshold',
    'sample_pairing',
    'no_downsample',
    'preload_melspecs'
  ]}
  
  model_config = {k: this_setting[k] for k in [
    'model_class',
    'drl',
    'from_teacher',
    'in_ch',
    'conv_ch',
    'att_ch',
    'out_ch',
    'embed_ch',
    'num_accents',
    'stack_factor',
    'dropout_rate'
  ]}

  train_config = {k: this_setting[k] for k in [
    'epochs',
    'batch_size',
    'main_loss',
    'learning_rate',
    'adam_betas',
    'DAL_weight',
    'OAL_weight',
    'att_weight_decay',
    'test_interval',
    'melspec_interval',
    'max_test_batches',
    'run_dir',
    'checkpoint_interval',
    'checkpoint_max',
    'from_checkpoint'
  ]}
  
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