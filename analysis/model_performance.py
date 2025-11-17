from data.util import load_config, recursive_to_device
from data.load import ParallelDatasetFactory
from network import KenkuTeacher, KenkuStudent, DRLKenkuTeacher, DRLKenkuStudent
from data.augment import get_augment_fns
from data.load import augment_collate_fn
from analysis.metrics import (
  mutual_information_gap, 
  mel_cepstral_distortion, 
  accent_entropy
)
from tqdm import tqdm
import pickle
import torch
from torch.utils.data import DataLoader
import os
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ =="__main__":
  
  parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
  
  parser.add_argument("model_dir", type=str, metavar='STR',
                      help="Path to the model directory containing model_config.json and the .pt checkpoint (first one found).")
  parser.add_argument('--dataset-dir', type=str, default='../Data/processed/VCTK', metavar='STR',
                      help="Directory containing the 'melspec' and 'transcript' folders, and 'speaker_info.csv'. Defaults to ../Data/processed/VCTK.")
  parser.add_argument('--batch-size', '-bs', type=int, default=16, metavar='INT',
                      help='Batch size. Defaults to 16.')
  parser.add_argument('--n-cores', type=int, default=None, metavar='INT',
                      help="Nr. of cores used for parallelization by the DataLoader. Defaults to os.cpu_count().")
  parser.add_argument('--min-samples', type=int, default=7, metavar='INT',
                      help='Minimum number of samples for a transcript to be included in the testset. Defaults to 7')
  parser.add_argument('--max-samples', type=int, default=10, metavar='INT',
                      help='Maximum amount of samples for a transcript to be included in the test set. '
                           'Functions like `train-set-threshold`. Defaults to 10.')
  parser.add_argument('--sample-pairing', type=str, default='product', metavar='STR',
                      help=('How samples are paired into source and target. Choose `product` for the Cartesian product. ' 
                            'Choose `random` to randomly pair a source sample to every target sample. Defaults to product.'))
  parser.add_argument('--no-downsample', action='store_true',
                      help='Disable downsampling of sentence samples. Results in skewed data but may not be an issue.')
  parser.add_argument('--scaler-filepath', type=str, default='../Data/processed/VCTK/norm_scaler.pkl', metavar='STR',
                      help='Path to the scaler file used to normalize the data. Defaults to `../Data/processed/VCTK/norm_scaler.pkl`.')
  
  args = parser.parse_args()
  args_dict = vars(args)
  
  model_dir = args_dict['model_dir']
  
  model_config = load_config(os.path.join(model_dir, 'model_config.json'))
  
  model_class = model_config['model_class'].lower().replace(' ', '')
  use_drl = 'drl' in model_class or model_config['drl']
  model_config['drl'] = use_drl
  
  if 'teacher' in model_class:
    if use_drl:
      model = DRLKenkuTeacher
    else: 
      model = KenkuTeacher
    is_student = False
    
  elif 'student' in model_class:
    if use_drl:
      model = DRLKenkuStudent
    else:
      model = KenkuStudent
    is_student = True
    
  else:
    raise ValueError('Incorrect model class. Use `teacher` or `student`.')
  
  if is_student:
    assert model_config['from_teacher'] is not None, \
      "The student model expects `from_teacher` to point to a teacher checkpoint, but it is `None`."
  
  model_init_args = model_config.copy()
  for pop_arg in ['model_class', 'from_teacher', 'drl']:
    model_init_args.pop(pop_arg, None)
    
    
  model = model(**model_init_args)
  # Init embed layer to avoid mismatched state dicts
  model._init_embed_layer(use_drl=use_drl)
  
  #=== Load checkpoint ===#
  checkpoint = None
  for file in os.listdir(model_dir):
    if file.endswith('.pt'):
      checkpoint_path = os.path.join(model_dir, file)
      checkpoint = torch.load(os.path.join(model_dir, file), map_location=device, weights_only=True)
      checkpoint = checkpoint['model']
      print(f'Loaded checkpoint from {checkpoint_path}.')
      break
  else:
    raise FileNotFoundError(f'No .pt checkpoint found in {model_dir}.')
  
  model.load_state_dict(checkpoint)
  print('Model state dict loaded from checkpoint.')
  
  factory     = ParallelDatasetFactory(dataset_dir=args_dict['dataset_dir'])
  _, test_set = factory.train_test_split(min_transcript_samples=args_dict['min_samples'],
                                         train_set_threshold=args_dict['max_samples'],
                                         sample_pairing=args_dict['sample_pairing'],
                                         downsample=not args_dict['no_downsample'])
  
  _, test_augment_fn = get_augment_fns('student' if is_student else 'teacher')
  
  test_loader = DataLoader(test_set, 
                           batch_size=args_dict['batch_size'], 
                           shuffle=True, 
                           drop_last=False, 
                           num_workers=(args_dict['n_cores'] or os.cpu_count()),
                           pin_memory=True,
                           persistent_workers=True,
                           prefetch_factor = 1,
                           collate_fn=augment_collate_fn(test_augment_fn))
  
  ######################
  ### Evaluate model ###
  ######################
  
  model.eval()
  
  src_age_gender = []
  tgt_age_gender = []
  src_latent_factors = []
  tgt_latent_factors = []
  
  #=== Mel-Cepstral Distortion ===#
  
  print('Calculating Mel-Cepstral Distortion (MCD) on test set...')
  
  scalar = None

  with open(args_dict['scaler_filepath'], "rb") as f:
    scalar = pickle.load(f)

  mel_mean = scalar.mean_
  mel_std = scalar.scale_
  
  print('Mel mean and std loaded from scaler.')
  
  total_mcd_sum = 0.0
  total_frame_count = 0

  for batch in tqdm(test_loader):
    src_mel, tgt_mel, src_mask, tgt_mask, src_info, tgt_info = recursive_to_device(batch, device)
    
    # Prepare model inputs
    forward_args = [src_mel]
    if not is_student:
      forward_args.append(tgt_mel)
    if use_drl:
      forward_args.extend([src_mask, tgt_mask])
    else:
      forward_args.extend([src_info, tgt_info])
    
    forward_kwargs = {'return_variational': True} if use_drl else {}
    
    # Forward pass
    with torch.no_grad():
      output = model(*forward_args, **forward_kwargs)

    pred_mel = output[0]
    
    # Record true labels from data and latent factors from model
    if use_drl:
      src_age_gender.append(torch.stack([(src_info[0] * 30. + 10.), src_info[1]]).to(dtype=torch.int).T)
      tgt_age_gender.append(torch.stack([(tgt_info[0] * 30. + 10.), tgt_info[1]]).to(dtype=torch.int).T)
      # Get the latent factors from the variational outputs
      _, _, (src_latent, _, _, _), (src_latent, _, _, _) = output
      src_latent_factors.append(output[2][0].detach().cpu())
      tgt_latent_factors.append(output[3][0].detach().cpu())
    
    # Get sum and count for this batch
    mcd_sum, frame_count = mel_cepstral_distortion(
      pred_mel, tgt_mel, mel_mean, mel_std, mask=tgt_mask, return_sum_and_count=True
    )

    total_mcd_sum += mcd_sum
    total_frame_count += frame_count

  # Calculate final average
  if total_frame_count > 0:
    dataset_mcd = total_mcd_sum / total_frame_count
    print(f"Dataset MCD: {dataset_mcd:.4f} dB (over {total_frame_count:.0f} frames)")
  else:
    print("No valid frames found!")

  #=== Mutual Information Gap ===#
  if use_drl:
    print('Calculating Mutual Information Gap (MIG) on test set...')
    
    all_latent_factors = torch.cat(src_latent_factors + tgt_latent_factors, dim=0)
    all_age_gender = torch.cat(src_age_gender + tgt_age_gender, dim=0)
    
    unique_ages = all_age_gender[:,0].unique().tolist()
    print(f'Unique ages in source set: {unique_ages}\nFor a total of {len(unique_ages)} unique ages.')

    mig = mutual_information_gap(
      all_latent_factors[:,:2].detach().cpu().numpy(),
      all_age_gender.detach().cpu().numpy(),
      [unique_ages, [0,1]],
      ['age', 'gender'],
    )
    
    print(f"MIG: \n{mig}")
    
    #=== Accent Entropy ===#
    
    print('Calculating Accent Entropy on test set...')
    
    entropy = accent_entropy(all_latent_factors[:,2:].detach().cpu())
    print(f'Accent Entropy: {entropy:.4f}')