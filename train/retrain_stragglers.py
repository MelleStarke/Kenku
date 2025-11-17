import sys
import os
import subprocess

counter = 0

run_nr = int(sys.argv[1])


for root, _, files in os.walk('/home3/s4984218/scratch/runs/hypertune_teacher/17378905'):
  highest_epoch = -1
  for file in files:
    if file == '.no_space':
      counter += 1
    
  if counter == run_nr:
    run_dir = os.path.dirname(root)
    print(f'Found run nr. {run_nr}: {run_dir}')

    folder_name = os.path.basename(run_dir)
      
    train_command = [
      'python', 
      '-m', 'train.train_model', 
      '--config-dir', f'"{run_dir}"',
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

    break
  
else:
  raise ValueError(f'Run nr {run_nr} not found in {counter} candidates')