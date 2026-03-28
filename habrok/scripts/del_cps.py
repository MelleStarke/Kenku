import sys, os

num_checkpoints = 0
num_no_space = 0

# Recursively delete files ending with '.pt'
for root, _, files in os.walk('/home3/s4984218/scratch/runs/hypertune_teacher/17378905'):
  highest_epoch = -1
  for file in files:
    if file == '.no_space':
      parent = os.path.dirname(root)
      for file in os.listdir(parent):
        if file.startswith('events.out'):
          tb_record_file = os.path.join(parent, file)
          print(f'File set for deletion: {tb_record_file}')
          
          # try:
          #   os.remove(tb_record_file)
          #   # print(f"Deleted: {file_path}")
          # except Exception as e:
          #   print(f"Error deleting {tb_record_file}: {e}")
            
          num_no_space += 1

    elif file.endswith('.pt'):
      file_path = os.path.join(root, file)
      # print(f'Checkpoint set for deletion: {file_path}')
      num_checkpoints += 1
      
      # try:
      #   os.remove(file_path)
      #   # print(f"Deleted: {file_path}")
      # except Exception as e:
      #   print(f"Error deleting {file_path}: {e}")
      
      # if file.endswith('.pt'):
      #   epoch = int(file.split('_')[0].replace('epoch', ''))
      #   if epoch > highest_epoch:
      #     highest_epoch = epoch
        
  # if highest_epoch == -1:
  #   continue
  
  # if highest_epoch < 30:
  #   print('no space:', highest_epoch)
  #   num_no_space += 1
  #   with open(os.path.join(root, '.no_space'), 'w+'):
  #     pass
    
  # elif highest_epoch < 59:
  #   print('timeout:', highest_epoch)
  #   num_timeout += 1
    
    
  # if highest_epoch == 36:
  #   print(root)

print(f"Total checkpoint files: {num_checkpoints}")
print(f"Total files with no space: {num_no_space}")

