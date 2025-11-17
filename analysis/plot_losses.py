import json
import os
import numpy as np
import matplotlib.pyplot as plt

models = {
  'teacher': 'Teacher',
  'student': 'Student',
  'drl_teacher': 'DRL Teacher'
}

loss_types = {
  'test': 'Test',
  'da': 'Diagonal Attention',
  'oa': 'Orthogonal Attention',
  'aa': 'Auxiliary Attention',
  'entr': 'Entropy',
  'dwkld': 'Dimension-wise KLD',
  'mi': 'Mutual Information',
  'tc': 'Total Correlation',
}

# Green: #029878
# Blue: #027684

for model_key, model_name in models.items():
  for loss_key, loss_name in loss_types.items():
    
    # if loss_key == 'train':
    #   continue
    
    if not os.path.exists(os.path.join('./analysis/losses', f'{model_key}_{loss_key}.json')):
      print(os.path.join('./analysis/losses', f'{model_key}_{loss_key}.json'), ' does not exist. Skipping...')
      continue
    
    losses = None
    with open(os.path.join('./analysis/losses', f'{model_key}_{loss_key}.json'), 'r') as f:
      losses = json.load(f)
    
    if losses is None or len(losses) == 1:
      raise ValueError(f'Loss data for ./losses/{model_key}_{loss_key}.json is invalid.')
    
    losses = np.array(losses)
    steps = losses[:,1]
    losses = losses[:,2]
    
    print(f"{model_name} {loss_name} Loss: Final value at step {steps[-1]} is {losses[-1]:.4f}\n"
          f"  Min value: {np.min(losses):.4f} at step {steps[np.argmin(losses)]}\n")
  
    fig = plt.figure(figsize=(8, 5))
    plt.plot(steps, losses, color=('#027684'))
    plt.title(f'{model_name}: {loss_name} Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # if loss_key == 'test' and os.path.exists(os.path.join('./analysis/losses', f'{model_key}_train.json')):
    #   losses = None
    #   with open(os.path.join('./analysis/losses', f'{model_key}_train.json'), 'r') as f:
    #     losses = json.load(f)
      
    #   if losses is None or len(losses) == 1:
    #     raise ValueError(f'Loss data for ./losses/{model_key}_train.json is invalid.')
      
    #   losses = np.array(losses)
    #   steps = losses[:,1]
    #   losses = losses[:,2]
      
    #   plt.plot(steps, losses, color=('#029878'))
    #   plt.title(f'{model_name}: Train and Test Loss')
    
    # plt.show()
    plt.tight_layout()
    fig.savefig(os.path.join('./analysis/plots', f'{model_key}_{loss_key}_loss.pdf'))
