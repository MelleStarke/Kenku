import torch
from torch.optim.lr_scheduler import LambdaLR, _warn_get_lr_called_within_step

import numpy as np

from typing import Union, List, Tuple, Optional
from typing_extensions import override

from network import KenkuStudent

def group_student_params(student_model: KenkuStudent, num_conv_layers: int = 8, format_for_optimizer=False):
  """
  Groups the parameters trnsfered from the teacher model into groups,
  from closest to the output layer to the furthest.
  This allows us to slowly unfreeze the transfered layers during training.
  
  Args:
      student_model (KenkuStudent): The student model containing the transfered layers.
      num_conv_layers (int): Number of convolutional layers in each KameBlock. Default is 8.
      format_for_optimizer (bool): If True, returns the parameter groups in a format suitable for passing to an optimizer.
                                   Each group is a dict with 'params' and 'name' keys. Default is False.
  
  Returns:
      thawing_groups (List[(str, Parameter)]): A list of parameter groups to be thawed incrementally.
      static_group (List[(str, Parameter)]): A list of parameters that remain frozen throughout training.
      untrained_group (List[(str, Parameter)]): A list of parameters that were not transfered and are trained from scratch.
  """
  
  def group_kameblock_params(kameblock, prefix):
    name_starts = [f'{prefix}.out_layer']
    for i in range(num_conv_layers):
        name_starts.append(f'{prefix}.conv_blocks.{num_conv_layers - 1 - i}')
      
    name_starts += [f'{prefix}.in_layer', f'{prefix}.embed_layer']
    return name_starts
    
  thawing_group_name_starts = group_kameblock_params(student_model, 'decoder') + \
                                group_kameblock_params(student_model, 'src_encoder')
  
  thawing_groups = [[] for _ in range(len(thawing_group_name_starts))]
  static_group = []
  untrained_group = []
  for name, param in student_model.named_parameters():
    if not param.requires_grad:
        static_group.append(param)
        continue
    
    for i, start in enumerate(thawing_group_name_starts):
      if name.startswith(start):
        thawing_groups[i].append(param)
        break
      
    else:
      untrained_group.append(param)
      
  # Freeze thawing groups to prevent accumulation of momentum etc.
  for group in thawing_groups:
    for param in group:
      param.requires_grad = False

  if format_for_optimizer:
    param_groups = [{
      'params': group,
      'name': name
    } for group, name in zip(thawing_groups + [untrained_group],
                       thawing_group_name_starts + ['<untrained>'])]
    return param_groups

  return thawing_groups, static_group, untrained_group
    
class IncrementalThawScheduler(LambdaLR):
  
  def __init__(self, 
               optimizer,
               total_steps: Optional[int] = None, 
               warmup_steps: Union[int, float] = 0.15, 
               thawing_steps: Union[int, float] = 0.5):
    """
    Scheduler that thaws groups of parameters incrementally during training.
    The thawing schedule is as follows:
    - Warmup Phase: No parameters are thawed.
    - Thawing Phase: Groups of parameters are thawed one by one at regular intervals.
                     Individual groups are thawed gradually (linearly) over a few steps.
    - Final Phase: All parameters are thawed.
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
        total_steps (int, optional): Total number of steps for training. If None, all other parameters must be integers.
        warmup_steps (int or float, optional): Number of steps (or fraction of total_steps) for the warmup phase. Default is 0.15.
        thawing_steps (int or float, optional): Number of steps (or fraction of total_steps) for the thawing phase. Default is 0.5.
                                                Each group is thawed gradually by linearly increasing the learning rate.
                                                Reaching the max when the next group begins thawing.
    """
    if total_steps is None:
      assert all(map(lambda x: isinstance(x, int), [warmup_steps, thawing_steps])), \
             "If the total_steps is unspecified, warmup_steps and thawing_steps must be integers."
    
    if isinstance(warmup_steps, float) and isinstance(thawing_steps, float):
      assert warmup_steps + thawing_steps <= 1.0, "The sum of warmup_steps and thawing_steps must be 1.0 or less."
  
    if isinstance(warmup_steps, float):
      warmup_steps = int(total_steps * warmup_steps)
    if isinstance(thawing_steps, float):
      thawing_steps = int(total_steps * thawing_steps)
    
    # Save the thawing groups. The last group in param_groups() is the untrained parameters.
    assert len(optimizer.param_groups) > 1, "The optimizer must have at least two parameter groups: one or more thawing groups and one untrained group."
    thawing_groups = optimizer.param_groups[:-1]
    n_thawing_groups = len(thawing_groups)
    
    thaw_start_step = warmup_steps
    thaw_interval = thawing_steps / n_thawing_groups
    
    def clipped_linear(start_step, end_step):
      return lambda step: max(0., min((step - start_step) / max(end_step - start_step, 1e-8), 1.))
    
    thawing_schedule = []
    for i in range(n_thawing_groups):
      start_step = int(thaw_start_step + i * thaw_interval)
      end_step = int(thaw_start_step + (i + 1) * thaw_interval) - 1
      thawing_schedule.append({
        'param_group_idx': i,
        'start_step': int(thaw_start_step + i * thaw_interval),
        'lambda': clipped_linear(start_step, end_step)
      })
    self.thawing_schedule = thawing_schedule
    
    lambdas = [part['lambda'] for part in thawing_schedule]
    # Append "always on" lambda for the untrained group
    lambdas.append(lambda step: 1.)
    
    super(IncrementalThawScheduler, self).__init__(optimizer, lambdas)
    
  @override
  def step(self, *args, **kwargs):
    assert self._step_count is not None, "self._step_count is None. This shouldn't be possible as LRScheduler._initial_step() should be called in the base init() fn."
    self.last_step = self._step_count
    
    # Thaw the next group if its start step is reached
    if len(self.thawing_schedule) > 0:
      next_thaw_group_step = self.thawing_schedule[0]['start_step']
      if self.last_step >= next_thaw_group_step:
        # Thaw the next group
        thaw_part = self.thawing_schedule.pop(0)
        param_group_idx = thaw_part['param_group_idx']
        for param in self.optimizer.param_groups[param_group_idx]['params']:
          param.requires_grad = True
    
    super(IncrementalThawScheduler, self).step(*args, **kwargs)
    
  @override
  def get_lr(self) -> list[float]:
    """Compute learning rate."""
    _warn_get_lr_called_within_step(self)

    return [
      base_lr * lmbda(self.last_step)
      for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)
    ]

    
if __name__ == "__main__":
    from network import KenkuTeacher
    ch = 32
    model = KenkuTeacher(ch, ch, ch, ch, 12, 11)
    
    model = model.to_student({})
    
    # thawing_groups, static_group, untrained_group = group_student_params(model)
    param_groups = group_student_params(model, format_for_optimizer=True)
    
    # all_names = sorted(np.ravel([[g[0] for g in group] for group in thawing_groups]).tolist() + [g[0] for g in static_group] + [g[0] for g in untrained_group])
    # model_param_names = sorted([name for name, param in model.named_parameters()])
    # assert all_names == model_param_names, "Some parameters are missing in the groups!"

    opt = torch.optim.Adam(param_groups, lr=0.001)
    
    total_steps = 400
    warmup_steps = 0.15
    thawing_steps = 0.5
    scheduler = IncrementalThawScheduler(opt, total_steps=total_steps, warmup_steps=warmup_steps, thawing_steps=thawing_steps)
    
    group_lrs = [[] for _ in range(len(param_groups))]
    
    for step in range(total_steps):
      opt.step()
      scheduler.step()
      lrs = scheduler.get_last_lr()
      for i, lr in enumerate(lrs):
        group_lrs[i].append(lr)
        
    import matplotlib.pyplot as plt
    for i, lrs in enumerate(group_lrs):
      plt.plot(lrs, label=f'{opt.param_groups[i]['name']}')
    plt.axvline(x=int(total_steps * warmup_steps) - 1, color='gray', linestyle='--')
    plt.axvline(x=int(total_steps * (warmup_steps + thawing_steps)) - 1, color='gray', linestyle='--')
    
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule with Incremental Thawing')
    plt.legend()
    plt.grid()
    plt.show()
    