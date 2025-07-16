import sys, os
from tensorboard.backend.event_processing.event_multiplexer import EventMultiplexer
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


hypertune_dir = "/home3/s4984218/scratch/runs/hypertune_teacher/17378905"

run_dict = {}

for root, _, files in os.walk(hypertune_dir):
  for file in files:
    if file.startswith('events.out'):
      runname = os.path.basename(root)
      run_dict[runname] = os.path.join(root, file)
      
em = EventMultiplexer(run_dict, size_guidance={'scalars': 4})
em.Reload()
tags = em.Tags()