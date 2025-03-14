from sys import argv
import os
import joblib

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from data.util import walk_files
from data.load import read_melspec


def melspec_to_img(melspec_path: str, img_path: str):
  if os.path.exists(img_path):
    return
  
  os.makedirs(os.path.dirname(img_path), exist_ok=True)
    
  melspec = read_melspec(melspec_path)
  
  fig = plt.figure(figsize=(10,5), dpi=80)
  
  plt.imshow(melspec, aspect='auto')
  plt.xlabel('Frame')
  plt.ylabel('Mel')
  plt.tight_layout()
  
  fig.savefig(img_path)
  plt.close()

def main(melspec_dir: str, img_dir: str):
  
  all_path_pairs = [
      [
          f,
          f.replace(melspec_dir, img_dir).replace('.h5', '.png')
      ]
      for f in walk_files(melspec_dir, '.h5')
  ]
  
  joblib.Parallel(n_jobs=16)(
      joblib.delayed(melspec_to_img)(*paths)
        for paths in tqdm(all_path_pairs, total=len(all_path_pairs))
  )
    

if __name__ == "__main__":
  melspec_dir = "../Data/processed/VCTK/melspec"
  img_dir     = "../Data/processed/VCTK/img"
  
  if len(argv) > 1:
    melspec_dir = argv[1]
  if len(argv) > 2:
    img_dir = argv[2]
    
  main(melspec_dir, img_dir)