import numpy as np
import os
import pandas as pd


DATA_PATH = "./../Data/"

#== Voice Cloning Tool Kit Corpus ==#
VCTK_PATH       = os.path.join(DATA_PATH, "/VCTK-Corpus-Alternative/VCTK-Corpus/")
VCTK_AUDIO_PATH = os.path.join(VCTK_PATH, "/wav48/")
VCTK_TEXT_PATH  = os.path.join(VCTK_PATH, "/txt/")
VCTK_CLASS_PATH = os.path.join(VCTK_PATH, "/speaker-info.txt")

#== Speech Accent Archive ==#
SAA_PATH       = os.path.join(DATA_PATH, "/SAA/")
SAA_AUDIO_PATH = os.path.join(SAA_PATH, "/recordings/recordings/")
SAA_TEXT_PATH  = os.path.join(SAA_PATH, "/reading-passage.txt")
SAA_CLASS_PATH = os.path.join(SAA_PATH, "speakers_all.csv")


def walk_files(root, extension):
    for path, dirs, files in os.walk(root):
        for file in files:
            if file.endswith(extension):
                yield os.path.join(path, file)
                
