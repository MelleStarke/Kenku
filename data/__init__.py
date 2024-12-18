import os


DATA_PATH  = "/home/user/Uni/m3/thesis/project/Data"
KENKU_PATH = "/home/user/Uni/m3/thesis/project/Kenku"

#=== Voice Cloning Tool Kit Corpus ===#
RAW_VCTK_PATH       = os.path.join(DATA_PATH, "raw/VCTK-Corpus")
RAW_VCTK_AUDIO_PATH = os.path.join(RAW_VCTK_PATH, "wav48")

VCTK_PATH           = os.path.join(DATA_PATH, "processed/VCTK")
VCTK_MELSPEC_PATH   = os.path.join(VCTK_PATH, "melspec")

#=== Speech Accent Archive ===#_
RAW_SAA_PATH        = os.path.join(DATA_PATH, "raw/SAA")
RAW_SAA_AUDIO_PATH  = os.path.join(RAW_SAA_PATH, "recordings/recordings")

SAA_PATH            = os.path.join(DATA_PATH, "processed/SAA")
SAA_MELSPEC_PATH    = os.path.join(SAA_PATH, "melspec")

def walk_files(root, extension):
    for path, dirs, files in os.walk(root):
        for file in files:
            if file.endswith(extension):
                yield os.path.join(path, file)
                
