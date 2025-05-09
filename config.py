import os
import torch

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------

# --- project root path --- #

PTH_ROOT = os.path.dirname(__file__) #root is in gdrive, where this file is supposed to be

# --- Data paths --- #

PTH_DATA = os.path.join('/home','default','Desktop', 'raw') # path to raw data in the vm for fast loading

# --- saving paths --- #

PTH_SAVES = os.path.join(PTH_ROOT, 'saves')
    
PTH_SAVED_MODELS = os.path.join(PTH_SAVES, 'models')

PTH_SAVED_FIGURES = os.path.join(PTH_SAVES, 'figures')

PTH_CHECKPOINTS = os.path.join('/home','default','Desktop', 'checkpoints')

# --- Others--- #
PTH_UTILS =  os.path.join(PTH_ROOT, 'utils')

# -----------------------------------------------------------------------------
# images and masks
# -----------------------------------------------------------------------------

IMG_NUM_CHANNELS = 1 # all images are grayscales

IMG_HEIGHT = 64

IMG_WIDTH = 64

# -----------------------------------------------------------------------------
# TRAINING
# -----------------------------------------------------------------------------
BATCH_SIZE = 32

TEST_SPLIT = .2

NUM_EPOCHS = 50

# -----------------------------------------------------------------------------
# Device option
# -----------------------------------------------------------------------------

if torch.cuda.is_available():
    AVAILABLE_DEVICE = 'cuda'
elif torch.backends.mps.is_available() :
    AVAILABLE_DEVICE = 'mps' 
else:
    AVAILABLE_DEVICE = 'cpu'