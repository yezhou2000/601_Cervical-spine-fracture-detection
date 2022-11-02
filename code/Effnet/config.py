import gc
import glob
import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom as dicom
import torch
import torchvision as tv
from sklearn.model_selection import GroupKFold
from torch.cuda.amp import GradScaler, autocast
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm.notebook import tqdm


plt.rcParams['figure.figsize'] = (20, 5)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 1000)

# Effnet
WEIGHTS = tv.models.efficientnet.EfficientNet_V2_S_Weights.DEFAULT
RSNA_2022_PATH = r'E:\rsna-2022-cervical-spine-fracture-detection\sampledata'
TRAIN_IMAGES_PATH = f'{RSNA_2022_PATH}/../train_images'
TEST_IMAGES_PATH = f'{RSNA_2022_PATH}/test_images'
EFFNET_MAX_TRAIN_BATCHES = 4000
EFFNET_MAX_EVAL_BATCHES = 200
ONE_CYCLE_MAX_LR = 0.0001
ONE_CYCLE_PCT_START = 0.3
SAVE_CHECKPOINT_EVERY_STEP = 1000
EFFNET_CHECKPOINTS_PATH = RSNA_2022_PATH+'\\effnetv2'
FRAC_LOSS_WEIGHT = 2.
N_FOLDS = 2
METADATA_PATH = RSNA_2022_PATH+'\\segeffnetv2'
os.environ["WANDB_MODE"] = "online"
os.environ['WANDB_API_KEY'] = '1aec9b0208e9ff0449fa27a58cd924037e664de2'
PREDICT_MAX_BATCHES = 1e9

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    BATCH_SIZE = 32
else:
    BATCH_SIZE = 2