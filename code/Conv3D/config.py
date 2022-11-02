import torch

# Hyper parameters
BATCH_SIZE = 2
LEARNING_RATE = 0.0001
N_EPOCHS = 20
PATIENCE = 3
EXPERIMENTAL = True
AUGMENTATIONS = True

# Config device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
