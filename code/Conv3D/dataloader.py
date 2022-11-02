import torch
import torchvision
import kornia
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import os
from glob import glob
import config


# Load metadata
train_df = pd.read_csv(r"E:\rsna-2022-cervical-spine-fracture-detection\sampledata\train.csv")
train_bbox = pd.read_csv(r"E:\rsna-2022-cervical-spine-fracture-detection\sampledata\train_bounding_boxes.csv")
test_df = pd.read_csv(r"E:\rsna-2022-cervical-spine-fracture-detection\sampledata\test.csv")
ss = pd.read_csv(r"E:\rsna-2022-cervical-spine-fracture-detection\sampledata\sample_submission.csv")

# Print dataframe shapes
print('train shape:', train_df.shape)
print('train bbox shape:', train_bbox.shape)
print('test shape:', test_df.shape)
print('ss shape:', ss.shape)
# print('')


# Dataset for train/valid sets only
class RSNADataset(Dataset):
    # Initialise
    def __init__(self, subset='train', df_table=train_df, transform=None):
        super().__init__()

        self.subset = subset
        self.df_table = df_table.reset_index(drop=True)
        self.transform = transform
        self.targets = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'patient_overall']

        # Identify files in each of the two datasets
        fh_paths = glob(os.path.join(r'E:\rsna-2022-cervical-spine-fracture-detection\sampledata\train_volumes\fh', "*.pt"))
        sh_paths = glob(os.path.join(r'E:\rsna-2022-cervical-spine-fracture-detection\sampledata\train_volumes\sh', "*.pt"))

        fh_list = []
        sh_list = []
        for i in fh_paths:
            fh_list.append(i.split('\\')[-1][:-3])

        for i in sh_paths:
            sh_list.append(i.split('\\')[-1][:-3])

        self.df_table_fh = self.df_table[self.df_table['StudyInstanceUID'].isin(fh_list)]
        self.df_table_sh = self.df_table[self.df_table['StudyInstanceUID'].isin(sh_list)]

        # Image paths
        self.volume_dir1 = r'E:\rsna-2022-cervical-spine-fracture-detection\sampledata\train_volumes\fh'  # <=1000 patient
        self.volume_dir2 = r'E:\rsna-2022-cervical-spine-fracture-detection\sampledata\train_volumes\sh'  # >1000 patient

        # Populate labels
        self.labels = self.df_table[self.targets].values

    # Get item in position given by index
    def __getitem__(self, index):
        if index in self.df_table_fh.index:
            patient = self.df_table_fh[self.df_table_fh.index == index]['StudyInstanceUID'].iloc[0]
            path = os.path.join(self.volume_dir1, f"{patient}.pt")
            vol = torch.load(path).to(torch.float32)
        else:
            patient = self.df_table_sh[self.df_table_sh.index == index]['StudyInstanceUID'].iloc[0]
            path = os.path.join(self.volume_dir2, f"{patient}.pt")
            vol = torch.load(path).to(torch.float32)

        # Data augmentations
        if self.transform:
            vol = self.transform(vol)

        return vol.unsqueeze(0), self.labels[index]

    # Length of dataset
    def __len__(self):
        return len(self.df_table['StudyInstanceUID'])


# Data augmentations (https://kornia.readthedocs.io/en/latest/augmentation.module.html#geometric)
if config.AUGMENTATIONS:
    augs = torchvision.transforms.Compose([
        kornia.augmentation.RandomRotation3D((0,0,30), resample='bilinear', p=0.5, same_on_batch=False, keepdim=True),
        #augmentation.RandomHorizontalFlip3D(same_on_batch=False, p=0.5, keepdim=True),
        ])
else:
    augs=None


# Train/valid datasets
experimental = config.EXPERIMENTAL
if experimental:
    train_table, valid_table = train_test_split(train_df, train_size=0.5, test_size=0.01, random_state=0)
    train_dataset = RSNADataset(subset='train', df_table=train_table, transform=augs)
    valid_dataset = RSNADataset(subset='valid', df_table=valid_table)
else:
    train_table, valid_table = train_test_split(train_df, train_size=0.85, test_size=0.15, random_state=0)
    train_dataset = RSNADataset(subset='train', df_table=train_table, transform=augs)
    valid_dataset = RSNADataset(subset='valid', df_table=valid_table)

# Dataloaders
train_loader = DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
