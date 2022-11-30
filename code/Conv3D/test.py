import os

import torch
from model import Conv3DNet
from config import device, BATCH_SIZE
import pandas as pd
from torch.utils.data import DataLoader, Dataset


# Dataset for test set only
class RSNADataset(Dataset):
    # Initialise
    def __init__(self, subset, df_table):
        super().__init__()

        self.subset = subset
        self.df_table = df_table

        # Image paths
        self.volume_dir = r'E:\rsna-2022-cervical-spine-fracture-detection\sampledata\test_volumes'

    # Get item in position given by index
    def __getitem__(self, index):
        # load 3d volume
        patient = self.df_table.loc[index, 'StudyInstanceUID']
        path = os.path.join(self.volume_dir, f'{patient}.pt')
        vol = torch.load(path).to(torch.float32)

        return (vol.unsqueeze(0), patient)

    # Length of dataset
    def __len__(self):
        return len(self.df_table)

def predict_fracture(file_path):
    # Load checkpoint
    PATH=r'C:\Users\ME\OneDrive\Classes\EC601\TeamProject\code\Conv3D\Conv3DNet.pt'
    if torch.cuda.is_available():
        checkpoint = torch.load(PATH)
    else:
        checkpoint = torch.load(PATH, map_location=torch.device('cpu'))

    model = Conv3DNet().to(device)
    # Load states
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    val_loss = checkpoint['val_loss']

    # Evaluation mode
    model.eval()
    model.to(device)

    # Test dataset
    # Load metadata
    # Fix mismatch with test_images folder
    test_df = pd.DataFrame(columns=['row_id', 'StudyInstanceUID', 'prediction_type'])
    filename = file_path[0].split("/")[-1][:-3]
    # for i in ['1.2.826.0.1.3680043.22327', '1.2.826.0.1.3680043.25399', '1.2.826.0.1.3680043.5876']:
    # for i in ['1.2.826.0.1.3680043.10001','1.2.826.0.1.3680043.1016','1.2.826.0.1.3680043.10005','1.2.826.0.1.3680043.10014', '1.2.826.0.1.3680043.10016','1.2.826.0.1.3680043.10261','1.2.826.0.1.3680043.10400', '1.2.826.0.1.3680043.12292']:
    for i in [filename]:
        for j in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'patient_overall']:
            # test_df = test_df.append({'row_id': i + '_' + j, 'StudyInstanceUID': i, 'prediction_type': j},
            #                          ignore_index=True)
            test_df = pd.concat([test_df, pd.DataFrame.from_records([{'row_id': i + '_' + j, 'StudyInstanceUID': i, 'prediction_type': j}])])
    # Sample submission
    ss = pd.DataFrame(test_df['row_id'])
    ss['fractured'] = 0.5

    # test_df = pd.read_csv(r"E:\rsna-2022-cervical-spine-fracture-detection\sampledata\test.csv")
    # ss = pd.read_csv(r"E:\rsna-2022-cervical-spine-fracture-detection\sampledata\sample_submission.csv")
    test_table = pd.DataFrame(pd.unique(test_df['StudyInstanceUID']),columns=['StudyInstanceUID'])
    test_dataset = RSNADataset(subset='test', df_table=test_table)
    # Dataloader
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    # Print final loss and epoch
    print('Final epoch:', epoch)
    print('Final loss:', loss)
    print('Final valid loss:', val_loss)
    test_df['fractured'] = 0.5
    with torch.no_grad():
        # Loop over batches
        for i, (imgs, patient) in enumerate(test_loader):
            print(f'Iteration {i + 1}/{len(test_loader)}')
            # Send to device
            imgs = imgs.to(device)

            # Make predictions
            preds = model(imgs)

            # Apply sigmoid
            sig = torch.nn.Sigmoid()
            preds = sig(preds)
            preds = preds.to('cpu')

            # Save preds
            test_df.loc[test_df['StudyInstanceUID'] == patient[0], 'fractured'] = preds.numpy().squeeze()

    print('Inference complete!')

    submission = test_df[['row_id','fractured']]
    submission.to_csv('submission1.csv', index=False)
    submission.head(3)
    return submission.head(3)