import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
import config

class EEGDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Assuming each row in the CSV represents one time step
        # and the columns 1:9 are different channels
        sample = self.data.iloc[idx, 1:9].values.astype(float)
        sample = sample.reshape(-1, 8)  # Reshape to [data_length, channels]

        blink = self.data.iloc[idx, -1]
        label = 1 if blink == "blink" else 0

        if self.transform:
            sample = self.transform(sample)

        return sample, label


def preprocess_data(sample):
    scaler = StandardScaler()
    return scaler.fit_transform(sample.reshape(1, -1)).astype('float32')


def get_dataLoader(fileName):
    dataset = EEGDataset(f'./data/{fileName}', transform=preprocess_data)
    data_loader = DataLoader(dataset, batch_size=config.input_size, shuffle=True)
    return data_loader

def get_shape(data_loader):
    batch_shape = None
    for batch in data_loader:
        inputs, labels = batch
        batch_shape = inputs.shape
        break 
    return batch_shape