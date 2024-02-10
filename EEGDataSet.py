import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class EEGDataSet(Dataset):
    def __init__(self, dataframe, labels=None):
        self.dataframe = dataframe
        self.eeg_data = self.dataframe.filter(regex='EEG').to_numpy()
        self.labels = labels

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        datapoint = self.eeg_data[idx]
        if self.labels is not None:
            label = self.labels[idx]
            return torch.tensor(datapoint, dtype=torch.float), torch.tensor(label, dtype=torch.long)
        return torch.tensor(datapoint, dtype=torch.float)
