import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class EEGDataSet(Dataset):
    def __init__(self, dataframe, segment_size=125, padding_value=0.0):
        self.dataframe = dataframe
        self.segment_size = segment_size
        self.padding_value = padding_value
        self.segments = self.create_segments()

    def create_segments(self):
        eeg_data = self.dataframe.filter(regex='EEG').to_numpy()
        num_samples = eeg_data.shape[0]
        num_segments = int(np.ceil(num_samples / self.segment_size))
        segments = []
        for i in range(num_segments):
            start_idx = i * self.segment_size
            end_idx = start_idx + self.segment_size
            segment = eeg_data[start_idx:end_idx]
            if len(segment) < self.segment_size:
                padding_amount = self.segment_size - len(segment)
                padding = np.full((padding_amount, eeg_data.shape[1]), self.padding_value)
                segment = np.vstack((segment, padding))
            segments.append(segment)
        print(segments)
        return segments

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = self.segments[idx]
        return torch.tensor(segment, dtype=torch.float)
