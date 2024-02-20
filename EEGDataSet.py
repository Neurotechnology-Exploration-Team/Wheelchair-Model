import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from PreProcessing import preprocess_signal
class EEGDataSet(Dataset):
    def __init__(self, file_paths, segment_size=125, padding_value=0.0):
        self.segment_size = segment_size
        self.padding_value = padding_value
        self.labels = []
        self.segments = []
        self.label_encoder = LabelEncoder()
        # Initialize an empty list to collect all labels
        all_labels = []
        # Collect labels from each file
        for file_path in file_paths:
            dataframe = pd.read_csv(file_path)
            all_labels.extend(dataframe['Label'].unique())
        # Fit the label encoder with all unique labels
        self.label_encoder.fit(all_labels)
        # Process each file
        for file_path in file_paths:
            dataframe = pd.read_csv(file_path)
            self.process_file(dataframe)

    def process_file(self, dataframe):
        # Encode the labels for the current dataframe
        dataframe['Label'] = self.label_encoder.transform(dataframe['Label'])
        self.labels.extend(dataframe['Label'].tolist())
        preprocessed_data = preprocess_signal(dataframe)
        self.segments += self.create_segments(preprocessed_data)

    def create_segments(self, eeg_data):
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
        return segments

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = self.segments[idx]
        label = self.labels[idx]  # Assume one label per segment for simplicity
        return torch.tensor(segment, dtype=torch.float), label
