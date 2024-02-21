import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class EEGDataSet(Dataset):
    def __init__(self, file_paths, exclude_labels=None):
        self.data = []
        self.labels = []
        self.label_encoder = LabelEncoder()  # Initialize the LabelEncoder
        all_labels = []

        # Collect all unique labels to fit the LabelEncoder
        for file_path in file_paths:
            dataframe = pd.read_csv(file_path)
            if 'Label' in dataframe.columns:
                all_labels.extend(dataframe['Label'].unique())

        # Fit the LabelEncoder with all collected labels
        self.label_encoder.fit(all_labels)

        # Now process each file
        for file_path in file_paths:
            dataframe = pd.read_csv(file_path)
            # Filter out rows with labels in exclude_labels
            if exclude_labels and 'Label' in dataframe.columns:
                dataframe = dataframe[~dataframe['Label'].isin(exclude_labels)]
            self.process_file(dataframe)

    def process_file(self, dataframe):
        # Assuming 'Timestamp' is in seconds and your sampling rate is known
        trimmed_df = dataframe  # No need for trimming if you're inputting each dataset individually

        # Processing labels
        if 'Label' in trimmed_df.columns:
            labels = self.label_encoder.transform(trimmed_df['Label'])
            self.labels.extend(labels)
        
        # Extracting EEG data
        eeg_data = trimmed_df.filter(regex='EEG').to_numpy()
        # Append each row of EEG data as a separate instance
        for row in eeg_data:
            self.data.append(row)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve EEG data and label for the given index
        eeg_data = self.data[idx]
        label = self.labels[idx]
        # Convert data and label to tensors
        return torch.tensor(eeg_data, dtype=torch.float), label
