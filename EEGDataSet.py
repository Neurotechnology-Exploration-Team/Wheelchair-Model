import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class EEGDataSet(Dataset):

    def __init__(self, file_paths, segment_size=125, padding_value=0.0, exclude_labels=None):
        self.segment_size = segment_size
        self.padding_value = padding_value
        self.segments = []
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
        sampling_rate = 256  # Example: 256 Hz, adjust this according to your actual sampling rate
        trim_samples = 4 * sampling_rate  # Number of samples to trim from start and end
        
        # Trim the dataframe based on the timestamp range
        if 'Timestamp' in dataframe.columns:
            start_time = dataframe['Timestamp'].iloc[trim_samples]  # Time after trimming start
            end_time = dataframe['Timestamp'].iloc[-trim_samples]  # Time before trimming end
            trimmed_df = dataframe[(dataframe['Timestamp'] >= start_time) & (dataframe['Timestamp'] <= end_time)]
        else:
            trimmed_df = dataframe  # No trimming if no Timestamp column

        if 'Label' in trimmed_df.columns:
            labels = self.label_encoder.transform(trimmed_df['Label'])
            self.labels.extend(labels)
        
        eeg_data = trimmed_df.filter(regex='EEG').to_numpy()
        self.segments.extend(self.create_segments(eeg_data))

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
        label = self.labels[idx] if self.labels else -1
        return torch.tensor(segment, dtype=torch.float), label
