import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from Processing import process_eeg_data

class EEGDataset(Dataset):
    def __init__(self, segment_length=0.5, fs=250):
        self.segment_length = segment_length
        self.fs = fs
        self.samples_per_segment = int(fs * segment_length)
        self.tests_data = []
        self.tests_labels = []

    def add_test(self, test_data):
        segmented_data, segmented_labels = self.preprocess_and_segment(test_data)
        self.tests_data.append(segmented_data)
        self.tests_labels.append(segmented_labels)

    def preprocess_and_segment(self, raw_data):
        segmented_data = []
        segmented_labels = []
        for i in range(0, len(raw_data), self.samples_per_segment):
            if i + self.samples_per_segment <= len(raw_data):
                segment = raw_data.iloc[i:i + self.samples_per_segment]
                # Assuming process_eeg_data function processes each segment
                segment_processed = process_eeg_data(segment, self.fs)
                segmented_data.append(segment_processed.drop('Label', axis=1).values)
                segmented_labels.append(segment['Label'].iloc[0])
        return np.array(segmented_data), np.array(segmented_labels)

    def __len__(self):
        # Total number of tests
        return len(self.tests_data)

    def __getitem__(self, test_idx):
        # Return the data and labels for a specific test
        return self.tests_data[test_idx], self.tests_labels[test_idx]

# FUCKKKKKK I HAVE TO CREATE A CUSTOM DATALOADER AND DATA SET TO PERSERVE THE TIME SERIES OF EACH TEST CAUSE MIXING IT WILL MESS WITH PREDICTIONS FUCK MY LIFE.