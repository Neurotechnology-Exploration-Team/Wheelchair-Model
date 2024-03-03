import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, BatchSampler
from Processing import process_eeg_data
from SingleDataSet import SingleDataSet
import torch
class EEGDataset(Dataset):
    def __init__(self, segment_length=0.5, fs=250):
        self.segment_length = segment_length
        self.fs = fs
        self.samples_per_segment = int(fs * segment_length)
        self.tests_data = []
        self.tests_labels = []
        self.label_mapping = {}  # Initialize label mapping
        self.tests_sizes = []
    def add_test(self, test_data):
        segmented_data, segmented_labels = self.preprocess_and_segment(test_data)
        self.tests_data.append(segmented_data)
        self.tests_labels.append(segmented_labels)
        self.tests_sizes.append(len(segmented_data))

    def preprocess_and_segment(self, raw_data):
        segmented_data = pd.DataFrame()
        segmented_labels = []
        label_collector = set()
        for i in range(0, len(raw_data), self.samples_per_segment):
            if i + self.samples_per_segment <= len(raw_data):
                segment = raw_data.iloc[i:i + self.samples_per_segment]
                segment_processed = process_eeg_data(segment, self.fs)
                segment_processed_df = pd.DataFrame(segment_processed)
                segmented_data = pd.concat([segmented_data, segment_processed_df], ignore_index=True)
                label = segment['Label'].iloc[0]
                segmented_labels.append(label)
                label_collector.add(label)

        # Update self.label_mapping here
        for label in label_collector:
            if label not in self.label_mapping:
                self.label_mapping[label] = len(self.label_mapping)

        # Convert labels using the updated label_mapping
        numeric_segmented_labels = [self.label_mapping[label] for label in segmented_labels]

        return segmented_data, np.array(numeric_segmented_labels)

    def get_test_sizes(self):
        return self.tests_sizes
    def __len__(self):
        return len(self.tests_data)
    def __getitem__(self, idx):
        # Create a new Dataset for the selected test data

        subset_dataset = SingleDataSet(self.tests_data[idx].drop("Label",axis=1), self.tests_labels[idx])
        # Wrap the subset dataset in a DataLoader
        subset_loader = DataLoader(subset_dataset, batch_size=64, shuffle=False)
        return subset_loader
