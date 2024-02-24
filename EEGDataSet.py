import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from PreProcessing import extract_ar_coefficients, extract_alpha_band_power, extract_mav
import config
class EEGDataSet(Dataset):
    def __init__(self, file_paths, exclude_labels=None):
        self.data = []
        self.labels = []
        self.label_encoder = LabelEncoder()  # Initialize the LabelEncoder
        all_labels = []
        #file_paths = [file_paths[0]]
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
        if 'Label' in dataframe.columns:
            labels = self.label_encoder.transform(dataframe['Label'])
            self.labels.extend(labels)

        eeg_data = dataframe.filter(regex='EEG').to_numpy()
        for row in eeg_data:
            # Assume these functions return numpy arrays
            ar_features = extract_ar_coefficients(row, order=2)
            mav_feature = extract_mav(row)
            alpha_bp_feature = extract_alpha_band_power(row, fs=config.fs)  # Assuming fs=256Hz is predefined
            features = np.concatenate((ar_features, [mav_feature], [alpha_bp_feature]))
            self.data.append(features)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Retrieve pre-processed features and label for a given index
        features_tensor = torch.tensor(self.data[index], dtype=torch.float)
        label = self.labels[index]  # Assuming labels are stored as integers
        return features_tensor, label