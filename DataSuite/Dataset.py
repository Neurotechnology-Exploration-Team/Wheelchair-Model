import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import glob
import config
from Processing import process_eeg_data
from sklearn.preprocessing import LabelEncoder
from AnalyzeData import analyze_dataset
from torch.utils.data import DataLoader, Subset
class EEGDataset(Dataset):
    def __init__(self, csv_file_path):
        #self.data = pd.read_csv(csv_file_path)
        # Assuming the first column is the timestamp and the rest are EEG channels
        #self.timestamps = self.data.iloc[:, 0]
        #self.eeg_data = self.data.iloc[:, 1:].values
        self.trim_amount = int(config.fs*5)
        self.labels = []
        self.label_encoder = LabelEncoder()
        self.fs = config.fs
        self.load_files_and_process(csv_file_path)
        self.handle_labels()

    def split_dataset(self, dataset, train_frac=0.75, valid_frac=0.15):
        # Calculate split sizes
        total_size = len(dataset)
        train_size = int(total_size * train_frac)
        valid_size = int(total_size * valid_frac)
        test_size = total_size - train_size - valid_size

        # Perform the split
        indices = torch.randperm(total_size).tolist()
        train_indices = indices[:train_size]
        valid_indices = indices[train_size:train_size + valid_size]
        test_indices = indices[train_size + valid_size:]

        # Append to existing datasets if they exist, else create new
        if hasattr(self, 'train_dataset'):
            self.train_dataset = pd.concat([self.train_dataset, dataset.iloc[train_indices]])
        else:
            self.train_dataset = dataset.iloc[train_indices]

        if hasattr(self, 'valid_dataset'):
            self.valid_dataset = pd.concat([self.valid_dataset, dataset.iloc[valid_indices]])
        else:
            self.valid_dataset = dataset.iloc[valid_indices]

        if hasattr(self, 'test_dataset'):
            self.test_dataset = pd.concat([self.test_dataset, dataset.iloc[test_indices]])
        else:
            self.test_dataset = dataset.iloc[test_indices]

        self.train_dataset.reset_index(drop=True, inplace=True)
        self.valid_dataset.reset_index(drop=True, inplace=True)
        self.test_dataset.reset_index(drop=True, inplace=True)

    def process_dataset(self, eeg_dataset):
        # Assuming eeg_dataset excludes timestamps and labels columns
        processed_data = process_eeg_data(eeg_dataset, self.fs)
        # Combine processed_data with labels
        processed_data_with_labels = processed_data.copy()
        #processed_data_with_labels['Label'] = labels
        return processed_data_with_labels

    def load_files_and_process(self, folder_path):
        file_paths = glob.glob(folder_path + '/*.csv')
        all_data = []  # This will hold all processed data with labels
        self.all_data_combined = []
        for file in file_paths:
            df = pd.read_csv(file)
            df_trimmed = df.iloc[self.trim_amount:-self.trim_amount]
            processed_data_with_labels = self.process_dataset(df_trimmed)

            self.split_dataset(processed_data_with_labels)
            self.all_data_combined.append(processed_data_with_labels)

        combined_df = pd.concat(self.all_data_combined, ignore_index=True)
        self.all_data_combined = combined_df

    def handle_labels(self):
        # Assuming each dataset has a 'Label' column with the labels
        # Fit the encoder on the labels of the training dataset
        if hasattr(self, 'train_dataset') and 'Label' in self.train_dataset.columns:
            self.label_encoder.fit(self.train_dataset['Label'])

            # Transform the labels in each dataset
            self.train_dataset['Label'] = self.label_encoder.transform(self.train_dataset['Label'])

            if hasattr(self, 'valid_dataset') and 'Label' in self.valid_dataset.columns:
                self.valid_dataset['Label'] = self.label_encoder.transform(self.valid_dataset['Label'])

            if hasattr(self, 'test_dataset') and 'Label' in self.test_dataset.columns:
                self.test_dataset['Label'] = self.label_encoder.transform(self.test_dataset['Label'])

    def run_data_check(self):
        analyze_dataset(self.test_dataset,self.label_encoder)

    def get_training_dataloader(self, batch_size=64, shuffle=True, num_workers=0):


        # If your dataset is very large and you wish to use only a part of it for quick experiments, you can use Subset
        # For example, to use the first 100 samples: subset = Subset(self, range(100))
        # Otherwise, use the full dataset: subset = self

        # Initialize DataLoader with the training dataset
        training_loader = DataLoader(
            self,  # Or 'subset' if you're using a subset of the dataset
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )

        return training_loader

    def get_validation_dataloader(self, batch_size=64, shuffle=True, num_workers=0):


        # Assuming self.valid_dataset is already populated and is a pandas DataFrame
        # Convert the valid_dataset DataFrame into a list of tuples (input_vector, label)
        # This step is necessary to align with the expected input format for PyTorch DataLoaders
        validation_data = [(row[2:].values.astype(np.float32), row['Label']) for _, row in
                           self.valid_dataset.iterrows()]

        # Creating a custom dataset for the validation data
        validation_dataset = torch.utils.data.TensorDataset(
            torch.tensor([i[0] for i in validation_data], dtype=torch.float),
            torch.tensor([i[1] for i in validation_data], dtype=torch.long))

        # Initialize DataLoader with the validation dataset
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )

        return validation_loader

    def get_test_dataloader(self, batch_size=64, shuffle=False, num_workers=0):
        from torch.utils.data import DataLoader, TensorDataset
        import torch

        # Assuming self.test_dataset is a pandas DataFrame with your test data
        test_data = [(row[1:].values.astype(np.float32), row['Label']) for _, row in self.test_dataset.iterrows()]

        # Convert the test data to a PyTorch TensorDataset
        test_features = torch.tensor([i[0] for i in test_data], dtype=torch.float)
        test_labels = torch.tensor([i[1] for i in test_data], dtype=torch.long)
        test_dataset = TensorDataset(test_features, test_labels)

        # Create the DataLoader
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )

        return test_loader

    def __len__(self):
        return len(self.train_dataset)

    def __getitem__(self, idx):
        # Assuming self.train_dataset is a DataFrame with 'Timestamp', 'Label', and 'EEG 1'-'EEG 8' columns
        row = self.train_dataset.iloc[idx]
        # Assuming EEG data starts from the 3rd column in your DataFrame
        eeg_data = row[2:].values.astype(np.float32)
        label = row['Label']
        return eeg_data, label


# FUCKKKKKK I HAVE TO CREATE A CUSTOM DATALOADER AND DATA SET TO PERSERVE THE TIME SERIES OF EACH TEST CAUSE MIXING IT WILL MESS WITH PREDICTIONS FUCK MY LIFE.