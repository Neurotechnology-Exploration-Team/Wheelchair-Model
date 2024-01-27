import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Create a custom dataset class
class EEGDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx, 1:-1].values.astype(float)  # Exclude timestamp and label
        blink = self.data.iloc[idx, -1]  # Assuming blink information is in the last column

        # Set label based on blink information
        label = 1 if blink == "blink" else 0

        if self.transform:
            sample = self.transform(sample)

        return sample, label

# Define a data transformation function if needed
def preprocess_data(sample):
    # You can apply preprocessing steps here, such as standardization
    scaler = StandardScaler()
    return scaler.fit_transform(sample.reshape(1, -1))

def get_dataLoader(fileName):
    # Load the dataset
    dataset = EEGDataset(f'./data/{fileName}', transform=preprocess_data)

    # Create a DataLoader
    batch_size = 8  # Set your desired batch size
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader