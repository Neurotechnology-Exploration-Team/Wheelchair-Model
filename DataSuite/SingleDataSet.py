from torch.utils.data import DataLoader, Dataset
import torch
class SingleDataSet(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert the selected row to a tensor
        a = self.data.iloc[idx].values
        data_row = torch.tensor(self.data.iloc[idx].values, dtype=torch.float32)
        # Convert the label to a tensor (assuming labels is a 1D numpy array of ints)
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # Use torch.long for integer labels

        return data_row, label