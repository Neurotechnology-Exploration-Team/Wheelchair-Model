import torch
import torch.nn as nn
import numpy as np

class EEGModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(EEGModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Use LSTM instead of RNN
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Additional fully connected layer for enhanced feature learning
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def predict(self, input_data):
        self.eval()
        with torch.no_grad():
            if isinstance(input_data, np.ndarray):
                input_data = torch.tensor(input_data, dtype=torch.float)
            if input_data.dim() == 2:
                input_data = input_data.unsqueeze(0)
            input_data = input_data.to(next(self.parameters()).device)
            output = self(input_data)
            predicted_class = output.argmax(dim=1)
            return predicted_class
