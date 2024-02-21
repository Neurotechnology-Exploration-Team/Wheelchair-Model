import torch
import torch.nn as nn
import numpy as np

class EEGModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(EEGModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTM layer for processing sequences of EEG data
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Fully connected layers for classification
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):
        # Add a batch dimension to the input if it's missing
        if x.dim() == 2:  # If input is unbatched (2-D), add a batch dimension
            x = x.unsqueeze(0)
        
        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate the LSTM
        out, _ = self.lstm(x, (h0, c0))
        # Decode the hidden state of the last time step
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def predict(self, input_data):
        # Ensure model is in evaluation mode
        self.eval()
        with torch.no_grad():
            # Convert input to tensor if necessary
            if isinstance(input_data, np.ndarray):
                input_data = torch.tensor(input_data, dtype=torch.float)
            # Add batch dimension if it's missing
            if input_data.dim() == 1:
                input_data = input_data.unsqueeze(0).unsqueeze(0)  # Add batch and sequence dimension
            elif input_data.dim() == 2:
                input_data = input_data.unsqueeze(0)  # Add batch dimension
                
            input_data = input_data.to(next(self.parameters()).device)
            # Forward pass
            output = self(input_data)
            # Get the predicted class
            predicted_class = output.argmax(dim=1)
            return predicted_class
