import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
class EEGModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(EEGModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM for sequence processing
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)

    import torch
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

    def forward(self, x, lengths=None):
        # Ensure x is 3-D (batch_size, seq_len, feature_size)
        # This check is important if you're dynamically adjusting to different input shapes
        if x.dim() == 2:
            # Adding a sequence length of 1 if it's not present
            x = x.unsqueeze(1)  # Adjusts 2D input to 3D by adding a seq_len of 1

        # Initializing hidden state and cell state for the LSTM
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Pack the sequence if lengths are provided
        if lengths is not None:
            # Ensures the LSTM does not process padding as part of the sequence
            x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, _ = self.lstm(x, (h0, c0))

        # Unpack the sequence if lengths were provided
        if lengths is not None:
            out, _ = pad_packed_sequence(out, batch_first=True)

        # Taking the output of the last time step
        # Note: For sequences of varying lengths, consider using the actual last time step per sequence
        # This assumes the last time step is relevant for all sequences, which may not always be ideal
        out = out[:, -1, :]

        # Passing the last time step's output through the fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out

    def predict(self, input_data):
        self.eval()
        with torch.no_grad():
            if isinstance(input_data, np.ndarray):
                input_data = torch.tensor(input_data, dtype=torch.float)
            if input_data.dim() == 1:
                input_data = input_data.unsqueeze(0).unsqueeze(0)
            elif input_data.dim() == 2:
                input_data = input_data.unsqueeze(0)
                
            input_data = input_data.to(next(self.parameters()).device)

            output = self(input_data)

            predicted_class = output.argmax(dim=1)
            return predicted_class
