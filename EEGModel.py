import torch
import torch.nn as nn
import numpy as np

class ESN(nn.Module):
    def __init__(self, input_size, hidden_size, sparsity=0.1, spectral_radius=0.95):
        super(ESN, self).__init__()
        self.hidden_size = hidden_size
        # Initialize input weights
        self.W_in = nn.Parameter(torch.randn(hidden_size, input_size), requires_grad=False)
        # Initialize reservoir weights
        W_res = torch.randn(hidden_size, hidden_size)
        W_res = torch.where(torch.rand(hidden_size, hidden_size) < sparsity, torch.zeros_like(W_res), W_res)
        radius = torch.max(torch.abs(torch.linalg.eigvals(W_res))).real
        self.W_res = nn.Parameter(W_res * (spectral_radius / radius), requires_grad=False)

    def forward(self, x):
        # Initialize hidden state
        h = torch.zeros(x.size(0), self.hidden_size).to(x.device)
        for t in range(x.size(1)):
            h = torch.tanh(self.W_in @ x[:, t, :].T + self.W_res @ h.T).T
        return h

class EEGModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(EEGModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.esn = ESN(input_size, hidden_size)  # Replace RNN with ESN
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.esn(x)  # This is now [batch_size, features], missing the sequence_length dimension
        out = self.fc(out)  # No need to index the non-existent sequence_length dimension
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
