import torch
import torch.nn as nn

class EEGModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, sparsity=0.1, spectral_radius=0.95):
        super(EEGModel, self).__init__()
        self.hidden_size = hidden_size
        
        self.W_in = nn.Parameter(torch.randn(hidden_size, input_size), requires_grad=False)
        W_res = torch.randn(hidden_size, hidden_size)
        W_res = torch.where(torch.rand(hidden_size, hidden_size) < sparsity, torch.zeros_like(W_res), W_res)
        radius = torch.max(torch.abs(torch.linalg.eigvals(W_res))).real
        self.W_res = nn.Parameter(W_res * (spectral_radius / radius), requires_grad=False)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        if hasattr(self, 'h'):
            if self.h.size(0) != batch_size:
                self.h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            self.h = torch.zeros(batch_size, self.hidden_size, device=x.device)

        x_transformed = torch.tanh(self.W_in @ x.T + self.W_res @ self.h.T)
        self.h = x_transformed.T
        
        out = self.fc(self.h)
        return out

    def reset_state(self):
        del self.h
