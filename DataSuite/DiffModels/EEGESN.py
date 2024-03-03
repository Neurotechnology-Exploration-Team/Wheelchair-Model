import torch
import torch.nn as nn



class EchoStateNetwork(torch.nn.Module):
    def __init__(self, input_size, reservoir_size, output_size, spectral_radius=0.95, device=None):
        super(EchoStateNetwork, self).__init__()
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize weights
        self.W_in = (torch.rand(reservoir_size, input_size, device=self.device) - 0.5)  # Input weights
        self.W_res = (torch.rand(reservoir_size, reservoir_size, device=self.device) - 0.5)  # Reservoir weights
        self.W_out = nn.Parameter(torch.zeros(output_size, reservoir_size + input_size, device=self.device), requires_grad=True)

        # Initialize the reservoir state with zeros
        self.reservoir = torch.zeros(self.reservoir_size, device=self.device)

        # Scale W_res to ensure the spectral radius condition
        radius = spectral_radius / torch.max(torch.abs(torch.linalg.eigvals(self.W_res))).item()
        self.W_res *= radius

    def forward(self, x):
        # Assuming x is of shape [batch_size, seq_length, input_size]
        batch_size, seq_length, _ = x.shape
        outputs = torch.zeros(batch_size, seq_length, self.output_size, device=self.device)

        for t in range(seq_length):
            xt = x[:, t, :]
            self.reservoir = torch.tanh(self.W_in @ xt.T + self.W_res @ self.reservoir)
            outputs[:, t] = self.W_out @ torch.cat((xt, self.reservoir), dim=1).T

        return outputs.squeeze()

    def reset_parameters(self):
        # Reset parameters if needed
        self.W_out.data.zero_()

    def reset_reservoir(self):
        self.reservoir = torch.zeros(self.reservoir_size, device=self.device)
    def train_model(self, train_loader, epochs=10, learning_rate=1e-3):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        self.train()
        for epoch in range(epochs):
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                #inputs = inputs.unsqueeze(1)
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
