import torch
import torch.nn as nn
import config



class EchoStateNetwork(nn.Module):
    def __init__(self, input_size, reservoir_size, output_size, spectral_radius=0.95):
        super(EchoStateNetwork, self).__init__()
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size

        # Initialize reservoir weights
        self.reservoir = nn.Linear(reservoir_size, reservoir_size, bias=False)
        self.reservoir.weight = nn.Parameter(torch.randn(reservoir_size, reservoir_size) * spectral_radius)

        # Initialize input weights
        self.input_weights = nn.Parameter(torch.randn(reservoir_size, input_size))

        # Output layer (trainable)
        self.output = nn.Linear(reservoir_size, output_size, bias=True)

    def forward(self, x):
        states = []
        state = torch.zeros(1, self.reservoir_size)
        
        # Feedforward through the reservoir
        for t in range(x.size(1)):
            state = torch.tanh(self.reservoir(state) + torch.matmul(x[:, t], self.input_weights.T))
            states.append(state)

        states = torch.cat(states, dim=0)
        output = self.output(states)
        return output

def train_esn(model, data_loader, criterion, epochs=10):
    model.output.weight.requires_grad = True
    model.output.bias.requires_grad = True

    optimizer = torch.optim.Adam([model.output.weight, model.output.bias], lr=0.001)
    
    for epoch in range(epochs):
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

def predict_esn(model, input_data):
    model.eval()
    with torch.no_grad():
        predictions = model(input_data)
        # Assuming Gaussian output
        mean, std_dev = predictions.chunk(2, dim=1)
        gaussian_output = torch.normal(mean, std_dev)
    return gaussian_output

def createModel():
    return EchoStateNetwork(config.input_size, config.reservoir_size, config.output_size)


def trainModel(esn,data):
    train_esn(esn, data, nn.MSELoss(), epochs=10)

def getSamplePred(esn,sampleEEG):
    return predict_esn(esn, sampleEEG)