import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import config

class EchoStateNetwork(nn.Module):
    def __init__(self, input_size, reservoir_size, output_size, spectral_radius=0.95, sparsity=0.1):
        super(EchoStateNetwork, self).__init__()
        self.reservoir = nn.Linear(reservoir_size, reservoir_size)
        
        self.init_reservoir(spectral_radius)
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.spectral_radius = spectral_radius
        self.input_weights = nn.Parameter(torch.randn(reservoir_size, input_size) / np.sqrt(input_size), requires_grad=False)
        reservoir_weights = torch.rand(reservoir_size, reservoir_size) - 0.5
        mask = (torch.rand(reservoir_size, reservoir_size) < sparsity).float()
        reservoir_weights *= mask
        eigenvalues = torch.linalg.eigvals(reservoir_weights).abs()
        reservoir_weights /= eigenvalues.max()
        reservoir_weights *= spectral_radius
        self.reservoir_weights = nn.Parameter(reservoir_weights, requires_grad=False)
        self.output_weights = nn.Linear(reservoir_size, output_size)
    def init_reservoir(self, spectral_radius):
        self.reservoir.weight.data = torch.randn(self.reservoir.weight.data.shape) * spectral_radius

    def forward(self, x):
        state = torch.zeros(x.size(0), self.reservoir_size) 
        outputs = []
        for t in range(x.size(1)):
            input_t = x[:, t]
            input_weighted = torch.matmul(input_t, self.input_weights.T).float()
            state = torch.tanh(self.reservoir(state).float() + input_weighted)
            outputs.append(state)
        outputs = torch.cat(outputs, dim=1)
        outputs = outputs.view(x.size(0), -1)
        if outputs.size(1) != self.reservoir_size:
            raise ValueError(f"Expected size of the second dimension of outputs is {self.reservoir_size}, but got {outputs.size(1)}")

        outputs = self.output_weights(outputs)
        return outputs








def train_esn(model, data_loader, criterion, epochs=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        for i, (inputs, labels) in enumerate(data_loader):
            inputs = Variable(inputs)
            labels = Variable(labels).float()  
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            outputs = outputs.view_as(labels)

            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(data_loader)}], Loss: {loss.item()}')

    print('Training complete')



def predict_esn(model, input_data):
    model.eval()
    with torch.no_grad():
        predictions = model(input_data)
        mean, std_dev = predictions.chunk(2, dim=1)
        gaussian_output = torch.normal(mean, std_dev)
    return gaussian_output

def createModel():
    return EchoStateNetwork(config.input_size, config.reservoir_size, config.output_size)


def trainModel(esn,data):
    train_esn(esn, data, nn.MSELoss(), epochs=10)

def getSamplePred(esn,sampleEEG):
    return predict_esn(esn, sampleEEG)