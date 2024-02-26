import torch


class EchoStateNetwork(torch.nn.Module):
    def __init__(self, input_size, reservoir_size, output_size, spectral_radius=0.95, device=None):
        super(EchoStateNetwork, self).__init__()
        # Initialize parameters
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize weights
        self.W_in = (torch.rand(reservoir_size, input_size, device=self.device) - 0.5)  # Uniform distribution
        self.W_res = (torch.rand(reservoir_size, reservoir_size, device=self.device) - 0.5)
        self.W_out = torch.zeros(output_size, reservoir_size, device=self.device)

        # Initialize the reservoir
        self.reservoir = torch.zeros(reservoir_size, device=self.device)

        # Scale W_res to ensure the spectral radius condition
        radius = spectral_radius / torch.max(torch.abs(torch.linalg.eigvals(self.W_res))).item()
        self.W_res *= radius

    def update_reservoir(self, input_vector):
        # Update the reservoir state


        try:
            input_vector = input_vector.to(self.device)
            self.reservoir = torch.tanh(self.W_in @ input_vector + self.W_res @ self.reservoir)
        except:
            #print(f"Input vector shape: {input_vector.shape}")
            #print(f"W_in shape: {self.W_in.shape}")
            #print(f"W_res shape: {self.W_res.shape}")
            #print(f"Reservoir shape: {self.reservoir.shape}")
            #print("Fucking Broken")
            pass

    def train(self, training_inputs, training_outputs, regularization_coefficient=1e-8):
        # Collect reservoir states
        states = []
        for input_vector in training_inputs:
            self.update_reservoir(input_vector)
            states.append(self.reservoir.unsqueeze(0))  # Add batch dimension
        states = torch.cat(states, dim=0).T  # Concatenate along batch dimension and transpose

        # Train W_out
        # Add regularization to avoid overfitting
        states_T = states.T  # Transpose for correct matrix multiplication
        inverse_term = torch.linalg.inv(
            states @ states_T + regularization_coefficient * torch.eye(self.reservoir_size, device=self.device))
        self.W_out = (training_outputs.T @ states_T) @ inverse_term

    def predict(self, input_vector):
        # Ensure input_vector is a tensor and on the correct device
        input_vector = torch.atleast_2d(input_vector).to(self.device)
        predictions = torch.empty((input_vector.size(0), self.output_size), device=self.device)

        # Process each input vector through the ESN
        for i, iv in enumerate(input_vector):
            self.update_reservoir(iv)
            predictions[i] = self.W_out @ self.reservoir

        # Return predictions
        return predictions.squeeze()
