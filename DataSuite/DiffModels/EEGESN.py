import numpy as np

class EchoStateNetwork:
    def __init__(self, input_size, reservoir_size, output_size, spectral_radius=0.95):
        # Initialize parameters
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        # Initialize weights
        self.W_in = np.random.rand(reservoir_size, input_size) - 0.5  # Uniform distribution
        self.W_res = np.random.rand(reservoir_size, reservoir_size) - 0.5
        self.W_out = np.zeros((output_size, reservoir_size))
        # Initialize the reservoir
        self.reservoir = np.zeros(reservoir_size)
        # Scale W_res to ensure the spectral radius condition
        self.W_res *= spectral_radius / np.max(np.abs(np.linalg.eigvals(self.W_res)))

    def update_reservoir(self, input_vector):
        # Update the reservoir state
        try:
            self.reservoir = np.tanh(np.dot(self.W_in, input_vector) + np.dot(self.W_res, self.reservoir))
        except:
            #print("input_vector")
            pass

    def train(self, training_inputs, training_outputs, regularization_coefficient=1e-8):
        # Collect reservoir states
        states = []
        for input_vector in training_inputs:
            self.update_reservoir(input_vector)
            states.append(self.reservoir)
        states = np.array(states).T  # Transpose to match the shape for training

        # Train W_out
        # Add regularization to avoid overfitting
        self.W_out = np.dot(np.dot(training_outputs.T, states.T), np.linalg.inv(np.dot(states, states.T) + regularization_coefficient * np.eye(self.reservoir_size)))

    def predict(self, input_vector):
        # Ensure input_vector is always a 2D array for consistent matrix operations
        input_vector = np.atleast_2d(input_vector)

        # Initialize an empty array to store predictions for each input vector
        predictions = np.empty((input_vector.shape[0], self.output_size))

        # Process each input vector through the ESN
        for i, iv in enumerate(input_vector):
            self.update_reservoir(iv)
            predictions[i] = np.dot(self.W_out, self.reservoir)

        # If predicting a single sample, return a 1D array
        if predictions.shape[0] == 1:
            return predictions.flatten()
        return predictions
