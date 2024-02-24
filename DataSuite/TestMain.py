from Dataset import EEGDataset
from DiffModels.EEGESN import EchoStateNetwork
import numpy as np


def train_esn(esn, training_loader, validation_loader=None, epochs=10):
    for epoch in range(epochs):
        esn_states = []
        targets = []
        processed_sequences = 0  # Track the number of sequences processed

        # Training phase
        for input_sequence, target_output in training_loader:
            # Process each sequence through the ESN to collect states
            current_states = []
            for input_vector in input_sequence:
                esn.update_reservoir(input_vector)
                current_states.append(esn.reservoir.copy())  # Collect reservoir state

            esn_states.extend(current_states)
            targets.extend(target_output)
            processed_sequences += 1  # Update processed sequences
            # Inside your for loop
            if processed_sequences % 500 == 0:  # Adjust the 100 to your preference
                print(f"Processed {processed_sequences} sequences in epoch {epoch + 1}.")

        # Convert lists to appropriate numpy arrays for training
        esn_states = np.array(esn_states)
        targets = np.array(targets)

        # Train the ESN's output weights
        esn.train(esn_states, targets)

        print(f"Epoch {epoch + 1}/{epochs} completed. Processed {processed_sequences} sequences.")

        ## Validation phase (optional)
        if validation_loader:
            validation_mse = validate_esn(esn, validation_loader)  # Assume validate_esn returns MSE
            print(f"Validation MSE: {validation_mse}")

        # Optionally, you could add more detailed performance metrics here


def validate_esn(esn, validation_loader):
    total_mse = 0
    total_samples = 0

    for input_sequence, target_output in validation_loader:
        predictions = []
        for input_vector in input_sequence:
            prediction = esn.predict(input_vector)
            predictions.append(prediction)

        predictions = np.array(predictions)
        mse = np.mean(
            (predictions - target_output.numpy()) ** 2)  # Ensure target_output is a numpy array for this operation
        total_mse += mse * len(input_sequence)
        total_samples += len(input_sequence)

    average_mse = total_mse / total_samples
    print(f"Validation MSE: {average_mse}")


def evaluate_model_accuracy(model, dataset_loader):
    correct_predictions = 0
    total_predictions = 0

    for input_vector, actual_label in dataset_loader:
        predicted_output = model.predict(input_vector)
        predicted_label = np.argmax(predicted_output, axis=1)  # Assuming output is scores for each class
        correct_predictions += (predicted_label == actual_label.numpy()).sum()
        total_predictions += actual_label.size(0)

    accuracy = correct_predictions / total_predictions
    print(f"Model Accuracy: {accuracy:.4f}")

def main():
    csv_file_path = '../cata'
    eeg_dataset = EEGDataset(csv_file_path)

    training_loader = eeg_dataset.get_training_dataloader(batch_size=64, shuffle=False)
    validation_loader = eeg_dataset.get_validation_dataloader(batch_size=64)
    esn = EchoStateNetwork(input_size=8, reservoir_size=100, output_size=13)

    train_esn(esn, training_loader, validation_loader=validation_loader, epochs=1)
    test_loader = eeg_dataset.get_test_dataloader(batch_size=64)
    evaluate_model_accuracy(esn, test_loader)


if __name__ == "__main__":
    main()
    #1278.1963002083849