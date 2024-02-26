from Dataset import EEGDataset
from DiffModels.EEGESN import EchoStateNetwork
import numpy as np
import pickle
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_esn(esn, training_loader, validation_loader=None, epochs=10):
    for epoch in range(epochs):
        esn_states = []
        targets = []
        processed_sequences = 0

        # Training phase
        for input_sequence, target_output in training_loader:
            current_states = []
            for input_vector in input_sequence:
                esn.update_reservoir(input_vector)
                current_states.append(esn.reservoir.clone())


            esn_states.extend(current_states)
            targets.extend(target_output)
            processed_sequences += 1
            #if processed_sequences % 500 == 0:
            #    print(f"Processed {processed_sequences} sequences in epoch {epoch + 1}.")

        esn_states = torch.stack(esn_states).to(device)
        targets = torch.tensor(targets).to(device)

        esn.train(esn_states, targets)

        print(f"Epoch {epoch + 1}/{epochs} completed. Processed {processed_sequences} sequences.")

        if validation_loader:
            validate_esn(esn, validation_loader)


def validate_esn(esn, validation_loader):
    total_mse = 0
    total_samples = 0

    for input_sequence, target_output in validation_loader:
        predictions = []
        for input_vector in input_sequence:
            prediction = esn.predict(input_vector)
            predictions.append(prediction)

        predictions = torch.stack(predictions)
        mse = torch.mean((predictions - target_output) ** 2).item()
        total_mse += mse * len(input_sequence)
        total_samples += len(input_sequence)

    average_mse = total_mse / total_samples
    print(f"Validation MSE: {average_mse}")


def evaluate_model_accuracy(model, dataset_loader):
    correct_predictions = 0
    total_predictions = 0

    for input_vector, actual_label in dataset_loader:
        predicted_output = model.predict(input_vector)
        predicted_label = torch.argmax(predicted_output, dim=1)
        correct_predictions += (predicted_label == actual_label).sum().item()
        total_predictions += actual_label.size(0)

    accuracy = correct_predictions / total_predictions
    print(f"Model Accuracy: {accuracy:.4f}")

def test_model_predictions(model, test_loader):
    for input_sequence, target_output in test_loader:
        predictions = []
        for input_vector in input_sequence:
            prediction = model.predict(input_vector)
            predictions.append(prediction)

        predictions = torch.stack(predictions)
        print("Model Predictions:", predictions)
        print("Actual Targets:", target_output.numpy())
        mse = torch.mean((predictions - target_output) ** 2).item()
        print(f"MSE: {mse}")
def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")
def load_model(filename):
    esn = EchoStateNetwork(input_size=8, reservoir_size=100,
                           output_size=1)  # 13 different possibilities but only one output maybe change in future for different percentages?
    return esn.load_state_dict(torch.load(filename))


def model_creation(training_loader, validation_loader):
    esn = EchoStateNetwork(input_size=8, reservoir_size=100, output_size=1)# 13 different possibilities but only one output maybe change in future for different percentages?
    train_esn(esn, training_loader, validation_loader=validation_loader, epochs=10)
    save_model(esn,"esnModel")
    return esn
def main():
    csv_file_path = '../cata'
    eeg_dataset = EEGDataset(csv_file_path)

    training_loader = eeg_dataset.get_training_dataloader(batch_size=1, shuffle=False)
    validation_loader = eeg_dataset.get_validation_dataloader(batch_size=1)
    model_creation(training_loader,validation_loader)

    #esn = load_model("esnModel")
    #test_loader = eeg_dataset.get_test_dataloader(batch_size=64)
    ##evaluate_model_accuracy(esn, test_loader)
    #test_model_predictions(esn, test_loader)


if __name__ == "__main__":
    main()
    #1278.1963002083849





    ##Ok we need to work with batches so that the feature extraction can work lets try a window and single datapoints