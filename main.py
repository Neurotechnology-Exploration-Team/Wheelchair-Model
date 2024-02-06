import pandas as pd
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from EEGDataSet import EEGDataSet
from EEGModel import EEGModel
from brainflow import BoardShim, BrainFlowInputParams, BoardIds
import time
def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in dataloader:
            inputs = data
            labels = torch.zeros(inputs.size(0), dtype=torch.long)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def train(model, train_loader, valid_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            inputs = data
            labels = torch.zeros(inputs.size(0), dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_valid_loss = validate(model, valid_loader, criterion)

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}')
def process_and_predict(data_segment, model):
    data_segment_tensor = torch.tensor(data_segment, dtype=torch.float).unsqueeze(0)
    return model.predict(data_segment_tensor)
def testing(model):

    params = BrainFlowInputParams()
    #params.serial_port = 'COM3'
    board_id = BoardIds.SYNTHETIC_BOARD.value
    board = BoardShim(board_id, params)
    board.prepare_session()

    board.start_stream()
    print("Started data stream.")
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    segment_length = 0.5
    samples_per_segment = int(sampling_rate * segment_length)



    #model.load_state_dict(torch.load('path_to_your_model.pth'))
    model.eval()

    start_time = time.time()
    while True:
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time > segment_length:
            data = board.get_board_data(125)
            if data.shape[1] >= samples_per_segment:
                eeg_channels = [1,2,3,4,5,6,7,8]
                data_segment = data[eeg_channels, -samples_per_segment:].T  # IMPORTANT BRAINFLOW GIVES DATA TRANSPOSED INCORRECTLY
                predicted_class = process_and_predict(data_segment, model)
                print(f"Predicted class: {predicted_class.item()}")
            start_time = current_time
    board.stop_stream()
    board.release_session()
    print("Data stream stopped and session released.")


def main():
    eeg_data_path = 'S002/Blink/trial_00/EEG_data.csv'
    eeg_data = pd.read_csv(eeg_data_path)
    dataset = EEGDataSet(eeg_data)

    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False)
    model = EEGModel(input_size=8, hidden_size=128, num_layers=2, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, train_loader, valid_loader, criterion, optimizer)


    testing(model)


if __name__ == "__main__":
    main()