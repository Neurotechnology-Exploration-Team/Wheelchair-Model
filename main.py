from EEGDataLoader import get_dataLoader, get_shape
import EEGModel
def trainer():
    dataLoader = get_dataLoader("collected_data.csv")
    #shape = get_shape(data_loader=dataLoader)
    #print(shape)
    model = EEGModel.createModel()
    EEGModel.trainModel(model,dataLoader)
    EEGModel.saveModel(model)
    print(model)


import time
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
# Import your model loading and data preprocessing functions
from EEGModel import loadModel  # Replace with your actual import
from EEGDataLoader import preprocess_data  # Replace with your actual import
import torch

from sklearn.preprocessing import StandardScaler
import numpy as np

def preprocess_and_reshape_data(data, expected_rows):
    # Preprocess the data (assuming preprocess_data is already defined)
    processed_data = preprocess_data(data)

    # Check if the processed data has the expected number of rows
    if processed_data.shape[0] != expected_rows:
        # Reshape the data to have the correct number of rows
        reshaped_data = processed_data.reshape(expected_rows, -1)
    else:
        reshaped_data = processed_data

    return reshaped_data


def tester():
    # Initialize BrainFlow
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    board_id = BoardIds.SYNTHETIC_BOARD.value  # Replace with your actual board ID
    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()

    # Load the trained model
    model = loadModel()
    model.eval()  # Set the model to evaluation mode
    data_buffer = np.zeros((8, 500))  # Buffer with 8 channels, 500 samples each

    start_time = time.time()
    while time.time() - start_time < 60:
        data = board.get_board_data()

        while data.shape[1] < 500:
            new_data = board.get_board_data()
            data = np.concatenate((data, new_data), axis=1)

        selected_channels = [0, 1, 2, 3, 4, 5, 6, 7]  # Using first 8 channels
        reshaped_data = data[selected_channels, :500]

        print("Reshaped data shape:", reshaped_data.shape)  # Debug print

        # Ensure tensor_data is (batch_size, channels, length)
        tensor_data = torch.tensor(reshaped_data, dtype=torch.float32).unsqueeze(0)

        print("Tensor data shape:", tensor_data.shape)  # Debug print

        with torch.no_grad():
            predictions = model(tensor_data)
        print(predictions)

        time.sleep(.5)

    board.stop_stream()
    board.release_session()





if __name__ == "__main__":
    trainer()
    #tester()
