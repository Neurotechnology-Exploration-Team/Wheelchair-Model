from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import numpy as np
import time

def realtime_eeg_cyton_processing(model, window_size, overlap, preprocess_fn, prediction_fn):
    params = BrainFlowInputParams()

    board_id = BoardIds.CYTON_BOARD.value
    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()

    buffer_length = window_size + overlap
    buffer = np.zeros((model.input_size, buffer_length))

    try:
        while True:

            data = board.get_board_data() 
            if data.size == 0:
                continue
            eeg_channels = BoardShim.get_eeg_channels(board_id)
            eeg_data = data[eeg_channels, :]
            buffer = np.roll(buffer, -eeg_data.shape[1], axis=1)
            buffer[:, -eeg_data.shape[1]:] = eeg_data
            if eeg_data.shape[1] >= window_size:
                window_data = buffer[:, -window_size:]
                preprocessed_data = preprocess_fn(window_data)
                input_data = preprocessed_data.reshape(1, -1, model.input_size)
                prediction = prediction_fn(model, input_data)

                # Handle the prediction (e.g., visualization, logging, decision making)
                # ...

            time.sleep(0.1) 
    finally:
        board.stop_stream()
        board.release_session()

# Define 'preprocess_fn' and 'prediction_fn' based on your requirements.
# Example usage
realtime_eeg_cyton_processing(esn_model, window_size=256, overlap=128, preprocess_fn, predict_esn)
