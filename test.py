import time
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds

def main():
    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    params.master_board=BoardIds.CYTON_BOARD.value
    params.file = './data/output_file.txt'
    #params.file = 'C:/Users/Alexb/Desktop/OpenBCI_GUI/data/EEG_Sample_Data/OpenBCI_GUI-v5-blinks-jawClench-alpha.txt'
    board = BoardShim(BoardIds.PLAYBACK_FILE_BOARD.value, params)
    board.prepare_session()
    board.start_stream()
    
    # Stream data for 10 seconds, printing in real time
    start_time = time.time()
    while time.time() - start_time < 10:
        time.sleep(1)  # Adjust the sleep time for more or less frequent updates
        data = board.get_current_board_data(256)  # Get the latest 256 data points
        print(data)

    board.stop_stream()
    board.release_session()

if __name__ == "__main__":
    main()
