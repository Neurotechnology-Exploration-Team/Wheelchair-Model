import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, lfilter
import config

def load_eeg_data(file_path):
    """
    Load EEG data from a CSV file.

    Parameters:
    file_path (str): Path to the CSV file containing EEG data.

    Returns:
    DataFrame: Pandas DataFrame containing the EEG data.
    """
    # TODO: Load the CSV file into a DataFrame
    return pd.read_csv(file_path)
def butter_bandpass_filter(data, order=5):
    def butter_bandpass(order=5):
        nyq = 0.5 * config.fs
        low = config.lowcut / nyq
        high = config.highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a
    b, a = butter_bandpass(config.fs, order=order)
    y = lfilter(b, a, data)
    return y

def preprocess_signal(data):
    """
    Preprocess the EEG signal data.

    Parameters:
    data (DataFrame): The EEG data to be preprocessed.

    Returns:
    DataFrame: Preprocessed EEG data.
    """
    # TODO: Implement your preprocessing steps here.
    # Maybe notch filtering for powerline interferance?

    # Example: Bandpass filtering, detrending, etc.
    zero_removed = data[data.iloc[:, 1:5].ne(0.00000).any(axis=1)]

    filtered_fully = butter_bandpass_filter(zero_removed)


    return filtered_fully

def segment_data(data, window_size, overlap_size):
    """
    Segment the data into overlapping windows.

    Parameters:
    data (DataFrame): The EEG data to be segmented.
    window_size (int): The size of each window in samples.
    overlap_size (int): The size of overlap between windows in samples.

    Returns:
    List[DataFrame]: A list of DataFrames, each containing a window of data.
    """
    segments = []
    # TODO: Implement the logic to create overlapping windows
    return segments


def normalize_data(data):
    """
    Normalize the data.

    Parameters:
    data (DataFrame): The EEG data to be normalized.

    Returns:
    DataFrame: Normalized EEG data.
    """
    # TODO: Implement normalization (e.g., StandardScaler from scikit-learn)
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(data), columns=data.columns)


def main():
    file_path = 'path_to_your_eeg_data.csv'  # Replace with your file path
    eeg_data = load_eeg_data(file_path)
    preprocessed_data = preprocess_signal(eeg_data)
    window_size = 125  # Example: 0.5 seconds * 250 Hz
    overlap_size = 62  # Example: 50% overlap
    segmented_data = segment_data(preprocessed_data, window_size, overlap_size)
    normalized_data = [normalize_data(window) for window in segmented_data]

    # TODO: Further processing, feature extraction, etc.

if __name__ == "__main__":
    main()
