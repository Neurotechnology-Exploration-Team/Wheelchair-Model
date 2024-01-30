import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
fs = 250.0
lowcut = 1.0
highcut = 60.0

def butter_bandpass_filter(data, order=5):

    def butter_bandpass(fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a
    b, a = butter_bandpass(fs, order=order)
    y = lfilter(b, a, data)
    return y

if __name__ == "__main__":


    # Read EEG data from CSV
    csv_file = 'data/collected_data.csv'  # Replace with your file path
    eeg_data = pd.read_csv(csv_file)
    #eeg_data = eeg_data[:len(eeg_data)/2]
    filtered_df = eeg_data[eeg_data.iloc[:, 1:5].ne(0.00000).any(axis=1)]
    # Select one channel (assuming your CSV has columns named as channels)
    channel_name = 'EEG_1'  # Replace with your actual channel name
    eeg_channel = filtered_df[channel_name][:int(len(filtered_df[channel_name]))].values

    # Apply the bandpass filter
    filtered_signal = butter_bandpass_filter(eeg_channel, order=6)

    # Time vector (assuming continuous sampling)
    t = np.arange(len(eeg_channel)) / fs

    # Plotting
    plt.figure(figsize=(122, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, eeg_channel, label='Original Signal')
    plt.title('Original EEG Signal')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t, filtered_signal, label='Filtered Signal', color='orange')
    plt.title('Filtered EEG Signal')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
