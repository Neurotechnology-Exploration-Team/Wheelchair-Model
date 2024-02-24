import pandas as pd
from pandas import DataFrame
from scipy.signal import butter, lfilter, welch
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.stats import skew, kurtosis

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def power_spectral_density(data, fs):
    f, Pxx_den = welch(data, fs, nperseg=len(data))
    return f, Pxx_den

def normalize_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data


def extract_band_power(psd, freqs, band):
    """Extract band power within a specific frequency range from PSD.

    Args:
        psd (np.array): Power spectral density of the signal.
        freqs (np.array): Frequencies corresponding to the PSD values.
        band (tuple): A tuple defining the lower and upper frequency of the band.

    Returns:
        float: The average power within the band.
    """
    band_freqs = (freqs >= band[0]) & (freqs <= band[1])
    band_power = np.mean(psd[band_freqs])
    return band_power


def hjorth_parameters(signal):
    """Compute Hjorth parameters: Activity, Mobility, and Complexity.

    Args:
        signal (np.array): The EEG signal (time series).

    Returns:
        tuple: A tuple containing Activity, Mobility, and Complexity.
    """
    activity = np.var(signal)
    gradient = np.diff(signal)
    mobility = np.sqrt(np.var(gradient) / activity)
    gradient2 = np.diff(gradient)
    complexity = np.sqrt(np.var(gradient2) / np.var(gradient)) / mobility
    return activity, mobility, complexity


def time_domain_features(signal):
    """Calculate basic time domain features of a signal.

    Args:
        signal (np.array): The EEG signal (time series).

    Returns:
        dict: A dictionary containing the features.
    """
    features = {
        'mean': np.mean(signal),
        'variance': np.var(signal),
        'skewness': skew(signal),
        'kurtosis': kurtosis(signal)
    }
    return features


def create_feature_vector(eeg_segment, fs):
    """Create a feature vector from an EEG segment.

    Args:
        eeg_segment (np.array): The EEG signal segment.
        fs (int): Sampling frequency.

    Returns:
        np.array: The feature vector.
    """
    f, psd_values = power_spectral_density(eeg_segment, fs)
    feature_vector = [
        extract_band_power(psd_values, f, (0.5, 4)),  # Delta
        extract_band_power(psd_values, f, (4, 8)),  # Theta
        extract_band_power(psd_values, f, (8, 12)),  # Alpha
        extract_band_power(psd_values, f, (12, 30)),  # Beta
        extract_band_power(psd_values, f, (30, 45)),  # Gamma
    ]
    activity, mobility, complexity = hjorth_parameters(eeg_segment)
    feature_vector.extend([activity, mobility, complexity])
    time_features = time_domain_features(eeg_segment)
    feature_vector.extend(list(time_features.values()))

    return np.array(feature_vector)


def process_eeg_data(eeg_data, fs):
    """
    Process raw EEG data to extract relevant features and normalize the data.

    Args:
        eeg_data (np.array): The raw EEG data, shape (channels, samples).
        fs (int): Sampling frequency of the EEG data.

    Returns:
        np.array: An array of feature vectors for each EEG channel.
    """
    processed_data = []
    df = pd.DataFrame()
    # Assuming eeg_data is a 2D array; process each channel separately
    for channel_data in eeg_data:
        if channel_data in ["Timestamp","Label"]:
            df[channel_data] = eeg_data[channel_data]
            continue
        # Step 1: Filter the data
        filtered_data = bandpass_filter(eeg_data[channel_data], lowcut=1, highcut=50, fs=fs)

        # Step 2: Normalize the filtered data
        normalized_data = normalize_data(filtered_data.reshape(-1, 1)).flatten()

        # Step 3: Extract features
        #feature_vector = create_feature_vector(normalized_data, fs)

        df[channel_data] = normalized_data

    return df
