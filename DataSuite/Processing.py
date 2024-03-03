import pandas as pd
from pandas import DataFrame
from scipy.signal import butter, lfilter, welch
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.stats import skew, kurtosis
from collections import Counter
from FeatureExtraction import extract_features
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
    """Calculate basic time domain features of a multichannel signal.

    Args:
        signal (np.array): The EEG signal (time series) with shape (channels, samples).

    Returns:
        dict: A dictionary containing the features for each channel.
    """
    # Initialize a dictionary to store features for each channel
    features = {}
    for channel_idx in range(signal.shape[0]):
        channel_signal = signal[channel_idx, :]
        channel_features = {
            'mean': np.mean(channel_signal),
            'variance': np.var(channel_signal),
            'skewness': skew(channel_signal),
            'kurtosis': kurtosis(channel_signal)
        }
        features[f'channel_{channel_idx}'] = channel_features

    return features


import numpy as np


def power_spectral_density_multichannel(eeg_data, fs):
    """
    Compute the Power Spectral Density (PSD) for multichannel EEG data using Welch's method.

    Args:
        eeg_data (np.array): Multichannel EEG data with shape (channels, samples).
        fs (int): Sampling frequency of the EEG data.

    Returns:
        f (np.array): Array of sample frequencies.
        psd_values (np.array): PSD values for each frequency and channel. Shape (channels, len(f)).
    """
    n_channels, _ = eeg_data.shape
    psd_values = []

    for i in range(n_channels):
        f, psd = welch(eeg_data[i, :], fs=fs)
        psd_values.append(psd)

    psd_values = np.array(psd_values)

    return f, psd_values

def extract_band_power_multichannel(psd_values, freqs, freq_band):
    """
    Extract band power from the PSD values for each channel in a specified frequency band.

    Args:
        psd_values (np.array): PSD values for each channel. Shape should be (channels, len(freqs)).
        freqs (np.array): Array of frequencies corresponding to the PSD values.
        freq_band (tuple): A tuple defining the lower and upper limits of the frequency band (low_freq, high_freq).

    Returns:
        np.array: Band power values for each channel within the specified frequency band.
    """
    # Find indices of frequencies within the desired band
    idx_band = np.where((freqs >= freq_band[0]) & (freqs <= freq_band[1]))[0]

    # Sum PSD values within the band for each channel
    band_power = np.sum(psd_values[:, idx_band], axis=1)

    return band_power


def hjorth_parameters_multichannel(eeg_data):
    """
    Calculate Hjorth parameters (Activity, Mobility, Complexity) for multichannel EEG data.

    Args:
        eeg_data (np.array): Multichannel EEG data with shape (channels, samples).

    Returns:
        tuple: A tuple containing arrays for Activity, Mobility, and Complexity for each channel.
    """
    n_channels, _ = eeg_data.shape

    activity = np.var(eeg_data, axis=1)  # Variance of the signal

    # First derivative of the signal
    gradient = np.diff(eeg_data, axis=1)
    mobility = np.sqrt(np.var(gradient, axis=1) / activity)

    # Second derivative of the signal
    gradient2 = np.diff(gradient, axis=1)
    mobility_derivative = np.sqrt(np.var(gradient2, axis=1) / np.var(gradient, axis=1))

    complexity = mobility_derivative / mobility

    return activity, mobility, complexity

def time_domain_features_multichannel(eeg_data):
    """
    Calculate basic time domain features (mean, variance, skewness, kurtosis) for multichannel EEG data.

    Args:
        eeg_data (np.array): Multichannel EEG data with shape (channels, samples).

    Returns:
        dict: A dictionary containing the features for each channel.
    """
    features = {}
    for i, channel_data in enumerate(eeg_data):
        channel_features = {
            'mean': np.mean(channel_data),
            'variance': np.var(channel_data),
            'skewness': skew(channel_data),
            'kurtosis': kurtosis(channel_data)
        }
        features[f'channel_{i}'] = channel_features

    return features
def create_feature_vector_multichannel(eeg_segment, fs):
    """Create a feature vector from a multichannel EEG segment.
    Args:
        eeg_segment (np.array): The EEG signal segment with shape (channels, samples).
        fs (int): Sampling frequency.
    Returns:
        np.array: The feature vector derived from the entire multichannel segment.
    """
    # Calculate the power spectral density for the entire segment
    f, psd_values = power_spectral_density_multichannel(eeg_segment, fs)

    # Extract band power from the entire PSD matrix and create feature vectors
    feature_vector = np.concatenate([
        np.ravel(extract_band_power_multichannel(psd_values, f, (0.5, 4))),  # Delta
        np.ravel(extract_band_power_multichannel(psd_values, f, (4, 8))),  # Theta
        np.ravel(extract_band_power_multichannel(psd_values, f, (8, 12))),  # Alpha
        np.ravel(extract_band_power_multichannel(psd_values, f, (12, 30))),  # Beta
        np.ravel(extract_band_power_multichannel(psd_values, f, (30, 45))),  # Gamma
    ])

    # Compute Hjorth parameters for the multichannel segment and concatenate
    activity, mobility, complexity = hjorth_parameters_multichannel(eeg_segment)
    hjorth_features = np.array([activity, mobility, complexity])

    # Compute time domain features for the entire segment and concatenate
    time_features = np.array(list(time_domain_features_multichannel(eeg_segment).values()))

    # Concatenate all features into one flat feature vector
    final_feature_vector = np.concatenate([feature_vector, hjorth_features, time_features])

    return final_feature_vector

def flatten_features(features):
    """Flatten the features dictionary into a list."""
    flat_features = []
    for channel_features in features.values():
        flat_features.extend(channel_features.values())
    return flat_features
def process_eeg_data(eeg_data, fs):
    """
    Process raw EEG data to extract relevant features and normalize the data.

    Args:
        eeg_data (np.array): The raw EEG data, shape (channels, samples).
        fs (int): Sampling frequency of the EEG data.

    Returns:
        pd.DataFrame: A DataFrame where each row corresponds to features from each channel, and the last column is the label.
    """

    most_common_label = Counter(eeg_data['Label']).most_common(1)[0][0]


    feature_vector= extract_features(eeg_data.iloc[:, 2:], fs)
    flattened_vector = flatten_features(feature_vector)

    df = pd.DataFrame(flattened_vector).transpose()
    df['Label'] = most_common_label

    return df