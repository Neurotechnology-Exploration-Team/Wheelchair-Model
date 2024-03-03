import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler


def bandpass_filter(data, fs, lowcut=1, highcut=50, order=5):
    from scipy.signal import butter, lfilter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data, axis=0)
    return y


def normalize_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data


def power_spectral_density(channel_data, fs):
    f, Pxx_den = welch(channel_data, fs, nperseg=len(channel_data))
    return np.mean(Pxx_den)


def hjorth_parameters(signal):
    activity = np.var(signal)
    gradient = np.diff(signal)
    mobility = np.sqrt(np.var(gradient) / activity)
    gradient2 = np.diff(gradient)
    complexity = np.sqrt(np.var(gradient2) / np.var(gradient)) / mobility
    return activity, mobility, complexity


def extract_features(segment, fs):
    # Assuming segment is a 2D numpy array with shape (timestamps, channels)
    # Preprocess: Filter and normalize each channel
    filtered_data = bandpass_filter(segment, fs)
    normalized_data = normalize_data(filtered_data)

    features = {}
    for i in range(normalized_data.shape[1]):  # Iterate through each channel
        channel_data = normalized_data[:, i]

        # Time-domain features
        mean_val = np.mean(channel_data)
        variance_val = np.var(channel_data)
        skewness_val = skew(channel_data)
        kurtosis_val = kurtosis(channel_data)

        # Frequency-domain feature (PSD)
        psd_val = power_spectral_density(channel_data, fs)

        # Hjorth parameters
        activity, mobility, complexity = hjorth_parameters(channel_data)

        features[f'channel_{i}'] = {
            'mean': mean_val,
            'variance': variance_val,
            'skewness': skewness_val,
            'kurtosis': kurtosis_val,
            'PSD': psd_val,
            'activity': activity,
            'mobility': mobility,
            'complexity': complexity
        }

    return features