import numpy as np
from scipy import signal

def log_specgram(audio, sample_rate, eps=1e-10):
    nperseg = 1764
    noverlap = 441
    freqs, times, spec = signal.spectrogram(audio, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)

def prepare_data(samples, num_of_samples=176400):
    print(f"Received samples: {samples}")
    samples = np.array(samples, dtype=np.float32)  # Convert directly to numpy array
    print(f"Converted samples to numpy array: {samples}")

    if len(samples) >= num_of_samples:
        data = samples[:num_of_samples]
    else:
        data = np.pad(samples, (0, num_of_samples - len(samples)), 'constant')
    return [data]

def extract_spectrogram_features(x_tr, sample_rate=44100):
    features = []
    for i in x_tr:
        _, _, spectrogram = log_specgram(i, sample_rate)
        mean = np.mean(spectrogram, axis=0)
        std = np.std(spectrogram, axis=0)
        spectrogram = (spectrogram - mean) / std
        features.append(spectrogram)
    return np.array(features)

def extract_features_from_samples(samples, sample_rate=44100):
    processed_data = prepare_data(samples)
    features = extract_spectrogram_features(processed_data, sample_rate)
    return features.tolist()  # Convert numpy array to list before returning
