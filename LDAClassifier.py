import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque
import mne
from scipy.signal import spectrogram
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# =============================================
# EEG Data Loader with Optional Trimming
# =============================================
class RealEEGDataLoader:
    def __init__(self, file_path, channels, rest_duration=30, window_size=1.0, max_duration=None):
        self.raw = mne.io.read_raw_edf(file_path, preload=True)
        self.sfreq = self.raw.info['sfreq']
        self.raw.pick(channels)

        # Limit duration if specified
        if max_duration is not None:
            max_samples = int(max_duration * self.sfreq)
            self.raw = self.raw.copy().crop(tmax=max_duration)

        data, _ = self.raw[:, :]
        self.data = np.mean(data, axis=0)

        self.nperseg = int(window_size * self.sfreq)
        self.window_size = window_size
        self.rest_duration = rest_duration

        freqs, _, Sxx = spectrogram(self.data, fs=self.sfreq,
                                   nperseg=self.nperseg,
                                   noverlap=0,
                                   mode='psd')

        mu_band = (freqs >= 8) & (freqs <= 12)
        beta_band = (freqs >= 13) & (freqs <= 30)

        self.mu_power = np.sum(Sxx[mu_band, :], axis=0)
        self.beta_power = np.sum(Sxx[beta_band, :], axis=0)

        self.rest_windows = int(rest_duration / window_size)
        self.times = np.arange(len(self.mu_power)) * window_size
        self.current_idx = 0

    def get_sample(self):
        if self.current_idx >= len(self.mu_power):
            return None
        mu = self.mu_power[self.current_idx]
        beta = self.beta_power[self.current_idx]
        t = self.times[self.current_idx]
        self.current_idx += 1
        return (mu, beta), t

# =============================================
# Classifier and Mapping Components
# =============================================
def get_dummy_lda():
    # Dummy LDA classifier trained on synthetic data
    X_train = np.random.rand(100, 2) * [20, 30]
    y_train = (X_train[:, 0] < 10).astype(int)  # Simulate intention/no-intention
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    return lda

def map_to_ems(class_output):
    return 8.0 if class_output == 1 else 2.0

class FastSmoother:
    def __init__(self, window_size=5):
        self.buffer = np.zeros(window_size)
        self.idx = 0
        self.full = False

    def update(self, value):
        if self.full:
            self.buffer[:-1] = self.buffer[1:]
            self.buffer[-1] = value
        else:
            self.buffer[self.idx] = value
            self.idx += 1
            if self.idx == len(self.buffer):
                self.full = True
        return self.buffer[:self.idx].mean() if not self.full else self.buffer.mean()

class RampLimiter:
    def __init__(self, max_rate=2):
        self.prev_value = 5.0
        self.max_delta = max_rate * 0.1

    def apply(self, target):
        delta = np.clip(target - self.prev_value, -self.max_delta, self.max_delta)
        self.prev_value += delta
        return self.prev_value

# =============================================
# Main Processing Loop
# =============================================
if __name__ == "__main__":
    FILE_PATH = "sub-01_task-motor-imagery_eeg.edf"
    CHANNELS = ["C3", "C4", "Cz", "FC3", "FC4", "CP3", "CP4"]
    WINDOW_SIZE = 1.0
    MAX_DURATION = 60  # seconds

    eeg_loader = RealEEGDataLoader(FILE_PATH, CHANNELS, window_size=WINDOW_SIZE, max_duration=MAX_DURATION)
    smoother = FastSmoother(window_size=5)
    ramp_limiter = RampLimiter(max_rate=2)
    lda_model = get_dummy_lda()

    max_windows = len(eeg_loader.mu_power)
    timestamps = np.zeros(max_windows)
    mu_values = np.zeros(max_windows)
    beta_values = np.zeros(max_windows)
    ems_values = np.zeros(max_windows)
    predictions = np.zeros(max_windows)
    time_taken = []

    idx = 0
    while True:
        start_time = time.time()
        result = eeg_loader.get_sample()
        if result is None:
            break
        (mu, beta), t = result

        if t < eeg_loader.rest_duration:
            final_ems = 0.0
            pred = 0
        else:
            X = np.array([[mu, beta]])
            pred = lda_model.predict(X)[0]
            raw_ems = map_to_ems(pred)
            smoothed_ems = smoother.update(raw_ems)
            safe_ems = ramp_limiter.apply(smoothed_ems)
            final_ems = np.clip(safe_ems, 2, 8)

        timestamps[idx] = t
        mu_values[idx] = mu
        beta_values[idx] = beta
        ems_values[idx] = final_ems
        predictions[idx] = pred
        idx += 1

        end_time = time.time()
        time_taken.append(end_time - start_time)
        time.sleep(WINDOW_SIZE * 0.9)

    timestamps = timestamps[:idx]
    mu_values = mu_values[:idx]
    beta_values = beta_values[:idx]
    ems_values = ems_values[:idx]
    predictions = predictions[:idx]

    print("Average time taken: ", np.mean(time_taken))
    print("Max time taken: ", np.max(time_taken))

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(timestamps, mu_values, label='Mu Power')
    plt.plot(timestamps, beta_values, label='Beta Power')
    plt.axvline(eeg_loader.rest_duration, color='r', linestyle='--')
    plt.legend()
    plt.ylabel('Power')
    plt.title('Mu and Beta Band Power')

    plt.subplot(3, 1, 2)
    plt.plot(timestamps, predictions, 'orange')
    plt.axvline(eeg_loader.rest_duration, color='r', linestyle='--')
    plt.ylabel('LDA Output')
    plt.yticks([0, 1])

    plt.subplot(3, 1, 3)
    plt.plot(timestamps, ems_values, 'red')
    plt.axvline(eeg_loader.rest_duration, color='r', linestyle='--')
    plt.ylabel('EMS Level')
    plt.xlabel('Time (s)')
    plt.ylim(0, 10)

    plt.tight_layout()
    plt.show()