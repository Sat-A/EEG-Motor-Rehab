import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque
import mne
from scipy.signal import spectrogram
from scipy.special import expit  # Sigmoid

class RealEEGDataLoader:
    def __init__(self, file_path, channels, rest_duration=30, window_size=1.0):
        self.raw = mne.io.read_raw_edf(file_path, preload=True)
        self.sfreq = self.raw.info['sfreq']
        self.raw.pick(channels)
        data, _ = self.raw[:, :]
        self.data = np.mean(data, axis=0)
        
        self.nperseg = int(window_size * self.sfreq)
        self.window_size = window_size
        self.rest_duration = rest_duration
        
        freqs, _, Sxx = spectrogram(self.data, fs=self.sfreq,
                                    nperseg=self.nperseg,
                                    noverlap=0,
                                    mode='psd')
        
        self.freqs = freqs
        self.Sxx = Sxx
        self.mu_band = (freqs >= 8) & (freqs <= 12)
        self.beta_band = (freqs >= 13) & (freqs <= 30)
        
        self.mu_power = np.sum(Sxx[self.mu_band, :], axis=0)
        self.beta_power = np.sum(Sxx[self.beta_band, :], axis=0)
        
        self.rest_windows = int(rest_duration / window_size)
        self.rest_mu = np.mean(self.mu_power[:self.rest_windows])
        self.rest_beta = np.mean(self.beta_power[:self.rest_windows])
        
        self.times = np.arange(len(self.mu_power)) * window_size
        self.current_idx = 0

    def get_sample(self):
        if self.current_idx >= len(self.mu_power):
            return None, None
        mu = self.mu_power[self.current_idx]
        beta = self.beta_power[self.current_idx]
        t = self.times[self.current_idx]
        self.current_idx += 1
        return (mu, beta), t

def simulate_lda_classification(mu_power, beta_power, rest_mu, rest_beta):
    # Normalised log bandpowers (simple features)
    mu_feat = np.log(mu_power / rest_mu + 1e-6)
    beta_feat = np.log(beta_power / rest_beta + 1e-6)
    
    # Simulated LDA weights (tuned to be reasonable)
    w = np.array([-1.5, -1.0])  # Less power in both → motor intention
    b = 1.2
    
    features = np.array([mu_feat, beta_feat])
    score = np.dot(w, features) + b
    prob = expit(score)  # Sigmoid to get probability [0, 1]
    return prob

def map_classifier_output_to_ems(probability, min_ems=2, max_ems=8):
    return np.clip(min_ems + probability * (max_ems - min_ems), min_ems, max_ems)

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
# Main Loop
# =============================================
if __name__ == "__main__":
    FILE_PATH = "sub-01_task-motor-imagery_eeg.edf"
    # CHANNELS = ["C3", "C4", "Cz", "FC3", "FC4", "CP3", "CP4"]
    CHANNELS = ["C3", "C4"]
    WINDOW_SIZE = 1.0

    eeg_loader = RealEEGDataLoader(FILE_PATH, CHANNELS)
    smoother = FastSmoother(window_size=5)
    ramp_limiter = RampLimiter(max_rate=2)

    max_windows = len(eeg_loader.mu_power)
    timestamps = np.zeros(max_windows)
    classifier_probs = np.zeros(max_windows)
    ems_values = np.zeros(max_windows)
    time_taken = []

    idx = 0
    while True:
        start_time = time.time()
        sample = eeg_loader.get_sample()
        if sample is None:
            continue
        (mu, beta), t = sample

        if t < eeg_loader.rest_duration:
            final_ems = 0.0
            prob = 0.0
        else:
            prob = simulate_lda_classification(mu, beta,
                                               eeg_loader.rest_mu,
                                               eeg_loader.rest_beta)
            raw_ems = map_classifier_output_to_ems(prob)
            smoothed_ems = smoother.update(raw_ems)
            safe_ems = ramp_limiter.apply(smoothed_ems)
            final_ems = safe_ems

        timestamps[idx] = t
        classifier_probs[idx] = prob
        ems_values[idx] = final_ems
        idx += 1

        end_time = time.time()
        time_taken.append(end_time - start_time)
        time.sleep(WINDOW_SIZE * 0.9)
        print(f"time taken: {end_time - start_time:.2e}")

    timestamps = timestamps[:idx]
    classifier_probs = classifier_probs[:idx]
    ems_values = ems_values[:idx]

    print("Average time taken: ", np.mean(time_taken))
    print("Max time taken: ", np.max(time_taken))

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(timestamps, classifier_probs, label='Motor Intent Probability')
    plt.axvline(eeg_loader.rest_duration, color='r', linestyle='--')
    plt.ylim(0, 1)
    plt.ylabel('Probability')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(timestamps, ems_values, 'red', label='EMS Intensity')
    plt.axvline(eeg_loader.rest_duration, color='r', linestyle='--')
    plt.ylabel('EMS Level')
    plt.xlabel('Time (s)')
    plt.ylim(0, 10)
    plt.legend()

    plt.tight_layout()
    plt.show()
