import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque
import mne
from scipy.signal import spectrogram

# =============================================
# Optimized EEG Data Processor
# =============================================
class RealEEGDataLoader:
    def __init__(self, file_path, channels, rest_duration=5, window_size=1.0):
        # Load and preprocess EEG data
        self.raw = mne.io.read_raw_edf(file_path, preload=True)
        self.sfreq = self.raw.info['sfreq']
        self.raw.pick(channels)
        
        # Convert to numpy array and average channels
        data, _ = self.raw[:, :]
        self.data = np.mean(data, axis=0)
        
        # Calculate window parameters
        self.nperseg = int(window_size * self.sfreq)
        self.window_size = window_size
        self.rest_duration = rest_duration
        
        # Compute spectrogram (vectorized)
        freqs, _, Sxx = spectrogram(self.data, fs=self.sfreq,
                                   nperseg=self.nperseg,
                                   noverlap=0,
                                   mode='psd')
        
        # Calculate band powers for all windows
        mu_band = (freqs >= 8) & (freqs <= 12)
        beta_band = (freqs >= 13) & (freqs <= 30)
        
        self.mu_power = np.sum(Sxx[mu_band, :], axis=0)
        self.beta_power = np.sum(Sxx[beta_band, :], axis=0)
        self.total_power = self.mu_power + self.beta_power
        
        # Calculate baseline power
        self.rest_windows = int(rest_duration/window_size)
        self.rest_power = np.mean(self.total_power[:self.rest_windows])
        
        # Store precomputed times
        self.times = np.arange(len(self.total_power)) * window_size
        self.current_idx = 0

    def get_sample(self):
        if self.current_idx >= len(self.total_power):
            return None, None
        power = self.total_power[self.current_idx]
        t = self.times[self.current_idx]
        self.current_idx += 1
        return power, t

# =============================================
# Optimized Controller Components
# =============================================
def compute_erd(current_power, rest_power):
    return max(0, min((1 - (current_power / rest_power)) * 100, 100))

def map_to_ems(erd_percent, max_erd=70):
    return np.clip(10 - (erd_percent / max_erd) * 9, 1, 10)

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
        self.max_delta = max_rate * 0.1  # Per 0.1s step
        
    def apply(self, target):
        delta = np.clip(target - self.prev_value, -self.max_delta, self.max_delta)
        self.prev_value += delta
        return self.prev_value

# =============================================
# Main Processing Loop
# =============================================
if __name__ == "__main__":
    # Configuration
    FILE_PATH = "sub-01_task-motor-imagery_eeg.edf"
    # CHANNELS = ["C3", "C4"]
    CHANNELS = ["C1", "C2", "C3", "C4", "Cz", "FC1", "FC2", "FC3", "FC4", "CP1", "CP2", "CP3", "CP4"]
    WINDOW_SIZE = 1.0
    
    # Initialize components
    eeg_loader = RealEEGDataLoader(FILE_PATH, CHANNELS)
    smoother = FastSmoother(window_size=5)
    ramp_limiter = RampLimiter(max_rate=2)
    
    # Preallocate arrays
    max_windows = len(eeg_loader.total_power)
    timestamps = np.zeros(max_windows)
    eeg_power = np.zeros(max_windows)
    erd_values = np.zeros(max_windows)
    ems_values = np.zeros(max_windows)
    
    idx = 0
    while True:
        power, t = eeg_loader.get_sample()
        if power is None:
            break
        
        # During rest period: zero EMS
        if t < eeg_loader.rest_duration:
            final_ems = 0.0
        else:
            # Compute ERD and EMS
            erd = compute_erd(power, eeg_loader.rest_power)
            raw_ems = map_to_ems(erd)
            smoothed_ems = smoother.update(raw_ems)
            safe_ems = ramp_limiter.apply(smoothed_ems)
            final_ems = np.clip(safe_ems, 2, 8)
        
        # Store results
        timestamps[idx] = t
        eeg_power[idx] = power
        erd_values[idx] = erd if t >= eeg_loader.rest_duration else 0
        ems_values[idx] = final_ems
        idx += 1
        
        # Simulate real-time (adjust sleep based on actual processing time)
        time.sleep(WINDOW_SIZE * 0.9)  # Compensate for processing time
    
    # Truncate unused preallocated space
    timestamps = timestamps[:idx]
    eeg_power = eeg_power[:idx]
    erd_values = erd_values[:idx]
    ems_values = ems_values[:idx]
    
    # Visualization
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(timestamps, eeg_power)
    plt.axvline(eeg_loader.rest_duration, color='r', linestyle='--')
    plt.ylabel('Power\n(mu + beta)')
    plt.title(f'EEG Power (Baseline: {eeg_loader.rest_power:.2f} µV²)')
    
    plt.subplot(3, 1, 2)
    plt.plot(timestamps, erd_values, 'orange')
    plt.axvline(eeg_loader.rest_duration, color='r', linestyle='--')
    plt.ylabel('ERD (%)')
    plt.ylim(0, 100)
    
    plt.subplot(3, 1, 3)
    plt.plot(timestamps, ems_values, 'red')
    plt.axvline(eeg_loader.rest_duration, color='r', linestyle='--')
    plt.ylabel('EMS Level')
    plt.xlabel('Time (s)')
    plt.ylim(0, 10)
    
    plt.tight_layout()
    plt.show()