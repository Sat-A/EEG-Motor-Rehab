import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque
import mne
from scipy.signal import welch

# =============================================
# Real EEG Data Processor (Compatibility Fix)
# =============================================
class RealEEGDataLoader:
    def __init__(self, file_path, channels, rest_duration=5, window_size=1.0):
        # Load and preprocess EEG data
        self.raw = mne.io.read_raw_edf(file_path, preload=True)
        self.sfreq = self.raw.info['sfreq']
        self.raw.pick(channels)  # Modern channel selection
        
        # Convert to numpy array and average channels
        data, _ = self.raw[:, :]
        self.data = np.mean(data, axis=0)  # Average across channels
        
        # Calculate window parameters
        self.nperseg = int(window_size * self.sfreq)
        self.n_overlap = 0  # No overlap between windows
        self.window_size = window_size
        
        # Split data into consecutive windows
        self.n_windows = len(self.data) // self.nperseg
        truncated_length = self.n_windows * self.nperseg
        self.data = self.data[:truncated_length]
        
        # Reshape into (n_windows, nperseg)
        self.windowed_data = self.data.reshape(self.n_windows, self.nperseg)
        
        # Calculate power for each window
        self.mu_power = np.zeros(self.n_windows)
        self.beta_power = np.zeros(self.n_windows)
        
        for i in range(self.n_windows):
            freqs, psd = welch(self.windowed_data[i], 
                              fs=self.sfreq,
                              nperseg=self.nperseg,
                              noverlap=self.n_overlap)
            self.mu_power[i] = self._band_power(freqs, psd, 8, 12)
            self.beta_power[i] = self._band_power(freqs, psd, 13, 30)
            
        self.total_power = self.mu_power + self.beta_power
        
        # Calculate baseline power (first 5 seconds)
        rest_windows = int(rest_duration/window_size)
        self.rest_power = np.mean(self.total_power[:rest_windows])
        
        self.current_index = 0

    def _band_power(self, freqs, psd, low, high):
        """Calculate power in specified frequency band"""
        mask = (freqs >= low) & (freqs <= high)
        return np.sum(psd[mask])

    def get_sample(self):
        """Get next power sample"""
        if self.current_index >= self.n_windows:
            return None
        val = self.total_power[self.current_index]
        self.current_index += 1
        return val

# =============================================
# Rest of the controller components remain unchanged
# =============================================
# =============================================
# Controller Components
# =============================================
def compute_erd(current_power, rest_power):
    erd = (1 - (current_power / rest_power)) * 100
    return max(0, min(erd, 100))  # Clamp between 0-100%

def map_to_ems(erd_percent, max_erd=70):
    ems = 10 - (erd_percent / max_erd) * 9
    return np.clip(ems, 1, 10)

class SmoothingFilter:
    def __init__(self, window_size=5):
        self.buffer = deque(maxlen=window_size)
        
    def update(self, value):
        self.buffer.append(value)
        return np.mean(self.buffer) if self.buffer else value

class RampLimiter:
    def __init__(self, max_rate=2):
        self.prev_value = 5  # Start at mid-range
        self.max_rate = max_rate
        
    def apply(self, target_value):
        delta = target_value - self.prev_value
        delta = np.clip(delta, -self.max_rate*0.1, self.max_rate*0.1)
        self.prev_value += delta
        return self.prev_value
    
# =============================================
# Main Processing Loop
# =============================================

if __name__ == "__main__":
    # Configuration
    FILE_PATH = "sub-01_task-motor-imagery_eeg.edf"
    CHANNELS = ["C3", "C4"]  # Use appropriate channels from your data
    WINDOW_SIZE = 1.0  # Seconds per analysis window
    
    # Initialize components
    eeg_loader = RealEEGDataLoader(FILE_PATH, CHANNELS, window_size=WINDOW_SIZE)
    smoother = SmoothingFilter(window_size=5)
    ramp_limiter = RampLimiter(max_rate=2)
    
    # Storage for visualization
    timestamps = []
    eeg_power = []
    erd_values = []
    ems_values = []
    
    # Process data
    current_time = 0.0
    while True:
        power = eeg_loader.get_sample()
        if power is None:
            break
        
        # Calculate ERD and EMS
        erd = compute_erd(power, eeg_loader.rest_power)
        raw_ems = map_to_ems(erd)
        smoothed_ems = smoother.update(raw_ems)
        safe_ems = ramp_limiter.apply(smoothed_ems)
        final_ems = np.clip(safe_ems, 2, 8)
        
        # Store results
        timestamps.append(current_time)
        eeg_power.append(power)
        erd_values.append(erd)
        ems_values.append(final_ems)
        
        current_time += WINDOW_SIZE
        time.sleep(WINDOW_SIZE)

    # Visualization (same as before)
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(timestamps, eeg_power, label='EEG Power')
    plt.ylabel('Combined Power\n(mu + beta)')
    plt.title('Real EEG Power Dynamics')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(timestamps, erd_values, 'orange', label='ERD%')
    plt.ylabel('ERD (%)')
    plt.ylim(0, 100)
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(timestamps, ems_values, 'red', label='EMS Intensity')
    plt.ylabel('EMS Level\n(1-10)')
    plt.xlabel('Time (seconds)')
    plt.ylim(0, 10)
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()