import numpy as np
import matplotlib.pyplot as plt
import time
import mne
from scipy.signal import spectrogram
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from collections import deque

# =============================================
# EEG Data Loader with Enhanced Feature Extraction
# =============================================
class RealEEGDataLoader:
    def __init__(self, file_path, channels, rest_duration=30, window_size=1.0, max_duration=None):
        self.raw = mne.io.read_raw_edf(file_path, preload=True)
        self.sfreq = self.raw.info['sfreq']
        self.raw.pick(channels)

        if max_duration is not None:
            self.raw.crop(tmax=max_duration)

        data, _ = self.raw[:, :]
        self.data = np.mean(data, axis=0)

        self.nperseg = int(window_size * self.sfreq)
        self.window_size = window_size
        self.rest_duration = rest_duration
        self.rest_windows = int(rest_duration / window_size)  # Define this first

        # Higher resolution spectrogram
        freqs, _, Sxx = spectrogram(self.data, fs=self.sfreq,
                                  nperseg=self.nperseg*2,
                                  noverlap=int(self.nperseg*1.5),
                                  mode='psd')

        # Define frequency bands with more detail
        self.mu_low = (freqs >= 8) & (freqs <= 10)
        self.mu_high = (freqs > 10) & (freqs <= 12)
        self.beta_low = (freqs >= 13) & (freqs <= 20)
        self.beta_high = (freqs > 20) & (freqs <= 30)
        self.gamma = (freqs >= 30) & (freqs <= 50)

        # Calculate band powers
        self.total_power = np.sum(Sxx, axis=0)  # Total power across all frequencies
        self.mu_power = np.sum(Sxx[self.mu_low | self.mu_high, :], axis=0)
        self.beta_power = np.sum(Sxx[self.beta_low | self.beta_high, :], axis=0)
        self.gamma_power = np.sum(Sxx[self.gamma, :], axis=0)
        
        # Calculate more sophisticated features
        self.beta_mu_ratio = (self.beta_power + 1e-6) / (self.mu_power + 1e-6)
        self.high_low_ratio = (self.beta_high + 1e-6) / (self.beta_low + 1e-6)
        
        # Now we can safely use self.rest_windows
        rest_mu_mean = np.mean(self.mu_power[:self.rest_windows])
        rest_mu_std = np.std(self.mu_power[:self.rest_windows])
        self.erd_score = (self.mu_power - rest_mu_mean) / (rest_mu_std + 1e-6)

        self.times = np.arange(len(self.total_power)) * window_size
        self.current_idx = 0

    def get_sample(self):
        if self.current_idx >= len(self.total_power):
            return None
        
        features = {
            'total_power': self.total_power[self.current_idx],
            'mu': self.mu_power[self.current_idx],
            'beta': self.beta_power[self.current_idx],
            'gamma': self.gamma_power[self.current_idx],
            'beta_mu_ratio': self.beta_mu_ratio[self.current_idx],
            'high_low_ratio': self.high_low_ratio[self.current_idx],
            'erd_score': self.erd_score[self.current_idx]
        }
        
        t = self.times[self.current_idx]
        self.current_idx += 1
        return features, t

# [Rest of the code remains the same...]

# =============================================
# High-Sensitivity Intention Classifier
# =============================================
class SensitiveIntentionClassifier:
    def __init__(self):
        self.lda = LinearDiscriminantAnalysis()
        self.scaler = StandardScaler()
        self._train_with_sensitive_patterns()
        
    def _train_with_sensitive_patterns(self):
        # Create synthetic data with more extreme patterns
        np.random.seed(42)
        n_samples = 1000
        
        # Class 0: No intention (baseline)
        X0 = np.column_stack([
            np.random.normal(100, 10, n_samples),  # total_power
            np.random.normal(15, 3, n_samples),    # mu
            np.random.normal(10, 2, n_samples),    # beta
            np.random.normal(5, 1, n_samples),     # gamma
            np.random.normal(0.7, 0.2, n_samples), # beta/mu
            np.random.normal(1.0, 0.3, n_samples), # high/low
            np.random.normal(0, 1, n_samples)      # ERD score
        ])
        
        # Class 1: Strong intention patterns
        X1 = np.column_stack([
            np.random.normal(80, 15, n_samples),   # reduced total power
            np.random.normal(8, 2, n_samples),     # strongly reduced mu
            np.random.normal(15, 4, n_samples),    # increased beta
            np.random.normal(8, 2, n_samples),     # increased gamma
            np.random.normal(2.0, 0.5, n_samples), # high beta/mu
            np.random.normal(1.5, 0.4, n_samples), # high beta high/low
            np.random.normal(-2, 1.5, n_samples)    # strong ERD
        ])
        
        X = np.vstack([X0, X1])
        y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
        
        # Train classifier with more sensitive weights
        X_scaled = self.scaler.fit_transform(X)
        self.lda.fit(X_scaled, y)
        
    def predict_intention(self, features):
        # Create feature vector
        X = np.array([[
            features['total_power'],
            features['mu'],
            features['beta'],
            features['gamma'],
            features['beta_mu_ratio'],
            features['high_low_ratio'],
            features['erd_score']
        ]])
        
        X_scaled = self.scaler.transform(X)
        
        # Get prediction and amplified decision score
        pred = self.lda.predict(X_scaled)[0]
        decision_score = self.lda.decision_function(X_scaled)[0] * 2  # Amplify changes
        
        # Sigmoid with steeper curve for more sensitivity
        confidence = 1 / (1 + np.exp(-decision_score*3))  # More sensitive sigmoid
        
        return pred, confidence

# =============================================
# EMS Controller with Pronounced Changes
# =============================================
def map_to_ems(intention, confidence):
    """More dramatic EMS response to detected intentions"""
    if intention == 1:  # Movement intention detected
        # Strong response (3-10) based on confidence
        intensity = 3 + (confidence * 7)  # More pronounced range
    else:
        # Minimal resting level (1-2)
        intensity = 1 + (confidence * 1)
    
    return np.clip(intensity, 1, 10)

class ResponsiveSmoother:
    """Smoother that allows quicker response to strong signals"""
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        self.confidence_buffer = deque(maxlen=window_size)
        
    def update(self, value, confidence):
        self.buffer.append(value)
        self.confidence_buffer.append(confidence)
        
        if len(self.buffer) < 2:  # Very responsive with minimum samples
            return value
            
        # Confidence-weighted average with emphasis on recent high-confidence samples
        weights = np.array(self.confidence_buffer)**2  # Square to emphasize high confidence
        values = np.array(self.buffer)
        return np.average(values, weights=weights)

class DynamicRampLimiter:
    """More aggressive ramping for high-confidence detections"""
    def __init__(self, base_rate=3):  # Higher base rate
        self.prev_value = 5.0
        self.base_rate = base_rate
        
    def apply(self, target, confidence):
        # Very responsive to high confidence
        max_delta = self.base_rate * 0.1 * (0.2 + confidence*1.8)  # Wider range
        delta = np.clip(target - self.prev_value, -max_delta, max_delta)
        self.prev_value += delta
        return self.prev_value

# =============================================
# Main Processing Loop
# =============================================
if __name__ == "__main__":
    FILE_PATH = "sub-01_task-motor-imagery_eeg.edf"
    CHANNELS = ["C3", "C4", "Cz", "FC3", "FC4", "CP3", "CP4"]
    WINDOW_SIZE = 1.0
    MAX_DURATION = 90  # seconds

    # Initialize components
    eeg_loader = RealEEGDataLoader(FILE_PATH, CHANNELS, window_size=WINDOW_SIZE, max_duration=MAX_DURATION)
    classifier = SensitiveIntentionClassifier()
    smoother = ResponsiveSmoother(window_size=5)
    ramp_limiter = DynamicRampLimiter(base_rate=4)  # More aggressive ramping

    # Data recording
    timestamps = []
    total_power = []
    predictions = []
    confidences = []
    ems_values = []
    time_taken = []

    print("Starting EEG processing with enhanced sensitivity...")
    start_time = time.time()

    while True:
        iteration_start = time.time()
        result = eeg_loader.get_sample()
        if result is None:
            break
            
        features, t = result

        if t < eeg_loader.rest_duration:
            # Rest period - no stimulation
            final_ems = 0.0
            pred = 0
            confidence = 0
        else:
            # Classify intention with sensitive detector
            pred, confidence = classifier.predict_intention(features)
            
            # Map to EMS intensity with pronounced changes
            raw_ems = map_to_ems(pred, confidence)
            
            # Apply responsive smoothing and aggressive ramping
            smoothed_ems = smoother.update(raw_ems, confidence)
            safe_ems = ramp_limiter.apply(smoothed_ems, confidence)
            final_ems = np.clip(safe_ems, 1, 10)

        # Record data
        timestamps.append(t)
        total_power.append(features['total_power'])
        predictions.append(pred)
        confidences.append(confidence)
        ems_values.append(final_ems)
        
        # Timing control
        iteration_time = time.time() - iteration_start
        time_taken.append(iteration_time)
        time_left = WINDOW_SIZE - iteration_time
        if time_left > 0:
            time.sleep(time_left)

    total_time = time.time() - start_time
    print(f"\nProcessing complete! Total time: {total_time:.1f}s")
    print(f"Average iteration time: {np.mean(time_taken)*1000:.1f}ms")
    print(f"Intention detected in {100*np.mean(predictions[eeg_loader.rest_windows:]):.1f}% of active windows")

    # Convert to arrays for plotting
    timestamps = np.array(timestamps)
    total_power = np.array(total_power)
    predictions = np.array(predictions)
    confidences = np.array(confidences)
    ems_values = np.array(ems_values)
    
    # Normalize power for plotting
    power_normalized = (total_power - np.min(total_power)) / (np.max(total_power) - np.min(total_power)) * 100

    # Visualization
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Total Power
    plt.subplot(4, 1, 1)
    plt.plot(timestamps, power_normalized, 'b-', label='Total Power')
    plt.axvline(eeg_loader.rest_duration, color='r', linestyle='--')
    plt.ylabel('Normalized Power')
    plt.title('EEG Total Power (Normalized)')

    # Plot 2: Intention Detection
    plt.subplot(4, 1, 2)
    plt.fill_between(timestamps, 0, predictions*100, color='orange', alpha=0.3, label='Intention')
    plt.plot(timestamps, confidences*100, 'orange', label='Confidence')
    plt.axvline(eeg_loader.rest_duration, color='r', linestyle='--')
    plt.ylabel('Detection (%)')
    plt.ylim(0, 110)
    plt.legend()

    # Plot 3: EMS Output
    plt.subplot(4, 1, 3)
    plt.plot(timestamps, ems_values, 'r-', linewidth=2)
    plt.axvline(eeg_loader.rest_duration, color='r', linestyle='--')
    plt.ylabel('EMS Intensity')
    plt.ylim(0, 10.5)
    plt.title('EMS Stimulation Intensity')

    # Plot 4: Combined View
    plt.subplot(4, 1, 4)
    plt.plot(timestamps, power_normalized, 'b-', alpha=0.5, label='Power')
    plt.plot(timestamps, confidences*100, 'orange', alpha=0.7, label='Confidence')
    plt.plot(timestamps, ems_values*10, 'r-', linewidth=2, label='EMS (x10)')
    plt.axvline(eeg_loader.rest_duration, color='r', linestyle='--')
    plt.ylabel('Combined View')
    plt.xlabel('Time (s)')
    plt.legend()

    plt.tight_layout()
    plt.show()