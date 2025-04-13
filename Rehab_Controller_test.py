import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque

# =============================================
# Synthetic EEG Data Generator (Simplified)
# =============================================
class SyntheticEEGGenerator:
    def __init__(self):
        self.rest_power = 10.0  # Baseline mu/beta power at rest
        self.noise_level = 0.5
        self.state = 'rest'  # or 'attempt'
        self.last_change = 0
        
    def get_sample(self):
        # Simulate state changes every 5 seconds
        if time.time() - self.last_change > 5:
            self.state = 'attempt' if self.state == 'rest' else 'rest'
            self.last_change = time.time()
        
        # Generate synthetic signal based on state
        if self.state == 'rest':
            base = self.rest_power + np.random.normal(0, self.noise_level)
        else:
            # Simulate ERD during movement attempt (60% suppression)
            base = self.rest_power * 0.4 + np.random.normal(0, self.noise_level*2)
            
        return max(base, 0.1)  # Prevent negative values

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
    # Initialize components
    eeg_gen = SyntheticEEGGenerator()
    smoother = SmoothingFilter(window_size=5)
    ramp_limiter = RampLimiter(max_rate=2)  # Max 2 units/sec change
    rest_power = eeg_gen.rest_power
    max_erd = 70  # From calibration
    
    # Storage for visualization
    timestamps = []
    eeg_data = []
    erd_values = []
    ems_values = []
    
    # Simulate 30 seconds of data
    start_time = time.time()
    while time.time() - start_time < 30:
        # Generate synthetic EEG data
        current_power = eeg_gen.get_sample()
        
        # Compute ERD
        erd = compute_erd(current_power, rest_power)
        
        # Map to EMS intensity
        raw_ems = map_to_ems(erd, max_erd)
        
        # Apply smoothing and safety limits
        smoothed_ems = smoother.update(raw_ems)
        safe_ems = ramp_limiter.apply(smoothed_ems)
        final_ems = np.clip(safe_ems, 2, 8)  # Absolute safety limits
        
        # Store data
        timestamps.append(time.time() - start_time)
        eeg_data.append(current_power)
        erd_values.append(erd)
        ems_values.append(final_ems)
        
        time.sleep(0.1)  # Simulate real-time processing (10Hz)

    # =============================================
    # Visualization
    # =============================================
    plt.figure(figsize=(12, 8))
    
    # Plot EEG Power
    plt.subplot(3, 1, 1)
    plt.plot(timestamps, eeg_data, label='EEG Power')
    plt.ylabel('Mu/Beta Power')
    plt.title('Synthetic EEG Signal')
    plt.grid(True)
    
    # Plot ERD%
    plt.subplot(3, 1, 2)
    plt.plot(timestamps, erd_values, 'orange', label='ERD%')
    plt.ylabel('ERD (%)')
    plt.ylim(0, 100)
    plt.grid(True)
    
    # Plot EMS Intensity
    plt.subplot(3, 1, 3)
    plt.plot(timestamps, ems_values, 'red', label='EMS Intensity')
    plt.ylabel('EMS (1-10)')
    plt.xlabel('Time (s)')
    plt.ylim(0, 10)
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()