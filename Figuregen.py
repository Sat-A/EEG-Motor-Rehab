import numpy as np
import matplotlib.pyplot as plt
import mne

def plot_eeg_signals(file_path, channels, duration=10, offset=100e-6):
    """
    Plot EEG signals with proper scaling and visibility
    
    Args:
        file_path: Path to EDF file
        channels: List of channel names (e.g., ['C3', 'C4'])
        duration: Time window to plot (seconds)
        offset: Vertical separation between channels (in volts)
    """
    # Load and prepare data
    raw = mne.io.read_raw_edf(file_path, preload=True)
    raw.pick(channels)
    raw.filter(1, 40)  # Basic bandpass filter to remove extreme artifacts
    
    # Get data array (in volts) and times
    data, times = raw[:, :int(duration * raw.info['sfreq'])]
    data *= 1e6  # Convert to microvolts for better scaling
    
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # Plot each channel with proper offset
    for i, (ch_data, ch_name) in enumerate(zip(data, channels)):
        offset_uV = i * offset * 1e6  # Convert offset to microvolts
        plt.plot(times, ch_data + offset_uV, label=ch_name, linewidth=1)
        
        # Add channel label
        plt.text(times[-1]+0.1, offset_uV, ch_name, 
                ha='left', va='center', fontweight='bold')
    
    # Format plot
    plt.title(f'EEG Signals ({duration} second window)', pad=20)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (μV)')
    plt.xlim([times[0], times[-1]])
    
    # Set yticks to show channel positions
    plt.yticks([i * offset * 1e6 for i in range(len(channels))], 
              [f'{ch}\n{offset*1e6:.0f}μV' for ch in channels])
    
    # Add grid and adjust layout
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Example usage with better default parameters
if __name__ == "__main__":
    FILE_PATH = "sub-01_task-motor-imagery_eeg.edf"
    CHANNELS = ["C3", "C4", "Cz", "FC3", "FC4", "CP3", "CP4"]
    
    plot_eeg_signals(
        file_path=FILE_PATH,
        channels=CHANNELS,
        duration=30,  # 100 second window
        offset=150e-6  # 150μV between channels
    )