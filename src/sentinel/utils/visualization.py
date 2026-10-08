"""
Visualization utilities for signals and events
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sp_signal
from typing import Tuple

def plot_thermal_scenario(
    time: np.ndarray,
    thermal_signature: np.ndarray,
    events: list,
    save_path: str = None
):
    """
    Plot thermal scenario with event markers
    
    Args:
        time: Time vector
        thermal_signature: Thermal signature data
        events: List of ThermalEvent objects
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(15, 6))
    
    ax.plot(time, thermal_signature, linewidth=0.8)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Temperature (K)', fontsize=12)
    ax.set_title('OPIR Thermal Scenario', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Mark events
    for event in events:
        ax.axvline(event.timestamp, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
        
        # Add label
        y_pos = ax.get_ylim()[1] * 0.85
        ax.text(
            event.timestamp, y_pos, 
            event.event_type.replace('_', ' ').title(),
            rotation=90, 
            verticalalignment='bottom',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved figure to {save_path}")
    
    return fig, ax


def plot_spectrogram(
    signal_data: np.ndarray,
    sample_rate: float,
    title: str = "RF Signal Spectrogram",
    save_path: str = None
):
    """
    Plot spectrogram of RF signal
    
    Args:
        signal_data: Complex or real signal
        sample_rate: Sample rate in Hz
        title: Plot title
        save_path: Optional save path
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Compute spectrogram
    f, t, Sxx = sp_signal.spectrogram(
        signal_data,
        fs=sample_rate,
        nperseg=1024,
        noverlap=512,
        scaling='spectrum'
    )
    
    # Convert to dB
    Sxx_db = 10 * np.log10(Sxx + 1e-12)
    
    # Plot
    im = ax.pcolormesh(t, f/1e6, Sxx_db, shading='gouraud', cmap='viridis')
    ax.set_ylabel('Frequency (MHz)', fontsize=12)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Power (dB)', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved spectrogram to {save_path}")
    
    return fig, ax


def plot_rf_pulse_train(
    time: np.ndarray,
    signal_data: np.ndarray,
    title: str = "Radar Pulse Train",
    save_path: str = None
):
    """
    Plot RF pulse train (time domain)
    
    Args:
        time: Time vector
        signal_data: Signal amplitude
        title: Plot title
        save_path: Optional save path
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    
    # Full signal
    axes[0].plot(time, np.abs(signal_data), linewidth=0.5)
    axes[0].set_xlabel('Time (s)', fontsize=12)
    axes[0].set_ylabel('Amplitude', fontsize=12)
    axes[0].set_title(f'{title} - Full View', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Zoomed view of first few pulses
    zoom_samples = min(5000, len(time))
    axes[1].plot(time[:zoom_samples], np.abs(signal_data[:zoom_samples]), linewidth=1)
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylabel('Amplitude', fontsize=12)
    axes[1].set_title(f'{title} - Zoomed View', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved pulse train to {save_path}")
    
    return fig, axes