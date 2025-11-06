"""
EEG preprocessing utilities for graph construction.
Extracts PSD (Power Spectral Density) features and AEC (Amplitude Envelope Correlation) edges.
Includes signal preprocessing: band-pass filter, notch filter, and normalization.
"""

import numpy as np
import torch
import warnings
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch, hilbert
from typing import List, Tuple, Dict, Optional
from torch_geometric.data import Data


def apply_bandpass_filter(
    eeg_signal: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: int,
    order: int = 4
) -> np.ndarray:
    """
    Apply Butterworth band-pass filter to EEG signal.
    
    Args:
        eeg_signal: (n_samples, n_channels) array
        lowcut: Low cutoff frequency in Hz
        highcut: High cutoff frequency in Hz
        fs: Sampling rate in Hz
        order: Filter order
    
    Returns:
        Filtered signal with same shape
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Design Butterworth filter
    b, a = butter(order, [low, high], btype='band')
    
    # Apply filter to each channel
    filtered_signal = np.zeros_like(eeg_signal)
    for ch in range(eeg_signal.shape[1]):
        filtered_signal[:, ch] = filtfilt(b, a, eeg_signal[:, ch])
    
    return filtered_signal


def apply_notch_filter(
    eeg_signal: np.ndarray,
    notch_freq: float,
    fs: int,
    quality_factor: float = 30.0
) -> np.ndarray:
    """
    Apply notch filter to remove powerline interference.
    
    Args:
        eeg_signal: (n_samples, n_channels) array
        notch_freq: Frequency to notch out (50 or 60 Hz)
        fs: Sampling rate in Hz
        quality_factor: Q factor (higher = narrower notch)
    
    Returns:
        Filtered signal with same shape
    """
    # Design notch filter
    b, a = iirnotch(notch_freq, quality_factor, fs)
    
    # Apply filter to each channel
    filtered_signal = np.zeros_like(eeg_signal)
    for ch in range(eeg_signal.shape[1]):
        filtered_signal[:, ch] = filtfilt(b, a, eeg_signal[:, ch])
    
    return filtered_signal


def apply_zscore_normalization(eeg_signal: np.ndarray) -> np.ndarray:
    """
    Apply z-score normalization per channel.
    
    Args:
        eeg_signal: (n_samples, n_channels) array
    
    Returns:
        Normalized signal with same shape
    """
    normalized_signal = np.zeros_like(eeg_signal)
    
    for ch in range(eeg_signal.shape[1]):
        channel_data = eeg_signal[:, ch]
        mean = np.mean(channel_data)
        std = np.std(channel_data)
        
        # Avoid division by zero
        if std > 1e-10:
            normalized_signal[:, ch] = (channel_data - mean) / std
        else:
            normalized_signal[:, ch] = channel_data - mean
    
    return normalized_signal


class EEGGraphBuilder:
    """Build PyTorch Geometric graphs from EEG signals."""
    
    def __init__(
        self,
        sampling_rate: int = 200,
        window_size: int = 10,
        stride: int = 5,
        bands: Optional[Dict[str, List[float]]] = None,
        aec_threshold: float = 0.5,
        nperseg_factor: int = 2,
        channels: Optional[List[str]] = None,
        # Preprocessing parameters
        apply_bandpass: bool = True,
        bandpass_low: float = 0.5,
        bandpass_high: float = 50.0,
        bandpass_order: int = 4,
        apply_notch: bool = True,
        notch_freq: float = 60.0,
        notch_q: float = 30.0,
        apply_normalize: bool = True
    ):
        """
        Args:
            sampling_rate: EEG sampling rate in Hz
            window_size: Window size in seconds
            stride: Stride in seconds for sliding window
            bands: Dictionary of frequency bands {name: [low, high]}
            aec_threshold: Minimum Amplitude Envelope Correlation for edge creation
            nperseg_factor: Factor for nperseg in Welch's method (for PSD)
            channels: List of channel names to use (excludes EKG)
            apply_bandpass: Whether to apply band-pass filter
            bandpass_low: Low cutoff for band-pass filter (Hz)
            bandpass_high: High cutoff for band-pass filter (Hz)
            bandpass_order: Butterworth filter order
            apply_notch: Whether to apply notch filter
            notch_freq: Notch frequency (50 or 60 Hz)
            notch_q: Quality factor for notch filter
            apply_normalize: Whether to apply z-score normalization
        """
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.stride = stride
        self.bands = bands or {
            'delta': [0.5, 4.0],
            'theta': [4.0, 8.0],
            'alpha': [8.0, 13.0],
            'beta': [13.0, 30.0],
            'gamma': [30.0, 50.0]
        }
        self.aec_threshold = aec_threshold
        self.nperseg = sampling_rate * nperseg_factor
        self.channels = channels
        
        # Preprocessing parameters
        self.apply_bandpass = apply_bandpass
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.bandpass_order = bandpass_order
        self.apply_notch = apply_notch
        self.notch_freq = notch_freq
        self.notch_q = notch_q
        self.apply_normalize = apply_normalize
        
        # Compute expected number of windows
        self.n_windows = int((50 - window_size) / stride) + 1  # Should be 9
    
    def preprocess_signal(self, eeg_signal: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing steps to raw EEG signal.
        
        Args:
            eeg_signal: (n_samples, n_channels) array
        
        Returns:
            Preprocessed signal with same shape
        """
        processed = eeg_signal.copy()
        
        # 1. Band-pass filter
        if self.apply_bandpass:
            processed = apply_bandpass_filter(
                processed,
                lowcut=self.bandpass_low,
                highcut=self.bandpass_high,
                fs=self.sampling_rate,
                order=self.bandpass_order
            )
        
        # 2. Notch filter
        if self.apply_notch:
            processed = apply_notch_filter(
                processed,
                notch_freq=self.notch_freq,
                fs=self.sampling_rate,
                quality_factor=self.notch_q
            )
        
        # 3. Z-score normalization
        if self.apply_normalize:
            processed = apply_zscore_normalization(processed)
        
        return processed
        
    def extract_temporal_windows(self, eeg_signal: np.ndarray) -> List[np.ndarray]:
        """
        Split EEG signal into overlapping temporal windows.
        
        Args:
            eeg_signal: (n_samples, n_channels) array, expected (10000, 19)
        
        Returns:
            List of windows, each (window_samples, n_channels)
        """
        window_samples = self.window_size * self.sampling_rate
        stride_samples = self.stride * self.sampling_rate
        
        windows = []
        for i in range(self.n_windows):
            start_idx = i * stride_samples
            end_idx = start_idx + window_samples
            window = eeg_signal[start_idx:end_idx, :]
            windows.append(window)
        
        return windows
    
    def compute_psd(self, eeg_window: np.ndarray) -> np.ndarray:
        """
        Compute Power Spectral Density for each channel across frequency bands.
        
        Args:
            eeg_window: (n_samples, n_channels) array
        
        Returns:
            (n_channels, n_bands) array of PSD values
        """
        n_channels = eeg_window.shape[1]
        n_bands = len(self.bands)
        psd_features = np.zeros((n_channels, n_bands))
        
        for ch_idx in range(n_channels):
            # Compute power spectral density using Welch's method
            freqs, psd = signal.welch(
                eeg_window[:, ch_idx],
                fs=self.sampling_rate,
                nperseg=min(self.nperseg, len(eeg_window[:, ch_idx]))
            )
            
            # Compute average PSD in each frequency band
            for band_idx, (band_name, (low, high)) in enumerate(self.bands.items()):
                # Find frequencies in this band
                idx_band = np.logical_and(freqs >= low, freqs <= high)
                
                if np.any(idx_band):
                    # Average PSD in this band (use mean instead of integral for normalization)
                    avg_psd = np.mean(psd[idx_band])
                    psd_features[ch_idx, band_idx] = avg_psd
                else:
                    psd_features[ch_idx, band_idx] = 0.0
        
        return psd_features
    
    def compute_aec_matrix(self, eeg_window: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Amplitude Envelope Correlation (AEC) between all channel pairs.
        
        AEC measures functional connectivity by correlating the amplitude envelopes
        of bandpass-filtered signals using the Hilbert transform.
        
        Args:
            eeg_window: (n_samples, n_channels) array
        
        Returns:
            edge_index: (2, n_edges) array
            edge_attr: (n_edges, 1) array of AEC values
        """
        n_channels = eeg_window.shape[1]
        edges = []
        weights = []
        
        # For AEC, we'll use the broadband signal (already preprocessed)
        # Extract amplitude envelopes using Hilbert transform
        envelopes = np.zeros_like(eeg_window)
        
        for ch_idx in range(n_channels):
            # Apply Hilbert transform to get analytic signal
            analytic_signal = hilbert(eeg_window[:, ch_idx])
            # Extract amplitude envelope (magnitude of analytic signal)
            envelopes[:, ch_idx] = np.abs(analytic_signal)
        
        # Compute correlation between amplitude envelopes for all channel pairs
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                # Skip if either channel has near-zero variance
                if np.std(envelopes[:, i]) < 1e-10 or np.std(envelopes[:, j]) < 1e-10:
                    continue
                
                # Compute Pearson correlation between envelopes
                # Suppress warnings for edge cases
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    
                    # Normalize envelopes (z-score)
                    env_i = (envelopes[:, i] - np.mean(envelopes[:, i])) / (np.std(envelopes[:, i]) + 1e-10)
                    env_j = (envelopes[:, j] - np.mean(envelopes[:, j])) / (np.std(envelopes[:, j]) + 1e-10)
                    
                    # Compute correlation
                    aec = np.corrcoef(env_i, env_j)[0, 1]
                
                # Check for NaN values
                if np.isnan(aec):
                    continue
                
                # Add edge if AEC exceeds threshold (take absolute value for correlation)
                aec_abs = np.abs(aec)
                if aec_abs > self.aec_threshold and not np.isnan(aec_abs):
                    # Add both directions (undirected graph)
                    edges.append([i, j])
                    edges.append([j, i])
                    weights.append(aec_abs)
                    weights.append(aec_abs)
        
        # Convert to arrays
        if edges:
            edge_index = np.array(edges, dtype=np.int64).T  # (2, n_edges)
            edge_attr = np.array(weights, dtype=np.float32).reshape(-1, 1)  # (n_edges, 1)
        else:
            # No edges above threshold - create empty arrays
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_attr = np.zeros((0, 1), dtype=np.float32)
        
        return edge_index, edge_attr
    
    def build_graph(self, eeg_window: np.ndarray, time_idx: int = 0, is_center: bool = False) -> Data:
        """
        Build a PyTorch Geometric graph from an EEG window.
        
        Args:
            eeg_window: (n_samples, n_channels) array
            time_idx: Temporal position index (0 to n_windows-1)
            is_center: Whether this window contains the labeled region
        
        Returns:
            PyG Data object with node features (PSD), edges (AEC), and temporal position
        """
        # Compute node features: Power Spectral Density
        psd_features = self.compute_psd(eeg_window)  # (n_channels, n_bands)
        x = torch.tensor(psd_features, dtype=torch.float)
        
        # Compute edges: Amplitude Envelope Correlation
        edge_index, edge_attr = self.compute_aec_matrix(eeg_window)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        # Create PyG Data object with positional encoding
        graph = Data(
            x=x, 
            edge_index=edge_index, 
            edge_attr=edge_attr,
            time_idx=torch.tensor([time_idx], dtype=torch.long),  # Temporal position
            is_center=torch.tensor([is_center], dtype=torch.bool)  # Label indicator
        )
        
        return graph
    
    def process_eeg_signal(self, eeg_signal: np.ndarray) -> List[Data]:
        """
        Process full EEG signal into a sequence of graphs.
        
        Args:
            eeg_signal: (n_samples, n_channels) array, expected (10000, 19)
        
        Returns:
            List of PyG Data objects (length = n_windows, expected 9)
        """
        # Apply preprocessing first
        preprocessed_signal = self.preprocess_signal(eeg_signal)
        
        # Extract temporal windows
        windows = self.extract_temporal_windows(preprocessed_signal)
        
        # Validate we got the expected number of windows
        if len(windows) != self.n_windows:
            print(f"Warning: Expected {self.n_windows} EEG windows, got {len(windows)}")
        
        # Determine center window index (for 9 windows, center is index 4)
        center_idx = self.n_windows // 2
        
        # Build graph for each window with positional encoding
        graphs = []
        for i, window in enumerate(windows):
            is_center = (i == center_idx)
            graph = self.build_graph(window, time_idx=i, is_center=is_center)
            graphs.append(graph)
        
        return graphs


def select_eeg_channels(eeg_df, channel_names: List[str]):
    """
    Select specific channels from EEG DataFrame and drop EKG.
    
    Args:
        eeg_df: DataFrame with EEG data
        channel_names: List of channel names to keep
    
    Returns:
        numpy array of shape (n_samples, n_channels)
    """
    # Select only the specified channels
    available_channels = [ch for ch in channel_names if ch in eeg_df.columns]
    
    if len(available_channels) != len(channel_names):
        missing = set(channel_names) - set(available_channels)
        print(f"Warning: Missing channels: {missing}")
    
    eeg_array = eeg_df[available_channels].values
    return eeg_array
