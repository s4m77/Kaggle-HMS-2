"""
Spectrogram preprocessing utilities for graph construction.
Aggregates frequency bins into bands and builds spatial graphs.
"""

import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Tuple
from torch_geometric.data import Data


class SpectrogramGraphBuilder:
    """Build PyTorch Geometric graphs from spectrograms."""
    
    def __init__(
        self,
        window_size: int = 10,
        stride: int = 5,
        regions: List[str] = None,
        bands: Dict[str, List[float]] = None,
        aggregation: str = 'mean',
        spatial_edges: List[List[int]] = None,
        apply_preprocessing: bool = True,
        clip_min: float = 1e-7,
        clip_max: float = 1e-4
    ):
        """
        Args:
            window_size: Window size in seconds
            stride: Stride in seconds for sliding window
            regions: List of region names (e.g., ['LL', 'RL', 'LP', 'RP'])
            bands: Dictionary of frequency bands {name: [low, high]}
            aggregation: Method to aggregate frequency bins ('mean' or 'max')
            spatial_edges: List of edge pairs for spatial connectivity
            apply_preprocessing: Whether to apply clip + log + normalize
            clip_min: Minimum value for clipping (default: 1e-7)
            clip_max: Maximum value for clipping (default: 1e-4)
        """
        self.window_size = window_size
        self.stride = stride
        self.regions = regions or ['LL', 'RL', 'LP', 'RP']
        self.bands = bands or {
            'delta': [0.5, 4.0],
            'theta': [4.0, 8.0],
            'alpha': [8.0, 13.0],
            'beta': [13.0, 20.0],
            'gamma': [20.0, 30.0]
        }
        self.aggregation = aggregation
        self.spatial_edges = spatial_edges or [
            [0, 2], [2, 0],  # LL <-> LP
            [1, 3], [3, 1],  # RL <-> RP
            [0, 1], [1, 0],  # LL <-> RL
            [2, 3], [3, 2],  # LP <-> RP
        ]
        self.apply_preprocessing = apply_preprocessing
        self.clip_min = clip_min
        self.clip_max = clip_max
        
        # Compute expected number of windows
        self.n_windows = int((600 - window_size) / stride) + 1  # Should be 119
        assert self.n_windows == 119
    
    def preprocess_spectrogram(self, spec_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing to raw spectrogram values:
        1. Clip values between clip_min and clip_max
        2. Take logarithm
        3. Normalize to mean=0, std=1
        
        Args:
            spec_df: DataFrame with spectrogram values
        
        Returns:
            Preprocessed DataFrame with same structure
        """
        if not self.apply_preprocessing:
            return spec_df
        
        # Create a copy to avoid modifying original
        processed_df = spec_df.copy()
        
        # Get all numeric columns (exclude 'time' column)
        numeric_cols = [col for col in processed_df.columns if col != 'time']
        
        if len(numeric_cols) == 0:
            return processed_df
        
        # 1. Clip values
        processed_df[numeric_cols] = processed_df[numeric_cols].clip(
            lower=self.clip_min,
            upper=self.clip_max
        )
        
        # 2. Take logarithm
        processed_df[numeric_cols] = np.log(processed_df[numeric_cols])
        
        # 3. Normalize to mean=0, std=1 (across all values in the spectrogram)
        all_values = processed_df[numeric_cols].values.flatten()
        mean = np.mean(all_values)
        std = np.std(all_values)
        
        if std > 1e-10:  # Avoid division by zero
            processed_df[numeric_cols] = (processed_df[numeric_cols] - mean) / std
        else:
            processed_df[numeric_cols] = processed_df[numeric_cols] - mean
        
        return processed_df
    
    def extract_temporal_windows(self, spec_df: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Split spectrogram into overlapping temporal windows.
        
        Args:
            spec_df: DataFrame with 'time' column and spectral features
        
        Returns:
            List of DataFrames, each covering window_size seconds
        """
        windows = []
        
        # Get min time from the dataframe
        min_time = spec_df['time'].min()
        
        for i in range(self.n_windows):
            # Calculate window boundaries
            window_start = min_time + (i * self.stride)
            window_end = window_start + self.window_size
            
            # Extract rows in this window
            window_df = spec_df[
                (spec_df['time'] >= window_start) & 
                (spec_df['time'] < window_end)
            ]
            
            if len(window_df) > 0:
                windows.append(window_df)
        
        return windows
    
    def aggregate_frequency_bands(self, spec_window: pd.DataFrame) -> np.ndarray:
        """
        Aggregate frequency bins into bands for each region.
        
        Args:
            spec_window: DataFrame with columns like 'LL_0.59', 'LL_0.78', etc.
        
        Returns:
            (n_regions, n_bands) array of aggregated power values
        """
        n_regions = len(self.regions)
        n_bands = len(self.bands)
        features = np.zeros((n_regions, n_bands))
        
        for region_idx, region in enumerate(self.regions):
            # Get all columns for this region
            region_cols = [col for col in spec_window.columns 
                          if col.startswith(f"{region}_")]
            
            if not region_cols:
                continue
            
            # Extract frequency values from column names
            freq_map = {}
            for col in region_cols:
                try:
                    freq = float(col.split('_')[1])
                    freq_map[freq] = col
                except (IndexError, ValueError):
                    continue
            
            # Aggregate into bands
            band_names = list(self.bands.keys())
            for band_idx, (band_name, (low, high)) in enumerate(self.bands.items()):
                # For the last band (gamma), use inclusive upper bound to capture all high frequencies
                is_last_band = (band_idx == len(band_names) - 1)
                
                # Find columns in this frequency range
                if is_last_band:
                    band_cols = [freq_map[freq] for freq in freq_map.keys() 
                                if low <= freq <= high]  # Inclusive upper bound for last band
                else:
                    band_cols = [freq_map[freq] for freq in freq_map.keys() 
                                if low <= freq < high]  # Exclusive upper bound to prevent overlap
                
                if band_cols:
                    # Get values across all timepoints in window
                    band_values = spec_window[band_cols].values
                    
                    # Aggregate spatially (across frequency bins) and temporally
                    if self.aggregation == 'mean':
                        power = np.mean(band_values)
                    elif self.aggregation == 'max':
                        power = np.max(band_values)
                    else:
                        power = np.mean(band_values)
                    
                    features[region_idx, band_idx] = power
                else:
                    features[region_idx, band_idx] = 0.0
        
        return features
    
    def build_graph(self, spec_window: pd.DataFrame, time_idx: int = 0, is_center: bool = False) -> Data:
        """
        Build a PyTorch Geometric graph from a spectrogram window.
        
        Args:
            spec_window: DataFrame with spectrogram data for one window
            time_idx: Temporal position index (0 to n_windows-1)
            is_center: Whether this window contains the labeled region
        
        Returns:
            PyG Data object with node features, spatial edges, and temporal position
        """
        # Aggregate frequency bands (n_regions, n_bands)
        features = self.aggregate_frequency_bands(spec_window)
        x = torch.tensor(features, dtype=torch.float)
        
        # Create spatial edges (fixed connectivity)
        edge_index = torch.tensor(self.spatial_edges, dtype=torch.long).T  # (2, n_edges)
        
        # Create PyG Data object with positional encoding
        graph = Data(
            x=x, 
            edge_index=edge_index,
            time_idx=torch.tensor([time_idx], dtype=torch.long),  # Temporal position
            is_center=torch.tensor([is_center], dtype=torch.bool)  # Label indicator
        )
        
        return graph
    
    def process_spectrogram(self, spec_df: pd.DataFrame) -> List[Data]:
        """
        Process full spectrogram into a sequence of graphs.
        
        Args:
            spec_df: DataFrame with 'time' column and spectral features
        
        Returns:
            List of PyG Data objects (length = n_windows, expected 119)
        """
        # Apply preprocessing (clip, log, normalize)
        preprocessed_df = self.preprocess_spectrogram(spec_df)
        
        # Extract temporal windows
        windows = self.extract_temporal_windows(preprocessed_df)
        
        # Validate we got the expected number of windows
        if len(windows) != self.n_windows:
            print(f"Warning: Expected {self.n_windows} spectrogram windows, got {len(windows)}")
        
        # Determine center window index (for 119 windows, center is index 59)
        center_idx = self.n_windows // 2
        
        # Build graph for each window with positional encoding
        graphs = []
        for i, window in enumerate(windows):
            is_center = (i == center_idx)
            graph = self.build_graph(window, time_idx=i, is_center=is_center)
            graphs.append(graph)
        
        return graphs


def filter_spectrogram_columns(spec_df: pd.DataFrame, regions: List[str]) -> pd.DataFrame:
    """
    Filter spectrogram DataFrame to keep only time column and region columns.
    
    Args:
        spec_df: Full spectrogram DataFrame
        regions: List of regions to keep
    
    Returns:
        Filtered DataFrame
    """
    # Keep time column and all columns for specified regions
    cols_to_keep = ['time']
    for region in regions:
        region_cols = [col for col in spec_df.columns if col.startswith(f"{region}_")]
        cols_to_keep.extend(region_cols)
    
    return spec_df[cols_to_keep]
