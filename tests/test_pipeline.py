"""
Pytest tests for the preprocessing pipeline.
Tests EEG and spectrogram processing on sample data.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from omegaconf import OmegaConf
from src.data.utils.eeg_process import EEGGraphBuilder, select_eeg_channels
from src.data.utils.spectrogram_process import SpectrogramGraphBuilder


@pytest.fixture
def config():
    """Load configuration fixture."""
    return OmegaConf.load("configs/graphs.yaml")


@pytest.fixture
def eeg_builder(config):
    """Create EEG graph builder fixture."""
    return EEGGraphBuilder(
        sampling_rate=config.eeg.sampling_rate,
        window_size=config.eeg.window_size,
        stride=config.eeg.stride,
        bands=dict(config.eeg.bands),
        coherence_threshold=config.eeg.coherence.threshold,
        nperseg_factor=config.eeg.coherence.nperseg_factor,
        channels=list(config.eeg.channels),
        # Preprocessing parameters
        apply_bandpass=config.eeg.preprocessing.bandpass_filter.enabled,
        bandpass_low=config.eeg.preprocessing.bandpass_filter.lowcut,
        bandpass_high=config.eeg.preprocessing.bandpass_filter.highcut,
        bandpass_order=config.eeg.preprocessing.bandpass_filter.order,
        apply_notch=config.eeg.preprocessing.notch_filter.enabled,
        notch_freq=config.eeg.preprocessing.notch_filter.frequency,
        notch_q=config.eeg.preprocessing.notch_filter.quality_factor,
        apply_normalize=config.eeg.preprocessing.normalize.enabled
    )


@pytest.fixture
def spec_builder(config):
    """Create spectrogram graph builder fixture."""
    return SpectrogramGraphBuilder(
        window_size=config.spectrogram.window_size,
        stride=config.spectrogram.stride,
        regions=list(config.spectrogram.regions),
        bands=dict(config.spectrogram.bands),
        aggregation=config.spectrogram.aggregation,
        spatial_edges=config.spectrogram.spatial_edges
    )


@pytest.fixture
def train_df(config):
    """Load training metadata fixture."""
    df = pd.read_csv(config.paths.train_csv)
    df.columns = df.columns.str.strip()
    return df


def test_config_loading(config):
    """Test that configuration loads correctly."""
    assert config.eeg.sampling_rate == 200
    assert config.eeg.window_size == 10
    assert config.eeg.stride == 5
    assert len(config.eeg.channels) == 19
    assert len(config.spectrogram.regions) == 4


def test_eeg_builder_initialization(eeg_builder):
    """Test EEG builder initialization."""
    assert eeg_builder.sampling_rate == 200
    assert eeg_builder.window_size == 10
    assert eeg_builder.stride == 5
    assert eeg_builder.n_windows == 9  # Expected number of windows


def test_spec_builder_initialization(spec_builder):
    """Test spectrogram builder initialization."""
    assert spec_builder.window_size == 10
    assert spec_builder.stride == 5
    assert spec_builder.n_windows == 119  # Expected number of windows


def test_metadata_loading(train_df):
    """Test that metadata loads correctly."""
    assert len(train_df) > 0
    required_columns = ['label_id', 'patient_id', 'eeg_id', 'spectrogram_id', 
                       'expert_consensus', 'eeg_label_offset_seconds', 
                       'spectrogram_label_offset_seconds']
    for col in required_columns:
        assert col in train_df.columns, f"Missing column: {col}"


def test_eeg_processing(config, eeg_builder, train_df):
    """Test EEG processing on first sample."""
    row = train_df.iloc[0]
    
    # Load EEG file
    eeg_path = f"{config.paths.train_eegs}/{row.eeg_id}.parquet"
    eeg_df = pd.read_parquet(eeg_path)
    
    # Extract window
    eeg_offset = int(row.eeg_label_offset_seconds)
    eeg_start = eeg_offset * 200
    eeg_end = eeg_start + (50 * 200)
    eeg_window_df = eeg_df.iloc[eeg_start:eeg_end]
    
    # Select channels
    eeg_array = select_eeg_channels(eeg_window_df, list(config.eeg.channels))
    
    # Assertions
    assert eeg_array.shape == (10000, 19), f"Expected (10000, 19), got {eeg_array.shape}"
    
    # Build graphs
    eeg_graphs = eeg_builder.process_eeg_signal(eeg_array)
    
    assert len(eeg_graphs) == 9, f"Expected 9 graphs, got {len(eeg_graphs)}"
    
    # Check first graph structure
    graph = eeg_graphs[0]
    assert graph.x.shape[0] == 19, f"Expected 19 nodes, got {graph.x.shape[0]}"
    assert graph.x.shape[1] == 5, f"Expected 5 features per node, got {graph.x.shape[1]}"
    assert graph.edge_index.shape[0] == 2, f"Expected edge_index shape[0]=2, got {graph.edge_index.shape[0]}"


def test_spectrogram_processing(config, spec_builder, train_df):
    """Test spectrogram processing on first sample."""
    row = train_df.iloc[0]
    
    # Load spectrogram file
    spec_path = f"{config.paths.train_spectrograms}/{row.spectrogram_id}.parquet"
    spec_df = pd.read_parquet(spec_path)
    
    # Extract window
    spec_offset = int(row.spectrogram_label_offset_seconds)
    spec_window_df = spec_df[
        (spec_df['time'] >= spec_offset) & 
        (spec_df['time'] < spec_offset + 600)
    ]
    
    assert len(spec_window_df) > 0, "Spectrogram window is empty"
    
    # Build graphs
    spec_graphs = spec_builder.process_spectrogram(spec_window_df)
    
    assert len(spec_graphs) == 119, f"Expected 119 graphs, got {len(spec_graphs)}"
    
    # Check first graph structure
    graph = spec_graphs[0]
    assert graph.x.shape[0] == 4, f"Expected 4 nodes, got {graph.x.shape[0]}"
    assert graph.x.shape[1] == 5, f"Expected 5 features per node, got {graph.x.shape[1]}"
    assert graph.edge_index.shape == (2, 8), f"Expected edge_index shape (2, 8), got {graph.edge_index.shape}"


def test_label_mapping(config, train_df):
    """Test that all labels in dataset are in label mapping."""
    unique_labels = train_df['expert_consensus'].str.strip().unique()
    label_to_index = dict(config.label_to_index)
    
    for label in unique_labels:
        assert label in label_to_index, f"Label '{label}' not in label_to_index mapping"


def test_eeg_band_power_computation(eeg_builder, train_df, config):
    """Test that band power computation returns valid values."""
    row = train_df.iloc[0]
    
    # Load and extract EEG
    eeg_path = f"{config.paths.train_eegs}/{row.eeg_id}.parquet"
    eeg_df = pd.read_parquet(eeg_path)
    eeg_offset = int(row.eeg_label_offset_seconds)
    eeg_window = eeg_df.iloc[eeg_offset*200:(eeg_offset+50)*200]
    eeg_array = select_eeg_channels(eeg_window, list(config.eeg.channels))
    
    # Get first 10s window
    first_window = eeg_array[:2000, :]
    
    # Compute band power
    band_power = eeg_builder.compute_band_power(first_window)
    
    assert band_power.shape == (19, 5), f"Expected (19, 5), got {band_power.shape}"
    assert not (band_power < 0).any(), "Band power should be non-negative"
    assert not pd.isna(band_power).any(), "Band power contains NaN values"


def test_eeg_coherence_computation(eeg_builder, train_df, config):
    """Test that coherence computation returns valid edges."""
    row = train_df.iloc[0]
    
    # Load and extract EEG
    eeg_path = f"{config.paths.train_eegs}/{row.eeg_id}.parquet"
    eeg_df = pd.read_parquet(eeg_path)
    eeg_offset = int(row.eeg_label_offset_seconds)
    eeg_window = eeg_df.iloc[eeg_offset*200:(eeg_offset+50)*200]
    eeg_array = select_eeg_channels(eeg_window, list(config.eeg.channels))
    
    # Get first 10s window
    first_window = eeg_array[:2000, :]
    
    # Compute coherence
    edge_index, edge_attr = eeg_builder.compute_coherence_matrix(first_window)
    
    assert edge_index.shape[0] == 2, "Edge index should have 2 rows"
    assert edge_attr.shape[1] == 1, "Edge attributes should have 1 feature"
    assert edge_attr.shape[0] == edge_index.shape[1], "Edge attr and index mismatch"
    
    # Check that all coherence values are in valid range
    if edge_attr.shape[0] > 0:
        assert (edge_attr >= 0).all(), "Coherence should be non-negative"
        assert (edge_attr <= 1).all(), "Coherence should be <= 1"


def test_eeg_preprocessing(eeg_builder):
    """Test that EEG preprocessing functions work correctly."""
    from src.data.utils.eeg_process import (
        apply_bandpass_filter,
        apply_notch_filter,
        apply_zscore_normalization
    )
    
    # Create synthetic signal
    fs = 200
    duration = 2  # seconds
    n_samples = fs * duration
    n_channels = 5
    
    # Generate signal with known frequencies
    t = np.linspace(0, duration, n_samples, endpoint=False)
    signal = np.zeros((n_samples, n_channels))
    
    for ch in range(n_channels):
        # Mix of frequencies: 2Hz (delta), 10Hz (alpha), 60Hz (powerline)
        signal[:, ch] = (
            np.sin(2 * np.pi * 2 * t) +
            np.sin(2 * np.pi * 10 * t) +
            0.5 * np.sin(2 * np.pi * 60 * t)
        )
    
    # Test bandpass filter
    filtered = apply_bandpass_filter(signal, lowcut=0.5, highcut=50, fs=fs, order=4)
    assert filtered.shape == signal.shape, "Bandpass should preserve shape"
    assert not np.array_equal(filtered, signal), "Bandpass should modify signal"
    
    # Test notch filter
    notched = apply_notch_filter(signal, notch_freq=60, fs=fs, quality_factor=30)
    assert notched.shape == signal.shape, "Notch should preserve shape"
    assert not np.array_equal(notched, signal), "Notch should modify signal"
    
    # Test normalization
    normalized = apply_zscore_normalization(signal)
    assert normalized.shape == signal.shape, "Normalization should preserve shape"
    
    # Check that each channel is normalized
    for ch in range(n_channels):
        mean = np.mean(normalized[:, ch])
        std = np.std(normalized[:, ch])
        assert abs(mean) < 1e-10, f"Mean should be ~0, got {mean}"
        assert abs(std - 1.0) < 1e-10, f"Std should be ~1, got {std}"
    
    # Test full preprocessing pipeline
    preprocessed = eeg_builder.preprocess_signal(signal)
    assert preprocessed.shape == signal.shape, "Preprocessing should preserve shape"
    assert not np.isnan(preprocessed).any(), "Preprocessing should not create NaN"
    assert not np.isinf(preprocessed).any(), "Preprocessing should not create Inf"

