"""
Pytest tests for checkpoint/resume functionality of make_dataset.py
"""

import sys
from pathlib import Path
import torch
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.make_dataset import get_processed_patients


@pytest.fixture
def processed_dir():
    """Get processed data directory."""
    return Path("data/processed")


def test_processed_dir_creation(processed_dir):
    """Test that processed directory can be created."""
    # This test just ensures the path is valid
    assert processed_dir.name == "processed"


def test_get_processed_patients_empty(tmp_path):
    """Test get_processed_patients with empty directory."""
    patients = get_processed_patients(tmp_path)
    assert isinstance(patients, set)
    assert len(patients) == 0


def test_get_processed_patients_with_files(tmp_path):
    """Test get_processed_patients with patient files."""
    # Create mock patient files
    (tmp_path / "patient_12345.pt").touch()
    (tmp_path / "patient_67890.pt").touch()
    (tmp_path / "other_file.pt").touch()
    
    patients = get_processed_patients(tmp_path)
    
    assert isinstance(patients, set)
    assert len(patients) == 2
    assert 12345 in patients
    assert 67890 in patients


def test_get_processed_patients_invalid_names(tmp_path):
    """Test that invalid filenames are skipped."""
    (tmp_path / "patient_abc.pt").touch()
    (tmp_path / "patient_.pt").touch()
    (tmp_path / "not_a_patient.pt").touch()
    
    patients = get_processed_patients(tmp_path)
    
    assert len(patients) == 0


@pytest.mark.skipif(
    not Path("data/processed").exists() or 
    len(list(Path("data/processed").glob("patient_*.pt"))) == 0,
    reason="No processed data available"
)
def test_processed_data_exists(processed_dir):
    """Test that processed data exists (skip if not available)."""
    existing_patients = get_processed_patients(processed_dir)
    
    assert len(existing_patients) > 0, "No processed patients found"
    
    # Check metadata
    metadata_path = processed_dir / "metadata.pt"
    if metadata_path.exists():
        metadata = torch.load(metadata_path)
        
        assert 'n_patients' in metadata
        assert 'n_samples' in metadata
        assert 'patient_ids' in metadata
        assert metadata['n_patients'] == len(existing_patients)


@pytest.mark.skipif(
    not Path("data/processed").exists() or 
    len(list(Path("data/processed").glob("patient_*.pt"))) == 0,
    reason="No processed data available"
)
def test_patient_file_structure(processed_dir):
    """Test structure of patient files."""
    existing_patients = get_processed_patients(processed_dir)
    
    if not existing_patients:
        pytest.skip("No processed patients available")
    
    # Load first patient
    patient_id = list(existing_patients)[0]
    patient_file = processed_dir / f"patient_{patient_id}.pt"
    
    patient_data = torch.load(patient_file)
    
    # Check it's a dictionary
    assert isinstance(patient_data, dict), "Patient data should be a dictionary"
    
    # Check structure of first sample
    if len(patient_data) > 0:
        first_label_id = list(patient_data.keys())[0]
        sample = patient_data[first_label_id]
        
        assert 'eeg_graphs' in sample
        assert 'spec_graphs' in sample
        assert 'target' in sample
        
        assert isinstance(sample['eeg_graphs'], list)
        assert isinstance(sample['spec_graphs'], list)
        assert isinstance(sample['target'], int)
        
        # Check expected lengths
        assert len(sample['eeg_graphs']) == 9, "Expected 9 EEG graphs"
        assert len(sample['spec_graphs']) == 119, "Expected 119 spectrogram graphs"
        
        # Check target is valid class
        assert 0 <= sample['target'] <= 5, f"Target {sample['target']} out of range [0, 5]"


@pytest.mark.skipif(
    not Path("data/processed").exists() or 
    len(list(Path("data/processed").glob("patient_*.pt"))) == 0,
    reason="No processed data available"
)
def test_metadata_consistency(processed_dir):
    """Test that metadata is consistent with actual files."""
    metadata_path = processed_dir / "metadata.pt"
    
    if not metadata_path.exists():
        pytest.skip("No metadata file available")
    
    metadata = torch.load(metadata_path)
    existing_patients = get_processed_patients(processed_dir)
    
    # Check counts match
    assert metadata['n_patients'] == len(existing_patients), \
        "Metadata patient count doesn't match actual files"
    
    # Check patient IDs match
    metadata_patient_ids = set(metadata['patient_ids'])
    assert metadata_patient_ids == existing_patients, \
        "Metadata patient IDs don't match actual files"
    
    # Check samples_per_patient is consistent
    assert len(metadata['samples_per_patient']) == len(existing_patients), \
        "samples_per_patient length doesn't match patient count"

