"""
Main script to preprocess raw EEG and Spectrogram data into PyTorch Geometric graphs.
Processes all data and saves graphs grouped by patient_id.
"""

import os
import sys

from pathlib import Path
from multiprocessing import Pool, cpu_count

import pandas as pd
import torch

from omegaconf import OmegaConf
from tqdm import tqdm

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.utils.eeg_process import EEGGraphBuilder, select_eeg_channels
from src.data.utils.spectrogram_process import SpectrogramGraphBuilder, filter_spectrogram_columns


def load_config(config_path: str = "configs/graphs.yaml"):
    config = OmegaConf.load(config_path)
    return config


def process_single_label(
    row,
    eeg_builder: EEGGraphBuilder,
    spec_builder: SpectrogramGraphBuilder,
    eeg_dir: str,
    spec_dir: str,
    eeg_channels: list,
    spec_regions: list,
    label_to_index: dict
):
    """
    Process a single labeled sample (one row from train.csv).
    
    Args:
        row: DataFrame row with metadata
        eeg_builder: EEGGraphBuilder instance
        spec_builder: SpectrogramGraphBuilder instance
        eeg_dir: Path to EEG parquet files
        spec_dir: Path to spectrogram parquet files
        eeg_channels: List of EEG channel names
        spec_regions: List of spectrogram regions
        label_to_index: Dictionary mapping label strings to indices
    
    Returns:
        Dictionary with processed data or None if failed
    """
    try:
        # === Load and extract EEG window ===
        eeg_path = os.path.join(eeg_dir, f"{row.eeg_id}.parquet")
        eeg_df = pd.read_parquet(eeg_path)
        
        # Extract 50-second window based on offset
        eeg_offset = int(row.eeg_label_offset_seconds)
        eeg_start_idx = eeg_offset * 200  # 200 Hz sampling rate
        eeg_end_idx = eeg_start_idx + (50 * 200)  # 50 seconds
        eeg_window_df = eeg_df.iloc[eeg_start_idx:eeg_end_idx]
        
        # Select channels and convert to numpy
        eeg_array = select_eeg_channels(eeg_window_df, eeg_channels)
        
        # Validate shape
        if eeg_array.shape[0] != 10000:
            print(f"Warning: EEG shape {eeg_array.shape} for label_id {row.label_id}")
            return None
        
        # Build EEG graphs (9 temporal windows)
        eeg_graphs = eeg_builder.process_eeg_signal(eeg_array)
        
        # === Load and extract Spectrogram window ===
        spec_path = os.path.join(spec_dir, f"{row.spectrogram_id}.parquet")
        spec_df = pd.read_parquet(spec_path)
        
        # Extract 600-second window based on offset
        spec_offset = int(row.spectrogram_label_offset_seconds)
        spec_window_df = spec_df[
            (spec_df['time'] >= spec_offset) & 
            (spec_df['time'] < spec_offset + 600)
        ]
        
        # Filter to keep only relevant regions
        spec_window_df = filter_spectrogram_columns(spec_window_df, spec_regions)
        
        # Validate
        if len(spec_window_df) == 0:
            print(f"Warning: Empty spectrogram for label_id {row.label_id}")
            return None
        
        # Build Spectrogram graphs (119 temporal windows)
        spec_graphs = spec_builder.process_spectrogram(spec_window_df)
        
        # === Get target label ===
        label_str = row.expert_consensus.strip()
        target = label_to_index.get(label_str, -1)
        
        if target == -1:
            print(f"Warning: Unknown label '{label_str}' for label_id {row.label_id}")
            return None
        
        # === Return processed data ===
        return {
            'eeg_graphs': eeg_graphs,
            'spec_graphs': spec_graphs,
            'target': target,
            'label_id': row.label_id,
            'patient_id': row.patient_id
        }
        
    except Exception as e:
        print(f"Error processing label_id {row.label_id}: {str(e)}")
        return None


def get_processed_patients(output_dir: Path) -> set:
    """
    Get set of already processed patient IDs.
    
    Args:
        output_dir: Directory containing processed patient files
    
    Returns:
        Set of patient IDs that have been processed
    """
    if not output_dir.exists():
        return set()
    
    processed_patients = set()
    for file_path in output_dir.glob("patient_*.pt"):
        # Extract patient_id from filename: patient_12345.pt -> 12345
        try:
            patient_id = int(file_path.stem.split('_')[1])
            processed_patients.add(patient_id)
        except (IndexError, ValueError):
            continue
    
    return processed_patients


def update_metadata(output_dir: Path, config):
    """
    Update metadata file with current state of processed data.
    
    Args:
        output_dir: Directory containing processed patient files
        config: OmegaConf configuration object
    """
    # Get all processed patients
    all_processed_patients = get_processed_patients(output_dir)
    
    if len(all_processed_patients) == 0:
        return
    
    # Count total samples across all patients
    total_samples = 0
    all_samples_per_patient = {}
    
    for patient_id in all_processed_patients:
        patient_path = output_dir / f"patient_{patient_id}.pt"
        if patient_path.exists():
            try:
                # Use weights_only=False for PyTorch Geometric Data objects
                patient_data = torch.load(patient_path, weights_only=False)
                n_samples = len(patient_data)
                all_samples_per_patient[patient_id] = n_samples
                total_samples += n_samples
            except Exception as e:
                print(f"Warning: Could not load patient {patient_id}: {e}")
                continue
    
    # Create metadata
    metadata = {
        'n_patients': len(all_processed_patients),
        'n_samples': total_samples,
        'patient_ids': sorted(list(all_processed_patients)),
        'samples_per_patient': all_samples_per_patient,
        'config': OmegaConf.to_container(config, resolve=True)
    }
    
    # Save metadata
    metadata_path = output_dir / "metadata.pt"
    torch.save(metadata, metadata_path)


def process_single_label_wrapper(args):
    """
    Wrapper for multiprocessing. Unpacks arguments and calls process_single_label.
    
    Args:
        args: Tuple of (row_dict, config_dict)
    
    Returns:
        Dictionary with processed data or None if failed
    """
    row_dict, config_dict = args
    
    # Convert dictionaries back to proper types
    row = pd.Series(row_dict)
    
    # Rebuild the builders (each process needs its own)
    eeg_builder = EEGGraphBuilder(
        sampling_rate=config_dict['eeg']['sampling_rate'],
        window_size=config_dict['eeg']['window_size'],
        stride=config_dict['eeg']['stride'],
        bands=config_dict['eeg']['bands'],
        coherence_threshold=config_dict['eeg']['coherence']['threshold'],
        nperseg_factor=config_dict['eeg']['coherence']['nperseg_factor'],
        channels=config_dict['eeg']['channels'],
        apply_bandpass=config_dict['eeg']['preprocessing']['bandpass_filter']['enabled'],
        bandpass_low=config_dict['eeg']['preprocessing']['bandpass_filter']['lowcut'],
        bandpass_high=config_dict['eeg']['preprocessing']['bandpass_filter']['highcut'],
        bandpass_order=config_dict['eeg']['preprocessing']['bandpass_filter']['order'],
        apply_notch=config_dict['eeg']['preprocessing']['notch_filter']['enabled'],
        notch_freq=config_dict['eeg']['preprocessing']['notch_filter']['frequency'],
        notch_q=config_dict['eeg']['preprocessing']['notch_filter']['quality_factor'],
        apply_normalize=config_dict['eeg']['preprocessing']['normalize']['enabled']
    )
    
    spec_builder = SpectrogramGraphBuilder(
        window_size=config_dict['spectrogram']['window_size'],
        stride=config_dict['spectrogram']['stride'],
        regions=config_dict['spectrogram']['regions'],
        bands=config_dict['spectrogram']['bands'],
        aggregation=config_dict['spectrogram']['aggregation'],
        spatial_edges=config_dict['spectrogram']['spatial_edges']
    )
    
    return process_single_label(
        row=row,
        eeg_builder=eeg_builder,
        spec_builder=spec_builder,
        eeg_dir=config_dict['paths']['train_eegs'],
        spec_dir=config_dict['paths']['train_spectrograms'],
        eeg_channels=config_dict['eeg']['channels'],
        spec_regions=config_dict['spectrogram']['regions'],
        label_to_index=config_dict['label_to_index']
    )


def process_all_data(config, n_workers=None):
    """
    Process all training data and save graphs grouped by patient_id.
    Supports resuming from interruption and multiprocessing.
    
    Args:
        config: OmegaConf configuration object
        n_workers: Number of worker processes. If None, uses CPU count - 1
    """
    print("=" * 80)
    print("HMS EEG/Spectrogram Graph Preprocessing Pipeline")
    print("=" * 80)
    
    # Set number of workers
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    print(f"\nUsing {n_workers} worker processes")
    
    # === Check for existing processed data ===
    output_dir = Path(config.paths.processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    existing_patients = get_processed_patients(output_dir)
    
    if existing_patients:
        print(f"\n[!] Found {len(existing_patients)} already processed patients")
        print(f"    Will skip these and resume from where we left off")
        
        # Find the last processed patient (to recompute in case of incomplete processing)
        last_patient = max(existing_patients)
        print(f"    Last processed patient: {last_patient}")
        print(f"    → Will recompute patient {last_patient} (in case of incomplete processing)")
        
        # Remove last patient from skip list so it gets reprocessed
        existing_patients.remove(last_patient)
        
        # Delete the last patient file
        last_patient_file = output_dir / f"patient_{last_patient}.pt"
        if last_patient_file.exists():
            last_patient_file.unlink()
            print(f"    → Deleted {last_patient_file.name} for reprocessing")
    
    # === Initialize builders ===
    print("\n[1/5] Initializing graph builders...")
    
    eeg_builder = EEGGraphBuilder(
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
    
    spec_builder = SpectrogramGraphBuilder(
        window_size=config.spectrogram.window_size,
        stride=config.spectrogram.stride,
        regions=list(config.spectrogram.regions),
        bands=dict(config.spectrogram.bands),
        aggregation=config.spectrogram.aggregation,
        spatial_edges=config.spectrogram.spatial_edges
    )
    
    print(f"  ✓ EEG: {eeg_builder.n_windows} windows per sample")
    print(f"  ✓ Spectrogram: {spec_builder.n_windows} windows per sample")
    
    # === Load metadata ===
    print("\n[2/5] Loading metadata...")
    train_df = pd.read_csv(config.paths.train_csv)
    # Strip whitespace from column names
    train_df.columns = train_df.columns.str.strip()
    print(f"  ✓ Loaded {len(train_df)} samples")
    print(f"  ✓ Unique patients: {train_df['patient_id'].nunique()}")
    print(f"  ✓ Unique EEG files: {train_df['eeg_id'].nunique()}")
    print(f"  ✓ Unique Spectrogram files: {train_df['spectrogram_id'].nunique()}")
    
    # Class distribution
    print("\n  Class distribution:")
    for label, count in train_df['expert_consensus'].value_counts().items():
        label_clean = label.strip()
        idx = config.label_to_index.get(label_clean, -1)
        print(f"    {label_clean:10s} (idx={idx}): {count:6d} samples")
    
    # === Process all samples ===
    print("\n[3/5] Processing samples...")
    
    # Get unique patients to process
    all_patients = set(train_df['patient_id'].unique())
    patients_to_process = all_patients - existing_patients
    
    print(f"  Total unique patients: {len(all_patients)}")
    print(f"  Already processed: {len(existing_patients)}")
    print(f"  To process: {len(patients_to_process)}")
    
    # Convert config to dict for multiprocessing
    config_dict = OmegaConf.to_container(config, resolve=True)
    
    # Process patients one at a time (to maintain grouping and incremental saves)
    failed_count = 0
    success_count = 0
    skipped_count = 0
    saved_patients_count = 0
    
    # Group train_df by patient_id
    grouped_by_patient = train_df.groupby('patient_id')
    
    print(f"\n  Processing patients with {n_workers} workers...")
    print(f"  Metadata will be updated every 100 patients")
    
    for patient_id in tqdm(sorted(all_patients), desc="Patients"):
        # Skip if this patient is already processed
        if patient_id in existing_patients:
            patient_samples = grouped_by_patient.get_group(patient_id)
            skipped_count += len(patient_samples)
            continue
        
        # Get all samples for this patient
        patient_samples = grouped_by_patient.get_group(patient_id)
        
        # Prepare arguments for multiprocessing
        args_list = [
            (row.to_dict(), config_dict) 
            for _, row in patient_samples.iterrows()
        ]
        
        # Process samples for this patient in parallel
        if n_workers > 1:
            with Pool(processes=n_workers) as pool:
                results = pool.map(process_single_label_wrapper, args_list)
        else:
            # Single-threaded for debugging
            results = [process_single_label_wrapper(args) for args in args_list]
        
        # Collect results into patient data
        patient_data = {}
        for result in results:
            if result is not None:
                label_id = result['label_id']
                patient_data[label_id] = {
                    'eeg_graphs': result['eeg_graphs'],
                    'spec_graphs': result['spec_graphs'],
                    'target': result['target']
                }
                success_count += 1
            else:
                failed_count += 1
        
        # Save patient file
        if patient_data:
            output_path = output_dir / f"patient_{patient_id}.pt"
            torch.save(patient_data, output_path)
            saved_patients_count += 1
            
            # Update metadata every 100 patients (instead of every patient)
            if saved_patients_count % 100 == 0:
                update_metadata(output_dir, config)
                print(f"\n  [Checkpoint] Saved {saved_patients_count} patients, updated metadata")
    
    print(f"\n  ✓ Successfully processed: {success_count} samples")
    print(f"  ⊘ Skipped (already done): {skipped_count} samples")
    print(f"  ✗ Failed: {failed_count} samples")
    print(f"  ✓ Saved {saved_patients_count} new patient files to {output_dir}")
    
    # === Final metadata update ===
    print("\n[4/5] Final metadata update...")
    update_metadata(output_dir, config)
    
    metadata_path = output_dir / "metadata.pt"
    metadata = torch.load(metadata_path, weights_only=False)
    
    print(f"  ✓ Updated metadata to {metadata_path}")
    print(f"  ✓ Total patients: {metadata['n_patients']}")
    print(f"  ✓ Total samples: {metadata['n_samples']}")
    
    print("\n" + "=" * 80)
    print("Preprocessing complete!")
    print("=" * 80)
    print(f"\nTo load data for a specific patient:")
    print(f"  data = torch.load('{output_dir}/patient_<patient_id>.pt')")
    print(f"\nTo load metadata:")
    print(f"  metadata = torch.load('{metadata_path}')")
    print()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess HMS EEG/Spectrogram data into graphs')
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=None,
        help=f'Number of worker processes (default: CPU count - 1 = {max(1, cpu_count() - 1)})'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/graphs.yaml',
        help='Path to configuration file (default: configs/graphs.yaml)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Process all data
    process_all_data(config, n_workers=args.workers)


if __name__ == "__main__":
    main()
