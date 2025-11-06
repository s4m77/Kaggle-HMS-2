"""Utilities to build and persist processed HMS graphs (patient_*.pt).

Builds per-patient files containing mappings:
    label_id -> { 'eeg_graphs': List[Data], 'spec_graphs': List[Data], 'target': int }

Intended usage:
- Called once before training to create missing patient files
- Subsequent trainings load from `data/processed` via HMSDataset/HMSDataModule
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
import torch

from src.data.utils.eeg_process import EEGGraphBuilder, select_eeg_channels
from src.data.utils.spectrogram_process import SpectrogramGraphBuilder


def _label_to_index(label: str) -> int:
    mapping = {
        'Seizure': 0,
        'LPD': 1,
        'GPD': 2,
        'LRDA': 3,
        'GRDA': 4,
        'Other': 5,
    }
    return mapping.get(str(label), 5)


def build_patient_file(
    patient_id: int,
    df_patient: pd.DataFrame,
    raw_data_dir: Path,
    eeg_builder: EEGGraphBuilder,
    spec_builder: SpectrogramGraphBuilder,
    save_dir: Path,
    overwrite: bool = False,
) -> Path:
    """Build and save one patient's graphs to a torch .pt file.

    The saved dict maps label_id -> sample dict with eeg/spec graphs and target class index.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"patient_{int(patient_id)}.pt"
    if out_path.exists() and not overwrite:
        return out_path

    result: Dict[int, Dict] = {}

    for _, row in df_patient.iterrows():
        label_id = int(row['label_id']) if 'label_id' in row else int(row.get('label_id', 0))
        eeg_id = int(row['eeg_id'])
        spec_id = int(row['spectrogram_id'])

        # Load raw EEG
        eeg_path = raw_data_dir / 'train_eegs' / f'{eeg_id}.parquet'
        eeg_df = pd.read_parquet(eeg_path)
        eeg_np = select_eeg_channels(eeg_df, eeg_builder.channels)

        # Build EEG graphs
        eeg_graphs = eeg_builder.process_eeg_signal(eeg_np)

        # Load raw spectrogram
        spec_path = raw_data_dir / 'train_spectrograms' / f'{spec_id}.parquet'
        spec_df = pd.read_parquet(spec_path)

        # Build spectrogram graphs
        spec_graphs = spec_builder.process_spectrogram(spec_df)

        # Target as class index (Lightning converts to one-hot for KLDiv)
        target = _label_to_index(row.get('expert_consensus', 'Other'))

        result[label_id] = {
            'eeg_graphs': eeg_graphs,
            'spec_graphs': spec_graphs,
            'target': target,
        }

    # Save
    torch.save(result, out_path)
    return out_path


def build_missing_processed(
    metadata: pd.DataFrame,
    raw_data_dir: Path,
    processed_dir: Path,
    eeg_builder: EEGGraphBuilder,
    spec_builder: SpectrogramGraphBuilder,
    patient_ids: Optional[Iterable[int]] = None,
    overwrite: bool = False,
) -> None:
    """Build any missing patient_*.pt files for the provided patient_ids.

    If patient_ids is None, builds for all unique patients in metadata.
    """
    raw_data_dir = Path(raw_data_dir)
    processed_dir = Path(processed_dir)

    # Determine patients to build
    if patient_ids is None:
        pids = sorted(set(int(pid) for pid in metadata['patient_id'].unique()))
    else:
        pids = sorted(set(int(pid) for pid in patient_ids))

    # Group metadata by patient
    by_patient = metadata.groupby('patient_id')

    try:
        from tqdm import tqdm  # type: ignore
        iterator = tqdm(pids, desc='Building processed graphs (patients)')
    except Exception:
        iterator = pids

    for pid in iterator:
        save_path = processed_dir / f"patient_{pid}.pt"
        if save_path.exists() and not overwrite:
            continue
        if pid not in by_patient.groups:
            continue
        df_patient = by_patient.get_group(pid)
        build_patient_file(
            pid,
            df_patient,
            raw_data_dir=raw_data_dir,
            eeg_builder=eeg_builder,
            spec_builder=spec_builder,
            save_dir=processed_dir,
            overwrite=overwrite,
        )
