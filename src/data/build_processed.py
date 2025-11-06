"""Utilities to build and persist processed HMS graphs (patient_*.pt).

Builds per-patient files containing mappings:
    label_id -> { 'eeg_graphs': List[Data], 'spec_graphs': List[Data], 'target': int }

Intended usage:
- Called once before training to create missing patient files
- Subsequent trainings load from `data/processed` via HMSDataset/HMSDataModule
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Any, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

import pandas as pd
import torch

from src.data.utils.eeg_process import EEGGraphBuilder, select_eeg_channels
from src.data.utils.spectrogram_process import SpectrogramGraphBuilder, filter_spectrogram_columns


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


def _make_builders_from_cfg(cfg: Dict[str, Any]) -> Tuple[EEGGraphBuilder, SpectrogramGraphBuilder]:
    """Construct EEG/Spectrogram builders from a plain dict config."""
    eeg_cfg = cfg['eeg']
    spec_cfg = cfg['spectrogram']
    eeg_b = EEGGraphBuilder(
        sampling_rate=eeg_cfg['sampling_rate'],
        window_size=eeg_cfg['window_size'],
        stride=eeg_cfg['stride'],
        bands=dict(eeg_cfg['bands']),
        aec_threshold=eeg_cfg['aec']['threshold'],
        nperseg_factor=eeg_cfg['psd']['nperseg_factor'],
        channels=list(eeg_cfg['channels']),
        apply_bandpass=eeg_cfg['preprocessing']['bandpass_filter']['enabled'],
        bandpass_low=eeg_cfg['preprocessing']['bandpass_filter']['lowcut'],
        bandpass_high=eeg_cfg['preprocessing']['bandpass_filter']['highcut'],
        bandpass_order=eeg_cfg['preprocessing']['bandpass_filter']['order'],
        apply_notch=eeg_cfg['preprocessing']['notch_filter']['enabled'],
        notch_freq=eeg_cfg['preprocessing']['notch_filter']['frequency'],
        notch_q=eeg_cfg['preprocessing']['notch_filter']['quality_factor'],
        apply_normalize=eeg_cfg['preprocessing']['normalize']['enabled'],
    )
    spec_b = SpectrogramGraphBuilder(
        window_size=spec_cfg['window_size'],
        stride=spec_cfg['stride'],
        regions=list(spec_cfg['regions']),
        bands=dict(spec_cfg['bands']),
        aggregation=spec_cfg['aggregation'],
        spatial_edges=list(spec_cfg.get('spatial_edges', [])) or None,
        apply_preprocessing=spec_cfg.get('preprocessing', {}).get('enabled', True),
        clip_min=spec_cfg.get('preprocessing', {}).get('clip_min', 1e-7),
        clip_max=spec_cfg.get('preprocessing', {}).get('clip_max', 1e-4),
    )
    return eeg_b, spec_b


def build_patient_file(
    patient_id: int,
    df_patient: pd.DataFrame,
    raw_data_dir: Path,
    eeg_builder: EEGGraphBuilder,
    spec_builder: SpectrogramGraphBuilder,
    save_dir: Path,
    overwrite: bool = False,
    label_to_index: Optional[Dict[str, int]] = None,
) -> Path:
    """Build and save one patient's graphs to a torch .pt file.

    Mirrors the logic in `make_graph_dataset.process_single_label`:
      - Extract a 50s EEG window based on `eeg_label_offset_seconds` then build 9 temporal EEG graphs.
      - Extract a 600s Spectrogram window based on `spectrogram_label_offset_seconds` then build 119 temporal Spectrogram graphs.
    This ensures consistency between online build path and legacy preprocessing.

    The saved dict maps label_id -> sample dict with eeg/spec graphs and target class index.
    Required columns in df_patient: label_id, eeg_id, spectrogram_id, expert_consensus,
        eeg_label_offset_seconds, spectrogram_label_offset_seconds.
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

        eeg_offset = int(row.get('eeg_label_offset_seconds', 0))
        spec_offset = int(row.get('spectrogram_label_offset_seconds', 0))

        # === Load & slice EEG (50s window) ===
        eeg_path = raw_data_dir / 'train_eegs' / f'{eeg_id}.parquet'
        eeg_df = pd.read_parquet(eeg_path)
        # Compute indices (sampling rate assumed = eeg_builder.sampling_rate, typically 200 Hz)
        sr = getattr(eeg_builder, 'sampling_rate', 200)
        eeg_start = eeg_offset * sr
        eeg_end = eeg_start + (50 * sr)
        eeg_window_df = eeg_df.iloc[eeg_start:eeg_end]
        eeg_np = select_eeg_channels(eeg_window_df, eeg_builder.channels)
        if eeg_np.shape[0] != 50 * sr:
            # Skip malformed sample (log & continue)
            print(f"Warning: EEG window shape {eeg_np.shape} for label_id {label_id} (expected {50*sr} samples)")
            continue
        eeg_graphs = eeg_builder.process_eeg_signal(eeg_np)

        # === Load & slice Spectrogram (600s window) ===
        spec_path = raw_data_dir / 'train_spectrograms' / f'{spec_id}.parquet'
        spec_df = pd.read_parquet(spec_path)
        # Filter time range [spec_offset, spec_offset + 600)
        if 'time' in spec_df.columns:
            spec_window_df = spec_df[(spec_df['time'] >= spec_offset) & (spec_df['time'] < spec_offset + 600)]
        else:
            # Fallback: assume entire file corresponds to needed window if no time column
            spec_window_df = spec_df.copy()
        # Restrict to configured regions similar to legacy pipeline
        try:
            spec_window_df = filter_spectrogram_columns(spec_window_df, spec_builder.regions)
        except Exception:
            pass
        if len(spec_window_df) == 0:
            print(f"Warning: Empty spectrogram window for label_id {label_id}")
            continue
        spec_graphs = spec_builder.process_spectrogram(spec_window_df)

        if label_to_index is not None:
            target = int(label_to_index.get(str(row.get('expert_consensus', 'Other')).strip(), 5))
        else:
            target = _label_to_index(row.get('expert_consensus', 'Other'))

        result[label_id] = {
            'eeg_graphs': eeg_graphs,
            'spec_graphs': spec_graphs,
            'target': target,
        }

    # Save
    # Atomic write: save to temp then rename
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(result, tmp_path)
    os.replace(tmp_path, out_path)
    return out_path


def _worker_build(args: Tuple[int, List[Dict[str, Any]], str, str, bool, Optional[Dict[str, Any]]]):
    """Top-level worker function for multiprocessing (must be picklable)."""
    pid, records, raw_dir_str, out_dir_str, ow, cfg = args
    raw_dir = Path(raw_dir_str)
    out_dir = Path(out_dir_str)
    # Reconstruct builders locally to avoid pickling heavy objects
    label_map = None
    if cfg is not None:
        eeg_b, spec_b = _make_builders_from_cfg(cfg)
        label_map = cfg.get('label_to_index') if isinstance(cfg, dict) else None
    else:
        eeg_b = EEGGraphBuilder()
        spec_b = SpectrogramGraphBuilder()
    # Reconstruct small DataFrame from lightweight records
    df_p = pd.DataFrame.from_records(records)
    return build_patient_file(
        pid,
        df_p,
        raw_data_dir=raw_dir,
        eeg_builder=eeg_b,
        spec_builder=spec_b,
        save_dir=out_dir,
        overwrite=ow,
        label_to_index=label_map,
    )

def build_missing_processed(
    metadata: pd.DataFrame,
    raw_data_dir: Path,
    processed_dir: Path,
    eeg_builder: Optional[EEGGraphBuilder] = None,
    spec_builder: Optional[SpectrogramGraphBuilder] = None,
    graph_config: Optional[Dict[str, Any]] = None,
    patient_ids: Optional[Iterable[int]] = None,
    overwrite: bool = False,
    num_workers: int = 0,
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

    # Helper to create builders from config (used in worker)
    def _make_builders_from_cfg(cfg: Dict[str, Any]) -> Tuple[EEGGraphBuilder, SpectrogramGraphBuilder]:
        eeg_cfg = cfg['eeg']
        spec_cfg = cfg['spectrogram']
        eeg_b = EEGGraphBuilder(
            sampling_rate=eeg_cfg['sampling_rate'],
            window_size=eeg_cfg['window_size'],
            stride=eeg_cfg['stride'],
            bands=dict(eeg_cfg['bands']),
            aec_threshold=eeg_cfg['aec']['threshold'],
            nperseg_factor=eeg_cfg['psd']['nperseg_factor'],
            channels=list(eeg_cfg['channels']),
            apply_bandpass=eeg_cfg['preprocessing']['bandpass_filter']['enabled'],
            bandpass_low=eeg_cfg['preprocessing']['bandpass_filter']['lowcut'],
            bandpass_high=eeg_cfg['preprocessing']['bandpass_filter']['highcut'],
            bandpass_order=eeg_cfg['preprocessing']['bandpass_filter']['order'],
            apply_notch=eeg_cfg['preprocessing']['notch_filter']['enabled'],
            notch_freq=eeg_cfg['preprocessing']['notch_filter']['frequency'],
            notch_q=eeg_cfg['preprocessing']['notch_filter']['quality_factor'],
            apply_normalize=eeg_cfg['preprocessing']['normalize']['enabled'],
        )
        spec_b = SpectrogramGraphBuilder(
            window_size=spec_cfg['window_size'],
            stride=spec_cfg['stride'],
            regions=list(spec_cfg['regions']),
            bands=dict(spec_cfg['bands']),
            aggregation=spec_cfg['aggregation'],
            spatial_edges=list(spec_cfg.get('spatial_edges', [])) or None,
            apply_preprocessing=spec_cfg.get('preprocessing', {}).get('enabled', True),
            clip_min=spec_cfg.get('preprocessing', {}).get('clip_min', 1e-7),
            clip_max=spec_cfg.get('preprocessing', {}).get('clip_max', 1e-4),
        )
        return eeg_b, spec_b


    # Prepare task list for patients needing build
    tasks: List[Tuple[int, List[Dict[str, Any]], str, str, bool, Optional[Dict[str, Any]]]] = []
    for pid in pids:
        save_path = processed_dir / f"patient_{pid}.pt"
        if save_path.exists() and not overwrite:
            continue
        if pid not in by_patient.groups:
            continue
        # Create small, picklable records list
        df_patient = by_patient.get_group(pid)[
            [
                'label_id', 'eeg_id', 'spectrogram_id', 'expert_consensus',
                'eeg_label_offset_seconds', 'spectrogram_label_offset_seconds'
            ]
        ].copy()
        records = df_patient.to_dict(orient='records')
        tasks.append((pid, records, str(raw_data_dir), str(processed_dir), overwrite, graph_config))

    if not tasks:
        return

    # Serial or parallel execution
    if num_workers and num_workers > 1:
        # Reduce thread oversubscription inside workers
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        os.environ.setdefault('MKL_NUM_THREADS', '1')
        os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
        try:
            from tqdm import tqdm  # type: ignore
            pbar = tqdm(total=len(tasks), desc='Building processed graphs (parallel)')
        except Exception:
            pbar = None
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(_worker_build, t) for t in tasks]
            for fut in as_completed(futures):
                _ = fut.result()
                if pbar is not None:
                    pbar.update(1)
        if pbar is not None:
            pbar.close()
    else:
        try:
            from tqdm import tqdm  # type: ignore
            iterator = tqdm(tasks, desc='Building processed graphs (serial)')
        except Exception:
            iterator = tasks
        for t in iterator:
            _worker_build(t)
