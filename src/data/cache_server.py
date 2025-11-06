"""In-memory patient cache server for HMS graphs.

Starts a multiprocessing Manager that holds a sanitized per-patient cache in
RAM and serves it to training processes via a proxy. This allows subsequent
runs to reuse the already-loaded graphs without paying disk I/O or deserialize
costs again. Stop the server manually when not needed.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Any, Iterable

import torch
import pandas as pd
from multiprocessing.managers import BaseManager

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable: Iterable, **kwargs):  # type: ignore
        return iterable


def _sanitize_patient_data(patient_data: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    for _lbl, item in patient_data.items():
        # EEG graphs
        for g in item.get('eeg_graphs', []):
            if hasattr(g, 'x') and g.x is not None:
                g.x = torch.nan_to_num(g.x, nan=0.0, posinf=0.0, neginf=0.0)
            if hasattr(g, 'edge_attr') and g.edge_attr is not None:
                g.edge_attr = torch.nan_to_num(g.edge_attr, nan=0.0, posinf=1.0, neginf=0.0)
        # Spec graphs
        for g in item.get('spec_graphs', []):
            if hasattr(g, 'x') and g.x is not None:
                x = torch.nan_to_num(g.x, nan=0.0, posinf=0.0, neginf=0.0)
                if x.dim() == 2 and x.size(1) == 5:
                    x = x[:, :4]
                g.x = x
    return patient_data


class PatientCache(dict):
    """Simple dict subclass so it can be proxied via BaseManager."""


class CacheManager(BaseManager):
    pass


def build_patient_list(train_csv: Path, n_folds: int, current_fold: int) -> Iterable[int]:
    df = pd.read_csv(train_csv)
    # Ensure fold assignment exists or fallback to all patients
    if 'fold' not in df.columns:
        return sorted(df['patient_id'].unique())
    train_df = df[df['fold'] != current_fold]
    val_df = df[df['fold'] == current_fold]
    # Load both train and val patients to cover the fold
    pids = set(train_df['patient_id'].unique()) | set(val_df['patient_id'].unique())
    return sorted(int(x) for x in pids)


def preload_cache(data_dir: Path, patient_ids: Iterable[int]) -> PatientCache:
    cache: PatientCache = PatientCache()
    pids = list(patient_ids)
    for pid in tqdm(pids, desc="CacheServer preloading", unit="patient"):
        path = data_dir / f"patient_{pid}.pt"
        if path.exists():
            data = torch.load(path, weights_only=False)
            cache[pid] = _sanitize_patient_data(data)
    return cache


def start_server(host: str, port: int, authkey: bytes, cache: PatientCache) -> None:
    CacheManager.register('get_cache', callable=lambda: cache)
    mgr = CacheManager(address=(host, port), authkey=authkey)
    srv = mgr.get_server()
    print(f"[CacheServer] Serving cache with {len(cache)} patients at {host}:{port}")
    srv.serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser(description="Start HMS Patient Cache Server")
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--train-csv", type=str, required=True)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--current-fold", type=int, default=0)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=50000)
    ap.add_argument("--authkey", type=str, default="hms-cache")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    train_csv = Path(args.train_csv)

    print("[CacheServer] Building patient list...")
    pids = build_patient_list(train_csv, args.n_folds, args.current_fold)
    print(f"[CacheServer] Preloading {len(pids)} patients from {data_dir}...")
    cache = preload_cache(data_dir, pids)
    print("[CacheServer] Preload complete.")

    start_server(args.host, args.port, args.authkey.encode("utf-8"), cache)


if __name__ == "__main__":
    main()
