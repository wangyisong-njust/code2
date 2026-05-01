from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from faultdg.config import resolve_path


REQUIRED_SPLITS = ("source_train", "source_val", "target_train", "target_test")


@dataclass
class DatasetBundle:
    source_train: "SignalWindowDataset"
    source_val: "SignalWindowDataset"
    target_train: "SignalWindowDataset"
    target_test: "SignalWindowDataset"
    label_to_index: Dict[str, int]
    index_to_label: Dict[int, str]
    domain_to_index: Dict[str, int]
    index_to_domain: Dict[int, str]
    input_dim: int
    stats_dim: int
    split_counts: Dict[str, int]


class SignalWindowDataset(Dataset):
    def __init__(
        self,
        signals: np.ndarray,
        stats: np.ndarray,
        labels: np.ndarray,
        domains: np.ndarray,
        sample_ids: list[str],
    ) -> None:
        self.signals = torch.as_tensor(signals, dtype=torch.float32)
        self.stats = torch.as_tensor(stats, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.domains = torch.as_tensor(domains, dtype=torch.long)
        self.sample_ids = sample_ids

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int):
        return (
            self.signals[index],
            self.stats[index],
            self.labels[index],
            self.domains[index],
            self.sample_ids[index],
        )


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    drop_last: bool,
) -> DataLoader:
    effective_drop_last = drop_last and len(dataset) >= batch_size
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=effective_drop_last,
    )


def add_awgn_numpy(signal: np.ndarray, snr_db: float) -> np.ndarray:
    signal = signal.astype(np.float32, copy=False)
    signal_power = float(np.mean(signal ** 2) + 1e-8)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = np.random.normal(0.0, math.sqrt(noise_power), size=signal.shape).astype(np.float32)
    return signal + noise


def add_awgn_torch(signal: torch.Tensor, snr_db: float) -> torch.Tensor:
    signal_power = torch.mean(signal ** 2, dim=1, keepdim=True) + 1e-8
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = torch.randn_like(signal) * torch.sqrt(noise_power)
    return signal + noise


def compute_stats_numpy(signal: np.ndarray) -> np.ndarray:
    eps = 1e-8
    signal = signal.astype(np.float32, copy=False)
    mean = float(np.mean(signal))
    centered = signal - mean
    std = float(np.std(signal) + eps)
    rms = float(np.sqrt(np.mean(signal ** 2) + eps))
    abs_mean = float(np.mean(np.abs(signal)) + eps)
    peak = float(np.max(np.abs(signal)) + eps)
    peak_to_peak = float(np.max(signal) - np.min(signal))
    skewness = float(np.mean(centered ** 3) / (std ** 3 + eps))
    kurt = float(np.mean(centered ** 4) / (std ** 4 + eps))
    crest = peak / (rms + eps)
    impulse = peak / (abs_mean + eps)
    shape = rms / (abs_mean + eps)

    magnitude = np.abs(np.fft.rfft(signal))
    freqs = np.linspace(0.0, 1.0, num=magnitude.shape[0], dtype=np.float32)
    mag_sum = float(np.sum(magnitude) + eps)
    centroid = float(np.sum(magnitude * freqs) / mag_sum)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * magnitude) / mag_sum))
    energy = magnitude ** 2
    total_energy = float(np.sum(energy) + eps)
    split_1 = max(1, energy.shape[0] // 3)
    split_2 = max(split_1 + 1, 2 * energy.shape[0] // 3)
    low_ratio = float(np.sum(energy[:split_1]) / total_energy)
    mid_ratio = float(np.sum(energy[split_1:split_2]) / total_energy)
    high_ratio = float(np.sum(energy[split_2:]) / total_energy)

    return np.asarray(
        [
            mean,
            std,
            rms,
            peak_to_peak,
            skewness,
            kurt,
            crest,
            impulse,
            shape,
            centroid,
            bandwidth,
            low_ratio,
            mid_ratio,
            high_ratio,
        ],
        dtype=np.float32,
    )


def compute_stats_torch(signals: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    mean = torch.mean(signals, dim=1)
    centered = signals - mean.unsqueeze(1)
    std = torch.std(signals, dim=1, unbiased=False) + eps
    rms = torch.sqrt(torch.mean(signals ** 2, dim=1) + eps)
    abs_mean = torch.mean(torch.abs(signals), dim=1) + eps
    peak = torch.amax(torch.abs(signals), dim=1) + eps
    peak_to_peak = torch.amax(signals, dim=1) - torch.amin(signals, dim=1)
    skewness = torch.mean(centered ** 3, dim=1) / (std ** 3 + eps)
    kurt = torch.mean(centered ** 4, dim=1) / (std ** 4 + eps)
    crest = peak / (rms + eps)
    impulse = peak / (abs_mean + eps)
    shape = rms / (abs_mean + eps)

    magnitude = torch.abs(torch.fft.rfft(signals, dim=1))
    freqs = torch.linspace(0.0, 1.0, steps=magnitude.shape[1], device=signals.device, dtype=signals.dtype)
    mag_sum = magnitude.sum(dim=1) + eps
    centroid = (magnitude * freqs.unsqueeze(0)).sum(dim=1) / mag_sum
    bandwidth = torch.sqrt(
        ((freqs.unsqueeze(0) - centroid.unsqueeze(1)) ** 2 * magnitude).sum(dim=1) / mag_sum
    )
    energy = magnitude ** 2
    total_energy = energy.sum(dim=1) + eps
    split_1 = max(1, energy.shape[1] // 3)
    split_2 = max(split_1 + 1, 2 * energy.shape[1] // 3)
    low_ratio = energy[:, :split_1].sum(dim=1) / total_energy
    mid_ratio = energy[:, split_1:split_2].sum(dim=1) / total_energy
    high_ratio = energy[:, split_2:].sum(dim=1) / total_energy

    return torch.stack(
        [
            mean,
            std,
            rms,
            peak_to_peak,
            skewness,
            kurt,
            crest,
            impulse,
            shape,
            centroid,
            bandwidth,
            low_ratio,
            mid_ratio,
            high_ratio,
        ],
        dim=1,
    )


def _normalize_signal(signal: np.ndarray, mode: str) -> np.ndarray:
    signal = signal.astype(np.float32, copy=False)
    if mode == "none":
        return signal
    if mode == "zscore":
        mean = float(np.mean(signal))
        std = float(np.std(signal))
        return (signal - mean) / max(std, 1e-8)
    if mode == "minmax":
        min_value = float(np.min(signal))
        max_value = float(np.max(signal))
        return (signal - min_value) / max(max_value - min_value, 1e-8)
    raise ValueError(f"Unsupported normalize mode: {mode}")


def _load_signal_file(path: Path, signal_key: Optional[str]) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        data = np.load(path)
    elif suffix in {".csv", ".txt"}:
        data = np.loadtxt(path, delimiter="," if suffix == ".csv" else None)
    elif suffix == ".mat":
        mat = loadmat(path)
        if signal_key:
            if signal_key not in mat:
                raise KeyError(f"signal_key='{signal_key}' not found in {path}")
            data = mat[signal_key]
        else:
            candidates: list[np.ndarray] = []
            for key, value in mat.items():
                if key.startswith("__"):
                    continue
                if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
                    flattened = np.asarray(value).squeeze()
                    if flattened.ndim == 1 and flattened.size >= 64:
                        candidates.append(flattened)
            if not candidates:
                raise ValueError(f"No 1D numeric array found in MAT file: {path}")
            data = max(candidates, key=lambda arr: arr.size)
    else:
        raise ValueError(f"Unsupported signal file format: {path}")

    signal = np.asarray(data, dtype=np.float32).squeeze()
    if signal.ndim != 1:
        signal = signal.reshape(-1)
    return signal.astype(np.float32, copy=False)


def _window_signal(signal: np.ndarray, window_size: int, stride: int, max_windows: Optional[int]) -> list[np.ndarray]:
    if signal.size < window_size:
        padded = np.zeros(window_size, dtype=np.float32)
        padded[: signal.size] = signal
        return [padded]

    windows = [signal[start : start + window_size] for start in range(0, signal.size - window_size + 1, stride)]
    if not windows:
        windows = [signal[:window_size]]
    if max_windows is not None and len(windows) > max_windows:
        indices = np.linspace(0, len(windows) - 1, num=max_windows, dtype=int)
        windows = [windows[index] for index in indices]
    return [np.asarray(window, dtype=np.float32) for window in windows]


def _validate_manifest(manifest: pd.DataFrame, config: dict) -> None:
    data_cfg = config["data"]
    required = {data_cfg["path_col"], data_cfg["label_col"], data_cfg["domain_col"]}
    missing = [column for column in required if column not in manifest.columns]
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")


def _split_groups(
    rows: pd.DataFrame,
    group_col: str,
    label_col: str,
    test_size: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    groups = rows[[group_col, label_col]].drop_duplicates(group_col)
    group_values = groups[group_col].astype(str).tolist()
    stratify = groups[label_col].tolist() if len(groups[label_col].unique()) > 1 else None

    if len(group_values) < 2:
        return set(group_values), set()

    try:
        train_groups, test_groups = train_test_split(
            group_values,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        train_groups, test_groups = train_test_split(group_values, test_size=test_size, random_state=seed)
    return set(train_groups), set(test_groups)


def _assign_splits(manifest: pd.DataFrame, config: dict) -> pd.DataFrame:
    data_cfg = config["data"]
    split_col = data_cfg.get("split_col", "split")
    if split_col in manifest.columns and manifest[split_col].fillna("").astype(str).str.len().gt(0).any():
        normalized = manifest.copy()
        normalized[split_col] = normalized[split_col].astype(str).str.strip().str.lower()
        return normalized

    path_col = data_cfg["path_col"]
    label_col = data_cfg["label_col"]
    domain_col = data_cfg["domain_col"]
    group_col = data_cfg.get("group_col", path_col)
    seed = int(config["training"]["seed"])
    source_domains = set(map(str, data_cfg["source_domains"]))
    target_domains = set(map(str, data_cfg["target_domains"]))
    source_val_ratio = float(data_cfg.get("source_val_ratio", 0.2))
    target_test_ratio = float(data_cfg.get("target_test_ratio", 0.5))

    auto = manifest.copy()
    if group_col not in auto.columns:
        auto[group_col] = auto[path_col].astype(str)
    auto[group_col] = auto[group_col].astype(str)
    auto[split_col] = "unused"

    source_rows = auto[auto[domain_col].astype(str).isin(source_domains)].copy()
    target_rows = auto[auto[domain_col].astype(str).isin(target_domains)].copy()

    source_train_groups, source_val_groups = _split_groups(source_rows, group_col, label_col, source_val_ratio, seed)
    target_train_groups, target_test_groups = _split_groups(target_rows, group_col, label_col, target_test_ratio, seed + 1)

    auto.loc[source_rows[group_col].isin(source_train_groups), split_col] = "source_train"
    auto.loc[source_rows[group_col].isin(source_val_groups), split_col] = "source_val"
    auto.loc[target_rows[group_col].isin(target_train_groups), split_col] = "target_train"
    auto.loc[target_rows[group_col].isin(target_test_groups), split_col] = "target_test"
    return auto


def _rows_to_dataset(
    rows: pd.DataFrame,
    split_name: str,
    manifest_root: Path,
    config: dict,
    label_to_index: Dict[str, int],
    domain_to_index: Dict[str, int],
) -> SignalWindowDataset:
    data_cfg = config["data"]
    project_root = Path(config["_meta"]["project_root"])
    path_col = data_cfg["path_col"]
    label_col = data_cfg["label_col"]
    domain_col = data_cfg["domain_col"]
    sample_id_col = data_cfg.get("sample_id_col", "sample_id")
    signal_key_col = data_cfg.get("signal_key_col", "signal_key")
    window_size = int(data_cfg["window_size"])
    stride = int(data_cfg["stride"])
    max_windows = data_cfg.get("max_windows_per_file")
    normalize = data_cfg.get("normalize", "zscore")

    signals: list[np.ndarray] = []
    stats: list[np.ndarray] = []
    labels: list[int] = []
    domains: list[int] = []
    sample_ids: list[str] = []

    for row in tqdm(rows.itertuples(index=False), total=len(rows), desc=f"Build {split_name}", leave=False):
        path_value = getattr(row, path_col)
        file_path = Path(path_value)
        if not file_path.is_absolute():
            manifest_candidate = (manifest_root / file_path).resolve()
            project_candidate = (project_root / file_path).resolve()
            if project_candidate.exists():
                file_path = project_candidate
            else:
                file_path = manifest_candidate
        signal_key = getattr(row, signal_key_col, None) if signal_key_col in rows.columns else None
        signal_key = None if pd.isna(signal_key) else str(signal_key)
        raw_signal = _load_signal_file(file_path, signal_key)
        windows = _window_signal(raw_signal, window_size, stride, max_windows)

        raw_sample_id = getattr(row, sample_id_col, file_path.stem) if sample_id_col in rows.columns else file_path.stem
        sample_prefix = str(raw_sample_id)
        label_index = label_to_index[str(getattr(row, label_col))]
        domain_index = domain_to_index[str(getattr(row, domain_col))]

        for window_index, window in enumerate(windows):
            normalized_window = _normalize_signal(window, normalize)
            signals.append(normalized_window)
            stats.append(compute_stats_numpy(normalized_window))
            labels.append(label_index)
            domains.append(domain_index)
            sample_ids.append(f"{sample_prefix}__w{window_index:04d}")

    if signals:
        signal_array = np.stack(signals).astype(np.float32)
        stats_array = np.stack(stats).astype(np.float32)
        label_array = np.asarray(labels, dtype=np.int64)
        domain_array = np.asarray(domains, dtype=np.int64)
    else:
        signal_array = np.zeros((0, window_size), dtype=np.float32)
        stats_array = np.zeros((0, 14), dtype=np.float32)
        label_array = np.zeros((0,), dtype=np.int64)
        domain_array = np.zeros((0,), dtype=np.int64)

    return SignalWindowDataset(signal_array, stats_array, label_array, domain_array, sample_ids)


def build_datasets(config: dict) -> DatasetBundle:
    data_cfg = config["data"]
    manifest_path = resolve_path(config, data_cfg["manifest"])
    manifest_root = manifest_path.parent
    manifest = pd.read_csv(manifest_path)
    _validate_manifest(manifest, config)
    manifest = _assign_splits(manifest, config)

    split_col = data_cfg.get("split_col", "split")
    label_col = data_cfg["label_col"]
    domain_col = data_cfg["domain_col"]
    source_domains = list(map(str, data_cfg["source_domains"]))
    target_domains = list(map(str, data_cfg["target_domains"]))
    relevant_domains = set(source_domains + target_domains)
    manifest = manifest[manifest[domain_col].astype(str).isin(relevant_domains)].copy()

    label_names = sorted(map(str, manifest[label_col].astype(str).unique().tolist()))
    domain_names = source_domains + [domain for domain in target_domains if domain not in source_domains]
    label_to_index = {name: index for index, name in enumerate(label_names)}
    domain_to_index = {name: index for index, name in enumerate(domain_names)}
    index_to_label = {index: name for name, index in label_to_index.items()}
    index_to_domain = {index: name for name, index in domain_to_index.items()}

    split_frames: Dict[str, pd.DataFrame] = {}
    split_counts: Dict[str, int] = {}
    for split_name in REQUIRED_SPLITS:
        split_rows = manifest[manifest[split_col].astype(str).str.lower() == split_name].copy()
        split_frames[split_name] = split_rows
        split_counts[split_name] = int(len(split_rows))

    datasets = {
        split_name: _rows_to_dataset(
            rows=split_rows,
            split_name=split_name,
            manifest_root=manifest_root,
            config=config,
            label_to_index=label_to_index,
            domain_to_index=domain_to_index,
        )
        for split_name, split_rows in split_frames.items()
    }

    stats_dim = int(datasets["source_train"].stats.shape[1])
    input_dim = int(data_cfg["window_size"])
    return DatasetBundle(
        source_train=datasets["source_train"],
        source_val=datasets["source_val"],
        target_train=datasets["target_train"],
        target_test=datasets["target_test"],
        label_to_index=label_to_index,
        index_to_label=index_to_label,
        domain_to_index=domain_to_index,
        index_to_domain=index_to_domain,
        input_dim=input_dim,
        stats_dim=stats_dim,
        split_counts=split_counts,
    )
