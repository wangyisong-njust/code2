from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from faultdg.data import compute_stats_torch


PU_DOMAIN_TO_SETTING = {
    "a": "N09_M07_F10",
    "b": "N15_M01_F10",
    "c": "N15_M07_F04",
    "d": "N15_M07_F10",
}

PU_SETTING_TO_DESC = {
    "N09_M07_F10": "900rpm / 0.7Nm / 1000N",
    "N15_M01_F10": "1500rpm / 0.1Nm / 1000N",
    "N15_M07_F04": "1500rpm / 0.7Nm / 400N",
    "N15_M07_F10": "1500rpm / 0.7Nm / 1000N",
}

PU_SETTING_METADATA = {
    "N09_M07_F10": {"rpm": 900, "torque_nm": 0.7, "radial_n": 1000},
    "N15_M01_F10": {"rpm": 1500, "torque_nm": 0.1, "radial_n": 1000},
    "N15_M07_F04": {"rpm": 1500, "torque_nm": 0.7, "radial_n": 400},
    "N15_M07_F10": {"rpm": 1500, "torque_nm": 0.7, "radial_n": 1000},
}

PU_LABEL_TO_NAME = {
    0: "healthy",
    1: "outer",
    2: "inner",
}


@dataclass
class PUTaskBundle:
    source_train: "ProcessedSignalDataset"
    source_val: "ProcessedSignalDataset"
    target_train: "ProcessedSignalDataset"
    target_val: "ProcessedSignalDataset"
    target_test: "ProcessedSignalDataset"
    label_names: list[str]
    domain_names: list[str]
    input_dim: int
    stats_dim: int
    source_domain_key: str
    target_domain_key: str
    source_setting: str
    target_setting: str
    source_metadata: dict[str, float | int]
    target_metadata: dict[str, float | int]


class ProcessedSignalDataset(Dataset):
    def __init__(
        self,
        signals: torch.Tensor,
        stats: torch.Tensor,
        labels: torch.Tensor,
        domain_index: int,
        sample_prefix: str,
    ) -> None:
        self.signals = signals.float()
        self.stats = stats.float()
        self.labels = labels.long()
        self.domains = torch.full((signals.shape[0],), domain_index, dtype=torch.long)
        self.sample_ids = [f"{sample_prefix}_{index:05d}" for index in range(signals.shape[0])]

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


def _normalize_signals(signals: torch.Tensor, mode: str) -> torch.Tensor:
    signals = signals.float()
    if mode == "none":
        return signals
    if mode == "zscore":
        mean = signals.mean(dim=1, keepdim=True)
        std = signals.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return (signals - mean) / std
    if mode == "minmax":
        min_values = signals.amin(dim=1, keepdim=True)
        max_values = signals.amax(dim=1, keepdim=True)
        return (signals - min_values) / (max_values - min_values).clamp_min(1e-6)
    raise ValueError(f"Unsupported normalize mode: {mode}")


def _build_stats(signals: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, signals.shape[0], batch_size):
            stop = min(start + batch_size, signals.shape[0])
            chunks.append(compute_stats_torch(signals[start:stop]))
    return torch.cat(chunks, dim=0)


def _load_pt_split(
    root: Path,
    split_name: str,
    domain_key: str,
    domain_index: int,
    normalize: str,
    max_samples: Optional[int] = None,
) -> ProcessedSignalDataset:
    path = root / f"{split_name}_{domain_key}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    signals = payload["samples"].float()
    labels = payload["labels"].long()

    if max_samples is not None and signals.shape[0] > max_samples:
        indices = torch.linspace(0, signals.shape[0] - 1, steps=max_samples).long()
        signals = signals[indices]
        labels = labels[indices]

    signals = _normalize_signals(signals, normalize)
    stats = _build_stats(signals)
    return ProcessedSignalDataset(
        signals=signals,
        stats=stats,
        labels=labels,
        domain_index=domain_index,
        sample_prefix=f"{split_name}_{domain_key}",
    )


def build_pu_task_bundle(
    root: str | Path,
    source_domain_key: str,
    target_domain_key: str,
    normalize: str = "zscore",
    max_samples_per_split: Optional[int] = None,
) -> PUTaskBundle:
    if source_domain_key not in PU_DOMAIN_TO_SETTING:
        raise KeyError(f"Unsupported source domain: {source_domain_key}")
    if target_domain_key not in PU_DOMAIN_TO_SETTING:
        raise KeyError(f"Unsupported target domain: {target_domain_key}")

    root_path = Path(root).resolve()
    domain_names = [PU_DOMAIN_TO_SETTING[key] for key in sorted(PU_DOMAIN_TO_SETTING)]
    label_names = [PU_LABEL_TO_NAME[index] for index in sorted(PU_LABEL_TO_NAME)]

    source_index = sorted(PU_DOMAIN_TO_SETTING).index(source_domain_key)
    target_index = sorted(PU_DOMAIN_TO_SETTING).index(target_domain_key)

    source_train = _load_pt_split(root_path, "train", source_domain_key, source_index, normalize, max_samples_per_split)
    source_val = _load_pt_split(root_path, "val", source_domain_key, source_index, normalize, max_samples_per_split)
    target_train = _load_pt_split(root_path, "train", target_domain_key, target_index, normalize, max_samples_per_split)
    target_val = _load_pt_split(root_path, "val", target_domain_key, target_index, normalize, max_samples_per_split)
    target_test = _load_pt_split(root_path, "test", target_domain_key, target_index, normalize, max_samples_per_split)

    return PUTaskBundle(
        source_train=source_train,
        source_val=source_val,
        target_train=target_train,
        target_val=target_val,
        target_test=target_test,
        label_names=label_names,
        domain_names=domain_names,
        input_dim=int(source_train.signals.shape[1]),
        stats_dim=int(source_train.stats.shape[1]),
        source_domain_key=source_domain_key,
        target_domain_key=target_domain_key,
        source_setting=PU_DOMAIN_TO_SETTING[source_domain_key],
        target_setting=PU_DOMAIN_TO_SETTING[target_domain_key],
        source_metadata=PU_SETTING_METADATA[PU_DOMAIN_TO_SETTING[source_domain_key]],
        target_metadata=PU_SETTING_METADATA[PU_DOMAIN_TO_SETTING[target_domain_key]],
    )


def build_pu_multisource_bundle(
    root: str | Path,
    source_domain_keys: list,
    target_domain_key: str,
    normalize: str = "zscore",
) -> PUTaskBundle:
    """Build a PU bundle whose source split is the concatenation of multiple domains.

    Useful for compound DG: train on N source domains, test on the held-out target.
    """
    if len(source_domain_keys) < 1:
        raise ValueError("At least one source domain is required.")
    for key in source_domain_keys:
        if key not in PU_DOMAIN_TO_SETTING:
            raise KeyError(f"Unsupported source domain: {key}")
    if target_domain_key not in PU_DOMAIN_TO_SETTING:
        raise KeyError(f"Unsupported target domain: {target_domain_key}")

    root_path = Path(root).resolve()
    domain_names = [PU_DOMAIN_TO_SETTING[key] for key in sorted(PU_DOMAIN_TO_SETTING)]
    label_names = [PU_LABEL_TO_NAME[index] for index in sorted(PU_LABEL_TO_NAME)]

    target_index = sorted(PU_DOMAIN_TO_SETTING).index(target_domain_key)

    train_parts: list[ProcessedSignalDataset] = []
    val_parts: list[ProcessedSignalDataset] = []
    for key in source_domain_keys:
        idx = sorted(PU_DOMAIN_TO_SETTING).index(key)
        train_parts.append(_load_pt_split(root_path, "train", key, idx, normalize))
        val_parts.append(_load_pt_split(root_path, "val", key, idx, normalize))

    def _concat(parts: list) -> ProcessedSignalDataset:
        signals = torch.cat([p.signals for p in parts], dim=0)
        stats = torch.cat([p.stats for p in parts], dim=0)
        labels = torch.cat([p.labels for p in parts], dim=0)
        domains = torch.cat([p.domains for p in parts], dim=0)
        prefix = "+".join(source_domain_keys)
        ds = ProcessedSignalDataset(
            signals=signals,
            stats=stats,
            labels=labels,
            domain_index=parts[0].domains[0].item() if len(parts[0]) else 0,
            sample_prefix=prefix,
        )
        ds.domains = domains
        return ds

    source_train = _concat(train_parts)
    source_val = _concat(val_parts)
    target_train = _load_pt_split(root_path, "train", target_domain_key, target_index, normalize)
    target_val = _load_pt_split(root_path, "val", target_domain_key, target_index, normalize)
    target_test = _load_pt_split(root_path, "test", target_domain_key, target_index, normalize)

    aggregated_metadata = {
        "rpm": float(np.mean([PU_SETTING_METADATA[PU_DOMAIN_TO_SETTING[k]]["rpm"] for k in source_domain_keys])),
        "torque_nm": float(np.mean([PU_SETTING_METADATA[PU_DOMAIN_TO_SETTING[k]]["torque_nm"] for k in source_domain_keys])),
        "radial_n": float(np.mean([PU_SETTING_METADATA[PU_DOMAIN_TO_SETTING[k]]["radial_n"] for k in source_domain_keys])),
    }

    return PUTaskBundle(
        source_train=source_train,
        source_val=source_val,
        target_train=target_train,
        target_val=target_val,
        target_test=target_test,
        label_names=label_names,
        domain_names=domain_names,
        input_dim=int(source_train.signals.shape[1]),
        stats_dim=int(source_train.stats.shape[1]),
        source_domain_key="+".join(source_domain_keys),
        target_domain_key=target_domain_key,
        source_setting="+".join(PU_DOMAIN_TO_SETTING[k] for k in source_domain_keys),
        target_setting=PU_DOMAIN_TO_SETTING[target_domain_key],
        source_metadata=aggregated_metadata,
        target_metadata=PU_SETTING_METADATA[PU_DOMAIN_TO_SETTING[target_domain_key]],
    )


def compute_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels.long(), minlength=num_classes).float().clamp_min(1.0)
    weights = counts.sum() / (counts * float(num_classes))
    return weights


def format_task_name(source_key: str, target_key: str) -> str:
    source_setting = PU_DOMAIN_TO_SETTING[source_key]
    target_setting = PU_DOMAIN_TO_SETTING[target_key]
    return f"{source_setting}_to_{target_setting}"
