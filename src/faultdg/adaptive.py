from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from faultdg.models import build_model
from faultdg.trainer import evaluate_classification_loss, train_method


def _trajectory_stability(losses: list[float], window: int) -> tuple[float, float]:
    """Return (relative_std, slope) of the last ``window`` loss values.

    - relative_std = std / |mean| (scale-free oscillation magnitude).
    - slope        = least-squares slope per epoch (negative ⇒ decreasing).

    These match the paper's "fast and stable decrease" criterion in Fig. 3-2:
    a candidate is considered stable when relative_std is small *and* slope is
    non-positive over the tail of the training trajectory.
    """
    tail = [float(x) for x in losses[-window:] if np.isfinite(x)]
    if len(tail) < 2:
        return 0.0, 0.0
    arr = np.asarray(tail, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    rel_std = std / max(abs(mean), 1e-8)
    epochs = np.arange(len(arr), dtype=np.float64)
    slope = float(np.polyfit(epochs, arr, 1)[0])
    return rel_std, slope


def _expand_width(hidden_dims: list[int], width_step: int, max_hidden_dim: int) -> list[int]:
    expanded: list[int] = []
    for index, hidden_dim in enumerate(hidden_dims):
        step = width_step if index == 0 else max(width_step // 2, 32)
        expanded.append(min(hidden_dim + step, max_hidden_dim))
    return expanded


def _expand_depth(hidden_dims: list[int], depth_scale: float, min_hidden_dim: int) -> list[int]:
    new_dim = max(int(hidden_dims[-1] * depth_scale), min_hidden_dim)
    if new_dim >= hidden_dims[-1]:
        new_dim = max(hidden_dims[-1] // 2, min_hidden_dim)
    return hidden_dims + [new_dim]


def select_hidden_dims(
    config: dict,
    input_dim: int,
    stats_dim: int,
    num_classes: int,
    source_train_loader: DataLoader,
    source_val_loader: DataLoader,
    target_train_loader: DataLoader,
    device: torch.device,
    checkpoint_dir: Path,
    label_names: list[str],
    domain_names: list[str],
    class_weights: Optional[torch.Tensor] = None,
) -> tuple[list[int], pd.DataFrame]:
    search_cfg = config.get("adaptive_search", {})
    if not search_cfg.get("enabled", True):
        return list(config["model"]["hidden_dims"]), pd.DataFrame()

    width_step = int(search_cfg.get("width_step", 128))
    max_hidden_dim = int(search_cfg.get("max_hidden_dim", 1024))
    depth_scale = float(search_cfg.get("depth_scale", 0.5))
    min_hidden_dim = int(search_cfg.get("min_hidden_dim", 128))
    max_stages = int(search_cfg.get("max_stages", 3))
    # Relative loss-delta criterion: accept a candidate if validation loss drops by
    # at least ``min_rel_loss_delta`` (e.g. 0.5%) relative to the current accepted
    # loss, OR by an absolute amount of ``min_loss_delta``. The relative criterion
    # makes the search trigger even when absolute loss values are small.
    min_loss_delta = float(search_cfg.get("min_loss_delta", 0.002))
    min_rel_loss_delta = float(search_cfg.get("min_rel_loss_delta", 0.005))
    patience = int(search_cfg.get("patience", 1))
    # Oscillation detection (Fig. 3-2 of the reference paper).
    # A candidate is "stable convergent" if the tail of its training-loss
    # trajectory has small relative variance AND non-positive slope. Unstable
    # candidates are flagged so the search prefers expanding structure rather
    # than locking in a noisy collapse.
    osc_window = int(search_cfg.get("oscillation_window", 4))
    osc_rel_std_thresh = float(search_cfg.get("oscillation_rel_std_thresh", 0.15))
    osc_slope_thresh = float(search_cfg.get("oscillation_slope_thresh", 0.0))
    osc_enabled = bool(search_cfg.get("oscillation_enabled", True))

    base_hidden_dims = list(config["model"]["hidden_dims"])
    current_hidden_dims = list(base_hidden_dims)
    accepted_loss: Optional[float] = None
    best_hidden_dims = list(base_hidden_dims)
    best_loss: Optional[float] = None
    rejects = 0
    records: list[dict[str, object]] = []

    for stage in range(max_stages + 1):
        if stage == 0:
            candidate_hidden_dims = list(current_hidden_dims)
            expansion = "base"
        elif stage % 2 == 1:
            candidate_hidden_dims = _expand_width(current_hidden_dims, width_step=width_step, max_hidden_dim=max_hidden_dim)
            expansion = "width"
        else:
            candidate_hidden_dims = _expand_depth(current_hidden_dims, depth_scale=depth_scale, min_hidden_dim=min_hidden_dim)
            expansion = "depth"

        search_config = copy.deepcopy(config)
        search_config["model"]["hidden_dims"] = candidate_hidden_dims
        search_config["training"]["pretrain_epochs"] = int(search_cfg.get("search_pretrain_epochs", 1))
        search_config["training"]["finetune_epochs"] = int(search_cfg.get("search_finetune_epochs", 3))

        model = build_model(search_config, input_dim=input_dim, stats_dim=stats_dim, num_classes=num_classes).to(device)
        checkpoint_path = checkpoint_dir / f"search_stage_{stage:02d}.pt"
        train_result = train_method(
            method="proposed",
            model=model,
            source_train_loader=source_train_loader,
            source_val_loader=source_val_loader,
            target_train_loader=target_train_loader,
            config=search_config,
            device=device,
            checkpoint_path=checkpoint_path,
            label_names=label_names,
            domain_names=domain_names,
            class_weights=class_weights,
        )
        val_loss = evaluate_classification_loss(
            model=model,
            data_loader=source_val_loader,
            device=device,
            use_stats=True,
            class_weights=class_weights,
        )
        train_losses = [float(rec.get("train_loss", float("nan"))) for rec in train_result.history]
        rel_std, slope = _trajectory_stability(train_losses, window=osc_window)
        is_stable = (rel_std <= osc_rel_std_thresh) and (slope <= osc_slope_thresh)

        if accepted_loss is None:
            improvement_abs = float("inf")
            improvement_rel = float("inf")
        else:
            improvement_abs = accepted_loss - val_loss
            improvement_rel = improvement_abs / max(abs(accepted_loss), 1e-8)
        loss_ok = (
            accepted_loss is None
            or improvement_abs >= min_loss_delta
            or improvement_rel >= min_rel_loss_delta
        )
        # If oscillation detection is on, an unstable trajectory at stage 0 is
        # treated as "must expand" (still record stage 0 as accepted to seed the
        # baseline). For later stages, instability blocks acceptance: we'd rather
        # try a wider/deeper variant than lock onto a noisy minimum.
        if osc_enabled and stage > 0 and not is_stable:
            accepted = False
        else:
            accepted = loss_ok
        records.append(
            {
                "stage": stage,
                "expansion": expansion,
                "hidden_dims": "-".join(map(str, candidate_hidden_dims)),
                "val_loss": val_loss,
                "accepted": bool(accepted),
                "improvement_abs": float(improvement_abs) if accepted_loss is not None else None,
                "improvement_rel": float(improvement_rel) if accepted_loss is not None else None,
                "traj_rel_std": float(rel_std),
                "traj_slope": float(slope),
                "traj_stable": bool(is_stable),
            }
        )

        if best_loss is None or val_loss < best_loss:
            best_loss = val_loss
            best_hidden_dims = list(candidate_hidden_dims)

        if accepted:
            accepted_loss = val_loss
            current_hidden_dims = candidate_hidden_dims
            rejects = 0
            continue

        rejects += 1
        if rejects > patience:
            break

    # Always return the structure with the lowest validation loss observed during the
    # search, even if it was not formally "accepted" by the delta criterion. This keeps
    # speed_shift from silently sticking to the base structure when the search is noisy.
    return best_hidden_dims, pd.DataFrame(records)
