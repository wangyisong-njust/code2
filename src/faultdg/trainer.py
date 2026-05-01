from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn, optim
from torch.utils.data import DataLoader

from faultdg.data import add_awgn_torch, compute_stats_torch


@dataclass
class EvaluationResult:
    metrics: Dict[str, float]
    predictions: pd.DataFrame
    confusion: np.ndarray
    features: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    domains: Optional[np.ndarray] = None


@dataclass
class TrainResult:
    history: List[Dict[str, float]]
    checkpoint_path: Path
    best_epoch: int


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _mmd_loss(source: torch.Tensor, target: torch.Tensor, kernel_gamma: float = 1.0) -> torch.Tensor:
    if source.size(0) == 0 or target.size(0) == 0:
        return source.new_tensor(0.0)
    xx = source @ source.t()
    yy = target @ target.t()
    xy = source @ target.t()
    rx = torch.diag(xx).unsqueeze(0).expand_as(xx)
    ry = torch.diag(yy).unsqueeze(0).expand_as(yy)
    rx_cross = torch.diag(xx).unsqueeze(1).expand(source.size(0), target.size(0))
    ry_cross = torch.diag(yy).unsqueeze(0).expand(source.size(0), target.size(0))
    k_xx = torch.exp(-kernel_gamma * (rx.t() + rx - 2 * xx))
    k_yy = torch.exp(-kernel_gamma * (ry.t() + ry - 2 * yy))
    k_xy = torch.exp(-kernel_gamma * (rx_cross + ry_cross - 2 * xy))
    return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()


def _coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if source.size(0) < 2 or target.size(0) < 2:
        return source.new_tensor(0.0)
    d = source.size(1)
    source_centered = source - source.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    source_cov = (source_centered.t() @ source_centered) / (source.size(0) - 1)
    target_cov = (target_centered.t() @ target_centered) / (target.size(0) - 1)
    return torch.mean((source_cov - target_cov) ** 2) / (4 * d * d)


def _discriminative_loss(
    latent: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
    intra_weight: float = 1.0,
    inter_weight: float = 1.0,
    squared: bool = False,
    adaptive: bool = False,
) -> torch.Tensor:
    """Class-compactness + class-separation loss.

    Paper Eq.(3-6): J_new = J + μ1·J1 − μ2·J2. Improvements:
    - ``squared=True`` switches the inter-class term to squared hinge max(0, m−d)²,
      which keeps a non-vanishing gradient until the margin is fully satisfied.
    - ``adaptive=True`` rebalances μ1·J1 and μ2·J2 each step so their detached
      magnitudes match, avoiding one term dominating once the other plateaus.
    """
    present_classes = torch.unique(labels)
    if present_classes.numel() < 2:
        return latent.new_tensor(0.0)

    centers: list[torch.Tensor] = []
    intra_terms: list[torch.Tensor] = []
    for class_index in present_classes:
        class_mask = labels == class_index
        class_features = latent[class_mask]
        class_center = class_features.mean(dim=0)
        centers.append(class_center)
        intra_terms.append(torch.mean(torch.sum((class_features - class_center) ** 2, dim=1)))

    intra_loss = torch.stack(intra_terms).mean()
    centers_tensor = torch.stack(centers, dim=0)
    pairwise_dist = torch.cdist(centers_tensor, centers_tensor, p=2)
    separation_mask = torch.triu(torch.ones_like(pairwise_dist), diagonal=1) > 0
    if not torch.any(separation_mask):
        separation_loss = latent.new_tensor(0.0)
    else:
        gap = F.relu(margin - pairwise_dist[separation_mask])
        separation_loss = (gap * gap).mean() if squared else gap.mean()

    if adaptive:
        intra_mag = intra_loss.detach().abs().clamp(min=1e-6)
        sep_mag = separation_loss.detach().abs().clamp(min=1e-6)
        balance = (intra_mag / sep_mag).clamp(min=1e-3, max=1e3)
        return intra_weight * intra_loss + inter_weight * balance * separation_loss
    return intra_weight * intra_loss + inter_weight * separation_loss


def _kl_sparsity(latent: torch.Tensor, rho: float) -> torch.Tensor:
    """KL(ρ‖ρ̂) over latent units (Eq.(9) of the reference paper).

    ρ̂_j is computed as the batch-mean of sigmoid(latent_j) so we get a value in
    (0,1) regardless of the encoder's output activation.
    """
    rho_hat = torch.sigmoid(latent).mean(dim=0).clamp(min=1e-6, max=1.0 - 1e-6)
    rho_t = torch.full_like(rho_hat, float(rho))
    kl = rho_t * torch.log(rho_t / rho_hat) + (1.0 - rho_t) * torch.log(
        (1.0 - rho_t) / (1.0 - rho_hat)
    )
    return kl.sum()


def _time_scale_signals(signals: torch.Tensor, scale: float) -> torch.Tensor:
    """Resample ``signals`` along the time axis as if rotating at a different rpm.

    With ``scale = source_rpm / target_rpm``:
    - ``scale > 1`` (source faster than target): we slow source signals toward the
      target rpm. ``sample_positions = base / scale`` stay in range, so we get a
      clean linear interpolation.
    - ``scale < 1`` (source slower than target): we'd need samples beyond the
      record, which don't exist. Previously we clamped, which silently turned the
      tail of the augmented signal into a constant — a destructive corruption that
      made ``a_to_*`` runs underperform across the full matrix. We now wrap the
      out-of-range positions back into the signal so the augmented record stays
      stationary in spectral content (a periodic-extension assumption is far less
      damaging than a flat tail) and the speed_aug branch produces useful gradient
      signal in both directions.
    """
    if abs(scale - 1.0) < 1e-4:
        return signals
    num_samples, signal_length = signals.shape
    base_positions = torch.arange(signal_length, device=signals.device, dtype=signals.dtype)
    raw_positions = base_positions / scale
    if scale >= 1.0:
        sample_positions = torch.clamp(raw_positions, 0.0, float(signal_length - 1))
    else:
        # Wrap (periodic extension) instead of clamp to avoid the constant tail.
        sample_positions = torch.remainder(raw_positions, float(signal_length))
    left_index = torch.floor(sample_positions).long()
    right_index = torch.clamp(left_index + 1, max=signal_length - 1)
    right_weight = sample_positions - left_index.to(signals.dtype)
    left_weight = 1.0 - right_weight

    left_values = signals[:, left_index]
    right_values = signals[:, right_index]
    scaled = left_values * left_weight.unsqueeze(0) + right_values * right_weight.unsqueeze(0)
    return scaled.reshape(num_samples, signal_length)


def _mixup(
    signals: torch.Tensor,
    stats: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0.0:
        return signals, stats, labels, labels, 1.0
    lam = float(np.random.beta(alpha, alpha))
    indices = torch.randperm(signals.size(0), device=signals.device)
    mixed_signals = lam * signals + (1.0 - lam) * signals[indices]
    mixed_stats = lam * stats + (1.0 - lam) * stats[indices]
    return mixed_signals, mixed_stats, labels, labels[indices], lam


def _cycle_loader(loader: Optional[DataLoader]):
    if loader is None or len(loader.dataset) == 0:
        return None
    return cycle(loader)


def _move_batch(batch, device: torch.device):
    signals, stats, labels, domains, sample_ids = batch
    return (
        signals.to(device, non_blocking=True),
        stats.to(device, non_blocking=True),
        labels.to(device, non_blocking=True),
        domains.to(device, non_blocking=True),
        list(sample_ids),
    )


def _autoencoder_pretrain(
    model: nn.Module,
    source_loader: DataLoader,
    target_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    denoise_std: float,
    sparsity_rho: float = 0.05,
    sparsity_weight: float = 0.0,
) -> None:
    if epochs <= 0:
        return

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    recon_criterion = nn.MSELoss()
    target_iter = _cycle_loader(target_loader)

    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        recon_sum = 0.0
        sparse_sum = 0.0
        batch_count = 0
        for source_batch in source_loader:
            source_signals, _, _, _, _ = _move_batch(source_batch, device)
            if target_iter is not None:
                target_signals, _, _, _, _ = _move_batch(next(target_iter), device)
                clean = torch.cat([source_signals, target_signals], dim=0)
            else:
                clean = source_signals

            noisy = clean + torch.randn_like(clean) * denoise_std
            noisy_stats = compute_stats_torch(noisy)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(noisy, noisy_stats, use_stats=False)
            recon_loss = recon_criterion(outputs["reconstruction"], clean.flatten(start_dim=1))
            if sparsity_weight > 0.0:
                sparse_loss = _kl_sparsity(outputs["latent"], sparsity_rho)
                loss = recon_loss + sparsity_weight * sparse_loss
                sparse_sum += float(sparse_loss.item())
            else:
                loss = recon_loss
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.item())
            recon_sum += float(recon_loss.item())
            batch_count += 1

        avg_loss = loss_sum / max(batch_count, 1)
        avg_recon = recon_sum / max(batch_count, 1)
        avg_sparse = sparse_sum / max(batch_count, 1)
        if sparsity_weight > 0.0:
            print(
                f"[pretrain] epoch={epoch + 1:02d}/{epochs:02d} "
                f"loss={avg_loss:.4f} recon={avg_recon:.4f} sparse={avg_sparse:.4f}"
            )
        else:
            print(f"[pretrain] epoch={epoch + 1:02d}/{epochs:02d} recon={avg_recon:.4f}")


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    label_names: list[str],
    domain_names: list[str],
    use_stats: bool,
    noise_db: Optional[float] = None,
    collect_features: bool = False,
) -> EvaluationResult:
    if len(data_loader.dataset) == 0:
        raise ValueError("Cannot evaluate with an empty dataset.")

    model.eval()
    logits_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    domains_list: list[torch.Tensor] = []
    sample_ids: list[str] = []
    feature_list: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in data_loader:
            signals, stats, labels, domains, batch_ids = _move_batch(batch, device)
            if noise_db is not None:
                signals = add_awgn_torch(signals, noise_db)
                if use_stats:
                    stats = compute_stats_torch(signals)

            outputs = model(signals, stats, use_stats=use_stats)
            logits_list.append(outputs["logits"].cpu())
            labels_list.append(labels.cpu())
            domains_list.append(domains.cpu())
            sample_ids.extend(batch_ids)
            if collect_features:
                feature_list.append(outputs["features"].cpu())

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0).numpy()
    domains = torch.cat(domains_list, dim=0).numpy()
    predictions = torch.argmax(logits, dim=1).numpy()

    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "num_samples": float(labels.shape[0]),
    }
    confusion = confusion_matrix(labels, predictions, labels=list(range(len(label_names))))

    prediction_df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "label_true": [label_names[index] for index in labels],
            "label_pred": [label_names[index] for index in predictions],
            "domain": [domain_names[index] for index in domains],
        }
    )
    prediction_df["correct"] = (prediction_df["label_true"] == prediction_df["label_pred"]).astype(int)
    prediction_df["confidence"] = torch.softmax(logits, dim=1).max(dim=1).values.numpy()

    features = np.concatenate([chunk.numpy() for chunk in feature_list], axis=0) if feature_list else None
    return EvaluationResult(
        metrics=metrics,
        predictions=prediction_df,
        confusion=confusion,
        features=features,
        labels=labels if collect_features else None,
        domains=domains if collect_features else None,
    )


def evaluate_classification_loss(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    use_stats: bool,
    class_weights: Optional[torch.Tensor] = None,
) -> float:
    if len(data_loader.dataset) == 0:
        raise ValueError("Cannot evaluate loss with an empty dataset.")

    model.eval()
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    loss_sum = 0.0
    batch_count = 0
    with torch.no_grad():
        for batch in data_loader:
            signals, stats, labels, _, _ = _move_batch(batch, device)
            outputs = model(signals, stats, use_stats=use_stats)
            loss = criterion(outputs["logits"], labels)
            loss_sum += float(loss.item())
            batch_count += 1
    return loss_sum / max(batch_count, 1)


def train_method(
    method: str,
    model: nn.Module,
    source_train_loader: DataLoader,
    source_val_loader: DataLoader,
    target_train_loader: Optional[DataLoader],
    config: dict,
    device: torch.device,
    checkpoint_path: Path,
    label_names: list[str],
    domain_names: list[str],
    class_weights: Optional[torch.Tensor] = None,
) -> TrainResult:
    training_cfg = config["training"]
    method_params = training_cfg.get("method_params", {})
    learning_rate = float(training_cfg["learning_rate"])
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    pretrain_epochs = int(training_cfg.get("pretrain_epochs", 0))
    finetune_epochs = int(training_cfg["finetune_epochs"])
    scheduler_gamma = float(training_cfg.get("scheduler_gamma", 0.5))
    denoise_std = float(method_params.get("denoise_std", 0.05))
    consistency_noise_db = float(method_params.get("consistency_noise_db", 6.0))
    discriminative_weight = float(method_params.get("discriminative_weight", 0.0))
    discriminative_margin = float(method_params.get("discriminative_margin", 8.0))
    discriminative_intra_weight = float(method_params.get("discriminative_intra_weight", 1.0))
    discriminative_inter_weight = float(method_params.get("discriminative_inter_weight", 1.0))
    discriminative_squared = bool(method_params.get("discriminative_squared", False))
    discriminative_adaptive = bool(method_params.get("discriminative_adaptive", False))
    pretrain_sparsity_rho = float(method_params.get("pretrain_sparsity_rho", 0.05))
    pretrain_sparsity_weight = float(method_params.get("pretrain_sparsity_weight", 0.0))
    speed_aug_weight = float(method_params.get("speed_aug_weight", 0.0))
    speed_scale = float(method_params.get("speed_scale", 1.0))
    speed_consistency_weight = float(method_params.get("speed_consistency_weight", 0.0))
    mixup_alpha = float(method_params.get("mixup_alpha", 0.2))
    coral_weight = float(method_params.get("coral_weight", 0.5))
    mmd_weight = float(method_params.get("mmd_weight", 0.5))
    recon_weight = float(method_params.get("recon_weight", 0.3))
    consistency_weight = float(method_params.get("consistency_weight", 0.2))

    dann_weight = float(method_params.get("dann_weight", 0.1))
    dann_lambda_max = float(method_params.get("dann_lambda_max", 1.0))

    protocol_mode = str(config.get("protocol", {}).get("mode", "uda")).lower()
    use_stats = method == "proposed"
    needs_target = method in {"coral", "mmd", "dann", "proposed"}
    if protocol_mode == "strict_dg":
        # Strict DG: never look at the target domain during training. Disable any
        # auxiliary loss that requires unlabeled target samples.
        target_train_loader = None
        needs_target = False
    target_iter = _cycle_loader(target_train_loader)
    if needs_target and target_iter is None:
        raise ValueError(f"Method '{method}' requires a non-empty target_train split.")

    _autoencoder_pretrain(
        model=model,
        source_loader=source_train_loader,
        target_loader=target_train_loader if needs_target else None,
        device=device,
        epochs=pretrain_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        denoise_std=denoise_std,
        sparsity_rho=pretrain_sparsity_rho,
        sparsity_weight=pretrain_sparsity_weight,
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(finetune_epochs // 2, 1),
        gamma=scheduler_gamma,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    recon_criterion = nn.MSELoss()
    history: list[dict[str, float]] = []
    best_score = -1.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    validation_loader = source_val_loader if len(source_val_loader.dataset) > 0 else source_train_loader

    for epoch in range(finetune_epochs):
        model.train()
        loss_sum = 0.0
        cls_sum = 0.0
        aux_sum = 0.0
        batch_count = 0

        for batch in source_train_loader:
            source_signals, source_stats, source_labels, _, _ = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)

            if method == "erm":
                outputs = model(source_signals, source_stats, use_stats=False)
                cls_loss = criterion(outputs["logits"], source_labels)
                aux_loss = source_signals.new_tensor(0.0)
            elif method == "mixup":
                mixed_signals, mixed_stats, labels_a, labels_b, lam = _mixup(
                    source_signals,
                    source_stats,
                    source_labels,
                    mixup_alpha,
                )
                outputs = model(mixed_signals, mixed_stats, use_stats=False)
                cls_loss = lam * criterion(outputs["logits"], labels_a) + (1.0 - lam) * criterion(
                    outputs["logits"],
                    labels_b,
                )
                aux_loss = source_signals.new_tensor(0.0)
            elif method in {"coral", "mmd", "dann"} and target_iter is None:
                # strict_dg fallback: source-only classification.
                outputs = model(source_signals, source_stats, use_stats=False)
                cls_loss = criterion(outputs["logits"], source_labels)
                aux_loss = source_signals.new_tensor(0.0)
            elif method in {"coral", "mmd"}:
                target_signals, target_stats, _, _, _ = _move_batch(next(target_iter), device)
                source_outputs = model(source_signals, source_stats, use_stats=False)
                target_outputs = model(target_signals, target_stats, use_stats=False)
                cls_loss = criterion(source_outputs["logits"], source_labels)
                if method == "coral":
                    aux_loss = coral_weight * _coral_loss(source_outputs["latent"], target_outputs["latent"])
                else:
                    aux_loss = mmd_weight * _mmd_loss(source_outputs["latent"], target_outputs["latent"])
            elif method == "dann":
                target_signals, target_stats, _, _, _ = _move_batch(next(target_iter), device)
                source_outputs = model(source_signals, source_stats, use_stats=False)
                target_outputs = model(target_signals, target_stats, use_stats=False)
                cls_loss = criterion(source_outputs["logits"], source_labels)
                progress = float(epoch + 1) / float(max(finetune_epochs, 1))
                lambda_ = dann_lambda_max * (2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0)
                domain_logits = torch.cat(
                    [
                        model.discriminate_domain(source_outputs["latent"], lambda_),
                        model.discriminate_domain(target_outputs["latent"], lambda_),
                    ],
                    dim=0,
                )
                domain_labels = torch.cat(
                    [
                        torch.zeros(source_outputs["latent"].size(0), dtype=torch.long, device=device),
                        torch.ones(target_outputs["latent"].size(0), dtype=torch.long, device=device),
                    ],
                    dim=0,
                )
                aux_loss = dann_weight * F.cross_entropy(domain_logits, domain_labels)
            elif method == "proposed":
                has_target = target_iter is not None
                if has_target:
                    target_signals, target_stats, _, _, _ = _move_batch(next(target_iter), device)
                source_clean = model(source_signals, source_stats, use_stats=True)
                target_clean = model(target_signals, target_stats, use_stats=True) if has_target else None

                source_noisy_signals = add_awgn_torch(source_signals, consistency_noise_db)
                source_noisy_stats = compute_stats_torch(source_noisy_signals)
                source_noisy = model(source_noisy_signals, source_noisy_stats, use_stats=True)
                if has_target:
                    target_noisy_signals = add_awgn_torch(target_signals, consistency_noise_db)
                    target_noisy_stats = compute_stats_torch(target_noisy_signals)
                    target_noisy = model(target_noisy_signals, target_noisy_stats, use_stats=True)

                cls_loss = criterion(source_clean["logits"], source_labels)
                # Align latent representations only. The handcrafted stats branch is intentionally
                # condition-sensitive and helps classification, but forcing full fused features to
                # match across conditions can hurt cross-condition transfer, especially for speed shift.
                if has_target:
                    align_loss = mmd_weight * _mmd_loss(source_clean["latent"], target_clean["latent"])
                else:
                    align_loss = source_signals.new_tensor(0.0)
                if has_target:
                    recon_loss = recon_weight * (
                        recon_criterion(source_noisy["reconstruction"], source_signals.flatten(start_dim=1))
                        + recon_criterion(target_noisy["reconstruction"], target_signals.flatten(start_dim=1))
                    )
                    consistency_loss = consistency_weight * (
                        F.mse_loss(
                            torch.softmax(source_noisy["logits"], dim=1),
                            torch.softmax(source_clean["logits"], dim=1).detach(),
                        )
                        + F.mse_loss(
                            torch.softmax(target_noisy["logits"], dim=1),
                            torch.softmax(target_clean["logits"], dim=1).detach(),
                        )
                    )
                else:
                    recon_loss = recon_weight * recon_criterion(
                        source_noisy["reconstruction"], source_signals.flatten(start_dim=1)
                    )
                    consistency_loss = consistency_weight * F.mse_loss(
                        torch.softmax(source_noisy["logits"], dim=1),
                        torch.softmax(source_clean["logits"], dim=1).detach(),
                    )
                discriminative_loss = source_signals.new_tensor(0.0)
                if discriminative_weight > 0.0:
                    discriminative_loss = discriminative_weight * _discriminative_loss(
                        source_clean["latent"],
                        source_labels,
                        margin=discriminative_margin,
                        intra_weight=discriminative_intra_weight,
                        inter_weight=discriminative_inter_weight,
                        squared=discriminative_squared,
                        adaptive=discriminative_adaptive,
                    )

                speed_cls_loss = source_signals.new_tensor(0.0)
                speed_consistency_loss = source_signals.new_tensor(0.0)
                if speed_aug_weight > 0.0 and abs(speed_scale - 1.0) >= 1e-4:
                    source_speed_signals = _time_scale_signals(source_signals, speed_scale)
                    source_speed_stats = compute_stats_torch(source_speed_signals)
                    source_speed = model(source_speed_signals, source_speed_stats, use_stats=True)
                    speed_cls_loss = speed_aug_weight * criterion(source_speed["logits"], source_labels)
                    if speed_consistency_weight > 0.0:
                        speed_consistency_loss = speed_consistency_weight * F.mse_loss(
                            torch.softmax(source_speed["logits"], dim=1),
                            torch.softmax(source_clean["logits"], dim=1).detach(),
                        )

                aux_loss = (
                    align_loss
                    + recon_loss
                    + consistency_loss
                    + discriminative_loss
                    + speed_cls_loss
                    + speed_consistency_loss
                )
            else:
                raise ValueError(f"Unsupported method: {method}")

            loss = cls_loss + aux_loss
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.item())
            cls_sum += float(cls_loss.item())
            aux_sum += float(aux_loss.item())
            batch_count += 1

        scheduler.step()
        val_result = evaluate_model(
            model=model,
            data_loader=validation_loader,
            device=device,
            label_names=label_names,
            domain_names=domain_names,
            use_stats=use_stats,
            noise_db=None,
            collect_features=False,
        )
        val_accuracy = float(val_result.metrics["accuracy"])
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": loss_sum / max(batch_count, 1),
                "train_cls_loss": cls_sum / max(batch_count, 1),
                "train_aux_loss": aux_sum / max(batch_count, 1),
                "val_accuracy": val_accuracy,
                "val_macro_f1": float(val_result.metrics["macro_f1"]),
            }
        )
        print(
            f"[{method}] epoch={epoch + 1:02d}/{finetune_epochs:02d} "
            f"loss={history[-1]['train_loss']:.4f} "
            f"cls={history[-1]['train_cls_loss']:.4f} "
            f"aux={history[-1]['train_aux_loss']:.4f} "
            f"val_acc={val_accuracy:.4f}"
        )

        if val_accuracy >= best_score:
            best_score = val_accuracy
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "method": method,
            "best_epoch": best_epoch,
            "history": history,
        },
        checkpoint_path,
    )
    return TrainResult(history=history, checkpoint_path=checkpoint_path, best_epoch=best_epoch)
