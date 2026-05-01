from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_method_accuracy(summary_df: pd.DataFrame, output_path: Path) -> None:
    if summary_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    metrics = [("accuracy", "Target Accuracy"), ("macro_f1", "Target Macro-F1")]
    x = np.arange(len(summary_df))

    for axis, (column, title) in zip(axes, metrics):
        values = summary_df[column].to_numpy()
        axis.bar(x, values, width=0.6)
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels(summary_df["method"].tolist(), rotation=15)
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        axis.set_ylim(0.0, min(1.0, max(values.max() * 1.15, 0.2)))

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_noise_curves(noise_df: pd.DataFrame, output_path: Path) -> None:
    if noise_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for method, method_rows in noise_df.groupby("method"):
        ordered = method_rows.sort_values("snr_db", ascending=False)
        ax.plot(ordered["snr_db"], ordered["accuracy"], marker="o", linewidth=2, label=method)
    ax.set_title("Target Accuracy Under AWGN")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(confusion: np.ndarray, label_names: Sequence[str], output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_embedding(
    features: np.ndarray,
    labels: np.ndarray,
    domains: np.ndarray,
    label_names: Sequence[str],
    domain_names: Sequence[str],
    output_path: Path,
    method: str,
    max_points: int,
    seed: int,
) -> None:
    if features is None or labels is None or domains is None or features.shape[0] < 3:
        return

    if features.shape[0] > max_points:
        indices = np.linspace(0, features.shape[0] - 1, num=max_points, dtype=int)
        features = features[indices]
        labels = labels[indices]
        domains = domains[indices]

    if method.lower() == "tsne":
        reducer = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto")
        embedding = reducer.fit_transform(features)
    else:
        embedding = PCA(n_components=2).fit_transform(features)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for label_index, label_name in enumerate(label_names):
        mask = labels == label_index
        if mask.any():
            axes[0].scatter(embedding[mask, 0], embedding[mask, 1], s=14, alpha=0.75, label=label_name)
    axes[0].set_title("Embedding By Fault Class")
    axes[0].legend(markerscale=1.4, fontsize=8)

    for domain_index, domain_name in enumerate(domain_names):
        mask = domains == domain_index
        if mask.any():
            axes[1].scatter(embedding[mask, 0], embedding[mask, 1], s=14, alpha=0.75, label=domain_name)
    axes[1].set_title("Embedding By Domain")
    axes[1].legend(markerscale=1.4, fontsize=8)

    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _reduce(features: np.ndarray, method: str, seed: int, max_points: int) -> np.ndarray:
    if features.shape[0] > max_points:
        indices = np.linspace(0, features.shape[0] - 1, num=max_points, dtype=int)
        features = features[indices]
    if method.lower() == "tsne":
        perplexity = min(30.0, max(5.0, float(features.shape[0]) / 4.0 - 1))
        reducer = TSNE(
            n_components=2,
            random_state=seed,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
        )
        return reducer.fit_transform(features)
    return PCA(n_components=2).fit_transform(features)


def plot_tsne_alignment(
    baseline_features: np.ndarray,
    baseline_labels: np.ndarray,
    baseline_domains: np.ndarray,
    proposed_features: np.ndarray,
    proposed_labels: np.ndarray,
    proposed_domains: np.ndarray,
    label_names: Sequence[str],
    domain_names: Sequence[str],
    output_path: Path,
    seed: int,
    max_points: int = 1500,
    baseline_name: str = "Baseline (ERM-SDAE)",
    proposed_name: str = "Adaptive-SDAE-3M",
) -> None:
    """Side-by-side t-SNE: baseline source/target vs proposed source/target."""
    if baseline_features is None or proposed_features is None:
        return
    if baseline_features.shape[0] < 4 or proposed_features.shape[0] < 4:
        return

    b_idx = np.linspace(0, baseline_features.shape[0] - 1, num=min(max_points, baseline_features.shape[0]), dtype=int)
    p_idx = np.linspace(0, proposed_features.shape[0] - 1, num=min(max_points, proposed_features.shape[0]), dtype=int)
    baseline_emb = _reduce(baseline_features[b_idx], "tsne", seed, max_points)
    proposed_emb = _reduce(proposed_features[p_idx], "tsne", seed, max_points)
    bl, bd = baseline_labels[b_idx], baseline_domains[b_idx]
    pl, pd_ = proposed_labels[p_idx], proposed_domains[p_idx]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10))
    palette_classes = sns.color_palette("tab10", n_colors=len(label_names))
    palette_domains = sns.color_palette("Set2", n_colors=len(domain_names))

    panels = [(baseline_emb, bl, bd, baseline_name), (proposed_emb, pl, pd_, proposed_name)]
    for col, (emb, lbls, doms, title) in enumerate(panels):
        for i, name in enumerate(label_names):
            mask = lbls == i
            if mask.any():
                axes[0, col].scatter(emb[mask, 0], emb[mask, 1], s=12, alpha=0.75,
                                     color=palette_classes[i % len(palette_classes)], label=name)
        axes[0, col].set_title(f"{title} — by class")
        axes[0, col].legend(markerscale=1.4, fontsize=8, loc="best")
        for i, name in enumerate(domain_names):
            mask = doms == i
            if mask.any():
                axes[1, col].scatter(emb[mask, 0], emb[mask, 1], s=12, alpha=0.75,
                                     color=palette_domains[i % len(palette_domains)], label=name)
        axes[1, col].set_title(f"{title} — by domain")
        axes[1, col].legend(markerscale=1.4, fontsize=8, loc="best")

    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("t-SNE: source/target alignment before vs after", fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_train_history(
    history_df: pd.DataFrame,
    output_path: Path,
    title: str,
    adaptive_events: Optional[Sequence[dict]] = None,
) -> None:
    if history_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    epochs = history_df["epoch"]
    axes[0].plot(epochs, history_df["train_loss"], label="train total", linewidth=2)
    if "train_cls_loss" in history_df.columns:
        axes[0].plot(epochs, history_df["train_cls_loss"], label="train cls", linewidth=1.4, linestyle="--")
    if "train_aux_loss" in history_df.columns:
        axes[0].plot(epochs, history_df["train_aux_loss"], label="train aux", linewidth=1.4, linestyle=":")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].set_title(f"{title} — losses")
    axes[0].grid(True, linestyle="--", alpha=0.35); axes[0].legend(fontsize=8)

    if "val_accuracy" in history_df.columns:
        axes[1].plot(epochs, history_df["val_accuracy"], label="val accuracy", linewidth=2, color="tab:green")
    if "val_macro_f1" in history_df.columns:
        axes[1].plot(epochs, history_df["val_macro_f1"], label="val macro-F1", linewidth=1.4, linestyle="--", color="tab:olive")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score"); axes[1].set_title(f"{title} — validation")
    axes[1].grid(True, linestyle="--", alpha=0.35); axes[1].legend(fontsize=8)

    if adaptive_events:
        for event in adaptive_events:
            for ax in axes:
                ax.axvline(x=float(event.get("epoch", 0)), color="tab:red", linestyle="--", alpha=0.6, linewidth=1.2)
            label = str(event.get("label", "adaptive"))
            axes[0].text(float(event.get("epoch", 0)), axes[0].get_ylim()[1] * 0.95, label,
                         color="tab:red", rotation=90, fontsize=8, va="top", ha="right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_bar(summary_df: pd.DataFrame, output_path: Path, title: str = "Ablation") -> None:
    if summary_df.empty:
        return
    df = summary_df.sort_values("accuracy", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.6 * len(df))))
    bars = ax.barh(df["variant"].astype(str), df["accuracy"].astype(float),
                   color=sns.color_palette("crest", n_colors=len(df)))
    for bar, value in zip(bars, df["accuracy"].astype(float)):
        ax.text(value + 0.005, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center", fontsize=8)
    ax.set_xlabel("Accuracy"); ax.set_title(title)
    ax.set_xlim(0.0, max(1.0, df["accuracy"].max() * 1.1))
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_f1_heatmap(f1_table: pd.DataFrame, output_path: Path, title: str) -> None:
    if f1_table.empty:
        return
    fig, ax = plt.subplots(figsize=(1.5 + 1.2 * f1_table.shape[1], 0.8 + 0.5 * f1_table.shape[0]))
    sns.heatmap(f1_table, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0.0, vmax=1.0,
                cbar_kws={"label": "F1"}, ax=ax)
    ax.set_title(title); ax.set_xlabel("Class"); ax.set_ylabel("Method")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_method_accuracy_with_std(summary_df: pd.DataFrame, output_path: Path, title: str) -> None:
    if summary_df.empty:
        return
    tasks = sorted(summary_df["task"].unique().tolist())
    methods = summary_df["method"].drop_duplicates().tolist()
    fig, ax = plt.subplots(figsize=(max(8, 1.5 + 1.6 * len(tasks)), 5))
    bar_width = 0.8 / max(len(methods), 1)
    palette = sns.color_palette("tab10", n_colors=len(methods))
    for mi, method in enumerate(methods):
        rows = summary_df[summary_df["method"] == method].set_index("task").reindex(tasks)
        means = rows["accuracy_mean"].to_numpy()
        stds = rows["accuracy_std"].fillna(0.0).to_numpy()
        positions = np.arange(len(tasks)) + mi * bar_width
        ax.bar(positions, means, width=bar_width, yerr=stds, capsize=3,
               label=method, color=palette[mi % len(palette)])
    ax.set_xticks(np.arange(len(tasks)) + (len(methods) - 1) * bar_width / 2)
    ax.set_xticklabels(tasks, rotation=15)
    ax.set_ylabel("Accuracy (mean ± std)"); ax.set_title(title)
    ax.set_ylim(0.0, 1.0); ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=9, ncol=min(3, len(methods)))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_noise_curves_with_std(noise_df: pd.DataFrame, output_path: Path, title: str) -> None:
    if noise_df.empty:
        return
    tasks = sorted(noise_df["task"].unique().tolist())
    fig, axes = plt.subplots(1, len(tasks) + 1, figsize=(5.0 * (len(tasks) + 1), 4.5), sharey=True)
    palette = sns.color_palette("tab10", n_colors=noise_df["method"].nunique())
    method_to_color = {m: palette[i] for i, m in enumerate(sorted(noise_df["method"].unique()))}

    panel_data = [(t, noise_df[noise_df["task"] == t]) for t in tasks]
    avg = (noise_df.groupby(["method", "snr_db"], as_index=False)
           .agg(accuracy_mean=("accuracy_mean", "mean"), accuracy_std=("accuracy_std", "mean")))
    panel_data.append(("avg over tasks", avg))

    for axis, (panel_title, rows) in zip(axes, panel_data):
        for method, mrows in rows.groupby("method"):
            ordered = mrows.sort_values("snr_db", ascending=False)
            x = ordered["snr_db"].to_numpy()
            y = ordered["accuracy_mean"].to_numpy()
            err = ordered["accuracy_std"].fillna(0.0).to_numpy() if "accuracy_std" in ordered.columns else np.zeros_like(y)
            color = method_to_color.get(method, "gray")
            axis.plot(x, y, marker="o", linewidth=2, label=method, color=color)
            axis.fill_between(x, y - err, y + err, alpha=0.15, color=color)
        axis.set_title(panel_title); axis.set_xlabel("SNR (dB)"); axis.set_ylabel("Accuracy")
        axis.grid(True, linestyle="--", alpha=0.35); axis.legend(fontsize=8)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
