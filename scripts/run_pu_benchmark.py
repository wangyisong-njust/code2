"""PU cross-condition benchmark with multi-seed reporting.

Pipeline (per task):
    1. Build PU bundle for source -> target.
    2. For seed in seeds:
        - Run adaptive structure search for the proposed method (seed 0 only by default).
        - Train each method, evaluate on clean target, evaluate under SNR sweeps.
        - For seed 0, persist features for t-SNE alignment plots.
    3. Aggregate (mean ± std) across seeds; write all CSVs and figures.

CSV layout under outputs/<run>/results/:
    summary.csv               # per (task, method, seed) row, raw results
    summary_agg.csv           # mean/std aggregation
    average_accuracy.csv      # mean ± std cross-task average per method
    noise_summary.csv         # per (task, method, snr, seed) raw
    noise_summary_agg.csv     # mean/std aggregation
    per_class_f1.csv          # per (task, method, class) per-class F1 (seed-mean)
    adaptive_search.csv       # search records
    task_metadata.csv         # task descriptors

Figures under outputs/<run>/figures/:
    task_accuracy.png         # bar chart with std error bars across tasks
    noise_accuracy.png        # per-task + averaged noise curves with std band
    <task>/<method>_confusion.png
    <task>/proposed_embedding.png
    <task>/tsne_alignment.png # baseline (ERM) vs proposed source/target alignment
    <task>/train_history.png  # proposed training curves
    <task>/per_class_f1.png   # per-class F1 heatmap
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from faultdg.adaptive import select_hidden_dims
from faultdg.config import (
    apply_runtime_overrides,
    ensure_dir,
    load_config,
    resolve_path,
    save_resolved_config,
)
from faultdg.data import make_loader
from faultdg.models import build_model
from faultdg.pu_data import (
    PU_SETTING_TO_DESC,
    build_pu_task_bundle,
    compute_class_weights,
)
from faultdg.trainer import evaluate_model, seed_everything, train_method
from faultdg.visuals import (
    plot_confusion,
    plot_embedding,
    plot_method_accuracy_with_std,
    plot_noise_curves_with_std,
    plot_per_class_f1_heatmap,
    plot_train_history,
    plot_tsne_alignment,
)


DISPLAY_NAMES = {
    "erm": "ERM-SDAE",
    "mixup": "Mixup-SDAE",
    "coral": "CORAL-SDAE",
    "mmd": "MMD-SDAE",
    "dann": "DANN-SDAE",
    "proposed": "Adaptive-SDAE-3M",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PU cross-condition benchmark with multi-seed reporting.")
    parser.add_argument("--config", default="configs/pu_adaptive_sdae.yaml")
    parser.add_argument("--output-dir", default=None,
                        help="Override experiment.output_dir from the config.")
    parser.add_argument("--data-root", default=None,
                        help="Override data.root from the config.")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Override the seeds list from the config.")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Subset of task names to run.")
    return parser.parse_args()


def write_env_file(output_root: Path) -> None:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (output_root / "env.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


def per_class_f1_from_predictions(predictions_df: pd.DataFrame, label_names: list) -> dict:
    name_to_index = {name: i for i, name in enumerate(label_names)}
    y_true = predictions_df["label_true"].map(name_to_index).to_numpy()
    y_pred = predictions_df["label_pred"].map(name_to_index).to_numpy()
    f1s = f1_score(y_true, y_pred, labels=list(range(len(label_names))), average=None, zero_division=0.0)
    return {label_names[i]: float(f1s[i]) for i in range(len(label_names))}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    apply_runtime_overrides(config, data_root=args.data_root, output_dir=args.output_dir)
    methods = args.methods or list(config["experiment"]["methods"])
    seeds = args.seeds or list(config["experiment"].get("seeds", [int(config["training"]["seed"])]))
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    output_root = resolve_path(config, config["experiment"]["output_dir"])
    checkpoints_dir = ensure_dir(output_root / "checkpoints")
    results_dir = ensure_dir(output_root / "results")
    figures_dir = ensure_dir(output_root / "figures")
    histories_dir = ensure_dir(output_root / "histories")
    predictions_dir = ensure_dir(output_root / "predictions")
    search_dir = ensure_dir(output_root / "adaptive_search")
    save_resolved_config(config, output_root / "resolved_config.json")
    write_env_file(output_root)

    data_root = resolve_path(config, config["data"]["root"])
    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["data"].get("num_workers", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device} torch={torch.__version__} seeds={seeds} methods={methods}")

    snr_levels = list(config["evaluation"].get("snr_db", []))
    embedding_method = str(config["evaluation"].get("embedding_method", "tsne"))
    embedding_max_points = int(config["evaluation"].get("embedding_max_points", 1500))

    summary_rows: list[dict[str, object]] = []
    noise_rows: list[dict[str, object]] = []
    search_rows: list[dict[str, object]] = []
    task_meta_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []

    requested_tasks = set(args.tasks) if args.tasks else None

    for task_cfg in config["experiment"]["tasks"]:
        task_name = str(task_cfg["name"])
        if requested_tasks is not None and task_name not in requested_tasks:
            continue
        source_domain = str(task_cfg["source_domain"])
        target_domain = str(task_cfg["target_domain"])
        bundle = build_pu_task_bundle(
            root=data_root,
            source_domain_key=source_domain,
            target_domain_key=target_domain,
            normalize=str(config["data"].get("normalize", "zscore")),
        )

        label_names = bundle.label_names
        domain_names = bundle.domain_names
        class_weights = compute_class_weights(bundle.source_train.labels, num_classes=len(label_names))

        source_rpm = float(bundle.source_metadata["rpm"])
        target_rpm = float(bundle.target_metadata["rpm"])
        task_speed_scale = source_rpm / target_rpm if target_rpm > 0 else 1.0
        if abs(task_speed_scale - 1.0) < 1e-4:
            task_speed_scale = 1.0

        task_meta = {
            "task": task_name,
            "source_domain": bundle.source_setting,
            "target_domain": bundle.target_setting,
            "source_desc": PU_SETTING_TO_DESC[bundle.source_setting],
            "target_desc": PU_SETTING_TO_DESC[bundle.target_setting],
            "source_rpm": source_rpm,
            "target_rpm": target_rpm,
            "speed_scale": task_speed_scale,
            "source_train_samples": len(bundle.source_train),
            "source_val_samples": len(bundle.source_val),
            "target_train_samples": len(bundle.target_train),
            "target_test_samples": len(bundle.target_test),
        }
        for label_index, label_name in enumerate(label_names):
            task_meta[f"source_train_{label_name}"] = int((bundle.source_train.labels == label_index).sum().item())
            task_meta[f"source_val_{label_name}"] = int((bundle.source_val.labels == label_index).sum().item())
            task_meta[f"target_train_{label_name}"] = int((bundle.target_train.labels == label_index).sum().item())
            task_meta[f"target_test_{label_name}"] = int((bundle.target_test.labels == label_index).sum().item())
        print(json.dumps(task_meta, ensure_ascii=False))
        task_meta_rows.append(task_meta)

        # Cache for tsne_alignment plot (collected on the first seed only).
        first_seed_features: dict[str, dict[str, np.ndarray]] = {}
        # The structure search runs once on seed 0 and the result is reused across
        # subsequent seeds to keep compute bounded.
        selected_hidden_dims: list = list(config["model"]["hidden_dims"])

        for seed_index, seed in enumerate(seeds):
            seed = int(seed)
            seed_everything(seed)

            task_config = copy.deepcopy(config)
            task_method_params = task_config["training"].setdefault("method_params", {})
            task_method_params["speed_scale"] = task_speed_scale
            task_config["training"]["seed"] = seed

            source_train_loader = make_loader(bundle.source_train, batch_size, True, num_workers, drop_last=True)
            source_val_loader = make_loader(bundle.source_val, batch_size, False, num_workers, drop_last=False)
            target_train_loader = make_loader(bundle.target_train, batch_size, True, num_workers, drop_last=True)
            target_test_loader = make_loader(bundle.target_test, batch_size, False, num_workers, drop_last=False)
            # Run structural search once per task (use first seed). For other seeds,
            # reuse the selected structure to keep multi-seed compute bounded.
            if "proposed" in methods and seed_index == 0:
                selected_hidden_dims, search_df = select_hidden_dims(
                    config=task_config,
                    input_dim=bundle.input_dim,
                    stats_dim=bundle.stats_dim,
                    num_classes=len(label_names),
                    source_train_loader=source_train_loader,
                    source_val_loader=source_val_loader,
                    target_train_loader=target_train_loader,
                    device=device,
                    checkpoint_dir=search_dir / task_name,
                    label_names=label_names,
                    domain_names=domain_names,
                    class_weights=None,
                )
                if not search_df.empty:
                    search_df["task"] = task_name
                    search_df["source_domain"] = bundle.source_setting
                    search_df["target_domain"] = bundle.target_setting
                    search_df["seed"] = seed
                    search_rows.extend(search_df.to_dict(orient="records"))

            for method in methods:
                run_config = copy.deepcopy(task_config)
                if method == "proposed":
                    run_config["model"]["hidden_dims"] = selected_hidden_dims
                    method_class_weights = None
                else:
                    method_class_weights = class_weights

                model = build_model(
                    run_config,
                    input_dim=bundle.input_dim,
                    stats_dim=bundle.stats_dim,
                    num_classes=len(label_names),
                ).to(device)

                ensure_dir(checkpoints_dir / task_name / method)
                checkpoint_path = checkpoints_dir / task_name / method / f"seed_{seed}.pt"
                train_result = train_method(
                    method=method,
                    model=model,
                    source_train_loader=source_train_loader,
                    source_val_loader=source_val_loader,
                    target_train_loader=target_train_loader,
                    config=run_config,
                    device=device,
                    checkpoint_path=checkpoint_path,
                    label_names=label_names,
                    domain_names=domain_names,
                    class_weights=method_class_weights,
                )

                history_df = pd.DataFrame(train_result.history)
                history_dir = ensure_dir(histories_dir / task_name / method)
                history_df.to_csv(history_dir / f"seed_{seed}.csv", index=False)

                # For the proposed method on the first seed, draw the training curve.
                if method == "proposed" and seed_index == 0:
                    plot_train_history(
                        history_df=history_df,
                        output_path=figures_dir / task_name / "train_history.png",
                        title=f"{DISPLAY_NAMES[method]} ({task_name}, seed={seed})",
                    )

                clean_eval = evaluate_model(
                    model=model,
                    data_loader=target_test_loader,
                    device=device,
                    label_names=label_names,
                    domain_names=domain_names,
                    use_stats=(method == "proposed"),
                    noise_db=None,
                    collect_features=(seed_index == 0 and method in {"erm", "proposed"}),
                )
                pred_dir = ensure_dir(predictions_dir / task_name / method)
                clean_eval.predictions.to_csv(pred_dir / f"seed_{seed}_clean.csv", index=False)

                # Confusion matrix only for the first seed to keep figures clean.
                if seed_index == 0:
                    plot_confusion(
                        clean_eval.confusion,
                        label_names=label_names,
                        output_path=figures_dir / task_name / f"{method}_confusion.png",
                        title=f"{DISPLAY_NAMES[method]} - {task_name}",
                    )

                f1_breakdown = per_class_f1_from_predictions(clean_eval.predictions, label_names)
                summary_rows.append(
                    {
                        "task": task_name,
                        "method": DISPLAY_NAMES[method],
                        "raw_method": method,
                        "seed": seed,
                        "accuracy": clean_eval.metrics["accuracy"],
                        "macro_f1": clean_eval.metrics["macro_f1"],
                        "best_epoch": train_result.best_epoch,
                        "source_domain": bundle.source_setting,
                        "target_domain": bundle.target_setting,
                        "hidden_dims": "-".join(map(str, run_config["model"]["hidden_dims"])),
                    }
                )
                for label_name, f1_value in f1_breakdown.items():
                    per_class_rows.append(
                        {
                            "task": task_name,
                            "method": DISPLAY_NAMES[method],
                            "raw_method": method,
                            "seed": seed,
                            "class": label_name,
                            "f1": float(f1_value),
                        }
                    )

                # Stash features for the t-SNE alignment plot.
                if seed_index == 0 and clean_eval.features is not None:
                    first_seed_features[method] = {
                        "features": clean_eval.features,
                        "labels": clean_eval.labels,
                        "domains": clean_eval.domains,
                    }

                if method == "proposed" and seed_index == 0:
                    plot_embedding(
                        features=clean_eval.features,
                        labels=clean_eval.labels,
                        domains=clean_eval.domains,
                        label_names=label_names,
                        domain_names=domain_names,
                        output_path=figures_dir / task_name / "proposed_embedding.png",
                        method=embedding_method,
                        max_points=embedding_max_points,
                        seed=seed,
                    )

                for snr_db in snr_levels:
                    noise_eval = evaluate_model(
                        model=model,
                        data_loader=target_test_loader,
                        device=device,
                        label_names=label_names,
                        domain_names=domain_names,
                        use_stats=(method == "proposed"),
                        noise_db=float(snr_db),
                        collect_features=False,
                    )
                    if seed_index == 0:
                        noise_eval.predictions.to_csv(
                            pred_dir / f"seed_{seed}_snr_{float(snr_db):+.1f}dB.csv",
                            index=False,
                        )
                    noise_rows.append(
                        {
                            "task": task_name,
                            "method": DISPLAY_NAMES[method],
                            "raw_method": method,
                            "seed": seed,
                            "snr_db": float(snr_db),
                            "accuracy": noise_eval.metrics["accuracy"],
                        }
                    )

        # Per-task t-SNE alignment plot: baseline (ERM) vs proposed.
        if "erm" in first_seed_features and "proposed" in first_seed_features:
            baseline = first_seed_features["erm"]
            proposed = first_seed_features["proposed"]
            plot_tsne_alignment(
                baseline_features=baseline["features"],
                baseline_labels=baseline["labels"],
                baseline_domains=baseline["domains"],
                proposed_features=proposed["features"],
                proposed_labels=proposed["labels"],
                proposed_domains=proposed["domains"],
                label_names=label_names,
                domain_names=domain_names,
                output_path=figures_dir / task_name / "tsne_alignment.png",
                seed=int(seeds[0]),
                max_points=embedding_max_points,
            )

    # ---------- aggregation ----------
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_dir / "summary.csv", index=False)
    summary_agg = (
        summary_df.groupby(["task", "method", "raw_method"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            n_seeds=("seed", "nunique"),
        )
    )
    summary_agg["accuracy_std"] = summary_agg["accuracy_std"].fillna(0.0)
    summary_agg["macro_f1_std"] = summary_agg["macro_f1_std"].fillna(0.0)
    summary_agg.to_csv(results_dir / "summary_agg.csv", index=False)

    average_df = (
        summary_agg.groupby(["method", "raw_method"], as_index=False)
        .agg(
            accuracy_mean=("accuracy_mean", "mean"),
            accuracy_std=("accuracy_mean", "std"),
        )
        .sort_values("accuracy_mean", ascending=False)
    )
    average_df["accuracy_std"] = average_df["accuracy_std"].fillna(0.0)
    average_df.to_csv(results_dir / "average_accuracy.csv", index=False)

    noise_df = pd.DataFrame(noise_rows)
    noise_df.to_csv(results_dir / "noise_summary.csv", index=False)
    noise_agg = (
        noise_df.groupby(["task", "method", "raw_method", "snr_db"], as_index=False)
        .agg(accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"), n_seeds=("seed", "nunique"))
    )
    noise_agg["accuracy_std"] = noise_agg["accuracy_std"].fillna(0.0)
    noise_agg.to_csv(results_dir / "noise_summary_agg.csv", index=False)

    per_class_df = pd.DataFrame(per_class_rows)
    per_class_df.to_csv(results_dir / "per_class_f1.csv", index=False)
    per_class_agg = (
        per_class_df.groupby(["task", "method", "class"], as_index=False)["f1"].mean()
    )
    per_class_agg.to_csv(results_dir / "per_class_f1_agg.csv", index=False)

    search_df = pd.DataFrame(search_rows)
    if not search_df.empty:
        search_df.to_csv(results_dir / "adaptive_search.csv", index=False)

    pd.DataFrame(task_meta_rows).to_csv(results_dir / "task_metadata.csv", index=False)

    # ---------- figures ----------
    plot_method_accuracy_with_std(summary_agg, figures_dir / "task_accuracy.png", title="PU cross-condition accuracy (mean ± std)")
    plot_noise_curves_with_std(noise_agg, figures_dir / "noise_accuracy.png", title="AWGN robustness (mean ± std)")

    for task_name, task_rows in per_class_agg.groupby("task"):
        pivot = task_rows.pivot(index="method", columns="class", values="f1")
        pivot = pivot[label_names]
        plot_per_class_f1_heatmap(
            pivot,
            output_path=figures_dir / task_name / "per_class_f1.png",
            title=f"Per-class F1 — {task_name}",
        )

    print("\nAggregated task accuracy:")
    print(summary_agg.to_string(index=False))
    print("\nAverage accuracy across tasks:")
    print(average_df.to_string(index=False))


if __name__ == "__main__":
    main()
