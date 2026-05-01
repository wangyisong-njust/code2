"""Per-module ablation for the three paper-aligned upgrades.

Variants compared on the three default tasks (speed_shift / load_shift / radial_shift)
across the configured seeds:

- ``full``           — experimental "all-upgrades-on" variant used by this ablation
                       (KL sparsity + squared hinge + adaptive μ-balance + oscillation gate).
- ``no_sparsity``    — disable KL sparsity in pretraining.
- ``no_disc_upgrade`` — revert the discriminative loss to plain (linear) hinge,
                       fixed μ1 = μ2 (i.e., neither ``squared`` nor ``adaptive``).
- ``no_oscillation`` — disable trajectory-stability gate in adaptive structure search.
- ``vanilla``        — all three upgrades off (closest to the original SDAE-3M before
                       these contributions). Establishes the contribution baseline.

Each variant is trained as the ``proposed`` method with the matching method_params /
adaptive_search overrides; everything else (SDAE backbone, statistics fusion, base
discriminative formulation, adaptive search) is held fixed.

Outputs under ``outputs/pu_module_ablation/``:
- results/summary.csv         per (variant, task, seed) raw row
- results/summary_agg.csv     mean ± std across seeds
- results/contribution.csv    Δ accuracy of (full − variant) per task
- figures/module_ablation.png bar chart of variant × task means
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from faultdg.adaptive import select_hidden_dims
from faultdg.config import apply_runtime_overrides, ensure_dir, load_config, resolve_path, save_resolved_config
from faultdg.data import make_loader
from faultdg.models import build_model
from faultdg.pu_data import build_pu_task_bundle
from faultdg.trainer import evaluate_model, seed_everything, train_method


FULL_PARAMS = {
    "pretrain_sparsity_weight": 0.001,
    "pretrain_sparsity_rho": 0.05,
    "discriminative_squared": True,
    "discriminative_adaptive": True,
}

FULL_SEARCH = {
    "oscillation_enabled": True,
}


VARIANTS = [
    {
        "name": "full",
        "params": dict(FULL_PARAMS),
        "search": dict(FULL_SEARCH),
    },
    {
        "name": "no_sparsity",
        "params": {**FULL_PARAMS, "pretrain_sparsity_weight": 0.0},
        "search": dict(FULL_SEARCH),
    },
    {
        "name": "no_disc_upgrade",
        "params": {**FULL_PARAMS, "discriminative_squared": False, "discriminative_adaptive": False},
        "search": dict(FULL_SEARCH),
    },
    {
        "name": "squared_only",  # squared hinge ON, adaptive μ-balance OFF
        "params": {**FULL_PARAMS, "discriminative_squared": True, "discriminative_adaptive": False},
        "search": dict(FULL_SEARCH),
    },
    {
        "name": "adaptive_only",  # linear hinge, adaptive μ-balance ON
        "params": {**FULL_PARAMS, "discriminative_squared": False, "discriminative_adaptive": True},
        "search": dict(FULL_SEARCH),
    },
    {
        "name": "no_oscillation",
        "params": dict(FULL_PARAMS),
        "search": {"oscillation_enabled": False},
    },
    {
        "name": "vanilla",
        "params": {
            **FULL_PARAMS,
            "pretrain_sparsity_weight": 0.0,
            "discriminative_squared": False,
            "discriminative_adaptive": False,
        },
        "search": {"oscillation_enabled": False},
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-module ablation of the 3 paper-aligned upgrades.")
    parser.add_argument("--config", default="configs/pu_adaptive_sdae.yaml")
    parser.add_argument("--output-dir", default="outputs/pu_module_ablation")
    parser.add_argument("--data-root", default=None,
                        help="Override data.root from the config.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Subset of variant names to run (default: all).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    apply_runtime_overrides(config, data_root=args.data_root)
    seeds = args.seeds or list(config["experiment"].get("seeds", [42]))
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    output_root = resolve_path(config, args.output_dir)
    checkpoints_dir = ensure_dir(output_root / "checkpoints")
    results_dir = ensure_dir(output_root / "results")
    figures_dir = ensure_dir(output_root / "figures")
    save_resolved_config(config, output_root / "resolved_config.json")

    data_root = resolve_path(config, config["data"]["root"])
    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["data"].get("num_workers", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device} torch={torch.__version__} seeds={seeds}")

    requested_tasks = set(args.tasks) if args.tasks else None
    requested_variants = set(args.variants) if args.variants else None
    variants_to_run = [v for v in VARIANTS if requested_variants is None or v["name"] in requested_variants]

    summary_rows: list[dict[str, object]] = []

    for task_cfg in config["experiment"]["tasks"]:
        task_name = str(task_cfg["name"])
        if requested_tasks is not None and task_name not in requested_tasks:
            continue

        bundle = build_pu_task_bundle(
            root=data_root,
            source_domain_key=str(task_cfg["source_domain"]),
            target_domain_key=str(task_cfg["target_domain"]),
            normalize=str(config["data"].get("normalize", "zscore")),
        )
        label_names = bundle.label_names
        domain_names = bundle.domain_names
        source_rpm = float(bundle.source_metadata["rpm"])
        target_rpm = float(bundle.target_metadata["rpm"])
        task_speed_scale = source_rpm / target_rpm if target_rpm > 0 else 1.0
        if abs(task_speed_scale - 1.0) < 1e-4:
            task_speed_scale = 1.0

        source_train_loader = make_loader(bundle.source_train, batch_size, True, num_workers, drop_last=True)
        source_val_loader = make_loader(bundle.source_val, batch_size, False, num_workers, drop_last=False)
        target_train_loader = make_loader(bundle.target_train, batch_size, True, num_workers, drop_last=True)
        target_test_loader = make_loader(bundle.target_test, batch_size, False, num_workers, drop_last=False)

        for variant in variants_to_run:
            variant_name = str(variant["name"])
            # Run a fresh structure search per (variant, task) since the search
            # acceptance criterion changes when oscillation gating is toggled.
            search_seed = int(seeds[0])
            seed_everything(search_seed)
            search_config = copy.deepcopy(config)
            search_config["training"].setdefault("method_params", {})["speed_scale"] = task_speed_scale
            search_config["training"]["method_params"].update(variant["params"])
            search_config.setdefault("adaptive_search", {}).update(variant["search"])
            search_config["training"]["seed"] = search_seed

            selected_hidden_dims, _ = select_hidden_dims(
                config=search_config,
                input_dim=bundle.input_dim,
                stats_dim=bundle.stats_dim,
                num_classes=len(label_names),
                source_train_loader=source_train_loader,
                source_val_loader=source_val_loader,
                target_train_loader=target_train_loader,
                device=device,
                checkpoint_dir=ensure_dir(output_root / "adaptive_search" / task_name / variant_name),
                label_names=label_names,
                domain_names=domain_names,
                class_weights=None,
            )

            for seed in seeds:
                seed = int(seed)
                seed_everything(seed)
                run_config = copy.deepcopy(search_config)
                run_config["training"]["seed"] = seed
                run_config["model"]["hidden_dims"] = selected_hidden_dims

                model = build_model(
                    run_config,
                    input_dim=bundle.input_dim,
                    stats_dim=bundle.stats_dim,
                    num_classes=len(label_names),
                ).to(device)
                checkpoint_path = ensure_dir(checkpoints_dir / task_name / variant_name) / f"seed_{seed}.pt"
                train_method(
                    method="proposed",
                    model=model,
                    source_train_loader=source_train_loader,
                    source_val_loader=source_val_loader,
                    target_train_loader=target_train_loader,
                    config=run_config,
                    device=device,
                    checkpoint_path=checkpoint_path,
                    label_names=label_names,
                    domain_names=domain_names,
                    class_weights=None,
                )
                clean_eval = evaluate_model(
                    model=model,
                    data_loader=target_test_loader,
                    device=device,
                    label_names=label_names,
                    domain_names=domain_names,
                    use_stats=True,
                    noise_db=None,
                    collect_features=False,
                )
                summary_rows.append(
                    {
                        "task": task_name,
                        "variant": variant_name,
                        "seed": seed,
                        "accuracy": float(clean_eval.metrics["accuracy"]),
                        "macro_f1": float(clean_eval.metrics["macro_f1"]),
                        "hidden_dims": "-".join(map(str, selected_hidden_dims)),
                    }
                )
                print(
                    f"[mod-ablation] task={task_name:>13s} variant={variant_name:>15s} "
                    f"seed={seed} acc={clean_eval.metrics['accuracy']:.4f} "
                    f"f1={clean_eval.metrics['macro_f1']:.4f}"
                )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_dir / "summary.csv", index=False)

    agg = (
        summary_df.groupby(["task", "variant"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            n_seeds=("seed", "nunique"),
        )
    )
    agg["accuracy_std"] = agg["accuracy_std"].fillna(0.0)
    agg["macro_f1_std"] = agg["macro_f1_std"].fillna(0.0)
    agg.to_csv(results_dir / "summary_agg.csv", index=False)

    # Δ table: full − variant per task, plus cross-task mean.
    pivot = agg.pivot(index="task", columns="variant", values="accuracy_mean")
    if "full" in pivot.columns:
        delta_rows = []
        for task in pivot.index:
            for variant in pivot.columns:
                if variant == "full":
                    continue
                delta_rows.append(
                    {
                        "task": task,
                        "variant_removed": variant,
                        "full_acc": float(pivot.at[task, "full"]),
                        "variant_acc": float(pivot.at[task, variant]),
                        "delta": float(pivot.at[task, "full"] - pivot.at[task, variant]),
                    }
                )
        delta_df = pd.DataFrame(delta_rows)
        delta_df.to_csv(results_dir / "contribution.csv", index=False)
        print("\nContribution Δ (full − variant), positive ⇒ removed module hurts:")
        print(delta_df.to_string(index=False))

    # Bar chart.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        tasks = list(pivot.index)
        variants = list(pivot.columns)
        x = np.arange(len(tasks))
        width = 0.8 / max(len(variants), 1)
        fig, ax = plt.subplots(figsize=(max(8, len(tasks) * 2.5), 4.5))
        for i, variant in enumerate(variants):
            means = pivot[variant].values
            std_pivot = agg.pivot(index="task", columns="variant", values="accuracy_std")
            stds = std_pivot[variant].reindex(tasks).values
            ax.bar(x + (i - (len(variants) - 1) / 2.0) * width, means, width,
                   yerr=stds, capsize=3, label=variant)
        ax.set_xticks(x)
        ax.set_xticklabels(tasks)
        ax.set_ylabel("target test accuracy")
        ax.set_title("Module ablation (mean ± std across seeds)")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures_dir / "module_ablation.png", dpi=150)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] failed to plot module_ablation.png: {exc}")

    print("\nAggregated module ablation:")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
