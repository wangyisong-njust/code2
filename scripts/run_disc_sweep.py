"""Grid sweep over the Module-2 discriminative loss hyperparameters.

For each (margin, weight) combination, trains the proposed method on the
``speed_shift`` task (the hardest cross-condition setting) and reports clean
target accuracy, noise robustness and the best-epoch validation score.

Outputs land under ``outputs/pu_disc_sweep/``:

    results/sweep.csv             # one row per (margin, weight, seed)
    results/sweep_agg.csv         # mean ± std across seeds
    results/best_config.json      # best (margin, weight) by mean accuracy
    figures/sweep_heatmap.png     # mean accuracy heatmap
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from faultdg.config import ensure_dir, load_config, resolve_path, save_resolved_config
from faultdg.data import make_loader
from faultdg.models import build_model
from faultdg.pu_data import build_pu_task_bundle
from faultdg.trainer import evaluate_model, seed_everything, train_method


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Module-2 discriminative loss grid sweep on speed_shift.")
    parser.add_argument("--config", default="configs/pu_adaptive_sdae.yaml")
    parser.add_argument("--output-dir", default="outputs/pu_disc_sweep")
    parser.add_argument("--margins", nargs="+", type=float, default=[4.0, 8.0, 12.0, 16.0])
    parser.add_argument("--weights", nargs="+", type=float, default=[0.005, 0.02, 0.05, 0.1])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 7])
    parser.add_argument("--source", default="d")
    parser.add_argument("--target", default="a")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    output_root = resolve_path(config, args.output_dir)
    results_dir = ensure_dir(output_root / "results")
    figures_dir = ensure_dir(output_root / "figures")
    save_resolved_config(config, output_root / "resolved_config.json")

    data_root = resolve_path(config, config["data"]["root"])
    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["data"].get("num_workers", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle = build_pu_task_bundle(
        root=data_root,
        source_domain_key=args.source,
        target_domain_key=args.target,
        normalize=str(config["data"].get("normalize", "zscore")),
    )
    label_names = bundle.label_names
    domain_names = bundle.domain_names
    source_rpm = float(bundle.source_metadata["rpm"])
    target_rpm = float(bundle.target_metadata["rpm"])
    speed_scale = source_rpm / target_rpm if target_rpm > 0 else 1.0

    rows: list[dict[str, float]] = []
    for margin, weight, seed in product(args.margins, args.weights, args.seeds):
        seed_everything(int(seed))
        run_config = copy.deepcopy(config)
        params = run_config["training"].setdefault("method_params", {})
        params["discriminative_margin"] = float(margin)
        params["discriminative_weight"] = float(weight)
        params["speed_scale"] = float(speed_scale)
        run_config["training"]["seed"] = int(seed)
        # Disable adaptive structure search to keep the sweep cheap and isolate Module 2.
        run_config["adaptive_search"] = {"enabled": False}

        source_train = make_loader(bundle.source_train, batch_size, True, num_workers, drop_last=True)
        source_val = make_loader(bundle.source_val, batch_size, False, num_workers, drop_last=False)
        target_train = make_loader(bundle.target_train, batch_size, True, num_workers, drop_last=True)
        target_test = make_loader(bundle.target_test, batch_size, False, num_workers, drop_last=False)

        model = build_model(
            run_config,
            input_dim=bundle.input_dim,
            stats_dim=bundle.stats_dim,
            num_classes=len(label_names),
        ).to(device)
        ckpt = ensure_dir(output_root / "checkpoints") / f"m{margin}_w{weight}_s{seed}.pt"
        train_method(
            method="proposed",
            model=model,
            source_train_loader=source_train,
            source_val_loader=source_val,
            target_train_loader=target_train,
            config=run_config,
            device=device,
            checkpoint_path=ckpt,
            label_names=label_names,
            domain_names=domain_names,
            class_weights=None,
        )
        clean_eval = evaluate_model(
            model=model,
            data_loader=target_test,
            device=device,
            label_names=label_names,
            domain_names=domain_names,
            use_stats=True,
            noise_db=None,
            collect_features=False,
        )
        rows.append(
            {
                "margin": float(margin),
                "weight": float(weight),
                "seed": int(seed),
                "accuracy": float(clean_eval.metrics["accuracy"]),
                "macro_f1": float(clean_eval.metrics["macro_f1"]),
            }
        )
        print(f"[sweep] margin={margin:>6.2f} weight={weight:>6.4f} seed={seed} acc={clean_eval.metrics['accuracy']:.4f}")

    sweep_df = pd.DataFrame(rows)
    sweep_df.to_csv(results_dir / "sweep.csv", index=False)
    agg = (
        sweep_df.groupby(["margin", "weight"], as_index=False)
        .agg(accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"))
    )
    agg["accuracy_std"] = agg["accuracy_std"].fillna(0.0)
    agg.to_csv(results_dir / "sweep_agg.csv", index=False)

    best_row = agg.sort_values("accuracy_mean", ascending=False).iloc[0]
    best_payload = {
        "best_margin": float(best_row["margin"]),
        "best_weight": float(best_row["weight"]),
        "best_accuracy_mean": float(best_row["accuracy_mean"]),
        "best_accuracy_std": float(best_row["accuracy_std"]),
        "task": f"{args.source}->{args.target}",
        "seeds": list(map(int, args.seeds)),
    }
    (results_dir / "best_config.json").write_text(json.dumps(best_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    pivot = agg.pivot(index="margin", columns="weight", values="accuracy_mean")
    fig, ax = plt.subplots(figsize=(1.5 + 1.1 * pivot.shape[1], 1 + 0.7 * pivot.shape[0]))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", cbar_kws={"label": "Accuracy"}, ax=ax)
    ax.set_title(f"Disc loss sweep — {args.source}->{args.target} (mean over {len(args.seeds)} seeds)")
    fig.tight_layout()
    fig.savefig(figures_dir / "sweep_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("\nBest configuration:")
    print(json.dumps(best_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
