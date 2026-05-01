"""Multi-source domain generalization on PU.

Supports two evaluation families:

- leave-one-out: source = the other 3 domains, target = the held-out domain
- pairs: source = any 2 domains, target = one of the remaining 2 domains

Outputs: outputs/pu_multisource/results/{summary.csv,summary_agg.csv,...}
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from faultdg.config import apply_runtime_overrides, ensure_dir, load_config, resolve_path, save_resolved_config
from faultdg.data import make_loader
from faultdg.models import build_model
from faultdg.pu_data import build_pu_multisource_bundle, compute_class_weights
from faultdg.trainer import evaluate_model, seed_everything, train_method
from faultdg.visuals import plot_method_accuracy_with_std


DOMAIN_KEYS = ["a", "b", "c", "d"]

DISPLAY_NAMES = {
    "erm": "ERM-SDAE",
    "mixup": "Mixup-SDAE",
    "coral": "CORAL-SDAE",
    "mmd": "MMD-SDAE",
    "dann": "DANN-SDAE",
    "proposed": "Adaptive-SDAE-3M",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-source DG on PU.")
    parser.add_argument("--config", default="configs/pu_adaptive_sdae.yaml")
    parser.add_argument("--output-dir", default="outputs/pu_multisource")
    parser.add_argument("--data-root", default=None,
                        help="Override data.root from the config.")
    parser.add_argument("--mode", choices=["leave-one-out", "pairs", "both"], default="leave-one-out")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    return parser.parse_args()


def build_tasks(mode: str) -> list:
    tasks = []
    if mode in ("leave-one-out", "both"):
        for tgt in DOMAIN_KEYS:
            sources = [k for k in DOMAIN_KEYS if k != tgt]
            tasks.append({"name": f"{'+'.join(sources)}_to_{tgt}", "sources": sources, "target": tgt})
    if mode in ("pairs", "both"):
        for src_pair in combinations(DOMAIN_KEYS, 2):
            for tgt in DOMAIN_KEYS:
                if tgt in src_pair:
                    continue
                tasks.append({"name": f"{'+'.join(src_pair)}_to_{tgt}", "sources": list(src_pair), "target": tgt})
    return tasks


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    apply_runtime_overrides(config, data_root=args.data_root)
    methods = args.methods or list(config["experiment"]["methods"])
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    output_root = resolve_path(config, args.output_dir)
    results_dir = ensure_dir(output_root / "results")
    figures_dir = ensure_dir(output_root / "figures")
    checkpoints_dir = ensure_dir(output_root / "checkpoints")
    save_resolved_config(config, output_root / "resolved_config.json")
    (output_root / "env.json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    data_root = resolve_path(config, config["data"]["root"])
    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["data"].get("num_workers", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tasks = build_tasks(args.mode)
    print(f"[multisource] mode={args.mode} num_tasks={len(tasks)}")

    rows: list[dict[str, object]] = []
    for task in tasks:
        bundle = build_pu_multisource_bundle(
            root=data_root,
            source_domain_keys=task["sources"],
            target_domain_key=task["target"],
            normalize=str(config["data"].get("normalize", "zscore")),
        )
        label_names = bundle.label_names
        domain_names = bundle.domain_names
        class_weights = compute_class_weights(bundle.source_train.labels, num_classes=len(label_names))

        for seed in args.seeds:
            seed_everything(int(seed))
            run_config = copy.deepcopy(config)
            run_config["training"]["seed"] = int(seed)
            run_config["adaptive_search"]["enabled"] = False  # use fixed structure for fairness across many tasks

            source_train = make_loader(bundle.source_train, batch_size, True, num_workers, drop_last=True)
            source_val = make_loader(bundle.source_val, batch_size, False, num_workers, drop_last=False)
            target_train = make_loader(bundle.target_train, batch_size, True, num_workers, drop_last=True)
            target_test = make_loader(bundle.target_test, batch_size, False, num_workers, drop_last=False)

            for method in methods:
                rc = copy.deepcopy(run_config)
                model = build_model(
                    rc, input_dim=bundle.input_dim, stats_dim=bundle.stats_dim, num_classes=len(label_names)
                ).to(device)
                method_class_weights = None if method == "proposed" else class_weights
                ckpt = ensure_dir(checkpoints_dir / task["name"] / method) / f"seed_{seed}.pt"
                train_method(
                    method=method,
                    model=model,
                    source_train_loader=source_train,
                    source_val_loader=source_val,
                    target_train_loader=target_train,
                    config=rc,
                    device=device,
                    checkpoint_path=ckpt,
                    label_names=label_names,
                    domain_names=domain_names,
                    class_weights=method_class_weights,
                )
                clean_eval = evaluate_model(
                    model=model,
                    data_loader=target_test,
                    device=device,
                    label_names=label_names,
                    domain_names=domain_names,
                    use_stats=(method == "proposed"),
                    noise_db=None,
                    collect_features=False,
                )
                rows.append(
                    {
                        "task": task["name"],
                        "sources": "+".join(task["sources"]),
                        "target": task["target"],
                        "method": DISPLAY_NAMES[method],
                        "raw_method": method,
                        "seed": int(seed),
                        "accuracy": float(clean_eval.metrics["accuracy"]),
                        "macro_f1": float(clean_eval.metrics["macro_f1"]),
                    }
                )
                print(f"[multi] {task['name']:>20s} {method:>10s} seed={seed} acc={clean_eval.metrics['accuracy']:.4f}")

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(results_dir / "summary.csv", index=False)
    summary_agg = (
        summary_df.groupby(["task", "method", "raw_method"], as_index=False)
        .agg(accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"), n_seeds=("seed", "nunique"))
    )
    summary_agg["accuracy_std"] = summary_agg["accuracy_std"].fillna(0.0)
    summary_agg.to_csv(results_dir / "summary_agg.csv", index=False)

    average_df = (
        summary_agg.groupby(["method", "raw_method"], as_index=False)
        .agg(accuracy_mean=("accuracy_mean", "mean"), accuracy_std=("accuracy_mean", "std"))
        .sort_values("accuracy_mean", ascending=False)
    )
    average_df["accuracy_std"] = average_df["accuracy_std"].fillna(0.0)
    average_df.to_csv(results_dir / "average_accuracy.csv", index=False)

    plot_method_accuracy_with_std(
        summary_agg, figures_dir / "task_accuracy.png",
        title=f"PU multi-source DG ({args.mode})",
    )

    print("\nMulti-source DG average accuracy:")
    print(average_df.to_string(index=False))


if __name__ == "__main__":
    main()
