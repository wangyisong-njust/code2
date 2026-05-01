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
from faultdg.visuals import plot_ablation_bar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation study for the PU speed-shift task.")
    parser.add_argument("--config", default="configs/pu_adaptive_sdae.yaml")
    parser.add_argument("--output-dir", default="outputs/pu_speed_ablation")
    parser.add_argument("--data-root", default=None,
                        help="Override data.root from the config.")
    return parser.parse_args()


def build_variants(base_config: dict, source_rpm: float, target_rpm: float) -> list[dict[str, object]]:
    moderate_speed_scale = (source_rpm / target_rpm) ** 0.5 if target_rpm > 0 else 1.0
    return [
        {
            "name": "fusion_fixed",
            "adaptive": False,
            "hidden_dims": [512, 256],
            "params": {
                "discriminative_weight": 0.0,
                "speed_aug_weight": 0.0,
                "speed_consistency_weight": 0.0,
                "speed_scale": 1.0,
            },
        },
        {
            "name": "fusion_adaptive",
            "adaptive": True,
            "params": {
                "discriminative_weight": 0.0,
                "speed_aug_weight": 0.0,
                "speed_consistency_weight": 0.0,
                "speed_scale": 1.0,
            },
        },
        {
            "name": "fusion_adaptive_disc",
            "adaptive": True,
            "params": {
                "discriminative_weight": 0.02,
                "discriminative_margin": 8.0,
                "speed_aug_weight": 0.0,
                "speed_consistency_weight": 0.0,
                "speed_scale": 1.0,
            },
        },
        {
            "name": "fusion_adaptive_speed",
            "adaptive": True,
            "params": {
                "discriminative_weight": 0.0,
                "speed_aug_weight": 0.20,
                "speed_consistency_weight": 0.05,
                "speed_scale": moderate_speed_scale,
            },
        },
        {
            "name": "fusion_adaptive_disc_speed",
            "adaptive": True,
            "params": {
                "discriminative_weight": 0.02,
                "discriminative_margin": 8.0,
                "speed_aug_weight": 0.20,
                "speed_consistency_weight": 0.05,
                "speed_scale": moderate_speed_scale,
            },
        },
    ]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    apply_runtime_overrides(config, data_root=args.data_root)
    seed_everything(int(config["training"]["seed"]))
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    output_root = resolve_path(config, args.output_dir)
    checkpoints_dir = ensure_dir(output_root / "checkpoints")
    results_dir = ensure_dir(output_root / "results")
    figures_dir = ensure_dir(output_root / "figures")
    histories_dir = ensure_dir(output_root / "histories")
    search_dir = ensure_dir(output_root / "adaptive_search")
    save_resolved_config(config, output_root / "resolved_config.json")

    data_root = resolve_path(config, config["data"]["root"])
    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["data"].get("num_workers", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle = build_pu_task_bundle(
        root=data_root,
        source_domain_key="d",
        target_domain_key="a",
        normalize=str(config["data"].get("normalize", "zscore")),
    )
    source_rpm = float(bundle.source_metadata["rpm"])
    target_rpm = float(bundle.target_metadata["rpm"])

    source_train_loader = make_loader(bundle.source_train, batch_size, True, num_workers, drop_last=True)
    source_val_loader = make_loader(bundle.source_val, batch_size, False, num_workers, drop_last=False)
    target_train_loader = make_loader(bundle.target_train, batch_size, True, num_workers, drop_last=True)
    target_test_loader = make_loader(bundle.target_test, batch_size, False, num_workers, drop_last=False)

    label_names = bundle.label_names
    domain_names = bundle.domain_names
    variants = build_variants(config, source_rpm, target_rpm)

    summary_rows: list[dict[str, object]] = []
    noise_rows: list[dict[str, object]] = []
    search_rows: list[dict[str, object]] = []

    for variant in variants:
        variant_name = str(variant["name"])
        run_config = copy.deepcopy(config)
        run_config["adaptive_search"]["enabled"] = bool(variant["adaptive"])
        run_config["training"]["method_params"].update(variant["params"])

        if bool(variant["adaptive"]):
            selected_hidden_dims, search_df = select_hidden_dims(
                config=run_config,
                input_dim=bundle.input_dim,
                stats_dim=bundle.stats_dim,
                num_classes=len(label_names),
                source_train_loader=source_train_loader,
                source_val_loader=source_val_loader,
                target_train_loader=target_train_loader,
                device=device,
                checkpoint_dir=search_dir / variant_name,
                label_names=label_names,
                domain_names=domain_names,
                class_weights=None,
            )
            if not search_df.empty:
                search_df["variant"] = variant_name
                search_rows.extend(search_df.to_dict(orient="records"))
            run_config["model"]["hidden_dims"] = selected_hidden_dims
        else:
            run_config["model"]["hidden_dims"] = list(variant["hidden_dims"])

        model = build_model(
            run_config,
            input_dim=bundle.input_dim,
            stats_dim=bundle.stats_dim,
            num_classes=len(label_names),
        ).to(device)
        checkpoint_path = checkpoints_dir / f"{variant_name}.pt"
        train_result = train_method(
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
        history_df = pd.DataFrame(train_result.history)
        history_df.to_csv(histories_dir / f"{variant_name}.csv", index=False)

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
                "variant": variant_name,
                "accuracy": clean_eval.metrics["accuracy"],
                "best_epoch": train_result.best_epoch,
                "hidden_dims": "-".join(map(str, run_config["model"]["hidden_dims"])),
                "speed_scale": float(run_config["training"]["method_params"].get("speed_scale", 1.0)),
                "discriminative_weight": float(run_config["training"]["method_params"].get("discriminative_weight", 0.0)),
                "speed_aug_weight": float(run_config["training"]["method_params"].get("speed_aug_weight", 0.0)),
                "speed_consistency_weight": float(
                    run_config["training"]["method_params"].get("speed_consistency_weight", 0.0)
                ),
            }
        )

        for snr_db in config["evaluation"].get("snr_db", []):
            noise_eval = evaluate_model(
                model=model,
                data_loader=target_test_loader,
                device=device,
                label_names=label_names,
                domain_names=domain_names,
                use_stats=True,
                noise_db=float(snr_db),
                collect_features=False,
            )
            noise_rows.append(
                {
                    "variant": variant_name,
                    "snr_db": float(snr_db),
                    "accuracy": noise_eval.metrics["accuracy"],
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values("accuracy", ascending=False)
    noise_df = pd.DataFrame(noise_rows).sort_values(["variant", "snr_db"], ascending=[True, False])
    search_df = pd.DataFrame(search_rows)

    summary_df.to_csv(results_dir / "summary.csv", index=False)
    noise_df.to_csv(results_dir / "noise_summary.csv", index=False)
    if not search_df.empty:
        search_df.to_csv(results_dir / "adaptive_search.csv", index=False)

    plot_ablation_bar(summary_df, figures_dir / "ablation_bar.png", title="Speed-shift ablation accuracy")

    print(json.dumps({"source_rpm": source_rpm, "target_rpm": target_rpm}, ensure_ascii=False))
    print("\nSpeed-shift ablation summary:")
    print(summary_df.to_string(index=False))
    print("\nNoise summary:")
    print(noise_df.to_string(index=False))


if __name__ == "__main__":
    main()
