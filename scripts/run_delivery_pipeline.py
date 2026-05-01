from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_DOC = PROJECT_ROOT / "docs" / "final_delivery.md"
EXPECTED_DATA_FILES = [
    f"{split}_{domain}.pt"
    for split in ("train", "val", "test")
    for domain in ("a", "b", "c", "d")
]
RUN_DIRS = {
    "benchmark": "pu_adaptive_sdae",
    "speed_ablation": "pu_speed_ablation",
    "disc_sweep": "pu_disc_sweep",
    "full_matrix": "pu_full_matrix",
    "multisource": "pu_multisource",
    "module_ablation": "pu_module_ablation",
}
EXPERIMENT_STEPS = set(RUN_DIRS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-command reproduction and delivery export pipeline.")
    parser.add_argument("--config", default="configs/pu_adaptive_sdae.yaml")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT),
                        help="Root directory for all experiment outputs.")
    parser.add_argument("--doc", default=str(DEFAULT_DOC),
                        help="Markdown report to refresh after the experiments finish.")
    parser.add_argument("--data-root", default="data/pu/ntu",
                        help="Dataset directory containing train/val/test × a/b/c/d .pt files.")
    parser.add_argument("--download-data", action="store_true",
                        help="Download the PU dataset automatically if it is missing.")
    parser.add_argument("--export-dir", default=None,
                        help="Assemble a delivery-ready directory after reproduction completes.")
    parser.add_argument("--zip-export", action="store_true",
                        help="Zip the exported delivery directory.")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=[
            "benchmark",
            "speed_ablation",
            "disc_sweep",
            "full_matrix",
            "multisource",
            "module_ablation",
            "refresh",
            "export",
        ],
        default=[
            "benchmark",
            "speed_ablation",
            "disc_sweep",
            "full_matrix",
            "multisource",
            "module_ablation",
            "refresh",
        ],
        help="Subset of pipeline steps to run.",
    )
    return parser.parse_args()


def resolve_local_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def dataset_missing(data_root: Path) -> list[str]:
    return [name for name in EXPECTED_DATA_FILES if not (data_root / name).exists()]


def ensure_dataset(data_root: Path, auto_download: bool) -> None:
    missing = dataset_missing(data_root)
    if not missing:
        return
    if auto_download:
        run([sys.executable, "scripts/download_pu_ntu.py", "--data-root", str(data_root)])
        missing = dataset_missing(data_root)
    if missing:
        raise FileNotFoundError(
            "PU dataset is incomplete under "
            f"{data_root}. Missing files: {', '.join(missing)}"
        )


def main() -> None:
    args = parse_args()
    if args.export_dir and "export" not in args.steps:
        args.steps = [*args.steps, "export"]
    output_root = resolve_local_path(args.output_root)
    doc_path = resolve_local_path(args.doc)
    data_root = resolve_local_path(args.data_root)

    if any(step in EXPERIMENT_STEPS for step in args.steps):
        run([sys.executable, "scripts/check_runtime.py", "--data-root", str(data_root)])
        ensure_dataset(data_root, auto_download=args.download_data)
        run([sys.executable, "scripts/check_runtime.py", "--strict", "--check-data", "--data-root", str(data_root)])

    base_cmd = [sys.executable]

    if "benchmark" in args.steps:
        run(
            base_cmd
            + [
                "scripts/run_pu_benchmark.py",
                "--config",
                args.config,
                "--output-dir",
                str(output_root / RUN_DIRS["benchmark"]),
                "--data-root",
                str(data_root),
            ]
        )

    if "speed_ablation" in args.steps:
        run(
            base_cmd
            + [
                "scripts/run_speed_ablation.py",
                "--config",
                args.config,
                "--output-dir",
                str(output_root / RUN_DIRS["speed_ablation"]),
                "--data-root",
                str(data_root),
            ]
        )

    if "disc_sweep" in args.steps:
        run(
            base_cmd
            + [
                "scripts/run_disc_sweep.py",
                "--config",
                args.config,
                "--output-dir",
                str(output_root / RUN_DIRS["disc_sweep"]),
                "--data-root",
                str(data_root),
            ]
        )

    if "full_matrix" in args.steps:
        run(
            base_cmd
            + [
                "scripts/run_pu_full_matrix.py",
                "--config",
                args.config,
                "--output-dir",
                str(output_root / RUN_DIRS["full_matrix"]),
                "--data-root",
                str(data_root),
                "--seeds",
                "42",
            ]
        )

    if "multisource" in args.steps:
        run(
            base_cmd
            + [
                "scripts/run_pu_multisource.py",
                "--config",
                args.config,
                "--output-dir",
                str(output_root / RUN_DIRS["multisource"]),
                "--data-root",
                str(data_root),
                "--mode",
                "leave-one-out",
                "--seeds",
                "42",
            ]
        )

    if "module_ablation" in args.steps:
        run(
            base_cmd
            + [
                "scripts/run_module_ablation.py",
                "--config",
                args.config,
                "--output-dir",
                str(output_root / RUN_DIRS["module_ablation"]),
                "--data-root",
                str(data_root),
            ]
        )

    if "refresh" in args.steps:
        run(
            base_cmd
            + [
                "scripts/refresh_final_delivery.py",
                "--output-root",
                str(output_root),
                "--doc",
                str(doc_path),
            ]
        )

    if "export" in args.steps:
        if not args.export_dir:
            raise ValueError("--export-dir is required when 'export' is included in --steps.")
        cmd = base_cmd + [
            "scripts/export_delivery_package.py",
            "--output-root",
            str(output_root),
            "--doc",
            str(doc_path),
            "--dest",
            str(resolve_local_path(args.export_dir)),
            "--overwrite",
        ]
        if args.zip_export:
            cmd.append("--zip")
        run(cmd)


if __name__ == "__main__":
    main()
