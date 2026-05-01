from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "pu" / "ntu"
EXPECTED_DATA_FILES = [
    f"{split}_{domain}.pt"
    for split in ("train", "val", "test")
    for domain in ("a", "b", "c", "d")
]
REQUIRED_MODULES = [
    "torch",
    "numpy",
    "pandas",
    "yaml",
    "sklearn",
    "scipy",
    "matplotlib",
    "seaborn",
    "tqdm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the local runtime and dataset layout.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--check-data", action="store_true", help="Check whether the 12 PU .pt files exist.")
    parser.add_argument("--strict", action="store_true", help="Exit with non-zero status when any check fails.")
    return parser.parse_args()


def check_modules() -> tuple[dict[str, str], list[str]]:
    versions: dict[str, str] = {}
    missing: list[str] = []
    for module_name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            missing.append(module_name)
            continue
        versions[module_name] = getattr(module, "__version__", "unknown")
    return versions, missing


def check_dataset(data_root: Path) -> list[str]:
    missing = []
    for file_name in EXPECTED_DATA_FILES:
        if not (data_root / file_name).exists():
            missing.append(file_name)
    return missing


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    versions, missing_modules = check_modules()

    print(f"python: {sys.version.split()[0]}")
    for module_name in REQUIRED_MODULES:
        if module_name in versions:
            print(f"{module_name}: {versions[module_name]}")
        else:
            print(f"{module_name}: MISSING")

    torch_module = None
    if "torch" in versions:
        torch_module = importlib.import_module("torch")
        print(f"cuda_available: {torch_module.cuda.is_available()}")
        if torch_module.cuda.is_available():
            print(f"cuda_device: {torch_module.cuda.get_device_name(0)}")

    dataset_missing: list[str] = []
    if args.check_data:
        dataset_missing = check_dataset(data_root)
        print(f"data_root: {data_root}")
        if dataset_missing:
            print(f"dataset_status: missing {len(dataset_missing)} files")
            for file_name in dataset_missing:
                print(f"  - {file_name}")
        else:
            print("dataset_status: complete")

    failed = bool(missing_modules) or bool(dataset_missing)
    if missing_modules:
        print("missing_modules:", ", ".join(missing_modules))
    if args.strict and failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
