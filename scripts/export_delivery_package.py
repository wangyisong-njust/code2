from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_DEST = PROJECT_ROOT / "deliverables" / "faultdg_delivery"
RESULT_DIRS = [
    "pu_adaptive_sdae",
    "pu_speed_ablation",
    "pu_disc_sweep",
    "pu_full_matrix",
    "pu_multisource",
    "pu_module_ablation",
]
ROOT_FILES = [
    "README.md",
    "environment.yml",
    "pyproject.toml",
    "参考论文思路.pdf",
]
ROOT_DIRS = [
    "configs",
    "docs",
    "scripts",
    "src",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble a delivery-ready package without public dataset files.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT),
                        help="Root directory that contains pu_adaptive_sdae/, pu_disc_sweep/, etc.")
    parser.add_argument("--doc", default="docs/final_delivery.md",
                        help="Final markdown report to include in the delivery package.")
    parser.add_argument("--dest", default=str(DEFAULT_DEST),
                        help="Destination directory for the assembled package.")
    parser.add_argument("--zip", action="store_true",
                        help="Create <dest>.zip after assembling the directory.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Remove the destination directory if it already exists.")
    return parser.parse_args()


def resolve_local_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def copy_tree(src: Path, dest: Path) -> None:
    shutil.copytree(
        src,
        dest,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            "checkpoints",
            "histories",
            "predictions",
            ".git",
            ".faultdg_tmp",
        ),
    )


def build_manifest(output_root: Path, dest: Path) -> dict[str, object]:
    return {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_repo": str(PROJECT_ROOT),
        "output_root": str(output_root),
        "delivery_root": str(dest),
        "result_directories": RESULT_DIRS,
        "root_files": ROOT_FILES,
        "root_dirs": ROOT_DIRS,
        "dataset_included": False,
        "quick_start": {
            "download_data": "python scripts/download_pu_ntu.py",
            "reproduce_all": "python scripts/run_delivery_pipeline.py --download-data --export-dir deliverables/faultdg_delivery",
        },
    }


def main() -> None:
    args = parse_args()
    output_root = resolve_local_path(args.output_root)
    doc_path = resolve_local_path(args.doc)
    dest = resolve_local_path(args.dest)

    if dest.exists():
        if not args.overwrite:
            raise FileExistsError(f"Destination already exists: {dest}. Use --overwrite to rebuild it.")
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    for file_name in ROOT_FILES:
        src = PROJECT_ROOT / file_name
        if src.exists():
            copy_file(src, dest / file_name)

    for dir_name in ROOT_DIRS:
        src = PROJECT_ROOT / dir_name
        if src.exists():
            copy_tree(src, dest / dir_name)

    if doc_path.exists():
        copy_file(doc_path, dest / "docs" / "final_delivery.md")

    data_readme = PROJECT_ROOT / "data" / "pu" / "README.md"
    if data_readme.exists():
        copy_file(data_readme, dest / "data" / "pu" / "README.md")

    outputs_dest = dest / "outputs"
    for run_name in RESULT_DIRS:
        src = output_root / run_name
        if not src.exists():
            raise FileNotFoundError(f"Missing result directory: {src}")
        copy_tree(src, outputs_dest / run_name)

    manifest = build_manifest(output_root, dest)
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.zip:
        archive_path = shutil.make_archive(str(dest), "zip", root_dir=dest.parent, base_dir=dest.name)
        print(f"created {archive_path}")

    print(f"delivery package ready: {dest}")


if __name__ == "__main__":
    main()
