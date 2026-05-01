"""All 12 PU cross-condition source->target pairs.

NTU's processed PU data only ships 3 fault classes (healthy/inner/outer), so a
fine-grained 5/7-class experiment is not possible without re-deriving from raw
mat files. Instead we run the *full* 12-pair cross-condition matrix to give a
much stronger appendix than the default 3 hand-picked tasks.

We reuse ``run_pu_benchmark.main`` by synthesising a config that lists all 12
tasks. By default we use 1 seed for budget reasons, but multi-seed is supported
via ``--seeds``.
"""
from __future__ import annotations

import argparse
import copy
import sys
import tempfile
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from faultdg.config import apply_runtime_overrides, ensure_dir, load_config


DOMAIN_KEYS = ["a", "b", "c", "d"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all 12 PU cross-condition pairs.")
    parser.add_argument("--config", default="configs/pu_adaptive_sdae.yaml")
    parser.add_argument("--output-dir", default="outputs/pu_full_matrix")
    parser.add_argument("--data-root", default=None,
                        help="Override data.root from the config.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--methods", nargs="+", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = load_config(args.config)
    apply_runtime_overrides(base, data_root=args.data_root)

    tasks = []
    for src in DOMAIN_KEYS:
        for tgt in DOMAIN_KEYS:
            if src == tgt:
                continue
            tasks.append({"name": f"{src}_to_{tgt}", "source_domain": src, "target_domain": tgt})

    materialised = copy.deepcopy(base)
    materialised.pop("_meta", None)
    materialised["experiment"]["output_dir"] = args.output_dir
    materialised["experiment"]["tasks"] = tasks
    materialised["experiment"]["seeds"] = list(args.seeds)
    if args.methods is not None:
        materialised["experiment"]["methods"] = list(args.methods)

    tmp_dir = ensure_dir(PROJECT_ROOT / ".faultdg_tmp")
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="pu_full_matrix_",
        dir=tmp_dir,
        delete=False,
        encoding="utf-8",
    ) as fp:
        tmp_config = Path(fp.name)
        fp.write(yaml.safe_dump(materialised, sort_keys=False, allow_unicode=True))

    try:
        # Delegate to the multi-seed benchmark by execing it as a script.
        sys.argv = [
            "run_pu_benchmark.py",
            "--config",
            str(tmp_config),
            "--output-dir",
            str(args.output_dir),
        ]
        if args.data_root is not None:
            sys.argv.extend(["--data-root", str(args.data_root)])
        if args.methods is not None:
            sys.argv.extend(["--methods", *args.methods])
        sys.argv.extend(["--seeds", *map(str, args.seeds)])

        import runpy

        runpy.run_path(str(PROJECT_ROOT / "scripts" / "run_pu_benchmark.py"), run_name="__main__")
    finally:
        tmp_config.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
