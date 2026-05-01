from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp) or {}
    config["_meta"] = {
        "config_path": str(config_path),
        "config_dir": str(config_path.parent),
        "project_root": str(config_path.parent.parent),
    }
    return config


def resolve_path(config: Dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path(config["_meta"]["project_root"]) / path).resolve()


def apply_runtime_overrides(
    config: Dict[str, Any],
    *,
    data_root: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Dict[str, Any]:
    if data_root is not None:
        config.setdefault("data", {})["root"] = str(data_root)
    if output_dir is not None:
        config.setdefault("experiment", {})["output_dir"] = str(output_dir)
    return config


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _relative_or_original(root: Path, value: str | Path) -> str:
    path = Path(value)
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def make_serializable_config(config: Dict[str, Any]) -> Dict[str, Any]:
    payload = copy.deepcopy(config)
    meta = payload.get("_meta")
    if not isinstance(meta, dict):
        return payload

    project_root = Path(meta.get("project_root", ".")).resolve()
    payload["_meta"] = {
        "config_path": _relative_or_original(project_root, meta.get("config_path", "")),
        "config_dir": _relative_or_original(project_root, meta.get("config_dir", "")),
        "project_root": ".",
    }
    return payload


def save_resolved_config(config: Dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(make_serializable_config(config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
