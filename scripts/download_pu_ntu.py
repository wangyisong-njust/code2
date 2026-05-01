from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.request import urlretrieve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "pu" / "ntu"
EXPECTED_FILES = [
    f"{split}_{domain}.pt"
    for split in ("train", "val", "test")
    for domain in ("a", "b", "c", "d")
]

DATASET_API = "https://researchdata.ntu.edu.sg/api/datasets/:persistentId/?persistentId=doi:10.21979/N9/X6M827"
FILE_API = "https://researchdata.ntu.edu.sg/api/access/datafile/{file_id}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the processed PU .pt splits from NTU Dataverse.")
    parser.add_argument("--data-root", default=str(DEFAULT_OUTPUT_ROOT),
                        help="Destination directory for the 12 PU .pt files.")
    return parser.parse_args()


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"Skip existing: {output_path.name}")
        return
    print(f"Downloading {output_path.name}")
    urlretrieve(url, output_path)


def main() -> None:
    args = parse_args()
    output_root = Path(args.data_root).resolve()
    meta_root = output_root.parent

    output_root.mkdir(parents=True, exist_ok=True)
    meta_path = meta_root / "ntu_dataset.json"
    download_file(DATASET_API, meta_path)
    payload = json.loads(meta_path.read_text())
    files = payload["data"]["latestVersion"]["files"]

    for file_item in files:
        label = file_item["label"]
        file_id = file_item["dataFile"]["id"]
        download_file(FILE_API.format(file_id=file_id), output_root / label)

    missing = [file_name for file_name in EXPECTED_FILES if not (output_root / file_name).exists()]
    if missing:
        raise FileNotFoundError(
            "Download finished but the PU dataset is incomplete. Missing files: "
            + ", ".join(missing)
        )
    print(f"Dataset files ready under {output_root}")


if __name__ == "__main__":
    sys.exit(main())
