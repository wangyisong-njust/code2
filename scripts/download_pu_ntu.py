from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.request import urlretrieve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "data" / "pu" / "ntu"
META_ROOT = PROJECT_ROOT / "data" / "pu"
EXPECTED_FILES = [
    f"{split}_{domain}.pt"
    for split in ("train", "val", "test")
    for domain in ("a", "b", "c", "d")
]

DATASET_API = "https://researchdata.ntu.edu.sg/api/datasets/:persistentId/?persistentId=doi:10.21979/N9/X6M827"
FILE_API = "https://researchdata.ntu.edu.sg/api/access/datafile/{file_id}"


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"Skip existing: {output_path.name}")
        return
    print(f"Downloading {output_path.name}")
    urlretrieve(url, output_path)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    meta_path = META_ROOT / "ntu_dataset.json"
    download_file(DATASET_API, meta_path)
    payload = json.loads(meta_path.read_text())
    files = payload["data"]["latestVersion"]["files"]

    for file_item in files:
        label = file_item["label"]
        file_id = file_item["dataFile"]["id"]
        download_file(FILE_API.format(file_id=file_id), OUTPUT_ROOT / label)

    missing = [file_name for file_name in EXPECTED_FILES if not (OUTPUT_ROOT / file_name).exists()]
    if missing:
        raise FileNotFoundError(
            "Download finished but the PU dataset is incomplete. Missing files: "
            + ", ".join(missing)
        )
    print(f"Dataset files ready under {OUTPUT_ROOT}")


if __name__ == "__main__":
    sys.exit(main())
