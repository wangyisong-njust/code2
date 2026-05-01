#!/usr/bin/env bash
# One-command reproduction wrapper.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
ENV_NAME="${FAULTDG_ENV_NAME:-faultdg}"
CONFIG_PATH="${FAULTDG_CONFIG:-configs/pu_adaptive_sdae.yaml}"
OUTPUT_ROOT="${FAULTDG_OUTPUT_ROOT:-outputs}"
DOC_PATH="${FAULTDG_DOC_PATH:-docs/final_delivery.md}"
DATA_ROOT="${FAULTDG_DATA_ROOT:-data/pu/ntu}"
EXPORT_DIR="${FAULTDG_EXPORT_DIR:-}"
AUTO_DOWNLOAD="${FAULTDG_AUTO_DOWNLOAD:-0}"
ZIP_EXPORT="${FAULTDG_ZIP_EXPORT:-0}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[error] Python executable '${PYTHON_BIN}' was not found."
    echo "Set PYTHON_BIN explicitly or activate the project environment first."
    exit 1
fi

if ! "${PYTHON_BIN}" -c "import torch, pandas, numpy, yaml, sklearn, scipy, matplotlib, seaborn, tqdm" >/dev/null 2>&1; then
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate "${ENV_NAME}" || {
            echo "[error] Failed to activate conda env '${ENV_NAME}'."
            echo "Create it with: conda env create -f environment.yml"
            exit 1
        }
        PYTHON_BIN="python"
    else
        echo "[error] Required Python packages are missing and conda is unavailable."
        echo "Create the environment with 'conda env create -f environment.yml' or install the dependencies manually."
        exit 1
    fi
fi

CMD=(
    "${PYTHON_BIN}" scripts/run_delivery_pipeline.py
    --config "${CONFIG_PATH}"
    --output-root "${OUTPUT_ROOT}"
    --doc "${DOC_PATH}"
    --data-root "${DATA_ROOT}"
)

if [[ "${AUTO_DOWNLOAD}" == "1" ]]; then
    CMD+=(--download-data)
fi

if [[ -n "${EXPORT_DIR}" ]]; then
    CMD+=(--export-dir "${EXPORT_DIR}")
    if [[ "${ZIP_EXPORT}" == "1" ]]; then
        CMD+=(--zip-export)
    fi
fi

echo "+ ${CMD[*]}"
"${CMD[@]}"
