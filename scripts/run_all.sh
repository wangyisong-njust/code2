#!/usr/bin/env bash
# One-shot reproduction of the final delivery package.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
ENV_NAME="${FAULTDG_ENV_NAME:-faultdg}"

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

"${PYTHON_BIN}" scripts/check_runtime.py || true
if ! "${PYTHON_BIN}" scripts/check_runtime.py --strict --check-data >/dev/null 2>&1; then
    echo "[error] PU dataset files are missing."
    echo "Run: ${PYTHON_BIN} scripts/download_pu_ntu.py"
    exit 1
fi

echo "=== [1/6] main 3-task benchmark (3 seeds) ==="
"${PYTHON_BIN}" scripts/run_pu_benchmark.py --config configs/pu_adaptive_sdae.yaml

echo "=== [2/6] speed-shift ablation ==="
"${PYTHON_BIN}" scripts/run_speed_ablation.py --config configs/pu_adaptive_sdae.yaml

echo "=== [3/6] Module-2 disc-loss grid sweep ==="
"${PYTHON_BIN}" scripts/run_disc_sweep.py --config configs/pu_adaptive_sdae.yaml

echo "=== [4/6] full 12-pair cross-condition matrix ==="
"${PYTHON_BIN}" scripts/run_pu_full_matrix.py --config configs/pu_adaptive_sdae.yaml --seeds 42

echo "=== [5/6] multi-source DG (LOO) ==="
"${PYTHON_BIN}" scripts/run_pu_multisource.py --config configs/pu_adaptive_sdae.yaml --mode leave-one-out --seeds 42

echo "=== [6/6] paper-aligned module ablation ==="
"${PYTHON_BIN}" scripts/run_module_ablation.py --config configs/pu_adaptive_sdae.yaml

echo "=== inject numbers into docs/final_delivery.md ==="
"${PYTHON_BIN}" scripts/refresh_final_delivery.py

echo "=== DONE ==="
