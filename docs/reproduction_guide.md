# 复现说明

仓库当前保留最终结果表、图和复现实验所需代码，不打包公开数据集，也不保留训练过程中产生的大体积中间产物。换机器复现时，建议严格按下面的顺序执行。

所有命令默认在仓库根目录执行。如果终端当前目录不是仓库根目录，先执行：

```bash
cd /path/to/code2
pwd
```

## 1. 环境准备

### 1.1 推荐环境

- Python：`3.8`
- PyTorch：`2.3.x`
- 操作系统：Linux
- 显卡：单卡 CUDA 即可；无 CUDA 也能运行，但全量实验会明显变慢

### 1.2 用 conda 创建环境

```bash
conda env create -f environment.yml
conda activate faultdg
```

如果希望沿用已有环境名，也可以显式指定新环境名称，例如：

```bash
conda env create -f environment.yml -n torch21new
conda activate torch21new
```

如果机器上已经有可用的 Python 3.8 环境，也可以直接安装：

```bash
pip install -e . --no-deps
pip install numpy pandas pyyaml scikit-learn scipy matplotlib seaborn tqdm
```

如果需要 GPU 版 PyTorch，可按本机 CUDA 版本单独安装官方对应包；本次结果是在 `torch 2.3.0` 环境下生成的。脚本默认只使用一个可见 GPU，不依赖固定的 GPU 数量。

环境创建完成后，建议立即执行一次：

```bash
python -c "import torch; print(torch.__version__)"
python -c "import faultdg; print('faultdg import ok')"
```

### 1.3 激活失败时的处理

如果 `conda activate faultdg` 报错，按顺序检查：

1. `conda --version` 是否可用。
2. 当前 shell 是否已初始化 conda。
   常见做法：

```bash
conda init bash
```

执行后重新打开终端，再运行：

```bash
conda activate faultdg
```

如果不希望改 shell 配置，也可以临时执行：

```bash
eval "$(conda shell.bash hook)"
conda activate faultdg
```

如果后续使用 `bash scripts/run_all.sh`，而环境名称不是 `faultdg`，需要显式指定：

```bash
FAULTDG_ENV_NAME=torch21new bash scripts/run_all.sh
```

如果机器上有多张 GPU，而只想指定其中一张运行，可以在命令前加：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_delivery_pipeline.py --download-data
```

如果只有一张 GPU，或准备在 CPU 上运行，不需要额外改代码。

## 2. 环境自检

环境装好后先跑一遍：

```bash
python scripts/check_runtime.py
```

它会检查：

- Python / torch / numpy / pandas 等核心依赖是否可导入
- 当前是否检测到 CUDA
- `data/pu/ntu/` 下是否已经存在 12 个 `.pt` 数据文件

如果要把“环境 + 数据”都作为硬检查项：

```bash
python scripts/check_runtime.py --strict --check-data
```

## 3. 数据准备

### 3.1 自动下载

PU 处理后数据不随交付包提供，需单独下载：

```bash
python scripts/download_pu_ntu.py
```

下载完成后，`data/pu/ntu/` 下应当有以下 12 个文件：

- `train_a.pt`
- `train_b.pt`
- `train_c.pt`
- `train_d.pt`
- `val_a.pt`
- `val_b.pt`
- `val_c.pt`
- `val_d.pt`
- `test_a.pt`
- `test_b.pt`
- `test_c.pt`
- `test_d.pt`

再次执行：

```bash
python scripts/check_runtime.py --strict --check-data
```

目录结构应当类似：

```text
data/pu/
├── ntu_dataset.json
└── ntu/
    ├── train_a.pt
    ├── train_b.pt
    ├── train_c.pt
    ├── train_d.pt
    ├── val_a.pt
    ├── val_b.pt
    ├── val_c.pt
    ├── val_d.pt
    ├── test_a.pt
    ├── test_b.pt
    ├── test_c.pt
    └── test_d.pt
```

### 3.2 下载异常处理

如果下载中断、网络超时或只下载到部分文件，直接再次执行：

```bash
python scripts/download_pu_ntu.py
```

脚本会跳过已经存在的文件，只补缺失文件。

如果 NTU 数据接口暂时不可用，可先手动下载同名 `.pt` 文件，再放到 `data/pu/ntu/`，文件名必须保持一致。

## 4. 复现顺序

### 4.1 一键复现

```bash
python scripts/run_delivery_pipeline.py --download-data
```

上面这条命令会按顺序完成：

- 环境自检
- 数据完整性检查
- 数据缺失时自动下载
- 6 组正式实验
- 刷新 `docs/final_delivery.md`

如果还需要直接组装不含公开数据集的交付目录，执行：

```bash
python scripts/run_delivery_pipeline.py \
  --download-data \
  --export-dir deliverables/faultdg_delivery \
  --zip-export
```

如果使用的是 shell 包装器，且环境名称不是 `faultdg`，改为：

```bash
FAULTDG_ENV_NAME=torch21new FAULTDG_AUTO_DOWNLOAD=1 bash scripts/run_all.sh
```

### 4.2 分步复现

```bash
# 主表 + 噪声 + 主图
python scripts/run_pu_benchmark.py --config configs/pu_adaptive_sdae.yaml

# speed_shift 消融
python scripts/run_speed_ablation.py --config configs/pu_adaptive_sdae.yaml

# 判别损失扫描
python scripts/run_disc_sweep.py --config configs/pu_adaptive_sdae.yaml

# 12 对全跨工况矩阵
python scripts/run_pu_full_matrix.py --config configs/pu_adaptive_sdae.yaml --seeds 42

# 多源 leave-one-out
python scripts/run_pu_multisource.py --config configs/pu_adaptive_sdae.yaml --mode leave-one-out --seeds 42

# 三个论文对齐增量的逐模块消融
python scripts/run_module_ablation.py --config configs/pu_adaptive_sdae.yaml

# 刷新最终报告中的自动数字块
python scripts/refresh_final_delivery.py --output-root outputs --doc docs/final_delivery.md

# 组装交付目录
python scripts/export_delivery_package.py --output-root outputs --dest deliverables/faultdg_delivery --zip
```

## 5. 结果目录说明

交付目录默认保留的是最终结果表和图：

- `outputs/pu_adaptive_sdae/results/`
- `outputs/pu_adaptive_sdae/figures/`
- `outputs/pu_speed_ablation/results/`
- `outputs/pu_speed_ablation/figures/`
- `outputs/pu_disc_sweep/results/`
- `outputs/pu_disc_sweep/figures/`
- `outputs/pu_full_matrix/results/`
- `outputs/pu_full_matrix/figures/`
- `outputs/pu_multisource/results/`
- `outputs/pu_multisource/figures/`
- `outputs/pu_module_ablation/results/`
- `outputs/pu_module_ablation/figures/`

训练时会重新生成 `checkpoints/`、`histories/`、`predictions/` 等过程文件；这些大体积文件不包含在仓库当前保留范围内。

一键脚本和分步脚本完成后，建议核对下面这些关键文件是否生成：

- `outputs/pu_adaptive_sdae/results/summary_agg.csv`
- `outputs/pu_adaptive_sdae/results/noise_summary_agg.csv`
- `outputs/pu_disc_sweep/results/sweep_agg.csv`
- `outputs/pu_full_matrix/results/summary_agg.csv`
- `outputs/pu_multisource/results/summary_agg.csv`
- `outputs/pu_module_ablation/results/summary_agg.csv`
- `docs/final_delivery.md`

如果执行了导出命令，还应当核对：

- `deliverables/faultdg_delivery/MANIFEST.json`
- `deliverables/faultdg_delivery/docs/final_delivery.md`
- `deliverables/faultdg_delivery/outputs/pu_adaptive_sdae/results/summary_agg.csv`

## 6. 常见问题

### 6.1 `ModuleNotFoundError`

说明当前 Python 不是项目环境。先确认：

```bash
which python
python -V
```

再重新激活环境：

```bash
conda activate faultdg
```

### 6.2 `data/pu/ntu/*.pt` 缺失

说明数据没下全，重新执行：

```bash
python scripts/download_pu_ntu.py
python scripts/check_runtime.py --strict --check-data
```

### 6.3 没有 GPU

可以直接在 CPU 上运行，但全量实验耗时会明显增加。若只检查流程是否跑通，建议先单独运行：

```bash
python scripts/run_pu_benchmark.py --config configs/pu_adaptive_sdae.yaml --tasks speed_shift
```

### 6.4 报告数字和 CSV 不一致

重新执行：

```bash
python scripts/refresh_final_delivery.py
```

该脚本会把 `outputs/` 中现有结果重新注入 `docs/final_delivery.md`。

### 6.5 `run_all.sh` 提示环境名不对

默认环境名写的是 `faultdg`。如果实际使用的是其他环境名，例如 `torch21new`，执行：

```bash
FAULTDG_ENV_NAME=torch21new bash scripts/run_all.sh
```

### 6.6 想把结果输出到别的目录

统一入口支持自定义输出根目录和报告路径，例如：

```bash
python scripts/run_delivery_pipeline.py \
  --download-data \
  --output-root outputs_rerun \
  --doc docs/final_delivery_rerun.md
```

分步脚本也都支持 `--output-dir`；涉及数据位置的脚本支持 `--data-root`。

### 6.7 想直接生成交付包

执行：

```bash
python scripts/export_delivery_package.py \
  --output-root outputs \
  --dest deliverables/faultdg_delivery \
  --zip
```

该命令会：

- 复制代码、配置、说明文档和最终结果目录
- 排除公开数据集文件
- 排除 `checkpoints/`、`histories/`、`predictions/` 等重跑中间件
- 生成 `MANIFEST.json`
