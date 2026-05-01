# FaultDG

基于 Paderborn (PU) 数据集的轴承跨工况故障分类复现实验仓库。主方法 `Adaptive-SDAE-3M` 由 SDAE 去噪、分类导向损失与结构自适应调整三模块组成，并配套 5 个对比方法、3 档噪声鲁棒性、12 对全跨工况矩阵、多源域泛化和模块消融。

仓库当前保留：

- 最终结果表和图
- 可复现实验代码
- 运行配置
- 复现文档

以下内容不随交付包保留：

- 公开数据集文件
- 训练过程中生成的大体积 checkpoint / prediction / history

## 主方法 Adaptive-SDAE-3M

由 3 个模块组成（沿原论文公式做了 3 项扩展尝试，逐模块消融见 `docs/final_delivery.md` §M）：

- **模块 1 — SDAE 去噪重构 + KL 稀疏预训练**：在源 + 目标域上联合做去噪自编码器预训练；**默认启用** Eq.(9) 形式的 KL 稀疏项 `J_spare = Σ KL(ρ‖ρ̂_j)`（`ρ̂_j = E[σ(z_j)]`），消融显示跨任务平均 +1.2 pp 收益、`-6 dB` 噪声鲁棒性领先 +3.7 pp。
- **模块 2 — 统计特征融合 + 分类导向判别损失**：把时频域统计量分支融入潜空间，叠加 intra-class 紧凑 + inter-class margin 判别约束（**线性 hinge + 固定 μ1=μ2**）。我们尝试的 squared-hinge 与基于 intra/inter 比值的自适应 μ1/μ2 平衡两项扩展在 PU 消融下未呈现稳健正向收益（`adaptive` 单独开启在 `speed_shift` 上掉 12 pp），代码保留为可选开关，**默认关闭**。
- **模块 3 — 验证损失驱动的结构自适应调整**：按"绝对 ∨ 相对 val_loss 下降阈值"决定是否扩宽或加深网络，并以"全过程最优 val_loss"作为最终结构兜底。我们尝试将论文图 3-2"快速稳定下降"形式化为"末段相对方差 + 斜率"双重判据；该判据在当前 PU 搜索预算下**不改变任何最终选型**（消融 Δ=0），因此只保留为搜索日志中的辅助记录，不作为有效增益点。

针对变转速场景额外增加：

- **速度缩放增强**：用 `source_rpm / target_rpm` 对源信号做时间轴线性插值，缓解速度变化引起的频谱漂移。

## 交付内容

| 实验 | 输出目录 | 说明 |
| --- | --- | --- |
| 主基准 | `outputs/pu_adaptive_sdae/` | 3 个默认任务 × 6 个方法 × 3 个 seeds，含主表 / 噪声 / 全部图 |
| speed_shift 模块消融 | `outputs/pu_speed_ablation/` | 5 个变体在 `d → a` 上的对比 + bar chart |
| Module-2 判别损失敏感性 | `outputs/pu_disc_sweep/` | margin × weight 网格热力图与最佳配置 |
| 全 12 对跨工况矩阵 | `outputs/pu_full_matrix/` | 4 域之间所有 source → target 配对 |
| 多源域泛化（leave-one-out） | `outputs/pu_multisource/` | 3 源 → 1 目标的 4 组 LOO 任务 |

## 对比方法

- `ERM-SDAE`：source-only 基线
- `Mixup-SDAE`：源域 Mixup 正则
- `CORAL-SDAE`：二阶矩对齐
- `MMD-SDAE`：RBF MMD 对齐
- `DANN-SDAE`：基于梯度反转的对抗域不变学习
- `Adaptive-SDAE-3M`（主方法）

所有对比方法共用同一份 SDAE 骨干和同一份预训练流程，方法之间只有损失函数和增强策略的差异，确保对比公平。

## 默认三任务（主表）

| task | source → target | 物理意义 |
| --- | --- | --- |
| `speed_shift` | `d → a` | `N15_M07_F10` → `N09_M07_F10`（变工况 / 变转速） |
| `load_shift`  | `d → b` | `N15_M07_F10` → `N15_M01_F10`（变负载扭矩） |
| `radial_shift`| `d → c` | `N15_M07_F10` → `N15_M07_F04`（变径向载荷） |

PU 工况映射（按数据集中字母字典序）：

- `a = N09_M07_F10`（900 rpm / 0.7 Nm / 1000 N）
- `b = N15_M01_F10`（1500 rpm / 0.1 Nm / 1000 N）
- `c = N15_M07_F04`（1500 rpm / 0.7 Nm / 400 N）
- `d = N15_M07_F10`（1500 rpm / 0.7 Nm / 1000 N）

## 协议

- 默认 `protocol.mode = uda`：训练时可使用无标签的 `target_train` 信号，但目标 `test` 始终不入训练。
- 在配置中切换 `protocol.mode = strict_dg` 后，所有需要目标分布的方法（CORAL / MMD / DANN / 主方法）会自动退化为 source-only，对应严格域泛化评测。

## 环境与数据

完整环境搭建、数据下载、自检、导出交付包和故障排查见 [`docs/reproduction_guide.md`](docs/reproduction_guide.md)。

最短路径如下：

```bash
conda env create -f environment.yml
conda activate faultdg
python scripts/run_delivery_pipeline.py --download-data
```

说明：

- 结果是在 `Python 3.8 + torch 2.3.0` 环境下生成的。
- 公开数据集不包含在仓库当前保留范围内，需单独下载到 `data/pu/ntu/`。
- 如果换机器复现，先跑 `scripts/check_runtime.py` 或直接用 `scripts/run_delivery_pipeline.py --download-data`。
- 下面所有命令默认都在仓库根目录执行。

## 一键复现

```bash
python scripts/run_delivery_pipeline.py --download-data
```

如果希望在跑完后直接整理出可交付目录并打包：

```bash
python scripts/run_delivery_pipeline.py \
  --download-data \
  --export-dir deliverables/faultdg_delivery \
  --zip-export
```

`bash scripts/run_all.sh` 也是同一套流程的 shell 包装器，支持通过环境变量改环境名、数据目录、输出目录和导出目录：

```bash
FAULTDG_ENV_NAME=torch21new \
FAULTDG_AUTO_DOWNLOAD=1 \
FAULTDG_EXPORT_DIR=deliverables/faultdg_delivery \
bash scripts/run_all.sh
```

## 分步复现

每个脚本都会写出独立的 `outputs/<run>/` 目录，包含 `env.json`、`resolved_config.json`、结果 CSV 与图：

```bash
# 1. 主基准（3 seeds、6 方法、3 任务）
python scripts/run_pu_benchmark.py --config configs/pu_adaptive_sdae.yaml

# 2. speed_shift 模块消融
python scripts/run_speed_ablation.py --config configs/pu_adaptive_sdae.yaml

# 3. Module-2 判别损失敏感性分析
python scripts/run_disc_sweep.py --config configs/pu_adaptive_sdae.yaml

# 4. 全 12 对跨工况矩阵
python scripts/run_pu_full_matrix.py --config configs/pu_adaptive_sdae.yaml --seeds 42

# 5. 多源 DG（leave-one-out）
python scripts/run_pu_multisource.py --config configs/pu_adaptive_sdae.yaml --mode leave-one-out --seeds 42

# 6. 把 outputs/ 下的最新数字注入 docs/final_delivery.md
python scripts/refresh_final_delivery.py --output-root outputs --doc docs/final_delivery.md

# 7. 组装不含公开数据集的交付目录
python scripts/export_delivery_package.py --output-root outputs --dest deliverables/faultdg_delivery --zip
```

## 输出目录结构

交付包中保留的是最终结果表和图。以 `outputs/pu_adaptive_sdae/` 为例：

```text
results/summary.csv               # 每个 (task, method, seed) 一行的原始结果
results/summary_agg.csv           # 跨 seed 聚合（mean ± std）
results/average_accuracy.csv      # 每个方法的跨任务平均（mean ± std）
results/noise_summary.csv         # 噪声实验原始行
results/noise_summary_agg.csv     # 噪声实验 mean ± std
results/per_class_f1.csv          # 每类 F1 原始行
results/per_class_f1_agg.csv      # 每类 F1 跨 seed 平均
results/adaptive_search.csv       # 自适应结构搜索全过程
results/task_metadata.csv         # 每任务的样本量、转速、speed_scale 等元信息
figures/task_accuracy.png         # 主图：mean ± std 准确率柱状图
figures/noise_accuracy.png        # 噪声曲线（按任务分面 + 平均面板，含 std 阴影）
figures/<task>/<method>_confusion.png
figures/<task>/proposed_embedding.png
figures/<task>/tsne_alignment.png # ERM 与主方法源 / 目标分布并排
figures/<task>/train_history.png  # 主方法训练曲线（损失 + val acc / F1）
figures/<task>/per_class_f1.png   # 每类 F1 热力图
env.json                          # python / torch / cuda / numpy 版本
resolved_config.json
```

如果重新执行训练脚本，`checkpoints/`、`histories/`、`predictions/` 等过程目录会自动重新生成；这些大体积目录不包含在仓库当前保留范围内。使用 `scripts/export_delivery_package.py` 时也会自动排除这些目录和公开数据集文件。

## 最终交付报告

完整的方法说明、所有结果、消融分析与任务覆盖清单见 [`docs/final_delivery.md`](docs/final_delivery.md)。

## 复现实操文档

详细环境配置、数据下载、自检命令、导出命令和常见问题处理见 [`docs/reproduction_guide.md`](docs/reproduction_guide.md)。

## 配置说明

- 多 seed：在配置里编辑 `experiment.seeds`（默认 `[42, 7, 2024]`）
- 噪声档位：编辑 `evaluation.snr_db`（默认 `[6.0, 0.0, -6.0]`）
- 结构搜索：`adaptive_search.*` 暴露宽度 / 深度步长以及绝对 / 相对损失下降阈值
- 方法超参（margin、各损失权重、Mixup α、MMD γ、DANN λ-max 等）在 `training.method_params` 下统一管理

## 方法引用

主方法参考论文 *An effective fault diagnosis approach for bearing using stacked de-noising auto-encoder with structure adaptive adjustment* 提出的三大核心思路：SDAE 骨干、分类导向损失、结构自适应调整。本仓库实现这三大核心思路，并扩展为面向 Paderborn 数据集的完整跨工况评测基准。
