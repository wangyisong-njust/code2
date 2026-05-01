## PU Data Directory

公开数据集文件不包含在仓库中。复现时请在仓库根目录执行：

```bash
python scripts/download_pu_ntu.py
python scripts/check_runtime.py --strict --check-data
```

如果希望把数据放到别的目录，可以显式指定：

```bash
python scripts/download_pu_ntu.py --data-root /path/to/pu_ntu
python scripts/check_runtime.py --strict --check-data --data-root /path/to/pu_ntu
```

下载完成后，目录结构应为：

```text
data/pu/
├── README.md
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

如果 `ntu/` 下 12 个 `.pt` 文件不完整，重新执行下载脚本即可，脚本会跳过已存在文件并补齐缺失文件。
