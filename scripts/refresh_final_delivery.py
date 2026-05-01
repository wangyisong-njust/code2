"""Inject the latest numbers from outputs/ into docs/final_delivery.md.

This script does NOT rewrite the document — it only replaces the "*(数字将在跑完后填入。)*"
placeholders with markdown tables drawn from the result CSVs. Idempotent: running
it twice in a row produces the same file.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOC = PROJECT_ROOT / "docs" / "final_delivery.md"

# Markers wrap each auto-refreshed block.
TAGS = {
    "MAIN": ("<!-- AUTO:MAIN:start -->", "<!-- AUTO:MAIN:end -->"),
    "NOISE": ("<!-- AUTO:NOISE:start -->", "<!-- AUTO:NOISE:end -->"),
    "ABLATION": ("<!-- AUTO:ABLATION:start -->", "<!-- AUTO:ABLATION:end -->"),
    "SWEEP": ("<!-- AUTO:SWEEP:start -->", "<!-- AUTO:SWEEP:end -->"),
    "FULL_MATRIX": ("<!-- AUTO:FULL_MATRIX:start -->", "<!-- AUTO:FULL_MATRIX:end -->"),
    "MODULE_ABLATION": ("<!-- AUTO:MODULE_ABLATION:start -->", "<!-- AUTO:MODULE_ABLATION:end -->"),
}


def replace_block(text: str, key: str, body: str) -> str:
    start, end = TAGS[key]
    if start not in text or end not in text:
        return text  # tag missing: leave document untouched.
    head, _, rest = text.partition(start)
    _, _, tail = rest.partition(end)
    return f"{head}{start}\n{body.strip()}\n{end}{tail}"


def fmt(value: float, digits: int = 4) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:.{digits}f}"


def render_main_summary(agg_path: Path) -> str:
    if not agg_path.exists():
        return "_(no summary_agg.csv yet)_"
    df = pd.read_csv(agg_path)
    pivot_mean = df.pivot(index="task", columns="method", values="accuracy_mean")
    pivot_std = df.pivot(index="task", columns="method", values="accuracy_std")
    methods = list(pivot_mean.columns)
    header = "| task | " + " | ".join(methods) + " |"
    align = "| --- |" + " ---: |" * len(methods)
    lines = [header, align]
    for task in pivot_mean.index:
        cells = []
        for method in methods:
            mean = pivot_mean.at[task, method]
            std = pivot_std.at[task, method]
            cells.append(f"{fmt(mean)} ± {fmt(std)}")
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_average(avg_path: Path) -> str:
    if not avg_path.exists():
        return "_(no average_accuracy.csv yet)_"
    df = pd.read_csv(avg_path).sort_values("accuracy_mean", ascending=False)
    lines = ["| method | mean accuracy | std |", "| --- | ---: | ---: |"]
    for _, row in df.iterrows():
        lines.append(f"| {row['method']} | {fmt(row['accuracy_mean'])} | {fmt(row['accuracy_std'])} |")
    return "\n".join(lines)


def render_noise(agg_path: Path) -> str:
    if not agg_path.exists():
        return "_(no noise_summary_agg.csv yet)_"
    df = pd.read_csv(agg_path)
    df = df.groupby(["method", "snr_db"], as_index=False).agg(
        accuracy_mean=("accuracy_mean", "mean"), accuracy_std=("accuracy_std", "mean")
    )
    snrs = sorted(df["snr_db"].unique(), reverse=True)
    methods = sorted(df["method"].unique())
    lines = ["| method | " + " | ".join(f"{snr:+.0f} dB" for snr in snrs) + " |",
             "| --- |" + " ---: |" * len(snrs)]
    for method in methods:
        cells = []
        for snr in snrs:
            r = df[(df["method"] == method) & (df["snr_db"] == snr)]
            if len(r) == 0:
                cells.append("—")
            else:
                cells.append(f"{fmt(r['accuracy_mean'].iloc[0])} ± {fmt(r['accuracy_std'].iloc[0])}")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_ablation(summary_path: Path) -> str:
    if not summary_path.exists():
        return "_(no pu_speed_ablation/summary.csv yet)_"
    df = pd.read_csv(summary_path).sort_values("accuracy", ascending=False)
    lines = ["| variant | accuracy |", "| --- | ---: |"]
    for _, row in df.iterrows():
        lines.append(f"| {row['variant']} | {fmt(row['accuracy'])} |")
    return "\n".join(lines)


def render_sweep_best(best_json: Path) -> str:
    if not best_json.exists():
        return "_(no best_config.json yet)_"
    payload = json.loads(best_json.read_text())
    return (
        f"- 最佳 `(margin={payload['best_margin']}, weight={payload['best_weight']})`，"
        f"mean accuracy = {fmt(payload['best_accuracy_mean'])} ± {fmt(payload['best_accuracy_std'])} "
        f"on `{payload['task']}`，seeds = `{payload['seeds']}`。"
    )


def render_full_matrix(agg_path: Path) -> str:
    if not agg_path.exists():
        return "_(no pu_full_matrix/summary_agg.csv yet)_"
    df = pd.read_csv(agg_path)
    pivot = df.pivot(index="task", columns="method", values="accuracy_mean")
    methods = list(pivot.columns)
    header = "| task | " + " | ".join(methods) + " |"
    align = "| --- |" + " ---: |" * len(methods)
    lines = [header, align]
    for task in pivot.index:
        cells = [fmt(pivot.at[task, m]) for m in methods]
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    avg = df.groupby("method", as_index=False)["accuracy_mean"].mean().sort_values("accuracy_mean", ascending=False)
    avg_block = ["", "**跨 12 对均值：**", "", "| method | mean accuracy |", "| --- | ---: |"]
    for _, row in avg.iterrows():
        avg_block.append(f"| {row['method']} | {fmt(row['accuracy_mean'])} |")
    return "\n".join(lines + avg_block)


def render_module_ablation(agg_path: Path, contribution_path: Path) -> str:
    if not agg_path.exists():
        return "_(no pu_module_ablation/summary_agg.csv yet)_"
    df = pd.read_csv(agg_path)
    pivot_mean = df.pivot(index="task", columns="variant", values="accuracy_mean")
    pivot_std = df.pivot(index="task", columns="variant", values="accuracy_std")
    # Order columns so that ``full`` is first and ``vanilla`` last when present.
    preferred = ["full", "no_sparsity", "no_disc_upgrade", "no_oscillation", "vanilla"]
    variants = [v for v in preferred if v in pivot_mean.columns] + [
        v for v in pivot_mean.columns if v not in preferred
    ]
    header = "| task | " + " | ".join(variants) + " |"
    align = "| --- |" + " ---: |" * len(variants)
    lines = ["**任务级 accuracy（mean ± std）：**", "", header, align]
    for task in pivot_mean.index:
        cells = []
        for variant in variants:
            mean = pivot_mean.at[task, variant]
            std = pivot_std.at[task, variant]
            cells.append(f"{fmt(mean)} ± {fmt(std)}")
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    # Cross-task mean per variant.
    avg = df.groupby("variant", as_index=False)["accuracy_mean"].mean()
    avg = avg.set_index("variant").reindex(variants).reset_index()
    lines += ["", "**跨任务平均：**", "", "| variant | mean accuracy |", "| --- | ---: |"]
    for _, row in avg.iterrows():
        lines.append(f"| {row['variant']} | {fmt(row['accuracy_mean'])} |")
    if contribution_path.exists():
        cdf = pd.read_csv(contribution_path)
        contrib = cdf.groupby("variant_removed", as_index=False)["delta"].mean()
        contrib = contrib.sort_values("delta", ascending=False)
        lines += [
            "",
            "**各模块贡献 Δ（full − 去掉该模块），正值代表移除会掉点：**",
            "",
            "| 移除的模块 | Δ accuracy（任务平均） |",
            "| --- | ---: |",
        ]
        for _, row in contrib.iterrows():
            lines.append(f"| {row['variant_removed']} | {fmt(row['delta'])} |")
    return "\n".join(lines)


def render_multisource(agg_path: Path) -> str:
    if not agg_path.exists():
        return "_(no pu_multisource/summary_agg.csv yet)_"
    df = pd.read_csv(agg_path)
    pivot = df.pivot(index="task", columns="method", values="accuracy_mean")
    methods = list(pivot.columns)
    header = "| task | " + " | ".join(methods) + " |"
    align = "| --- |" + " ---: |" * len(methods)
    lines = [header, align]
    for task in pivot.index:
        cells = [fmt(pivot.at[task, m]) for m in methods]
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    avg = df.groupby("method", as_index=False)["accuracy_mean"].mean().sort_values("accuracy_mean", ascending=False)
    avg_block = ["", "**跨 LOO 均值：**", "", "| method | mean accuracy |", "| --- | ---: |"]
    for _, row in avg.iterrows():
        avg_block.append(f"| {row['method']} | {fmt(row['accuracy_mean'])} |")
    return "\n".join(lines + avg_block)


def main() -> None:
    text = DOC.read_text()

    main_block = (
        "**任务级 accuracy（mean ± std，3 seeds）：**\n\n"
        + render_main_summary(PROJECT_ROOT / "outputs/pu_adaptive_sdae/results/summary_agg.csv")
        + "\n\n**跨任务平均（mean ± std）：**\n\n"
        + render_average(PROJECT_ROOT / "outputs/pu_adaptive_sdae/results/average_accuracy.csv")
    )

    noise_block = (
        "**AWGN 鲁棒性（mean ± std，3 seeds，跨任务平均）：**\n\n"
        + render_noise(PROJECT_ROOT / "outputs/pu_adaptive_sdae/results/noise_summary_agg.csv")
    )

    ablation_block = render_ablation(PROJECT_ROOT / "outputs/pu_speed_ablation/results/summary.csv")

    sweep_block = render_sweep_best(PROJECT_ROOT / "outputs/pu_disc_sweep/results/best_config.json")

    matrix_block = render_full_matrix(PROJECT_ROOT / "outputs/pu_full_matrix/results/summary_agg.csv")

    multisrc_block = render_multisource(PROJECT_ROOT / "outputs/pu_multisource/results/summary_agg.csv")

    module_ablation_block = render_module_ablation(
        PROJECT_ROOT / "outputs/pu_module_ablation/results/summary_agg.csv",
        PROJECT_ROOT / "outputs/pu_module_ablation/results/contribution.csv",
    )

    # Replace each tagged block with its rendered content.
    text = replace_block(text, "MAIN", main_block)
    text = replace_block(text, "NOISE", noise_block)
    text = replace_block(text, "ABLATION", ablation_block)
    text = replace_block(text, "SWEEP", sweep_block)
    text = replace_block(text, "FULL_MATRIX", matrix_block + "\n\n**多源 DG（LOO）：**\n\n" + multisrc_block)
    text = replace_block(text, "MODULE_ABLATION", module_ablation_block)

    DOC.write_text(text, encoding="utf-8")
    print(f"updated {DOC}")


if __name__ == "__main__":
    main()
