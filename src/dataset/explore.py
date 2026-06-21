#!/usr/bin/env python3
"""
Data Distribution Exploration for Thesis Figure
================================================
Draws the prompt length (prompt_tokens) and response length (completion_tokens)
distributions for the three 14b datasets located in profiling_results/generated/.

Datasets:
  - LMSYS-Chat      : qwen2.5-14b-instruct-port-8014-processed-lmsys-chat-generated.jsonl
  - RAG-ShareGPT    : qwen2.5-14b-instruct-port-8014-rag-dataset-sharegpt-generated.jsonl
  - Code-Feedback   : qwen2.5-coder-14b-instruct-code-feedback-generated.jsonl

Output: fig/dataset_distribution.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import matplotlib.font_manager as _fm

# Register Times New Roman from system TTF files (needed on Linux where the font
# cache may not include msttcorefonts automatically).
_TNR_TTFS = [
    "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Italic.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold_Italic.ttf",
]
for _ttf in _TNR_TTFS:
    if Path(_ttf).exists():
        _fm.fontManager.addfont(_ttf)

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "pdf.fonttype": 42,  # embeddable fonts for IEEE / ACM submissions
        "ps.fonttype": 42,
    }
)

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "profiling_results" / "generated"
FIG_DIR = REPO_ROOT / "fig"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATASETS: dict[str, Path] = {
    "LMSYS-Chat": DATA_DIR
    / "qwen2.5-14b-instruct-port-8014-processed-lmsys-chat-generated.jsonl",
    "RAG-ShareGPT": DATA_DIR
    / "qwen2.5-14b-instruct-port-8014-rag-dataset-sharegpt-generated.jsonl",
    "Code-Feedback": DATA_DIR
    / "qwen2.5-coder-14b-instruct-code-feedback-generated.jsonl",
}

# Colour palette (colour-blind friendly)
PALETTE = {
    "LMSYS-Chat": "#4C72B0",
    "RAG-ShareGPT": "#DD8452",
    "Code-Feedback": "#55A868",
}

# ── Data loading ───────────────────────────────────────────────────────────────

def load_lengths(path: Path) -> tuple[list[int], list[int]]:
    """Return (prompt_tokens, completion_tokens) lists from a JSONL file."""
    prompt_lens: list[int] = []
    comp_lens: list[int] = []
    print(f"Loading {path.name} …", end=" ", flush=True)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                prompt_lens.append(int(obj["prompt_tokens"]))
                comp_lens.append(int(obj["completion_tokens"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    print(f"{len(prompt_lens):,} records.")
    return prompt_lens, comp_lens


# ── Plotting helpers ───────────────────────────────────────────────────────────

def _percentile_clip(data: list[int], pct: float = 99.5) -> np.ndarray:
    """Return array clipped at *pct*-th percentile to suppress extreme outliers."""
    arr = np.array(data, dtype=float)
    cap = np.percentile(arr, pct)
    return arr[arr <= cap]


def plot_kde_hist(
    ax: plt.Axes,
    datasets: dict[str, list[int]],
    xlabel: str,
    title: str,
    clip_pct: float = 97.0,
    bins: int = 60,
) -> None:
    """Overlay histogram + KDE for each dataset on *ax*."""
    for name, data in datasets.items():
        arr = _percentile_clip(data, clip_pct)
        color = PALETTE[name]

        # Histogram (normalised density)
        ax.hist(
            arr,
            bins=bins,
            density=True,
            alpha=0.25,
            color=color,
            linewidth=0,
        )

        # KDE via numpy (no scipy dependency)

        # Simple Gaussian KDE using numpy
        bandwidth = 1.06 * np.std(arr) * len(arr) ** (-1 / 5)  # Silverman's rule
        x_min, x_max = arr.min(), arr.max()
        x_grid = np.linspace(x_min, x_max, 512)
        # Evaluate KDE on grid
        diff = x_grid[:, np.newaxis] - arr[np.newaxis, :]  # (512, N)
        kernel = np.exp(-0.5 * (diff / bandwidth) ** 2)
        kde_vals = kernel.mean(axis=1) / (bandwidth * np.sqrt(2 * np.pi))

        ax.plot(x_grid, kde_vals, color=color, linewidth=2, label=name)

        # Vertical median line
        median = np.median(arr)
        ax.axvline(
            median,
            color=color,
            linewidth=1.2,
            linestyle="--",
            alpha=0.75,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax.legend(framealpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)




# ── Summary statistics ─────────────────────────────────────────────────────────

def print_stats(name: str, data: list[int], label: str) -> None:
    arr = np.array(data, dtype=float)
    print(
        f"  [{name}] {label}: "
        f"n={len(arr):,}  mean={arr.mean():.1f}  median={np.median(arr):.0f}  "
        f"std={arr.std():.1f}  p95={np.percentile(arr, 95):.0f}  "
        f"p99={np.percentile(arr, 99):.0f}  max={arr.max():.0f}"
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # Load data
    prompt_data: dict[str, list[int]] = {}
    comp_data: dict[str, list[int]] = {}

    for name, path in DATASETS.items():
        p, c = load_lengths(path)
        prompt_data[name] = p
        comp_data[name] = c

    # Print summary statistics
    print("\n── Summary Statistics ───────────────────────────────────────────────────────")
    for name in DATASETS:
        print_stats(name, prompt_data[name], "Prompt tokens")
        print_stats(name, comp_data[name], "Completion tokens")
    print()

    # ── Figure layout: 1 row × 2 columns ──────────────────────────────────────
    # KDE histograms: prompt (left) | response (right)
    fig, axes = plt.subplots(
        1, 2,
        figsize=(13, 5),
    )

    plot_kde_hist(
        axes[0],
        prompt_data,
        xlabel="Prompt Length (tokens)",
        title="(a) Prompt Length Distribution",
    )
    plot_kde_hist(
        axes[1],
        comp_data,
        xlabel="Response Length (tokens)",
        title="(b) Response Length Distribution",
    )

    fig.tight_layout()

    # Save
    out_png = FIG_DIR / "dataset_distribution.png"
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    print(f"Saved → {out_png}")

    plt.show()


if __name__ == "__main__":
    main()
