"""Generate comparison tables and figures directly from saved training output."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
VARIANTS = ["baseline", "depthwise", "se", "depthwise_se"]


def main():
    runs = {}
    for variant in VARIANTS:
        directory = ROOT / "results" / variant
        if not (directory / "metrics.json").exists():
            continue
        runs[variant] = {
            "config": json.loads((directory / "config.json").read_text(encoding="utf-8")),
            "metrics": json.loads((directory / "metrics.json").read_text(encoding="utf-8")),
            "history": json.loads((directory / "history.json").read_text(encoding="utf-8")),
        }
    output = ROOT / "analysis"
    output.mkdir(exist_ok=True)
    summary = []
    for name, run in runs.items():
        metrics = run["metrics"]
        summary.append({
            "variant": name,
            "test_accuracy": metrics["test"]["accuracy"],
            "test_correct": metrics["test"]["correct"],
            "test_total": metrics["test"]["total"],
            "parameters": metrics["profile"]["parameters"],
            "macs_per_image": metrics["profile"]["macs_per_image"],
            "latency_ms_per_image": metrics["profile"]["latency_ms_per_image"],
            "best_epoch": metrics["best_epoch"],
            "best_val_accuracy": metrics["best_val_accuracy"],
        })
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    colors = ["#252b39", "#3276a8", "#b96a39", "#3a927b"]
    fig, ax = plt.subplots(figsize=(6.5, 3.6), dpi=200)
    for name, color in zip(VARIANTS, colors):
        if name not in runs:
            continue
        history = runs[name]["history"]
        ax.plot([row["epoch"] for row in history],
                [100 * row["val_accuracy"] for row in history],
                label=name, color=color, linewidth=1.6)
    ax.set(xlabel="Epoch", ylabel="Validation accuracy (%)")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "validation_curves.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 3.8), dpi=200)
    for row, color in zip(summary, colors):
        ax.scatter(row["parameters"] / 1e6, 100 * row["test_accuracy"],
                   s=80, color=color, label=row["variant"])
    ax.set(xlabel="Parameters (million)", ylabel="Test accuracy (%)")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "accuracy_parameters.png", bbox_inches="tight")
    plt.close(fig)

    if "baseline" in runs:
        baseline = np.asarray(runs["baseline"]["metrics"]["test"]["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=200)
        matrix = baseline / baseline.sum(axis=1, keepdims=True)
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
        labels = list(runs["baseline"]["metrics"]["test"]["per_class_accuracy"])
        ax.set_xticks(range(10), labels, rotation=55, ha="right")
        ax.set_yticks(range(10), labels)
        ax.set(xlabel="Predicted", ylabel="True")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row fraction")
        fig.tight_layout()
        fig.savefig(output / "baseline_confusion.png", bbox_inches="tight")
        plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
