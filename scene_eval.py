"""Evaluate CIFAR-10.1 and deterministic blur/low-light conditions."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageFilter

from model import build_model
from run import CLASSES, MEAN, ROOT, STD


VARIANTS = ("baseline", "depthwise", "se", "depthwise_se")
CONDITIONS = ("clean", "blur_r1", "lowlight_0p45", "blur_r1_lowlight_0p45")
COLORS = ("#27364B", "#447A9C", "#BB704C", "#4C907B")


def inputs(images):
    blur = np.stack([np.asarray(Image.fromarray(image).filter(
        ImageFilter.GaussianBlur(radius=1.0))) for image in images])
    dark = np.rint(images.astype(np.float32) * 0.45).clip(0, 255).astype(np.uint8)
    blur_dark = np.rint(blur.astype(np.float32) * 0.45).clip(0, 255).astype(np.uint8)
    return dict(zip(CONDITIONS, (images, blur, dark, blur_dark)))


def normalize(batch, device):
    x = torch.from_numpy(batch.copy()).permute(0, 3, 1, 2).to(device,
                                                               dtype=torch.float32) / 255.0
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.inference_mode()
def predict(model, images, device, batch_size=256):
    model.eval()
    predictions = []
    confidences = []
    for start in range(0, len(images), batch_size):
        logits = model(normalize(images[start:start + batch_size], device))
        probabilities = logits.softmax(1)
        confidence, prediction = probabilities.max(1)
        predictions.extend(prediction.cpu().tolist())
        confidences.extend(confidence.cpu().tolist())
    return np.asarray(predictions), np.asarray(confidences)


def plot_accuracy(records, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    x = np.arange(len(CONDITIONS))
    width = 0.18
    fig, ax = plt.subplots(figsize=(7.2, 3.7), dpi=220)
    for i, (variant, color) in enumerate(zip(VARIANTS, COLORS)):
        y = [100 * records[variant][condition]["accuracy"] for condition in CONDITIONS]
        ax.bar(x + (i - 1.5) * width, y, width=width, label=variant,
               color=color, linewidth=0)
    ax.set_xticks(x, ["Clean", "Gaussian blur\nradius 1", "Brightness\n0.45",
                      "Blur + brightness\n0.45"])
    ax.set(ylabel="CIFAR-10.1 accuracy (%)", ylim=(0, 100))
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, 1.01), columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(output / "condition_accuracy.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.7), dpi=220)
    for i, (variant, color) in enumerate(zip(VARIANTS, COLORS)):
        clean = records[variant]["clean"]["accuracy"]
        drops = [100 * (clean - records[variant][condition]["accuracy"])
                 for condition in CONDITIONS[1:]]
        ax.bar(np.arange(3) + (i - 1.5) * width, drops, width=width,
               label=variant, color=color, linewidth=0)
    ax.set_xticks(range(3), ["Blur", "Low light", "Combined"])
    ax.set(ylabel="Accuracy loss vs clean (percentage points)")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, 1.01), columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(output / "condition_accuracy_drop.png", bbox_inches="tight")
    plt.close(fig)


def plot_examples(images, labels, conditions, records, output):
    clean_predictions = [np.asarray(records[variant]["clean"]["predictions"])
                         for variant in VARIANTS]
    indices = [int(np.flatnonzero(
        (labels == cls) & np.logical_and.reduce(
            [pred == cls for pred in clean_predictions]))[0]) for cls in range(10)]
    display_indices = [indices[i] for i in (0, 1, 3, 5, 8)]
    fig, axes = plt.subplots(5, 4, figsize=(7.5, 8.2), dpi=180)
    for row, index in enumerate(display_indices):
        for col, condition in enumerate(CONDITIONS):
            ax = axes[row, col]
            ax.imshow(conditions[condition][index], interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            predicted = records["baseline"][condition]["predictions"][index]
            color = "#2C745B" if predicted == int(labels[index]) else "#AA4A42"
            ax.set_xlabel(CLASSES[predicted], fontsize=8, color=color, labelpad=2)
            if col == 0:
                ax.set_ylabel(CLASSES[int(labels[index])], fontsize=8, rotation=0,
                              ha="right", va="center", labelpad=28)
            if row == 0:
                ax.set_title(condition.replace("_", " "), fontsize=9)
    fig.text(0.01, 0.995, "True label at left; baseline prediction below each image",
             va="top", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output / "condition_examples.png", bbox_inches="tight")
    plt.close(fig)
    return indices


def plot_prediction_shift(records, output):
    clean = np.bincount(records["baseline"]["clean"]["predictions"], minlength=10)
    combined = np.bincount(
        records["baseline"]["blur_r1_lowlight_0p45"]["predictions"], minlength=10)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), dpi=220, sharey=True)
    for ax, counts, title, color in zip(axes, (clean, combined),
                                         ("Clean", "Blur + low light"),
                                         ("#27364B", "#BB704C")):
        ax.barh(np.arange(10), counts, color=color, height=0.65)
        ax.set_title(title)
        ax.set(xlabel="Predicted image count", xlim=(0, 1500))
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.17)
    axes[0].set_yticks(np.arange(10), CLASSES)
    fig.tight_layout()
    fig.savefig(output / "prediction_shift.png", bbox_inches="tight")
    plt.close(fig)


def main():
    data_dir = ROOT / "data" / "cifar10.1"
    images = np.load(data_dir / "cifar10.1_v6_data.npy")
    labels = np.load(data_dir / "cifar10.1_v6_labels.npy")
    assert images.shape == (2000, 32, 32, 3) and labels.shape == (2000,)
    conditions = inputs(images)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = {}
    for variant in VARIANTS:
        checkpoint = torch.load(ROOT / "results" / variant / "best.pt",
                                map_location=device, weights_only=False)
        model = build_model(variant).to(device)
        model.load_state_dict(checkpoint["model"])
        records[variant] = {}
        for condition, altered in conditions.items():
            predictions, confidences = predict(model, altered, device)
            correct = int((predictions == labels).sum())
            counts = np.bincount(labels * 10 + predictions, minlength=100).reshape(10, 10)
            records[variant][condition] = {
                "correct": correct, "total": len(labels),
                "accuracy": correct / len(labels),
                "mean_confidence": float(confidences.mean()),
                "per_class_accuracy": {
                    name: float(counts[i, i] / max(1, counts[i].sum()))
                    for i, name in enumerate(CLASSES)},
                "predictions": predictions.tolist(),
            }
            print(variant, condition, correct, len(labels), flush=True)
    output = ROOT / "analysis" / "scene"
    output.mkdir(parents=True, exist_ok=True)
    payload = {"dataset": "CIFAR-10.1 v6", "source":
               "https://github.com/modestyachts/CIFAR-10.1",
               "conditions": {
                   "clean": "original RGB uint8",
                   "blur_r1": "PIL GaussianBlur radius=1.0",
                   "lowlight_0p45": "round(original RGB * 0.45), uint8",
                   "blur_r1_lowlight_0p45": "round(GaussianBlur RGB * 0.45), uint8",
               }, "batch_size": 256, "results": records}
    (output / "scene_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_accuracy(records, output)
    plot_prediction_shift(records, output)
    indices = plot_examples(images, labels, conditions, records, output)
    sample_dir = ROOT / "examples" / "scene_inputs"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for index in indices:
        Image.fromarray(images[index]).save(sample_dir /
                                         f"cifar10_1_{index:04d}_{CLASSES[int(labels[index])]}.png")
    (output / "sample_indices.json").write_text(json.dumps(indices), encoding="utf-8")


if __name__ == "__main__":
    main()
