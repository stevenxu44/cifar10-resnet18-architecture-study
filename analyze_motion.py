"""Evaluate the three ResNet-18 checkpoints on paired motion-blur inputs."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from analyze_robust import bootstrap_ci
from model import build_model
from motion_blur import motion_images
from run import ROOT
from scene_eval import inputs, predict


MODEL_DIRS = {"original": "baseline", "gaussian_trained": "robust_baseline",
              "motion_trained": "motion_robust"}
CONDITIONS = ("clean", "blur_r1", "motion_horizontal_5",
              "motion_vertical_5", "motion_diag_down_5",
              "motion_horizontal_7", "motion_diag_down_7",
              "motion_shallow_down_7")


def main():
    data = ROOT / "data" / "cifar10.1"
    images = np.load(data / "cifar10.1_v6_data.npy")
    labels = np.load(data / "cifar10.1_v6_labels.npy")
    all_inputs = inputs(images)
    for length, direction in ((5, "horizontal"), (5, "vertical"),
                              (5, "diag_down"), (7, "horizontal"),
                              (7, "diag_down"), (7, "shallow_down")):
        all_inputs[f"motion_{direction}_{length}"] = motion_images(
            images, length, direction)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    for name, folder in MODEL_DIRS.items():
        checkpoint = torch.load(ROOT / "results" / folder / "best.pt",
                                map_location=device, weights_only=False)
        model = build_model(checkpoint["variant"]).to(device)
        model.load_state_dict(checkpoint["model"])
        models[name] = model.eval()
    rows = []
    for condition in CONDITIONS:
        predictions = {}
        results = {}
        for name, model in models.items():
            predicted, _ = predict(model, all_inputs[condition], device)
            predictions[name] = predicted
            correct = int((predicted == labels).sum())
            results[name] = {"correct": correct, "total": len(labels),
                             "accuracy": correct / len(labels)}
        before = (predictions["gaussian_trained"] == labels).astype(np.int8)
        after = (predictions["motion_trained"] == labels).astype(np.int8)
        difference = after - before
        lo, hi = bootstrap_ci(difference, seed=2029)
        rows.append({"condition": condition, "models": results,
                     "motion_vs_gaussian_difference_percentage_points":
                         100 * float(difference.mean()),
                     "motion_vs_gaussian_paired_bootstrap_95pct_ci_percentage_points":
                         [100 * lo, 100 * hi],
                     "gained_images": int((difference == 1).sum()),
                     "lost_images": int((difference == -1).sum())})
        print(condition, {k: v["correct"] for k, v in results.items()}, flush=True)
    output = ROOT / "analysis" / "motion"
    output.mkdir(parents=True, exist_ok=True)
    (output / "motion_metrics.json").write_text(
        json.dumps({"dataset": "CIFAR-10.1 v6", "count": len(labels),
                    "bootstrap_repetitions": 3000, "bootstrap_seed": 2029,
                    "conditions": CONDITIONS, "rows": rows}, indent=2), encoding="utf-8")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    x = np.arange(len(rows))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 4), dpi=220)
    for i, (name, color) in enumerate((("original", "#27364B"),
                                        ("gaussian_trained", "#7469A8"),
                                        ("motion_trained", "#3B947F"))):
        values = [100 * row["models"][name]["accuracy"] for row in rows]
        ax.bar(x + (i - 1) * width, values, width=width,
               color=color, label=name.replace("_", " "))
    labels_short = ["Clean", "Gaussian\nr1", "Motion\nH5", "Motion\nV5",
                    "Motion\nD5", "Motion\nH7", "Motion\nD7", "Motion\nShallow7"]
    ax.set_xticks(x, labels_short)
    ax.set(ylabel="CIFAR-10.1 accuracy (%)", ylim=(0, 100))
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout()
    fig.savefig(output / "motion_comparison.png", bbox_inches="tight")
    plt.close(fig)
    lines = ["# 运动模糊增强实验", "",
             "从高斯模糊增强模型继续训练 15 个 epoch；训练时随机施加水平、垂直及两种对角方向的核长 3/5 运动模糊，同时保留高斯模糊和清晰图。使用 CIFAR-10 的 45,000/5,000 训练/验证划分，按清晰图、高斯模糊和水平/垂直核长 5 验证指标选权重。",
             "", "| 条件 | 原基线 | 高斯增强模型 | 运动增强模型 | 相对上一模型（百分点） |", "| --- | ---: | ---: | ---: | ---: |"]
    display = {"clean": "清晰图", "blur_r1": "高斯模糊，半径 1",
               "motion_horizontal_5": "运动模糊：水平，核长 5",
               "motion_vertical_5": "运动模糊：垂直，核长 5",
               "motion_diag_down_5": "运动模糊：对角，核长 5",
               "motion_horizontal_7": "运动模糊：水平，核长 7",
               "motion_diag_down_7": "运动模糊：对角，核长 7",
               "motion_shallow_down_7": "运动模糊：约 27°，核长 7"}
    for row in rows:
        m = row["models"]
        lines.append(f"| {display[row['condition']]} | {100*m['original']['accuracy']:.2f}% | "
                     f"{100*m['gaussian_trained']['accuracy']:.2f}% | "
                     f"{100*m['motion_trained']['accuracy']:.2f}% | "
                     f"{row['motion_vs_gaussian_difference_percentage_points']:+.2f} |")
    selected = next(row for row in rows if row["condition"] == "motion_horizontal_5")
    lo, hi = selected["motion_vs_gaussian_paired_bootstrap_95pct_ci_percentage_points"]
    result = json.loads((ROOT / "results" / "motion_robust" / "metrics.json")
                        .read_text(encoding="utf-8"))
    lines += ["", f"水平核长 5 相对高斯增强模型的提升为 {selected['motion_vs_gaussian_difference_percentage_points']:+.2f} 个百分点，逐图配对 bootstrap 95% 区间 [{lo:+.2f}, {hi:+.2f}]。",
              f"原 CIFAR-10 测试集清晰图准确率：运动增强模型 {100*result['cifar10_test']['clean']['accuracy']:.2f}%（{result['cifar10_test']['clean']['correct']}/10000）。",
              "核长 7 未参与训练或选权重；约 27° 方向也未参与训练。表内 CIFAR-10.1 数字由保存的最佳权重重新推理得到。",
              "", "对比图：`analysis/motion/motion_comparison.png`。示例输入：`examples/motion_inputs/`；六模型逐图卡片：`output/motion_cards/`。",
              "完整逐条件计数及配对区间：`analysis/motion/motion_metrics.json`；训练配置、权重和官方 CIFAR-10 测试结果：`results/motion_robust/`。"]
    (ROOT / "运动模糊增强结果.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
