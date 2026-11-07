"""Paired robustness comparison on CIFAR-10.1, sourced from actual predictions."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from model import build_model
from run import ROOT
from scene_eval import CONDITIONS, inputs, predict


def bootstrap_ci(difference, seed=2027, repetitions=3000):
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(0, repetitions, 100):
        indices = rng.integers(0, len(difference), size=(100, len(difference)))
        estimates.extend(difference[indices].mean(axis=1).tolist())
    return [float(x) for x in np.quantile(estimates, [0.025, 0.975])]


def main():
    source = ROOT / "data" / "cifar10.1"
    images = np.load(source / "cifar10.1_v6_data.npy")
    labels = np.load(source / "cifar10.1_v6_labels.npy")
    conditions = inputs(images)
    baseline = json.loads((ROOT / "analysis" / "scene" / "scene_metrics.json")
                          .read_text(encoding="utf-8"))["results"]["baseline"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ROOT / "results" / "robust_baseline" / "best.pt",
                            map_location=device, weights_only=False)
    model = build_model(checkpoint["variant"]).to(device)
    model.load_state_dict(checkpoint["model"])
    rows = []
    for condition in CONDITIONS:
        before = np.asarray(baseline[condition]["predictions"])
        after, _ = predict(model, conditions[condition], device)
        before_correct = before == labels
        after_correct = after == labels
        difference = after_correct.astype(np.int8) - before_correct.astype(np.int8)
        rows.append({
            "condition": condition, "total": len(labels),
            "baseline_correct": int(before_correct.sum()),
            "robust_correct": int(after_correct.sum()),
            "baseline_accuracy": float(before_correct.mean()),
            "robust_accuracy": float(after_correct.mean()),
            "difference_percentage_points": float(100 * difference.mean()),
            "difference_95pct_paired_bootstrap_ci_percentage_points":
                [float(100 * x) for x in bootstrap_ci(difference)],
            "gained_images": int((difference == 1).sum()),
            "lost_images": int((difference == -1).sum()),
            "baseline_cat_predictions": int((before == 3).sum()),
            "robust_cat_predictions": int((after == 3).sum()),
        })
    output = ROOT / "analysis" / "robust"
    output.mkdir(parents=True, exist_ok=True)
    (output / "paired_metrics.json").write_text(
        json.dumps({"dataset": "CIFAR-10.1 v6", "source":
                    "https://github.com/modestyachts/CIFAR-10.1",
                    "bootstrap_repetitions": 3000, "bootstrap_seed": 2027,
                    "rows": rows}, indent=2), encoding="utf-8")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=220)
    ax.bar(x - 0.18, [100 * row["baseline_accuracy"] for row in rows],
           width=0.36, color="#27364B", label="Original ResNet-18")
    ax.bar(x + 0.18, [100 * row["robust_accuracy"] for row in rows],
           width=0.36, color="#7469A8", label="Blur-trained ResNet-18")
    ax.set_xticks(x, ["Clean", "Blur", "Low light", "Blur + low light"])
    ax.set(ylabel="CIFAR-10.1 accuracy (%)", ylim=(0, 100))
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout()
    fig.savefig(output / "before_after_accuracy.png", bbox_inches="tight")
    plt.close(fig)
    lines = ["# 模糊鲁棒性改进实测", "",
             "同一 ResNet-18 结构，从原基线最佳权重出发，使用 CIFAR-10 训练集的 45,000 张图加入随机模糊和亮度增强，训练 20 个 epoch。权重只依据原训练集所划 5,000 张验证图选取；CIFAR-10.1 v6 的 2,000 张图用于下面的前后对照。该数据集在阶段 4 已用于发现模糊失效，因而此处分析针对已观察到的场景问题。",
             "", "| 条件 | 原模型 | 改进模型 | 差值（百分点） | 配对 bootstrap 95% 区间 |", "| --- | ---: | ---: | ---: | ---: |"]
    names = {"clean": "原图", "blur_r1": "模糊", "lowlight_0p45": "低照度",
             "blur_r1_lowlight_0p45": "模糊 + 低照度"}
    for row in rows:
        lo, hi = row["difference_95pct_paired_bootstrap_ci_percentage_points"]
        lines.append(f"| {names[row['condition']]} | {100*row['baseline_accuracy']:.2f}% | "
                     f"{100*row['robust_accuracy']:.2f}% | "
                     f"{row['difference_percentage_points']:+.2f} | "
                     f"[{lo:+.2f}, {hi:+.2f}] |")
    original = json.loads((ROOT / "results" / "baseline" / "metrics.json")
                          .read_text(encoding="utf-8"))
    robust = json.loads((ROOT / "results" / "robust_baseline" / "metrics.json")
                        .read_text(encoding="utf-8"))
    lines += ["", f"原 CIFAR-10 测试集清晰图准确率：原模型 {100*original['test']['accuracy']:.2f}%，改进模型 {100*robust['cifar10_test']['clean']['accuracy']:.2f}%。",
              f"独立图像叠加退化时，预测为 `cat` 的数量从 {rows[3]['baseline_cat_predictions']} 张降至 {rows[3]['robust_cat_predictions']} 张（共 2000 张）。",
              "", "区间采用逐图配对 bootstrap，3000 次重采样；它描述这批图像上的抽样波动。"]
    other_file = output / "other_blurs.json"
    if other_file.exists():
        other = json.loads(other_file.read_text(encoding="utf-8"))["results"]
        lines += ["", "## 其他模糊类型", "",
                  "这些变换未直接用于权重选择，仍使用同一批 CIFAR-10.1 图像。",
                  "", "| 模糊条件 | 原模型 | 改进模型 |", "| --- | ---: | ---: |"]
        other_names = {"box_blur_r1": "方框模糊，半径 1",
                       "motion_blur_horizontal_5": "水平运动模糊，核长 5",
                       "gaussian_blur_r1p5": "高斯模糊，半径 1.5"}
        for key, name in other_names.items():
            row = other[key]
            lines.append(f"| {name} | {100*row['baseline']['accuracy']:.2f}% | "
                         f"{100*row['blur_trained']['accuracy']:.2f}% |")
        lines += ["", "方框模糊和更强的高斯模糊也获得明显提升；运动模糊提升较小，改进模型在该条件下仍有较多误判。因而当前结果支持特定模糊条件下的鲁棒性改善，不支持对所有实际相机模糊作统一效果声明。"]
    lines += ["", "原始数字：`analysis/robust/paired_metrics.json`、`analysis/robust/other_blurs.json`、`results/robust_baseline/metrics.json`。对照图：`analysis/robust/before_after_accuracy.png`；应用卡片：`output/robust_blurred_cards/`。"]
    (ROOT / "模糊鲁棒性改进结果.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
