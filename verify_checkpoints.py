"""Recompute clean CIFAR-10 accuracy from every saved main checkpoint."""
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import build_model
from run import MEAN, ROOT, STD, evaluate
from scene_eval import predict


VARIANTS = ("baseline", "depthwise", "se", "depthwise_se",
            "robust_baseline", "motion_robust")


def main():
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    dataset = datasets.CIFAR10(ROOT / "data", train=False, download=False,
                               transform=transform)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for variant in VARIANTS:
        folder = ROOT / "results" / variant
        checkpoint = torch.load(folder / "best.pt", map_location=device,
                                weights_only=False)
        model = build_model(checkpoint["variant"]).to(device)
        model.load_state_dict(checkpoint["model"])
        metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        if "test" in metrics:
            observed = evaluate(model, loader, device)
            expected = metrics["test"]
            correct = observed["correct"]
        else:
            predictions, _ = predict(model, np.asarray(dataset.data), device)
            expected = metrics["cifar10_test"]["clean"]
            correct = int((predictions == np.asarray(dataset.targets)).sum())
        print(json.dumps({"model": variant, "observed": correct,
                          "recorded": expected["correct"],
                          "total": len(dataset)}), flush=True)
        if correct != expected["correct"]:
            raise AssertionError(f"Accuracy mismatch: {variant}")


if __name__ == "__main__":
    main()
