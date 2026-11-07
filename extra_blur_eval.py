"""Check whether the blur-trained model transfers to other deterministic blurs."""
import json

import numpy as np
import torch
from PIL import Image, ImageFilter

from model import build_model
from run import ROOT
from scene_eval import predict


def main():
    data = ROOT / "data" / "cifar10.1"
    images = np.load(data / "cifar10.1_v6_data.npy")
    labels = np.load(data / "cifar10.1_v6_labels.npy")
    filters = {
        "box_blur_r1": ImageFilter.BoxBlur(radius=1),
        "motion_blur_horizontal_5": ImageFilter.Kernel(
            (5, 5), [0, 0, 0, 0, 0] * 2 + [1, 1, 1, 1, 1] +
            [0, 0, 0, 0, 0] * 2, scale=5),
        "gaussian_blur_r1p5": ImageFilter.GaussianBlur(radius=1.5),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    for name, folder in (("baseline", "baseline"),
                         ("blur_trained", "robust_baseline")):
        checkpoint = torch.load(ROOT / "results" / folder / "best.pt",
                                map_location=device, weights_only=False)
        model = build_model(checkpoint["variant"]).to(device)
        model.load_state_dict(checkpoint["model"])
        models[name] = model.eval()
    results = {}
    for condition, image_filter in filters.items():
        altered = np.stack([np.asarray(Image.fromarray(image).filter(image_filter))
                            for image in images])
        results[condition] = {}
        for name, model in models.items():
            predictions, _ = predict(model, altered, device)
            correct = int((predictions == labels).sum())
            results[condition][name] = {"correct": correct, "total": len(labels),
                                        "accuracy": correct / len(labels)}
            print(condition, name, correct, len(labels))
    output = ROOT / "analysis" / "robust" / "other_blurs.json"
    output.write_text(json.dumps({"dataset": "CIFAR-10.1 v6",
                                  "conditions": {
                                      "box_blur_r1": "Pillow BoxBlur radius=1",
                                      "motion_blur_horizontal_5": "Pillow 5x5 horizontal mean kernel",
                                      "gaussian_blur_r1p5": "Pillow GaussianBlur radius=1.5",
                                  }, "results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
