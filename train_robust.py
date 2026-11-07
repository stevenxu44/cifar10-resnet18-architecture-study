"""Fine-tune the trained baseline for blur robustness on CIFAR-10 train/val only."""
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import ImageEnhance, ImageFilter
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import build_model
from run import MEAN, ROOT, STD, seed_all
from scene_eval import CONDITIONS, inputs, predict


class BlurLight:
    def __call__(self, image):
        if random.random() < 0.60:
            image = image.filter(ImageFilter.GaussianBlur(
                radius=random.uniform(0.7, 1.3)))
        if random.random() < 0.20:
            image = ImageEnhance.Brightness(image).enhance(
                random.uniform(0.45, 1.0))
        return image


@torch.inference_mode()
def condition_scores(model, conditions, labels, device):
    scores = {}
    for name, images in conditions.items():
        predictions, _ = predict(model, images, device)
        correct = int((predictions == labels).sum())
        scores[name] = {"correct": correct, "total": len(labels),
                        "accuracy": correct / len(labels)}
    return scores


def main():
    seed_all(2027)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    output = ROOT / "results" / "robust_baseline"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(ROOT / "results" / "baseline" / "best.pt",
                            map_location=device, weights_only=False)
    model = build_model("baseline").to(device)
    model.load_state_dict(checkpoint["model"])
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        BlurLight(), transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    train_data = datasets.CIFAR10(ROOT / "data", train=True,
                                  download=False, transform=transform)
    permutation = torch.randperm(50000,
                                 generator=torch.Generator().manual_seed(2026))
    train_ids = permutation[:45000].tolist()
    val_ids = permutation[45000:].tolist()
    train_loader = DataLoader(Subset(train_data, train_ids), batch_size=128,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_images = np.asarray(train_data.data)[val_ids]
    val_labels = np.asarray(train_data.targets)[val_ids]
    val_conditions = inputs(val_images)
    test_data = datasets.CIFAR10(ROOT / "data", train=False, download=False)
    test_conditions = inputs(np.asarray(test_data.data))
    test_labels = np.asarray(test_data.targets)
    new_dir = ROOT / "data" / "cifar10.1"
    new_images = np.load(new_dir / "cifar10.1_v6_data.npy")
    new_labels = np.load(new_dir / "cifar10.1_v6_labels.npy")
    new_conditions = inputs(new_images)
    config = {
        "base_checkpoint": "results/baseline/best.pt",
        "train_count": 45000, "val_count": 5000, "split_seed": 2026,
        "seed": 2027, "epochs": 20, "batch_size": 128,
        "optimizer": "SGD momentum=0.9 nesterov=True weight_decay=5e-4",
        "initial_lr": 0.01, "lr_schedule": "cosine to zero",
        "blur_probability": 0.60, "blur_radius_uniform": [0.7, 1.3],
        "brightness_probability": 0.20,
        "brightness_factor_uniform": [0.45, 1.0],
        "selection": "max 0.3*val_clean+0.4*val_blur+0.3*val_combined with val_clean>=0.90",
        "device": str(device), "torch": torch.__version__,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                                weight_decay=5e-4, nesterov=True)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss()
    history = []
    best_score = float("-inf")
    for epoch in range(20):
        lr = 0.005 * (1 + math.cos(math.pi * epoch / 20))
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        train_loss = 0.0
        train_correct = 0
        total = 0
        start = time.perf_counter()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            n = len(labels)
            total += n
            train_loss += loss.item() * n
            train_correct += (logits.argmax(1) == labels).sum().item()
        validation = condition_scores(model, val_conditions, val_labels, device)
        score = (0.3 * validation["clean"]["accuracy"] +
                 0.4 * validation["blur_r1"]["accuracy"] +
                 0.3 * validation["blur_r1_lowlight_0p45"]["accuracy"])
        eligible = validation["clean"]["accuracy"] >= 0.90
        row = {"epoch": epoch + 1, "lr": lr,
               "train_loss": train_loss / total,
               "train_accuracy": train_correct / total,
               "validation": validation, "selection_score": score,
               "eligible": eligible, "seconds": time.perf_counter() - start}
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if eligible and score > best_score:
            best_score = score
            torch.save({"model": model.state_dict(), "variant": "baseline",
                        "epoch": epoch + 1, "selection_score": score,
                        "validation": validation}, output / "best.pt")
        print(json.dumps({"epoch": epoch + 1, "val_clean": validation["clean"]["accuracy"],
                          "val_blur": validation["blur_r1"]["accuracy"],
                          "val_combined": validation["blur_r1_lowlight_0p45"]["accuracy"],
                          "score": score, "eligible": eligible,
                          "seconds": row["seconds"]}), flush=True)
    if best_score == float("-inf"):
        raise RuntimeError("No checkpoint met the clean-validation accuracy constraint")
    best = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    result = {"best_epoch": best["epoch"],
              "validation": best["validation"],
              "cifar10_test": condition_scores(model, test_conditions, test_labels, device),
              "cifar10_1_v6": condition_scores(model, new_conditions, new_labels, device)}
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"result": result}), flush=True)


if __name__ == "__main__":
    main()
