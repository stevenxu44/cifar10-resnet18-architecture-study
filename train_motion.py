"""Fine-tune the blur-trained ResNet-18 with varied motion-blur kernels."""
import json
import math
import random
import time

import numpy as np
import torch
from PIL import ImageEnhance, ImageFilter
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import build_model
from motion_blur import apply_motion, motion_images
from run import MEAN, ROOT, STD, seed_all
from scene_eval import inputs, predict


class MixedBlur:
    def __call__(self, image):
        choice = random.random()
        if choice < 0.35:
            image = apply_motion(image, random.choice((3, 5)),
                                 random.choice(("horizontal", "vertical",
                                                "diag_down", "diag_up")))
        elif choice < 0.70:
            image = image.filter(ImageFilter.GaussianBlur(
                radius=random.uniform(0.7, 1.3)))
        if random.random() < 0.20:
            image = ImageEnhance.Brightness(image).enhance(
                random.uniform(0.45, 1.0))
        return image


@torch.inference_mode()
def scores(model, conditions, labels, device):
    result = {}
    for name, images in conditions.items():
        predicted, _ = predict(model, images, device)
        correct = int((predicted == labels).sum())
        result[name] = {"correct": correct, "total": len(labels),
                        "accuracy": correct / len(labels)}
    return result


def selection(validation):
    weights = {"clean": 0.15, "blur_r1": 0.20,
               "blur_r1_lowlight_0p45": 0.15,
               "motion_horizontal_5": 0.25,
               "motion_vertical_5": 0.25}
    score = sum(weight * validation[name]["accuracy"]
                for name, weight in weights.items())
    eligible = (validation["clean"]["accuracy"] >= 0.92 and
                validation["blur_r1"]["accuracy"] >= 0.85)
    return score, eligible


def main():
    seed_all(2028)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    source = ROOT / "results" / "robust_baseline" / "best.pt"
    checkpoint = torch.load(source, map_location=device, weights_only=False)
    model = build_model("baseline").to(device)
    model.load_state_dict(checkpoint["model"])
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(), MixedBlur(),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    dataset = datasets.CIFAR10(ROOT / "data", train=True,
                               download=False, transform=transform)
    ids = torch.randperm(50000,
                         generator=torch.Generator().manual_seed(2026))
    train_ids, val_ids = ids[:45000].tolist(), ids[45000:].tolist()
    train_loader = DataLoader(Subset(dataset, train_ids), batch_size=128,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_images = np.asarray(dataset.data)[val_ids]
    val_labels = np.asarray(dataset.targets)[val_ids]
    val_conditions = inputs(val_images)
    val_conditions["motion_horizontal_5"] = motion_images(
        val_images, 5, "horizontal")
    val_conditions["motion_vertical_5"] = motion_images(
        val_images, 5, "vertical")
    # Only selection conditions are evaluated each epoch.
    val_conditions.pop("lowlight_0p45")
    output = ROOT / "results" / "motion_robust"
    output.mkdir(parents=True, exist_ok=True)
    config = {"source_checkpoint": "results/robust_baseline/best.pt",
              "seed": 2028, "split_seed": 2026,
              "train_count": 45000, "val_count": 5000,
              "epochs": 15, "batch_size": 128,
              "motion_probability": 0.35,
              "motion_train_lengths": [3, 5],
              "motion_train_directions": ["horizontal", "vertical", "diag_down", "diag_up"],
              "gaussian_probability": 0.35,
              "gaussian_radius_uniform": [0.7, 1.3],
              "brightness_probability": 0.20,
              "brightness_factor_uniform": [0.45, 1.0],
              "clean_probability_before_brightness": 0.30,
              "optimizer": "SGD momentum=0.9 nesterov=True weight_decay=5e-4",
              "initial_lr": 0.005, "selection":
              "0.15 clean + 0.20 gaussian + 0.15 gaussian_lowlight + 0.25 horizontal5 + 0.25 vertical5; clean>=0.92, gaussian>=0.85",
              "device": str(device), "torch": torch.__version__}
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    starting_validation = scores(model, val_conditions, val_labels, device)
    best_score, _ = selection(starting_validation)
    torch.save({"model": model.state_dict(), "variant": "baseline",
                "epoch": 0, "validation": starting_validation,
                "selection_score": best_score}, output / "best.pt")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9,
                                weight_decay=5e-4, nesterov=True)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss()
    history = []
    print(json.dumps({"epoch": 0, "validation": starting_validation,
                      "score": best_score}), flush=True)
    for epoch in range(15):
        lr = 0.0025 * (1 + math.cos(math.pi * epoch / 15))
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        total = 0
        loss_sum = 0.0
        correct = 0
        start = time.perf_counter()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            n = len(labels)
            total += n
            loss_sum += loss.item() * n
            correct += (logits.argmax(1) == labels).sum().item()
        validation = scores(model, val_conditions, val_labels, device)
        score, eligible = selection(validation)
        row = {"epoch": epoch + 1, "lr": lr,
               "train_loss": loss_sum / total,
               "train_accuracy": correct / total,
               "validation": validation, "selection_score": score,
               "eligible": eligible, "seconds": time.perf_counter() - start}
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if eligible and score > best_score:
            best_score = score
            torch.save({"model": model.state_dict(), "variant": "baseline",
                        "epoch": epoch + 1, "validation": validation,
                        "selection_score": score}, output / "best.pt")
        print(json.dumps({"epoch": epoch + 1,
                          "clean": validation["clean"]["accuracy"],
                          "gaussian": validation["blur_r1"]["accuracy"],
                          "motion_h5": validation["motion_horizontal_5"]["accuracy"],
                          "motion_v5": validation["motion_vertical_5"]["accuracy"],
                          "score": score, "eligible": eligible,
                          "seconds": row["seconds"]}), flush=True)
    best = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    cifar_test = datasets.CIFAR10(ROOT / "data", train=False, download=False)
    cifar_images = np.asarray(cifar_test.data)
    cifar_labels = np.asarray(cifar_test.targets)
    cifar_conditions = inputs(cifar_images)
    cifar_conditions = {key: cifar_conditions[key] for key in ("clean", "blur_r1")}
    new_dir = ROOT / "data" / "cifar10.1"
    new_images = np.load(new_dir / "cifar10.1_v6_data.npy")
    new_labels = np.load(new_dir / "cifar10.1_v6_labels.npy")
    new_conditions = inputs(new_images)
    for length, direction in ((5, "horizontal"), (5, "vertical"),
                              (5, "diag_down"), (7, "horizontal"),
                              (7, "diag_down")):
        new_conditions[f"motion_{direction}_{length}"] = motion_images(
            new_images, length, direction)
    result = {"best_epoch": best["epoch"],
              "validation": best["validation"],
              "cifar10_test": scores(model, cifar_conditions, cifar_labels, device),
              "cifar10_1_v6": scores(model, new_conditions, new_labels, device)}
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"result": result}), flush=True)


if __name__ == "__main__":
    main()
