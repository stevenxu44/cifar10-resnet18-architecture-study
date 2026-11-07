"""Train, evaluate, profile, and classify images with CIFAR ResNet-18 variants."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import build_model


ROOT = Path(__file__).resolve().parent
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog",
           "frog", "horse", "ship", "truck"]
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_loaders(batch_size, workers, seed, limit_train=0):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    train_set = datasets.CIFAR10(ROOT / "data", train=True,
                                 download=True, transform=train_transform)
    val_set = datasets.CIFAR10(ROOT / "data", train=True,
                               download=False, transform=test_transform)
    test_set = datasets.CIFAR10(ROOT / "data", train=False,
                                download=False, transform=test_transform)
    permutation = torch.randperm(50000,
                                 generator=torch.Generator().manual_seed(2026))
    train_ids = permutation[:45000].tolist()
    val_ids = permutation[45000:].tolist()
    if limit_train:
        train_ids = train_ids[:limit_train]
    train_loader = DataLoader(Subset(train_set, train_ids), batch_size=batch_size,
                              shuffle=True, num_workers=workers,
                              pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(Subset(val_set, val_ids), batch_size=batch_size,
                            shuffle=False, num_workers=workers,
                            pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=workers,
                             pin_memory=torch.cuda.is_available())
    return train_loader, val_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    confusion = torch.zeros(10, 10, dtype=torch.long)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += criterion(logits, labels).item()
        predictions = logits.argmax(1)
        correct += (predictions == labels).sum().item()
        total += len(labels)
        counts = torch.bincount((labels * 10 + predictions).cpu(),
                                minlength=100).reshape(10, 10)
        confusion += counts
    per_class = (confusion.diag() / confusion.sum(1).clamp(min=1)).tolist()
    return {"loss": loss_sum / total, "accuracy": correct / total,
            "correct": correct, "total": total,
            "per_class_accuracy": dict(zip(CLASSES, per_class)),
            "confusion_matrix": confusion.tolist()}


def profile(model, device):
    from thop import profile as thop_profile
    model.eval()
    sample = torch.randn(1, 3, 32, 32, device=device)
    macs, _ = thop_profile(model, inputs=(sample,), verbose=False)
    parameters = sum(p.numel() for p in model.parameters())
    # Synchronize around timing; single-image latency is hardware-dependent.
    with torch.inference_mode():
        for _ in range(30):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return {"parameters": parameters, "macs_per_image": int(macs),
            "latency_ms_per_image": 1000 * (time.perf_counter() - start) / 100}


def train(args):
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    train_loader, val_loader, test_loader = get_loaders(
        args.batch_size, args.workers, args.seed, args.limit_train)
    model = build_model(args.variant).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=5e-4, nesterov=True)
    warmup_epochs = min(5, max(1, args.epochs // 10))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max(1, args.epochs - warmup_epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss()
    output = ROOT / "results" / (args.tag or args.variant)
    output.mkdir(parents=True, exist_ok=True)
    config = {"variant": args.variant, "epochs": args.epochs,
              "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
              "split_seed": 2026, "train_count": len(train_loader.dataset),
              "val_count": len(val_loader.dataset), "test_count": len(test_loader.dataset),
              "device": str(device), "torch": torch.__version__,
              "warmup_epochs": warmup_epochs}
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    history = []
    best_accuracy = -1.0
    for epoch in range(args.epochs):
        if epoch < warmup_epochs:
            warmup_lr = args.lr * (epoch + 1) / warmup_epochs
            for group in optimizer.param_groups:
                group["lr"] = warmup_lr
        model.train()
        loss_sum = 0.0
        correct = 0
        count = 0
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
            count += len(labels)
            loss_sum += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
        validation = evaluate(model, val_loader, device)
        row = {"epoch": epoch + 1, "train_loss": loss_sum / count,
               "train_accuracy": correct / count,
               "val_loss": validation["loss"],
               "val_accuracy": validation["accuracy"],
               "lr": optimizer.param_groups[0]["lr"],
               "seconds": time.perf_counter() - start}
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            torch.save({"model": model.state_dict(), "variant": args.variant,
                        "epoch": epoch + 1, "val_accuracy": best_accuracy},
                       output / "best.pt")
        print(json.dumps(row), flush=True)
        if epoch >= warmup_epochs - 1:
            scheduler.step()
    checkpoint = torch.load(output / "best.pt", map_location=device,
                            weights_only=False)
    model.load_state_dict(checkpoint["model"])
    result = {"best_epoch": checkpoint["epoch"],
              "best_val_accuracy": checkpoint["val_accuracy"],
              "test": evaluate(model, test_loader, device),
              "profile": profile(model, device)}
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"result": result}, ensure_ascii=False), flush=True)


def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    model = build_model(checkpoint["variant"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((32, 32)), transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    source = Path(args.input)
    paths = sorted(p for p in source.iterdir() if p.suffix.lower() in
                   {".jpg", ".jpeg", ".png", ".bmp"}) if source.is_dir() else [source]
    if not paths or any(not path.is_file() for path in paths):
        raise ValueError(f"No supported images at {source}")
    rows = []
    with torch.inference_mode():
        for path in paths:
            image = Image.open(path).convert("RGB")
            probabilities = torch.softmax(model(transform(image).unsqueeze(0).to(device)), 1)[0]
            value, index = probabilities.max(0)
            row = {"image": str(path), "class": CLASSES[index.item()],
                   "confidence": value.item()}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    training.add_argument("--variant", choices=["baseline", "depthwise", "se", "depthwise_se"], required=True)
    training.add_argument("--epochs", type=int, default=80)
    training.add_argument("--batch-size", type=int, default=128)
    training.add_argument("--workers", type=int, default=2)
    training.add_argument("--lr", type=float, default=0.1)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--limit-train", type=int, default=0)
    training.add_argument("--tag", default="")
    inference = sub.add_parser("predict")
    inference.add_argument("--weights", required=True)
    inference.add_argument("--input", required=True)
    inference.add_argument("--output", default=str(ROOT / "output" / "predictions.json"))
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
