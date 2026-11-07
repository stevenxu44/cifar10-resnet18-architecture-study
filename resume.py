"""Continue an interrupted model-only run with an explicitly reset optimizer."""
import json
import math
import time
from pathlib import Path

import torch
from torch import nn

from model import build_model
from run import ROOT, evaluate, get_loaders, profile, seed_all


def main():
    source = ROOT / "results" / "baseline_seed123"
    output = ROOT / "results" / "baseline_seed123_resumed"
    output.mkdir(parents=True, exist_ok=True)
    history = json.loads((source / "history.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(source / "best.pt", map_location="cpu", weights_only=False)
    start_epoch = int(checkpoint["epoch"])
    assert start_epoch == 52 and len(history) == 53
    history = history[:start_epoch]
    seed_all(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    train_loader, val_loader, test_loader = get_loaders(128, 2, 123)
    model = build_model("baseline").to(device)
    model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                                weight_decay=5e-4, nesterov=True)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss()
    config = {
        "variant": "baseline", "seed": 123, "target_epochs": 80,
        "resume_from_epoch": start_epoch,
        "resume_from_best_val_accuracy": checkpoint["val_accuracy"],
        "optimizer_reinitialized": True,
        "original_epoch_53_discarded": True,
        "batch_size": 128, "split_seed": 2026,
        "train_count": len(train_loader.dataset),
        "val_count": len(val_loader.dataset),
        "test_count": len(test_loader.dataset),
        "device": str(device), "torch": torch.__version__,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    best_accuracy = float(checkpoint["val_accuracy"])
    torch.save(checkpoint, output / "best.pt")
    for epoch in range(start_epoch, 80):
        # Match the remaining cosine schedule; momentum and AMP scaler restart.
        lr = 0.05 * (1 + math.cos(math.pi * (epoch + 1 - 5) / 75))
        for group in optimizer.param_groups:
            group["lr"] = lr
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
               "lr": lr, "seconds": time.perf_counter() - start}
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            torch.save({"model": model.state_dict(), "variant": "baseline",
                        "epoch": epoch + 1, "val_accuracy": best_accuracy},
                       output / "best.pt")
        print(json.dumps(row), flush=True)
    best = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    result = {"best_epoch": best["epoch"],
              "best_val_accuracy": best["val_accuracy"],
              "test": evaluate(model, test_loader, device),
              "profile": profile(model, device)}
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"result": result}), flush=True)


if __name__ == "__main__":
    main()
