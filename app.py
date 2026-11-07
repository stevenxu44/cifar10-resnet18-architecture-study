"""Classify a user image or folder and save readable result cards."""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision import transforms

from model import build_model
from run import CLASSES, MEAN, ROOT, STD


VARIANTS = ("baseline", "depthwise", "se", "depthwise_se")
COLORS = {"baseline": "#27364B", "depthwise": "#447A9C",
          "se": "#BB704C", "depthwise_se": "#4C907B",
          "robust_baseline": "#7469A8", "motion_robust": "#3B947F"}
DISPLAY_NAMES = {"baseline": "ResNet-18 baseline", "depthwise": "depthwise",
                 "se": "SE", "depthwise_se": "depthwise + SE",
                 "robust_baseline": "blur-trained ResNet-18",
                 "motion_robust": "motion-trained ResNet-18"}


def font(size):
    path = Path("C:/Windows/Fonts/segoeui.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def card(image, name, predictions, variants, destination):
    height = 520 + max(0, len(variants) - 4) * 88
    canvas = Image.new("RGB", (880, height), "#F8F9FB")
    draw = ImageDraw.Draw(canvas)
    draw.text((36, 25), "CIFAR-10 image classification", fill="#202B39", font=font(27))
    draw.text((36, 65), name, fill="#677381", font=font(17))
    draw.rounded_rectangle((35, 108, 424, height - 52), radius=14, fill="white",
                           outline="#E1E6EA", width=2)
    interpolation = Image.Resampling.NEAREST if max(image.size) <= 64 else Image.Resampling.LANCZOS
    preview = ImageOps.contain(image, (340, height - 210), method=interpolation)
    canvas.paste(preview, (229 - preview.width // 2,
                           (108 + height - 52 - preview.height) // 2))
    draw.text((35, height - 40), f"Input: {image.width} x {image.height} px",
              fill="#677381", font=font(15))
    for i, variant in enumerate(variants):
        top = 112 + i * 88
        draw.rounded_rectangle((455, top, 845, top + 72), radius=11,
                               fill="white", outline="#E1E6EA", width=2)
        draw.rectangle((455, top + 10, 461, top + 62), fill=COLORS[variant])
        item = predictions[variant]
        draw.text((480, top + 8), DISPLAY_NAMES[variant],
                  fill="#596574", font=font(16))
        draw.text((480, top + 34), item["class"], fill="#202B39", font=font(20))
        draw.text((755, top + 38), f"{item['confidence']:.1%}",
                  fill=COLORS[variant], font=font(17))
    draw.text((455, height - 38), "Predictions restricted to CIFAR-10 classes",
              fill="#677381", font=font(14))
    canvas.save(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(ROOT / "examples" / "scene_inputs"))
    parser.add_argument("--output", default=str(ROOT / "output" / "scene_cards"))
    parser.add_argument("--include-robust", action="store_true")
    parser.add_argument("--include-motion", action="store_true")
    args = parser.parse_args()
    source = Path(args.input)
    paths = sorted(p for p in source.iterdir() if p.suffix.lower() in
                   {".png", ".jpg", ".jpeg", ".bmp"}) if source.is_dir() else [source]
    if not paths:
        raise ValueError(f"No supported images at {source}")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variants = VARIANTS + (("robust_baseline",) if args.include_robust else ())
    variants += (("motion_robust",) if args.include_motion else ())
    models = {}
    for variant in variants:
        checkpoint = torch.load(ROOT / "results" / variant / "best.pt",
                                map_location=device, weights_only=False)
        model = build_model(checkpoint["variant"]).to(device)
        model.load_state_dict(checkpoint["model"])
        models[variant] = model.eval()
    transform = transforms.Compose([
        transforms.Resize((32, 32)), transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    results = []
    with torch.inference_mode():
        for path in paths:
            image = Image.open(path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            predictions = {}
            for variant, model in models.items():
                probabilities = model(tensor).softmax(1)[0]
                confidence, index = probabilities.max(0)
                predictions[variant] = {"class": CLASSES[index.item()],
                                        "confidence": confidence.item()}
            destination = output / f"{path.stem}_result.png"
            card(image, path.name, predictions, variants, destination)
            results.append({"image": str(path), "card": str(destination),
                            "predictions": predictions})
            print(path.name, {v: p["class"] for v, p in predictions.items()})
    (output / "predictions.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
