"""Create a small, labeled gallery of motion-blurred CIFAR-10.1 examples."""
from pathlib import Path

from PIL import Image

from motion_blur import apply_motion
from run import ROOT


def main():
    source = ROOT / "examples" / "scene_inputs"
    output = ROOT / "examples" / "motion_inputs"
    output.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.glob("*.png")):
        image = Image.open(path).convert("RGB")
        for length, direction, suffix in ((5, "horizontal", "h5"),
                                          (7, "shallow_down", "shallow7")):
            destination = output / f"{path.stem}_{suffix}.png"
            apply_motion(image, length, direction).save(destination)
            print(destination)


if __name__ == "__main__":
    main()
