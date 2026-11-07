"""Download the two public datasets used by the project."""
import hashlib
from pathlib import Path
from urllib.request import urlretrieve

from torchvision import datasets

from run import ROOT


CIFAR10_1 = {
    "cifar10.1_v6_data.npy":
        "2997188e5816f5bd545dc77771b6227828c28146049fcecf3fa10775474cacc6",
    "cifar10.1_v6_labels.npy":
        "ae40beda001693674edc94d925ee8268cfe68905f8f9aff800c8dcdfcd6c9448",
}
BASE_URL = "https://raw.githubusercontent.com/modestyachts/CIFAR-10.1/master/datasets/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    print("Checking CIFAR-10 training and test data", flush=True)
    datasets.CIFAR10(ROOT / "data", train=True, download=True)
    datasets.CIFAR10(ROOT / "data", train=False, download=True)
    folder = ROOT / "data" / "cifar10.1"
    folder.mkdir(parents=True, exist_ok=True)
    for name, expected in CIFAR10_1.items():
        target = folder / name
        if target.exists() and sha256(target) == expected:
            print(f"Verified {name}", flush=True)
            continue
        temporary = folder / f"{name}.part"
        try:
            urlretrieve(BASE_URL + name, temporary)
            if sha256(temporary) != expected:
                raise ValueError(f"SHA-256 mismatch: {name}")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Downloaded and verified {name}", flush=True)


if __name__ == "__main__":
    main()
