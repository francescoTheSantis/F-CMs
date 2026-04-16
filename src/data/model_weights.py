from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.error import URLError

from torch.hub import download_url_to_file


RESNET18_IMAGENET_FILENAME = "resnet18-5c106cde.pth"
RESNET18_IMAGENET_SHA256 = "5c106cde386e87d4033832f2996f5493238eda96ccf559d1d62760c4de0613f8"
RESNET18_IMAGENET_URLS = [
    "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "https://huggingface.co/ManyOtherFunctions/face-parse-bisent/resolve/main/resnet18-5c106cde.pth",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_resnet18_imagenet_weights(model_directory: str | Path) -> Path:
    model_directory = Path(model_directory)
    model_directory.mkdir(parents=True, exist_ok=True)
    destination = model_directory / RESNET18_IMAGENET_FILENAME

    if destination.exists():
        if _sha256(destination) == RESNET18_IMAGENET_SHA256:
            return destination
        print(f"Checksum mismatch for {destination}. Re-downloading the weights file.")
        destination.unlink()

    last_error = None
    for url in RESNET18_IMAGENET_URLS:
        try:
            print(f"Downloading {RESNET18_IMAGENET_FILENAME} from {url}")
            download_url_to_file(url, str(destination), progress=True)
            if _sha256(destination) != RESNET18_IMAGENET_SHA256:
                raise RuntimeError(f"Checksum mismatch after downloading from {url}")
            return destination
        except (URLError, RuntimeError, OSError) as exc:
            last_error = exc
            if destination.exists():
                destination.unlink()

    raise RuntimeError(
        f"Unable to download {RESNET18_IMAGENET_FILENAME} to {destination}"
    ) from last_error
