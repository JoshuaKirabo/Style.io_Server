"""
Josh's matcher: FashionCLIP turns an image into 512 numbers, then the trained head adjusts them towards how people
pick stand-ins. The app only ever calls `Matcher.embed`.

Everything here has to process images exactly the way training did (rule 1), so it copies `load_image` and the
embedding steps from Style.io_ML/baseline.py and the head from train_colab.ipynb. scripts/check_same_numbers.py
proves the numbers match.
"""

import hashlib
import io
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor

FASHION_CLIP = "patrickjohncyh/fashion-clip"

# Pillow refuses images bigger than this many pixels. A crafted file can claim to be enormous and eat all the memory
# (a "decompression bomb"); real phone photos and redraws are far below this.
Image.MAX_IMAGE_PIXELS = 40_000_000


class Head(nn.Module):
    """The same layer as in train_colab.ipynb: 512 -> 128 -> 512, added back to FashionCLIP's numbers."""

    def __init__(self, size=512, hidden=128, dropout=0.1):
        super().__init__()
        self.down = nn.Linear(size, hidden)
        self.up = nn.Linear(hidden, size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.up(self.drop(torch.relu(self.down(x))))
        return F.normalize(x, dim=-1)


def open_image(data: bytes) -> Image.Image:
    """Bytes from an upload, opened the way baseline.load_image opens a file. Raises ValueError if it isn't a real image."""
    try:
        image = Image.open(io.BytesIO(data))
        image.verify()                        # checks the file is a whole, valid image without decoding it all
        image = Image.open(io.BytesIO(data))  # verify() leaves the image unusable, so it's opened again
    except Exception as error:
        raise ValueError("not a readable image") from error
    if image.format not in {"JPEG", "PNG", "WEBP"}:
        raise ValueError(f"{image.format} isn't accepted, only JPEG, PNG or WebP")

    # From here on, the same steps as baseline.load_image.
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA", "P"):
        image = Image.alpha_composite(Image.new("RGBA", image.size, "white"), image.convert("RGBA"))
    return image.convert("RGB")


class Matcher:
    def __init__(self, head_path: Path, device: str = "cpu"):
        self.device = device
        self.clip = CLIPModel.from_pretrained(FASHION_CLIP).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(FASHION_CLIP)

        # weights_only=True loads plain numbers and refuses anything that could run code.
        self.head = Head().to(device).eval()
        self.head.load_state_dict(torch.load(head_path, map_location=device, weights_only=True))

        # A short fingerprint of the head file, stored next to every saved vector, so the database knows which head
        # made it and which rows to recompute after a retrain.
        self.head_version = f"{head_path.stem}-{hashlib.sha256(head_path.read_bytes()).hexdigest()[:8]}"

    @torch.no_grad()
    def embed(self, images: list[Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (raw, matched): FashionCLIP's numbers, and the same after the head. Both are length 1."""
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        output = self.clip.get_image_features(**inputs)
        if not torch.is_tensor(output): output = output.pooler_output   # newer transformers wraps the numbers
        raw = output.float()
        raw = raw / raw.norm(dim=1, keepdim=True)
        return raw.cpu(), self.head(raw).cpu()
