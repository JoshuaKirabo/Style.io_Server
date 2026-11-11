"""
Proves the server turns images into the same numbers as training did (rule 1). If this fails, the app would be
matching with a slightly different model from the one that was tested.

    uv run python scripts/check_same_numbers.py

It sends real crops through the server's /embed (in-process, no network) and compares the answers with:
1. raw: Style.io_ML/baseline.py's embed, the function every score so far came from;
2. matched: the head applied to those numbers, the way train_colab.ipynb does it.
It also sends a fake image and a huge one, which must be turned away.
"""

import io
import sys
from pathlib import Path

import torch
from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Style.io_ML"))
sys.path.insert(0, str(ROOT / "Style.io_Server"))
from baseline import embed as baseline_embed, load_model   # noqa: E402

from app.main import app   # noqa: E402

samples = sorted((ROOT / "Style.io_Data" / "tune").glob("*/redrawn/pieces/*.png"))[:4]
samples += sorted((ROOT / "Style.io_Data" / "tune").glob("*/closet/*/*.png"))[:2]   # rough crops too
if not samples: raise SystemExit("No sample images found in Style.io_Data")

with TestClient(app) as client:   # starts the server's lifespan, so the model loads as it would for real
    matcher = sys.modules["app.main"].matcher
    print("health:", client.get("/health").json())

    served_raw, served_matched = [], []
    for path in samples:
        reply = client.post("/embed", files={"file": (path.name, path.read_bytes(), "image/png")})
        assert reply.status_code == 200, reply.text
        served_raw.append(reply.json()["raw"]); served_matched.append(reply.json()["matched"])
    served_raw, served_matched = torch.tensor(served_raw), torch.tensor(served_matched)

    model, processor = load_model("fashion-clip", matcher.device)
    expected_raw = torch.from_numpy(baseline_embed(samples, model, processor, matcher.device))
    with torch.no_grad(): expected_matched = matcher.head(expected_raw.to(matcher.device)).cpu()

    raw_gap = (served_raw - expected_raw).abs().max().item()
    matched_gap = (served_matched - expected_matched).abs().max().item()
    print(f"{len(samples)} images. Largest difference, raw: {raw_gap:.2e}, matched: {matched_gap:.2e}")
    # Tiny differences come from batching and the GPU's rounding; anything above 1e-4 would be a real mismatch.
    assert raw_gap < 1e-4 and matched_gap < 1e-4, "the server's numbers don't match training's"

    fake = client.post("/embed", files={"file": ("cat.png", b"not an image at all", "image/png")})
    assert fake.status_code == 415, fake.status_code
    huge = io.BytesIO(); Image.new("RGB", (9000, 9000), "white").save(huge, "PNG")   # 81M pixels, over the guard
    bomb = client.post("/embed", files={"file": ("big.png", huge.getvalue(), "image/png")})
    assert bomb.status_code == 415, bomb.status_code
    print("Fake and oversized images were turned away.")

print("All good: the server matches training.")
