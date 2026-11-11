"""
The model service. The Next.js app sends it a garment image and gets numbers back; it stores nothing.

Run it locally:
    uv run uvicorn app.main:app --port 8000
Then open http://localhost:8000/docs to try it in the browser.

On Cloud Run it's a private service: only the Next.js app's service account can call it (set up at deploy time,
not in this code).
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, UploadFile

from app.model import Matcher, open_image

HEAD_PATH = Path(os.environ.get("HEAD_PATH", Path(__file__).parent.parent / "models" / "head-fashion-clip-redrawn-seed0.pt"))
MAX_UPLOAD_BYTES = 10 * 1024 * 1024   # 10 MB: bigger than any redraw or phone photo needs

matcher: Matcher | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loads the model once when the server starts, not on every request (that's the slow part of a cold start).
    global matcher
    device = os.environ.get("DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
    matcher = Matcher(HEAD_PATH, device)
    yield


app = FastAPI(title="Style.io matcher", lifespan=lifespan)


@app.get("/health")
def health():
    """Cloud Run checks this to know the server is up. Also says which head is loaded."""
    return {"ok": matcher is not None, "head": matcher.head_version if matcher else None}


@app.post("/embed")
async def embed(file: UploadFile):
    """One garment image in, two sets of 512 numbers out: FashionCLIP's raw numbers and the head's version."""
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES: raise HTTPException(413, "image is over 10 MB")
    try:
        image = open_image(data)   # the file name and stated type are ignored; only the bytes count
    except ValueError as error:
        raise HTTPException(415, str(error))

    raw, matched = matcher.embed([image])
    return {"head": matcher.head_version, "raw": raw[0].tolist(), "matched": matched[0].tolist()}
