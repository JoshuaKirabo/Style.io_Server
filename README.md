# Style.io matcher service

A small FastAPI service holding the matcher: FashionCLIP plus the trained head from `Style.io_ML/train_colab.ipynb`.
The Next.js app sends it one garment image at a time and gets numbers back. It stores nothing.

## Run it on the Mac

```sh
uv sync
uv run uvicorn app.main:app --port 8000
```

Open http://localhost:8000/docs to try `/embed` in the browser.

## Check it matches training

```sh
uv run python scripts/check_same_numbers.py
```

Sends real crops through `/embed` and compares the answers with `Style.io_ML/baseline.py` (the function every score
came from) and the notebook's head. It also checks that fake and oversized images are turned away. Run it after any
change to `app/model.py`, a new head, or new library versions.

## Endpoints

| | |
|---|---|
| `GET /health` | `{"ok": true, "head": "<head version>"}`. Cloud Run uses it to know the server is up. |
| `POST /embed` | Form upload, field `file`: one JPEG, PNG or WebP, up to 10 MB. Returns `head` (which head made the numbers), `raw` (FashionCLIP's 512 numbers) and `matched` (the same after the head). |

Store both: `raw` never changes, so after a retrain the `matched` numbers can be recomputed from it without the image.

## A new head

Train in the notebook, put the new `.pt` file in `models/`, and point `HEAD_PATH` at it (or replace the default file).
The head version in every reply changes with it. Head files are kept out of git (`.gitignore`).

## Security

- **Uploads:** 10 MB limit; only real JPEG, PNG or WebP (checked from the bytes, the file name is ignored); images over
  40 million pixels are refused, which stops "decompression bomb" files.
- **The head** is loaded with `weights_only=True`, so a tampered file can't run code.
- **Nothing is kept:** no images are saved or logged.
- **On Cloud Run** (not set up yet): deploy as a private service that only the Next.js app's service account can call,
  with a maximum instance count and a request timeout.
