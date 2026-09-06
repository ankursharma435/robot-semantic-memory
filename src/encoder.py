"""
encoder.py

Wraps a Matryoshka-trained, multimodal (image + text) embedding model.

Model choice: jinaai/jina-clip-v2
  - ~0.86B params total (561M text tower + 304M vision tower), CLIP-style,
    trained with Matryoshka Representation Learning so any prefix of the
    output vector (64 / 128 / 256 / 512 / 768 / 1024 dims) is itself a
    valid embedding.
  - Image and text land in the SAME space, which is exactly what lets a
    natural-language query ("red mug") retrieve a stored image embedding.

IMPORTANT NUANCE (see deck slide 09):
  Truncating the vector does NOT make the encoder's forward pass cheaper.
  The backbone runs once regardless of how many dimensions you keep.
  What truncation buys you is cheaper STORAGE and cheaper SIMILARITY
  COMPARISON downstream (smaller vectors = less to store, fewer FLOPs per
  dot product, smaller index). So: always encode at full resolution once,
  then slice + re-normalize for whichever tier you're writing to or
  comparing against. That's what `truncate()` below does.
"""

from __future__ import annotations
import numpy as np
from PIL import Image

_MODEL = None
FULL_DIM = 1024
TIER_DIMS = {"short": 64, "medium": 256, "long": 768}


def _get_model():
    """Lazy-load the model once per process. Loading is the slow part
    (a few seconds, more on first run while weights download) — do this
    once at program start, not per frame."""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(
            "jinaai/jina-clip-v2",
            trust_remote_code=True,
        )
    return _MODEL


def truncate(vec: np.ndarray, dim: int) -> np.ndarray:
    """Take the first `dim` numbers of an MRL embedding and re-normalize.
    Re-normalizing matters: a raw prefix of a unit vector is no longer
    unit length, and cosine-similarity comparisons assume unit vectors."""
    v = vec[:dim]
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def embed_image(image: Image.Image) -> np.ndarray:
    """Encode a PIL image at full (1024-dim) resolution. Slice with
    truncate() afterwards for whichever tier you need."""
    model = _get_model()
    vec = model.encode([image], normalize_embeddings=True)[0]
    return np.asarray(vec, dtype=np.float32)


def embed_text(text: str) -> np.ndarray:
    """Encode a natural-language query into the same space as images."""
    model = _get_model()
    vec = model.encode([text], normalize_embeddings=True)[0]
    return np.asarray(vec, dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Both inputs are assumed pre-normalized (unit length)."""
    return float(np.dot(a, b))
