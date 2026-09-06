"""
nvidia_nim.py

Two REST clients against NVIDIA's hosted NIM microservices at
build.nvidia.com. Both use the same NVIDIA_API_KEY.

IMPORTANT: NVIDIA's exact request/response JSON shape can differ slightly
per model and does shift over time. Each model's page on build.nvidia.com
has an "API" tab with a working Python/cURL sample for THAT model — if a
call here fails, that sample is the ground truth to check first, not this
file. The two spots most likely to need a tweak are marked below.

Nothing in this file has been tested against a live key (this was written
without network access) — treat the first run as the real test, and paste
any error back if something doesn't match.
"""

from __future__ import annotations
import base64
import os
import time
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("NVIDIA_API_KEY", "")
BASE_URL = os.environ.get("NVIDIA_INTEGRATE_BASE_URL", "https://integrate.api.nvidia.com/v1")

# The fixed-width baseline embedder.
#
# CHANGED 2026-09-04: NV-CLIP (nvidia/nvclip) is DISCONTINUED by NVIDIA and
# 404s on this account ("Function '3072eebf-...': Not found for account").
# Replaced with nvidia/llama-nemotron-embed-vl-1b-v2, verified live:
#   - multimodal: embeds BOTH text and images, in one shared space
#   - fixed width 2048 dims, with no meaningful truncation — which is
#     exactly the property this comparison exists to highlight
#   - ~0.8s per call
#   - cross-modal Recall@1 4/4 on the synthetic test scenes (jina-clip-v2
#     managed 3/4 on the same set), so it is a STRONG baseline, not a
#     strawman -- worth saying out loud when presenting
# This is a better comparison than NV-CLIP was: it is current rather than
# discontinued, and at 2048 dims the storage contrast with the 64/256/768
# tiers is larger and still entirely honest.
#
# NVCLIP_MODEL is still read first, for backwards compatibility with older
# .env files.
BASELINE_EMBED_MODEL = (os.environ.get("NVCLIP_MODEL")
                         or os.environ.get("BASELINE_EMBED_MODEL")
                         or "nvidia/llama-nemotron-embed-vl-1b-v2")
BASELINE_EMBED_DIM = 2048   # fixed; there is no cheaper prefix to take

# Deprecated alias, kept so nothing that imported it breaks.
NVCLIP_MODEL = BASELINE_EMBED_MODEL
# Default changed 2026-09-03 from meta/llama-3.2-90b-vision-instruct: that model
# read-timed-out on this account even for a text-only request (tried 90s and 150s).
# The 11b variant serves the identical payload in ~0.6s.
VLM_MODEL = os.environ.get("VLM_MODEL", "meta/llama-3.2-11b-vision-instruct")

_HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json",
}


def _require_key():
    if not API_KEY:
        raise RuntimeError(
            "NVIDIA_API_KEY is not set. Copy .env.example to .env and fill it in, "
            "or `export NVIDIA_API_KEY=nvapi-...` before running."
        )


def _image_to_data_uri(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _embed_baseline(payload_input, input_type: str) -> tuple[np.ndarray, float]:
    """Shared call for the fixed-width baseline embedder.

    `input_type` is the retriever convention this model family uses:
    "query" for the thing you're searching WITH, "passage" for the thing
    you're searching OVER. Images go in as "passage" — "image" is rejected.
    """
    _require_key()
    url = f"{BASE_URL}/embeddings"
    payload = {
        "model": BASELINE_EMBED_MODEL,
        "input": [payload_input],
        "input_type": input_type,
    }
    t0 = time.perf_counter()
    resp = requests.post(url, headers=_HEADERS, json=payload, timeout=45)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if resp.status_code == 404:
        raise RuntimeError(
            f"Baseline embedder '{BASELINE_EMBED_MODEL}' returned 404: this "
            "account cannot reach that function. NOTE: nvidia/nvclip is "
            "DISCONTINUED and will always 404 — if that's what's configured, "
            "clear NVCLIP_MODEL from .env so the current default "
            "(nvidia/llama-nemotron-embed-vl-1b-v2) is used. Verify what your "
            "account can reach with GET /v1/models. "
            f"Server said: {resp.text[:200]}"
        )
    if resp.status_code == 503 and "VLM serving" in resp.text:
        raise RuntimeError(
            f"'{BASELINE_EMBED_MODEL}' is text-only on this endpoint "
            f"(server: image inputs require VLM serving to be enabled). Pick a "
            "multimodal embedder — nvidia/llama-nemotron-embed-vl-1b-v2 is "
            "verified working for both text and images."
        )
    resp.raise_for_status()
    vec = np.asarray(resp.json()["data"][0]["embedding"], dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec, elapsed_ms


def embed_image_baseline(image_bytes: bytes) -> tuple[np.ndarray, float]:
    """Embed one image with NVIDIA's fixed-width baseline embedder.

    This model is NOT Matryoshka-trained — it always returns one fixed-width
    2048-dim vector. That's the entire point of calling it: it's the live,
    on-camera comparison baseline. There is no truncate() equivalent that
    gives you a smaller *meaningful* vector back, which is precisely the
    capability the local MRL encoder has and this one doesn't.
    """
    return _embed_baseline(_image_to_data_uri(image_bytes), "passage")


def embed_text_baseline(text: str) -> tuple[np.ndarray, float]:
    """Embed a text query with the same baseline model, for a like-for-like
    cross-modal comparison against the local encoder."""
    return _embed_baseline(text, "query")


# Deprecated name kept so older callers/notebooks don't break.
embed_image_nvclip = embed_image_baseline


ROBOT_PERCEPTION_PROMPT = (
    "You are the vision system of a mobile robot. You are looking at the "
    "robot's current camera frame. The robot's memory could not answer this "
    "request with enough confidence, so it has escalated to you.\n\n"
    "Request: {query}\n\n"
    "Reply in at most two sentences. Say what you can actually see in the "
    "frame that is relevant to the request, and where it is (left / right / "
    "centre, near / far). If the thing asked about is not visible, say so "
    "plainly. Do not explain what you are or what you cannot do."
)


def robot_perception_prompt(query: str) -> str:
    """Frame a user query as a robot-perception task.

    Added 2026-09-03. Passing the bare query straight through produced
    replies that were useless on stage — asked to "pick up the banana" the
    VLM answered "I'm a large language model, I cannot physically interact
    with the world... consider purchasing one from a store", and asked about
    a fire extinguisher it hallucinated a "fire extinguisher tree". Both are
    correct behaviour for an unframed chat prompt and terrible demo content.
    Giving it the robot's role and a length limit fixes that without
    pretending the model is something it isn't.
    """
    return ROBOT_PERCEPTION_PROMPT.format(query=query)


def call_vlm_escalation(image_bytes: bytes, instruction: str) -> tuple[str, float]:
    """The escalation-path stand-in. Sends the current frame + the
    instruction that triggered ESCALATE to a real, large, NVIDIA-hosted
    vision-language model and returns (response_text, latency_ms).

    This is deliberately NOT a stub — the whole point of this function is
    to produce a genuine network round-trip against a genuinely large model,
    so the latency you measure here is real, comparable evidence for the
    "expensive path" side of your metrics, standing in for the class of
    call a real on-device VLA (e.g. NVIDIA Isaac GR00T) would make.

    ADJUST HERE if the call fails: some vision models on the catalog expect
    the image under `image_url` with a data URI (used below, OpenAI-style);
    others may want a different content-block shape — check the "API" tab
    for your chosen VLM_MODEL.
    """
    _require_key()
    url = f"{BASE_URL}/chat/completions"
    payload = {
        "model": VLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": _image_to_data_uri(image_bytes)}},
                ],
            }
        ],
        "max_tokens": 200,
    }
    t0 = time.perf_counter()
    resp = requests.post(url, headers=_HEADERS, json=payload, timeout=60)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    return text, elapsed_ms


if __name__ == "__main__":
    # Minimal standalone smoke test — run this file directly, on its own,
    # before wiring it into the full demo. Isolates NIM connectivity
    # problems from everything else.
    import sys

    if len(sys.argv) < 2:
        print("Usage: python nvidia_nim.py path/to/test_image.jpg")
        sys.exit(1)

    with open(sys.argv[1], "rb") as f:
        img_bytes = f.read()

    print(f"Testing baseline embedder ({BASELINE_EMBED_MODEL})...")
    try:
        vec, ms = embed_image_baseline(img_bytes)
        print(f"  image OK — {vec.shape[0]}-dim fixed-width vector in {ms:.1f}ms")
        tvec, tms = embed_text_baseline("a red mug")
        print(f"  text  OK — {tvec.shape[0]}-dim in {tms:.1f}ms, "
              f"cross-modal similarity to the image = {float(vec @ tvec):.3f}")
    except Exception as e:
        print(f"  FAILED: {e}")

    print("Testing VLM escalation call...")
    try:
        text, ms = call_vlm_escalation(
            img_bytes, robot_perception_prompt("What object is in front of me?"))
        print(f"  OK in {ms:.1f}ms — response: {text}")
    except Exception as e:
        print(f"  FAILED: {e}")
