# robot-semantic-memory

## What this is

A semantic memory system for a robot, built on Matryoshka Representation
Learning (MRL) embeddings, for an NVIDIA GTC Berlin Golden Ticket contest
submission (deadline: Sept 10, 2026). Core idea: a small, cheap MRL-trained
encoder (jina-clip-v2) does continuous "cheap" perception and memory
lookup; an expensive model (stood in for by an NVIDIA-hosted VLM NIM, in
place of a real on-device VLA like NVIDIA Isaac GR00T) only gets called
when a confidence/novelty "gate" decides the cheap path can't answer.

Demo runs on a MacBook (no NVIDIA GPU) — NVIDIA coverage comes from
hosted NIM microservices (a multimodal embedder, Riva ASR/TTS, a
vision-language model),
not local CUDA. TensorRT/Triton/on-device GR00T are the stated "next step,
once the Jetson Orin Nano Super arrives" — don't try to make those work
locally, that's out of scope for this demo.

## Architecture

```
src/
  encoder.py       - jina-clip-v2 wrapper: embed_image, embed_text, truncate(vec, dim)
                     IMPORTANT: truncation saves storage/comparison cost, NOT encoder
                     compute — the backbone runs once regardless of dim kept.
  memory_store.py  - tiered vector store: short=64dim, medium=256dim, long=768dim,
                     brute-force numpy cosine search (no faiss, arm64-portable)
  gate.py          - decide(similarity, requires_manipulation) -> CHEAP | ESCALATE
  nvidia_nim.py    - REST calls: the fixed-width baseline embedder (was NV-CLIP,
                     now llama-nemotron-embed-vl-1b-v2 -- NV-CLIP is discontinued)
                     and a VLM chat-completions call (the escalation-path stand-in)
  riva_voice.py    - Riva ASR/TTS over gRPC (NVIDIA's actual protocol, not REST) —
                     record_clip(), transcribe(), speak()
  metrics.py       - MetricsLogger: escalation rate, blended cost vs. an
                     "always-escalate" baseline, tiered vs. fixed-width storage,
                     and the encode-vs-search split of the cheap path
  frame_source.py  - frames from a chosen camera index, a folder of stills, or a
                     live file written by camera_bridge.py (--watch), so the demo
                     runs even without camera permission

demo_webcam.py   - offline-only baseline demo, NO keys needed, keep this working
                   as the fallback no matter what else changes
demo_live.py     - full demo wiring in the baseline embedder, Riva voice, real VLM
                   escalation, and metrics on top of demo_webcam.py's core loop
camera_bridge.py - run from Terminal.app; republishes camera frames to a file so a
                   process without camera permission can consume them (--watch)
evaluate_tiers.py - Recall@1 per tier on a labeled folder of real test photos,
                   plus --calibrate to set gate thresholds from measured data
verify_offline.py - camera-free/keyless integration smoke test; run first on a
                   new machine. --live also exercises NIM + Riva
tests/test_core.py - 44 hermetic unit tests (no model, network, or camera)
```

## Status as of 2026-09-04 — first run against a live key

All three previously-suspected failure points turned out to be **wrong
leads**. What was actually broken was different in every case. Don't go
re-debugging the payload shapes; they're fine.

| Previously suspected | Reality |
|---|---|
| NV-CLIP request JSON shape | Shape was fine. **NV-CLIP is DISCONTINUED by NVIDIA** and always 404s. Replaced 2026-09-04 with `nvidia/llama-nemotron-embed-vl-1b-v2`: multimodal, fixed-width **2048** dims, ~0.8–5s, cross-modal Recall@1 **4/4** on the synthetic scenes. Set via `BASELINE_EMBED_MODEL`. Note `NVCLIP_MODEL` is read FIRST for back-compat — leaving it set resurrects the dead model. |
| `RIVA_TTS_VOICE` wrong | Voice `Magpie-Multilingual.EN-US.Aria` was always valid. The real fault: `RIVA_*_FUNCTION_ID` in `.env` held `nvapi-` **API keys instead of function UUIDs**, so gRPC reached the server and died `StatusCode.INTERNAL`. Fixed. |
| VLM `image_url` content-block format | Format was always correct. The real fault: `meta/llama-3.2-90b-vision-instruct` **read-times-out** on this account even text-only (tried 90s and 150s). Switched to `meta/llama-3.2-11b-vision-instruct`, ~0.6–2.4s. |

### Still open (not code bugs)

- **Camera permission is per-process on macOS.** Run the demos from
  Terminal.app, not from an editor/agent — see "Running the demo" below.
- **Gate thresholds are uncalibrated.** They print
  `PROVISIONAL defaults (not calibrated)` at startup until you run
  `evaluate_tiers.py --calibrate --write-thresholds thresholds.json`
  against real photos.

### Real bug found and fixed: cross-tier score comparison

Both demos used to pick the answer by `max()`-ing similarity across tiers.
Truncating an MRL embedding **systematically raises** cosine similarity, so
that always returned the 64-dim short tier — which is stored unlabeled and
had the worst Recall@1. Symptoms were `label='None'` on every query and
broken novelty detection (an object never seen still read as CHEAP).
Measured, same pair: `long 0.354 / medium 0.386 / short 0.435`.
Fixed by `MemoryStore.best_match()` (tier priority, per-tier thresholds).
Regression-tested in `tests/test_core.py::TestBestMatchRegression` —
including a test that asserts the *old* `max()` behaviour would still pick
short, so the fixture can't silently stop covering the bug.

### Measured, and important for the submission

On this MacBook the cheap/expensive comparison **inverts**: the local
jina-clip-v2 text tower (561M params, CPU, ~0.5–1.8s) is *slower* than the
hosted 11B VLM on datacentre GPUs (~1.0s). `metrics.py` now prints a loud
warning when that happens. The claims that do survive on this hardware:
memory lookup is **~0.16ms** (≈16,000x cheaper than escalating), tiered
storage is **4x** smaller than the 2048-dim fixed-width baseline, and
nothing needs the network. `metrics.print_report()` is deliberately ordered
to put these architectural claims FIRST and the blended wall-clock last,
under a "do NOT lead with this" heading.
Wall-clock-on-a-laptop is the wrong axis until the encoder runs on the
Jetson under TensorRT.

## Test order — don't jump straight to demo_live.py

0. `python -m unittest discover -s tests` — 44 hermetic unit tests, no
   model/network/camera, <1s. Run this first; it catches regressions in the
   memory/gate/metrics logic instantly.
1. `python verify_offline.py` — integration smoke test, real encoder, no
   camera and no keys needed. Add `--live` to include NIM + Riva.
2. `python demo_webcam.py` — no keys, confirms the core loop. Needs camera
   permission; `--images DIR` runs it without a camera.
3. `python src/nvidia_nim.py path/to/test_image.jpg` — baseline embedder + VLM.
4. `python src/riva_voice.py` — mic/ASR/TTS in isolation.
5. `python demo_live.py` — full integration, only once the above pass.
6. `python evaluate_tiers.py test_objects/manifest.json --calibrate` — real
   Recall@1 numbers, and the thresholds to set from them.

## Running the demo (macOS camera permission)

`cv2.VideoCapture` is gated by per-process TCC permission. A process whose
app bundle doesn't declare `NSCameraUsageDescription` gets **no permission
prompt and no System Settings entry** — the request dies before TCC
registers it, and every camera index fails with
`not authorized to capture video`. It *hangs* rather than erroring under
ffmpeg, so a permission problem can look like a freeze.

Launch from **Terminal.app**, which acts as the responsible app for its
children and does get prompted:

```bash
cd ~/Documents/robot-semantic-memory && python3 demo_webcam.py
```

Cameras on the current dev machine (`ffmpeg -f avfoundation -list_devices true -i ""`):
`[0] S65VC Webcam` (USB monitor), `[1] FaceTime HD Camera`. Index 0 is the
default; use `--camera 1` for the built-in, or `ROBOT_CAMERA_INDEX`.

### Giving a permission-less process camera access

If something that CAN'T be granted the camera needs live frames (an agent
or editor in its own app bundle), run the bridge from Terminal.app and
point the consumer at its output file:

```bash
python3 camera_bridge.py                                  # in Terminal.app; leave running
python3 demo_live.py --watch .camera_feed/latest.jpg      # anywhere else
python3 verify_offline.py --watch .camera_feed/latest.jpg --live
```

The bridge writes the newest frame atomically (temp file + `os.replace`),
so a reader never sees a half-written JPEG. Ctrl-C revokes access and
deletes the file. It is not a permission bypass — Terminal still has to be
granted the camera first.

## Env vars

All in `.env` (copy from `.env.example`): `NVIDIA_API_KEY`,
`RIVA_ASR_FUNCTION_ID`, `RIVA_TTS_FUNCTION_ID`, `RIVA_TTS_VOICE`,
`BASELINE_EMBED_MODEL`, `VLM_MODEL`. Currently using `nvidia/parakeet-ctc-0.6b-asr`
for ASR and `nvidia/magpie-tts-multilingual` for TTS.

## Ground rules

- Don't break `demo_webcam.py` — it's the safety-net fallback if any NIM
  integration is flaky on demo day.
- Keep `.env` out of version control if this becomes a git repo.
- If a NIM call's request/response shape turns out to differ from what's
  in the code, fix it in place and leave a short comment on what changed
  and why — future-me (and the actual me, in the next chat) will want to
  know if the catalog's schema shifted.
