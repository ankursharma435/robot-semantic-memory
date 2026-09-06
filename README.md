# robot-semantic-memory — starter scaffold

A minimal, runnable version of the concept from the deck: a small
Matryoshka (MRL) multimodal encoder feeding a tiered vector memory, with
a gate that decides when a decision is cheap enough to act on memory
alone, versus when it should escalate to a (for now, stubbed) VLA.

This is deliberately hardware-agnostic. It runs on your laptop with a
webcam today, and ports to the Jetson + chassis later with the changes
listed at the bottom — the encoder/memory/gate logic doesn't change.

## What's here

```
robot-semantic-memory/
  requirements.txt
  .env.example           # copy to .env, fill in your NVIDIA keys/IDs
  src/
    encoder.py            # jina-clip-v2 wrapper: embed_image, embed_text, truncate
    memory_store.py        # tiered (short/medium/long) vector store, brute-force cosine search
    gate.py                 # confidence/novelty thresholds -> CHEAP vs ESCALATE decision
    nvidia_nim.py            # fixed-width baseline embedder + VLM escalation, NIM REST
    riva_voice.py             # voice in/out via NVIDIA Riva Speech NIM, gRPC
    metrics.py                # logs decisions, computes the value-prop numbers
    frame_source.py           # camera index / stills folder / live --watch file
  demo_webcam.py           # original offline-only demo — no keys needed, safe fallback
  demo_live.py             # full NVIDIA-integrated demo (baseline embedder + Riva + VLM + metrics)
  evaluate_tiers.py        # Recall@1 per tier, and --calibrate for gate thresholds
  verify_offline.py        # camera-free/keyless integration smoke test — run this first
  camera_bridge.py         # publish camera frames to a file (for permission-less consumers)
  tests/test_core.py       # 44 hermetic unit tests: no model, no network, no camera
```

## Setup

```bash
cd robot-semantic-memory
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# now edit .env — see "Which NVIDIA keys you need" below
```

First run of `demo_webcam.py`/`demo_live.py` will download the
jina-clip-v2 weights (~a few hundred MB) — needs network once, then it's
cached locally.

## Which NVIDIA keys you need

1. **`NVIDIA_API_KEY`** — one key covers the baseline embedder, the VLM call,
   and Riva auth. Get it free at `build.nvidia.com` (sign up, verify
   email, open any model page, "Get API Key").
2. **`RIVA_ASR_FUNCTION_ID`** and **`RIVA_TTS_FUNCTION_ID`** — these are
   function **UUIDs**, NOT API keys. Putting an `nvapi-…` key here is a
   silent trap: the gRPC call connects and then fails with
   `StatusCode.INTERNAL`. List the real ones for your account with:
   ```bash
   curl -s https://api.nvcf.nvidia.com/v2/nvcf/functions \
     -H "Authorization: Bearer $NVIDIA_API_KEY" | python3 -m json.tool
   ```
   Currently in use: `d8dd4e9b-fbf5-4fb0-9dba-8cf436c8d965`
   (`ai-parakeet-ctc-0_6b-asr`) and `877104f7-e885-42b9-8de8-f6e4c6303969`
   (`ai-magpie-tts-multilingual`).
3. **`BASELINE_EMBED_MODEL`** / **`VLM_MODEL`** — model identifier strings.
   Confirm against `GET /v1/models` on `integrate.api.nvidia.com`. Two traps
   found the hard way: **`nvidia/nvclip` is discontinued** and always 404s
   (use `nvidia/llama-nemotron-embed-vl-1b-v2`, which is multimodal and
   fixed-width 2048), and `meta/llama-3.2-90b-vision-instruct`
   read-times-out on this account even for a text-only request (use the
   `11b` variant). A leftover `NVCLIP_MODEL=` line takes precedence over
   `BASELINE_EMBED_MODEL`, so delete it rather than commenting it out.

## Recommended order of testing (don't jump straight to the full demo)

0. `python -m unittest discover -s tests` — 44 unit tests, no model,
   network or camera needed. Under a second. Start here.
1. `python verify_offline.py` — integration smoke test with the real
   encoder; no camera and no keys. Add `--live` to include NIM and Riva.
   **Run this first on a new machine** — it names the broken layer.
2. `python demo_webcam.py` — the core loop, no keys. Needs camera
   permission (see "Run the demo"); `--images DIR` runs it without one.
3. `python src/nvidia_nim.py path/to/any_test_image.jpg` — the baseline
   embedder and the VLM call in isolation.
4. `python src/riva_voice.py` — mic recording, ASR and TTS in isolation.
5. `python demo_live.py` — the full thing, once the above pass.
6. `python evaluate_tiers.py test_objects/manifest.json --calibrate
   --write-thresholds thresholds.json` — real Recall@1 per tier for the
   metrics recap, and the gate thresholds measured from your own photos.

Verified against a live key on 2026-09-04: all of the above passes —
44/44 unit tests, and the baseline embedder, VLM escalation, Riva TTS and
Riva ASR all confirmed against live endpoints.

If you need live camera frames in a process that can't be granted the
camera, run `python3 camera_bridge.py` from Terminal.app and pass
`--watch .camera_feed/latest.jpg` to any of the above.

## Run the demo

**Launch from Terminal.app.** macOS gates the camera per process, and a
process whose app bundle doesn't declare `NSCameraUsageDescription` gets no
permission prompt *and* no System Settings entry — every camera index then
fails with `not authorized to capture video` (under ffmpeg it hangs rather
than erroring, so it can look like a freeze). Terminal is the responsible
app for what it launches and does get prompted:

```bash
cd ~/Documents/robot-semantic-memory && python3 demo_webcam.py
```

Allow the prompt on first run. To pick a camera when you have several:

```bash
ffmpeg -f avfoundation -list_devices true -i ""   # list them with indices
python3 demo_webcam.py --camera 1                  # or ROBOT_CAMERA_INDEX=1
python3 demo_webcam.py --images test_objects       # no camera at all
```

Click the video window so it has keyboard focus, then:

- `w a s d` — move a *simulated* robot position (stand-in for odometry)
- `m` — memorize the current frame into medium-term memory (you'll be
  asked for a label in the terminal, e.g. "red mug on shelf")
- `space` — type a text query (e.g. "where is the red mug") and watch it:
  1. get embedded into the same space as the images
  2. get compared against everything in memory
  3. produce a gate decision (CHEAP → prints the coordinates to navigate
     to; ESCALATE → hits the VLA stub, which just prints what it would
     have called)

Every ~10 frames also gets written into short-term memory automatically,
so you can see that tier filling and then decaying (it prunes anything
older than 30 seconds).

## Try this to build intuition

1. Memorize an object ("m") with a clear label.
2. Move the simulated position elsewhere ("w"/"a"/"s"/"d" a few times).
3. Query for that object by description ("space") — you should get a
   CHEAP decision pointing back at the coordinates you memorized it at.
4. Query for something you never memorized — you should get an ESCALATE
   decision because similarity to everything stored falls below the
   novelty threshold.
5. Query using the word "pick" or "grasp" — notice it escalates
   regardless of similarity, because that's a manipulation task memory
   alone can't finish (see `gate.py`'s `requires_manipulation` flag).
6. Open `gate.py` and change `CONFIDENCE_THRESHOLD` / `NOVELTY_THRESHOLD`
   — see how it changes which queries stay cheap. Better: run
   `evaluate_tiers.py --calibrate` and set them from measured data instead
   of by feel. The gate prints which thresholds are in force at startup.
7. Watch the `per-tier top-1` line on each query. Scores go *up* as
   dimensions go *down* — that's truncation bias, not the 64-dim tier
   being better, and it's why the tiers get separate thresholds. See
   `MemoryStore.best_match()`.

## What this scaffold intentionally does NOT do yet

- **No real VLA call.** `gate.call_vla_stub()` just prints. Wiring in an
  actual model (a served OpenVLA/π0 checkpoint, local or over the
  network) is a separate, later step — get the memory/gate logic solid
  first, since it's the cheap 90% of the system you'll be running
  constantly.
- **No real spatial metadata.** Position is a keyboard-controlled toy
  variable. On the robot this comes from wheel odometry (encoder ticks)
  or visual odometry, not from you pressing "w."
- **No ANN index.** `memory_store.py` does a brute-force numpy scan.
  Fine up to low thousands of entries — swap in `hnswlib` or similar only
  if you actually outgrow that.

## Path to the real Jetson + chassis build

1. **Swap the camera source.** Add a Jetson CSI branch to
   `src/frame_source.py` (GStreamer via OpenCV, or `jetson-utils`) — the
   demos already go through `FrameSource`, so nothing else changes.
   Everything downstream (`embed_image`, `MemoryStore`, `gate.decide`) is
   untouched.
2. **Replace simulated position with real odometry.** Read wheel
   encoder ticks (or IMU + simple dead reckoning) into `(x, y)` instead
   of the `w/a/s/d` stand-in.
3. **Automate "memorize" instead of a keypress.** Once frames are
   arriving continuously, decide programmatically when to consolidate
   short-term → medium-term (e.g. when a region of the scene has been
   stable/novel for N consecutive frames — this is your first real
   "event detection" logic, replacing the manual `m` key).
4. **Benchmark on-device.** Re-run the encoder on the actual Jetson Orin
   Nano Super and measure real latency/power — the deck's numbers are
   illustrative; this is where you get your own.

   This is the step that fixes the one genuinely awkward measurement in
   the current build. On a MacBook CPU the *cheap* path is **slower** than
   the *expensive* one: jina-clip-v2's text tower is 561M params on CPU
   (~0.5–1.8s per query) while the hosted 11B VLM answers from datacentre
   GPUs in ~1.0s. `metrics.py` prints a loud warning when the comparison
   inverts like this. The memory lookup itself is ~0.08ms — roughly
   14,000x cheaper than escalating — so the architecture is sound; it's
   the local encoder that needs the Jetson and TensorRT. Until then, quote
   the per-stage numbers, not the blended one.
5. **Wire in the VLA.** Replace `call_vla_stub()` with a real inference
   call once you've picked a model and a serving strategy (on-device
   quantized, or a call out to a workstation/cloud endpoint — worth
   deciding based on the latency numbers you just measured).
6. **Add the motor control loop.** The cheap path's "navigate toward
   (x, y)" is currently just a `print()`. This is where it becomes an
   actual differential-drive controller.

Each of these is a self-contained next step — you don't need the
chassis to make progress on 1–3, and you don't need the VLA decided to
make progress on the memory system, which is exactly the point of
building it in this order.
