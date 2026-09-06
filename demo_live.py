"""
demo_live.py

The full NVIDIA-maxed demo. Builds on demo_webcam.py's core loop (which
still works standalone, offline, with no keys — keep that as your safe
fallback) and adds:

  - NV-CLIP comparison beat  ('c')  — same frame, NVIDIA's fixed-width
    baseline embedding next to your truncatable one
  - Voice query via Riva ASR ('v')  — speak instead of type
  - Spoken responses via Riva TTS   — automatic after every query
  - Real escalation calls          — ESCALATE now hits an actual
    NVIDIA-hosted VLM instead of printing a stub
  - Live metrics logging           — printed on demand ('p') and a full
    report + JSON saved automatically on quit

Controls:
  w/a/s/d   move simulated position
  m         memorize current frame (typed label, as before)
  space     type a text query
  v         speak a query instead (records 4s, transcribes via Riva)
  c         NV-CLIP comparison beat on the current frame
  p         print metrics report so far
  q / ESC   quit -> prints report, saves metrics JSON

Run the standalone smoke tests in src/nvidia_nim.py and src/riva_voice.py
FIRST, in isolation, before running this. Debugging NIM/Riva connectivity
issues is much easier one client at a time than inside the full loop.
"""

import argparse
import sys
import os
import time
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import cv2
from PIL import Image

from encoder import embed_image, embed_text
from memory_store import MemoryStore, TIER_DIMS
from gate import decide, Decision, TIER_CONFIDENCE_THRESHOLDS, threshold_summary
from metrics import MetricsLogger
from frame_source import FrameSource, add_source_args
import nvidia_nim
import riva_voice


def frame_to_jpeg_bytes(frame) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Could not encode frame to JPEG")
    return buf.tobytes()


def run_query(query_text: str, frame, pos, store: MemoryStore, metrics: MetricsLogger):
    """Shared path for both typed and spoken queries."""
    # Time the encode and the search SEPARATELY. On CPU the text encode is
    # ~99.9% of the cheap path, and reporting only the total made the gate
    # look barely worthwhile. See metrics.DecisionEvent for the numbers.
    t0 = time.perf_counter()
    q_vec = embed_text(query_text)
    encode_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    # Tier priority, NOT max() across tiers — narrow MRL prefixes inflate
    # cosine scores, so max() always picked the unlabeled short tier.
    # See MemoryStore.best_match() for the measurements behind this.
    best_tier, best_score, best_entry = store.best_match(
        q_vec, TIER_CONFIDENCE_THRESHOLDS)
    search_ms = (time.perf_counter() - t1) * 1000
    cheap_latency_ms = encode_ms + search_ms

    print(f"  query encoded in {encode_ms:.1f}ms, memory searched in {search_ms:.3f}ms")

    if best_entry is None:
        print("  no memories stored yet.")
        speak_safe("I don't have any memories yet.")
        metrics.log_decision(query_text, "none", 0.0, "escalate", cheap_latency_ms,
                              encode_ms=encode_ms, search_ms=search_ms)
        return

    results = store.query_all_tiers(q_vec, top_k=1)
    print("  per-tier top-1: " + "  ".join(
        f"{t}({TIER_DIMS[t]}d)={m[0][0]:.3f}" for t, m in results.items() if m))

    print(f"  best match: tier={best_tier} label='{best_entry.label}' "
          f"pos=({best_entry.x:.1f},{best_entry.y:.1f}) similarity={best_score:.3f}")

    requires_manip = any(w in query_text.lower() for w in ("pick", "grasp", "hold"))
    result = decide(best_score, requires_manipulation=requires_manip, tier=best_tier)
    print(f"  GATE DECISION: {result.decision.value.upper()} — {result.reason}")

    if result.decision == Decision.CHEAP:
        response = f"Found it. Navigate to ({best_entry.x:.1f}, {best_entry.y:.1f})."
        print(f"  -> cheap path: {response}")
        speak_safe(response)
        metrics.log_decision(query_text, best_tier, best_score, "cheap", cheap_latency_ms,
                              encode_ms=encode_ms, search_ms=search_ms)
    else:
        print("  -> escalating to NVIDIA-hosted VLM (stand-in for on-device VLA)...")
        try:
            frame_bytes = frame_to_jpeg_bytes(frame)
            response_text, escalate_ms = nvidia_nim.call_vlm_escalation(
                frame_bytes, nvidia_nim.robot_perception_prompt(query_text))
            print(f"  VLM responded in {escalate_ms:.1f}ms: {response_text}")
            print(f"  RATIO: escalation took {escalate_ms / cheap_latency_ms:.1f}x the "
                  f"whole cheap path, {escalate_ms / search_ms:,.0f}x the memory lookup")
            speak_safe(response_text[:200])
            metrics.log_decision(query_text, best_tier, best_score, "escalate",
                                  cheap_latency_ms, escalate_latency_ms=escalate_ms,
                                  encode_ms=encode_ms, search_ms=search_ms)
        except Exception as e:
            print(f"  VLM call failed: {e}")


def speak_safe(text: str):
    """Wraps Riva TTS so a network hiccup never crashes the live demo."""
    try:
        riva_voice.speak(text)
    except Exception as e:
        print(f"  (TTS skipped — {e})")


def run_nvclip_comparison(frame, metrics: MetricsLogger):
    """The 'c' beat: same frame through the local truncatable encoder and
    through NVIDIA's fixed-width hosted embedder, side by side.

    Renamed in spirit 2026-09-04 — the baseline is no longer NV-CLIP, which
    NVIDIA discontinued. Function name kept so muscle memory and the 'c' key
    still work.
    """
    print(f"  comparing local MRL encoder vs {nvidia_nim.BASELINE_EMBED_MODEL}...")
    frame_bytes = frame_to_jpeg_bytes(frame)

    t0 = time.perf_counter()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    local_vec = embed_image(Image.fromarray(rgb))
    local_ms = (time.perf_counter() - t0) * 1000

    try:
        base_vec, base_ms = nvidia_nim.embed_image_baseline(frame_bytes)
    except Exception as e:
        print(f"  baseline embedder failed: {e}")
        print(f"  (local encoder still fine: {local_vec.shape[0]} dims in {local_ms:.1f}ms)")
        return

    print(f"  local  (MRL, truncatable): {local_vec.shape[0]:5d} dims in {local_ms:7.1f}ms")
    print(f"  hosted (fixed width):      {base_vec.shape[0]:5d} dims in {base_ms:7.1f}ms")
    print()
    print("  The point of this beat is the STORAGE column, not the latency:")
    bpf = 4  # float32
    for tier, dim in TIER_DIMS.items():
        print(f"    local kept at {tier:6} = {dim:4d} dims = {dim * bpf:5d} bytes/frame")
    print(f"    hosted, no option    = {base_vec.shape[0]:4d} dims = "
          f"{base_vec.shape[0] * bpf:5d} bytes/frame")
    ratio = base_vec.shape[0] / TIER_DIMS["short"]
    print(f"    --> short-term tier is {ratio:.0f}x smaller per frame than the")
    print(f"        hosted embedding, which has no cheaper prefix to take.")


def main():
    parser = add_source_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()

    try:
        source = FrameSource.open(camera=args.camera, images=args.images,
                                    watch=args.watch)
    except RuntimeError as e:
        print(e)
        return
    print(f"Frame source: {source.description}")

    # Warm the encoder before the loop — see the same note in demo_webcam.py.
    print("Loading encoder...")
    t0 = time.time()
    embed_text("warmup")
    print(f"Encoder ready in {time.time() - t0:.1f}s")
    print(threshold_summary())

    store = MemoryStore(short_ttl_seconds=30.0)
    metrics = MetricsLogger()
    pos = [0.0, 0.0]
    step = 0.5
    frame_count = 0
    SHORT_TERM_EVERY_N_FRAMES = 10
    latest_frame = None

    print("Ready. w/a/s/d move · m memorize · space type-query · v voice-query")
    print("       c NV-CLIP compare · p print metrics · q/ESC quit")

    while True:
        ok, frame = source.read()
        if not ok:
            break
        latest_frame = frame
        frame_count += 1

        cv2.putText(frame, f"pos=({pos[0]:.1f},{pos[1]:.1f})  "
                            f"short={len(store._entries['short'])}  medium={len(store._entries['medium'])}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.imshow("robot-semantic-memory — NVIDIA-integrated demo", frame)

        if frame_count % SHORT_TERM_EVERY_N_FRAMES == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            vec = embed_image(Image.fromarray(rgb))
            store.add("short", vec, pos[0], pos[1])
            store.prune_short_term()

        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('w'):
            pos[1] += step
        elif key == ord('s'):
            pos[1] -= step
        elif key == ord('a'):
            pos[0] -= step
        elif key == ord('d'):
            pos[0] += step
        elif key == ord('m'):
            label = input("Label for this memory (e.g. 'red mug on shelf'): ").strip()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            vec = embed_image(Image.fromarray(rgb))
            store.add("medium", vec, pos[0], pos[1], label=label)
            print(f"  -> memorized '{label}' at ({pos[0]:.1f}, {pos[1]:.1f})")
        elif key == ord(' '):
            query_text = input("Query (e.g. 'where is the red mug'): ").strip()
            if query_text:
                run_query(query_text, frame, pos, store, metrics)
        elif key == ord('v'):
            try:
                clip = riva_voice.record_clip(4.0)
                query_text, asr_ms = riva_voice.transcribe(clip)
                print(f"  heard (in {asr_ms:.1f}ms): '{query_text}'")
                if query_text:
                    run_query(query_text, frame, pos, store, metrics)
            except Exception as e:
                print(f"  voice query failed: {e}")
        elif key == ord('c'):
            run_nvclip_comparison(frame, metrics)
        elif key == ord('p'):
            metrics.print_report(
                tier_counts=store.stats(),
                tier_dims=TIER_DIMS,
                fixed_width_dim=nvidia_nim.BASELINE_EMBED_DIM,
            )

    source.release()
    cv2.destroyAllWindows()

    print("\nSession ended.")
    metrics.print_report(tier_counts=store.stats(), tier_dims=TIER_DIMS,
                          fixed_width_dim=nvidia_nim.BASELINE_EMBED_DIM)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics.save_json(f"metrics_session_{ts}.json")


if __name__ == "__main__":
    main()
