"""
demo_webcam.py

Runs the whole software loop (encoder -> tiered memory -> query -> gate)
using a laptop webcam, standing in for the robot's camera before the
Jetson + chassis arrive. This lets you validate the concept end to end
in software first.

Controls (click the video window so it has keyboard focus):
  w / a / s / d   move a simulated robot position (just for demo metadata)
  m               memorize the current frame into medium-term memory
                  (you'll be prompted in the terminal for a label)
  space           run a text query against memory (typed in the terminal)
  q / ESC         quit

What happens automatically:
  - Every frame is embedded and written into SHORT-term memory (decays
    after `short_ttl_seconds`), simulating the "always watching" tier.
  - Pressing 'm' consolidates the current view into MEDIUM-term memory
    with a label and the simulated (x, y) position — this is your
    stand-in for "event: object placed here."
  - The query path embeds your typed text, searches memory, and runs it
    through the gate to decide CHEAP vs ESCALATE.
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import cv2
from PIL import Image

from encoder import embed_image, embed_text
from memory_store import MemoryStore
from gate import (decide, Decision, call_vla_stub, TIER_CONFIDENCE_THRESHOLDS,
                   threshold_summary)
from frame_source import FrameSource, add_source_args


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

    # Warm the encoder BEFORE entering the loop. _get_model() is lazy, so the
    # first embed used to cost ~14s — which landed on the first frame and
    # froze the video window on startup. Paying it here keeps the loop smooth.
    print("Loading encoder (first run downloads ~1.6GB of weights)...")
    t0 = time.time()
    embed_text("warmup")
    print(f"Encoder ready in {time.time() - t0:.1f}s")
    print(threshold_summary())

    store = MemoryStore(short_ttl_seconds=30.0)
    pos = [0.0, 0.0]          # simulated robot position
    step = 0.5
    frame_count = 0
    SHORT_TERM_EVERY_N_FRAMES = 10   # don't embed literally every frame — still plenty for a demo

    print("Ready. w/a/s/d move · m memorize · space query · q/ESC quit")

    while True:
        ok, frame = source.read()
        if not ok:
            break
        frame_count += 1

        cv2.putText(frame, f"pos=({pos[0]:.1f},{pos[1]:.1f})  short_mem={len(store._entries['short'])}  medium_mem={len(store._entries['medium'])}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.imshow("robot-semantic-memory demo", frame)

        if frame_count % SHORT_TERM_EVERY_N_FRAMES == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            vec = embed_image(pil_img)
            store.add("short", vec, pos[0], pos[1])
            store.prune_short_term()

        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):  # q or ESC
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
            pil_img = Image.fromarray(rgb)
            vec = embed_image(pil_img)
            store.add("medium", vec, pos[0], pos[1], label=label)
            print(f"  -> memorized '{label}' at ({pos[0]:.1f}, {pos[1]:.1f})")
        elif key == ord(' '):
            query_text = input("Query (e.g. 'where is the red mug'): ").strip()
            if not query_text:
                continue
            t0 = time.time()
            q_vec = embed_text(query_text)
            # Tier priority, NOT max() across tiers — narrow MRL prefixes
            # inflate cosine scores, so max() always picked the unlabeled
            # short tier. See MemoryStore.best_match() for the measurements.
            best_tier, best_score, best_entry = store.best_match(
                q_vec, TIER_CONFIDENCE_THRESHOLDS)
            elapsed_ms = (time.time() - t0) * 1000

            print(f"  query embedded + searched in {elapsed_ms:.1f}ms")
            if best_entry is None:
                print("  no memories stored yet.")
                continue

            # Per-tier scores, for eyeballing how much the truncation costs.
            results = store.query_all_tiers(q_vec, top_k=1)
            print("  per-tier top-1: " + "  ".join(
                f"{t}={m[0][0]:.3f}" for t, m in results.items() if m))

            print(f"  best match: tier={best_tier} label='{best_entry.label}' "
                  f"pos=({best_entry.x:.1f},{best_entry.y:.1f}) similarity={best_score:.3f}")

            result = decide(
                best_score,
                requires_manipulation="pick" in query_text.lower() or "grasp" in query_text.lower(),
                tier=best_tier,
            )
            print(f"  GATE DECISION: {result.decision.value.upper()} — {result.reason}")

            if result.decision == Decision.CHEAP:
                print(f"  -> cheap path: navigate toward ({best_entry.x:.1f}, {best_entry.y:.1f})")
            else:
                call_vla_stub(query_text)

    source.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
