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
  c         baseline-embedder comparison on the current frame
  1-5       show a full-screen presentation card (0 = back to live camera)
  p         print metrics report so far
  q / ESC   quit -> prints report, saves metrics JSON

Run the standalone smoke tests in src/nvidia_nim.py and src/riva_voice.py
FIRST, in isolation, before running this. Debugging NIM/Riva connectivity
issues is much easier one client at a time than inside the full loop.
"""

# Required on Python 3.9: annotations like `HUD | None` are PEP 604 syntax,
# which 3.9 tries to evaluate at runtime and rejects with
# "unsupported operand type(s) for |". Deferring annotation evaluation makes
# them plain strings, so the file runs on 3.9 and on 3.10+ alike.
from __future__ import annotations

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
from hud import HUD
import nvidia_nim
import riva_voice


def frame_to_jpeg_bytes(frame) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Could not encode frame to JPEG")
    return buf.tobytes()


def run_query(query_text: str, frame, pos, store: MemoryStore, metrics: MetricsLogger,
               hud: HUD | None = None):
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
    per_tier_scores = {t: (m[0][0] if m else 0.0)
                       for t, m in store.query_all_tiers(q_vec, top_k=1).items()}

    if best_entry is None:
        print("  no memories stored yet.")
        speak_safe("I don't have any memories yet.")
        metrics.log_decision(query_text, "none", 0.0, "escalate", cheap_latency_ms,
                              encode_ms=encode_ms, search_ms=search_ms)
        return

    print("  per-tier top-1: " + "  ".join(
        f"{t}({TIER_DIMS[t]}d)={v:.3f}" for t, v in per_tier_scores.items()))

    print(f"  best match: tier={best_tier} label='{best_entry.label}' "
          f"pos=({best_entry.x:.1f},{best_entry.y:.1f}) similarity={best_score:.3f}")

    requires_manip = any(w in query_text.lower() for w in ("pick", "grasp", "hold"))
    result = decide(best_score, requires_manipulation=requires_manip, tier=best_tier)
    print(f"  GATE DECISION: {result.decision.value.upper()} — {result.reason}")

    if hud:
        hud.set_result(per_tier_scores, best_tier, result.decision.value, result.reason,
                        best_entry.label, (best_entry.x, best_entry.y),
                        encode_ms, search_ms)

    if result.decision == Decision.CHEAP:
        response = f"Found it. Navigate to ({best_entry.x:.1f}, {best_entry.y:.1f})."
        print(f"  -> cheap path: {response}")
        speak_safe(response, hud)
        metrics.log_decision(query_text, best_tier, best_score, "cheap", cheap_latency_ms,
                              encode_ms=encode_ms, search_ms=search_ms)
    else:
        print("  -> escalating to NVIDIA-hosted VLM (stand-in for on-device VLA)...")
        try:
            frame_bytes = frame_to_jpeg_bytes(frame)
            response_text, escalate_ms = nvidia_nim.call_vlm_escalation(
                frame_bytes, nvidia_nim.robot_perception_prompt(query_text))
            print(f"  VLM responded in {escalate_ms:.1f}ms: {response_text}")
            if hud:
                hud.set_vlm(escalate_ms, response_text.strip())
            print(f"  RATIO: escalation took {escalate_ms / cheap_latency_ms:.1f}x the "
                  f"whole cheap path, {escalate_ms / search_ms:,.0f}x the memory lookup")
            speak_safe(response_text[:200], hud)
            metrics.log_decision(query_text, best_tier, best_score, "escalate",
                                  cheap_latency_ms, escalate_latency_ms=escalate_ms,
                                  encode_ms=encode_ms, search_ms=search_ms)
        except Exception as e:
            print(f"  VLM call failed: {e}")


def speak_safe(text: str, hud=None):
    """Wraps Riva TTS so a network hiccup never crashes the live demo."""
    try:
        riva_voice.speak(text)
        if hud:
            hud.light("Riva TTS")
    except Exception as e:
        print(f"  (TTS skipped — {e})")


def run_nvclip_comparison(frame, metrics: MetricsLogger, hud=None):
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
    # Typing on camera is the enemy of a clean recording: input() blocks the
    # frame loop, so the video window freezes while you type in the terminal,
    # and you have to switch windows mid-take. These presets let 'm' and the
    # 7/8/9 keys run without any typing at all.
    parser.add_argument("--labels", default="",
                        help="comma-separated labels for successive 'm' presses, "
                             "so memorizing needs no typing "
                             "(e.g. \"the green mug,the blue mug,the orange\")")
    parser.add_argument("--queries", default="",
                        help="comma-separated preset queries bound to keys 7, 8, 9")
    args = parser.parse_args()

    hud = HUD(TIER_DIMS)

    # Mirror everything this demo prints into the window's log strip, so one
    # window is enough to record. Patched onto builtins rather than defined
    # as a local `print`, because a local of that name makes every earlier
    # print() in this function an UnboundLocalError -- and patching builtins
    # also catches the prints inside run_query() and the other helpers, which
    # a function-local shadow would have missed.
    import builtins
    _real_print = builtins.print

    def _tee_print(*a, **kw):
        _real_print(*a, **kw)
        msg = " ".join(str(x) for x in a).strip()
        if msg:
            hud.log(msg)

    builtins.print = _tee_print

    preset_labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    # Each preset label gets its OWN simulated position. Without this every
    # object is memorized wherever the position happens to be -- and in a
    # recorded take where w/a/s/d was never pressed, that is (0,0) for all of
    # them. "Found it, navigate to (0.0, 0.0)" is a meaningless answer when
    # every object is at (0,0), and it undercuts the whole point of recalling
    # a location. Spread them instead, so each recall names a distinct spot.
    PRESET_POSITIONS = [(1.5, 0.0), (0.0, 2.0), (-1.5, 0.5), (0.5, -2.0),
                        (2.0, 1.5), (-2.0, -1.0)]
    preset_queries = [s.strip() for s in args.queries.split(",") if s.strip()]
    label_i = 0

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
    print("       c baseline compare · p print metrics · q/ESC quit")
    print("       1-5 show a presentation card · 0 back to live camera")
    if preset_labels:
        print(f"       m memorizes with preset labels: {preset_labels}")
    if preset_queries:
        for i, q in enumerate(preset_queries[:3]):
            print(f"       {7 + i} -> \"{q}\"")
    print()
    print("  Recording order: 1 (problem) · 2 (innovation) · 0 · memorize with m")
    print("  · 0 · space query · hold up the unseen object · space query")
    print("  · 3 (why it matters) · p (metrics) · 4 (stack) · 5 (github) · q")

    while True:
        ok, frame = source.read()
        if not ok:
            break
        latest_frame = frame
        frame_count += 1

        hud.set_memory(store.stats())
        cv2.imshow("robot-semantic-memory — NVIDIA-integrated demo", hud.compose(frame))

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
            if label_i < len(preset_labels):
                label = preset_labels[label_i]
                if label_i < len(PRESET_POSITIONS):
                    pos[0], pos[1] = PRESET_POSITIONS[label_i]
                label_i += 1
            elif preset_labels:
                # Presets given but exhausted. Do NOT fall back to input():
                # that blocks the frame loop and looks like the demo hanging
                # mid-recording. Auto-name instead and carry on.
                label_i += 1
                label = f"object {label_i}"
                print(f"  (presets exhausted — auto-labelled '{label}')")
            else:
                label = input("Label for this memory (e.g. 'red mug on shelf'): ").strip()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            vec = embed_image(Image.fromarray(rgb))
            store.add("medium", vec, pos[0], pos[1], label=label)
            store.add("long", vec, pos[0], pos[1], label=label)
            hud.set_memory(store.stats())
            hud.toast(f"MEMORIZED  {label}", 2.5)
            print(f"  -> memorized '{label}' at ({pos[0]:.1f}, {pos[1]:.1f})")
        elif key == ord(' '):
            query_text = input("Query (e.g. 'where is the red mug'): ").strip()
            if query_text:
                hud.begin_query(query_text)
                cv2.imshow("robot-semantic-memory — NVIDIA-integrated demo",
                           hud.compose(frame))
                cv2.waitKey(1)
                run_query(query_text, frame, pos, store, metrics, hud)
        elif key == ord('v'):
            try:
                clip = riva_voice.record_clip(4.0)
                query_text, asr_ms = riva_voice.transcribe(clip)
                print(f"  heard (in {asr_ms:.1f}ms): '{query_text}'")
                if query_text:
                    hud.begin_query(query_text, spoken=True)
                    cv2.imshow("robot-semantic-memory — NVIDIA-integrated demo",
                               hud.compose(frame))
                    cv2.waitKey(1)
                    run_query(query_text, frame, pos, store, metrics, hud)
            except Exception as e:
                print(f"  voice query failed: {e}")
        elif key == ord('c'):
            run_nvclip_comparison(frame, metrics, hud)
        elif key in (ord('7'), ord('8'), ord('9')):
            idx = key - ord('7')
            if idx < len(preset_queries):
                query_text = preset_queries[idx]
                print(f"\nQuery (preset {idx + 1}): {query_text}")
                hud.show_card(None)
                hud.begin_query(query_text)
                cv2.imshow("robot-semantic-memory — NVIDIA-integrated demo",
                           hud.compose(frame))
                cv2.waitKey(1)
                run_query(query_text, frame, pos, store, metrics, hud)
            else:
                print(f"  no preset query bound to key {chr(key)} "
                      f"(pass --queries to set them)")
        elif key in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5')):
            # Presentation cards, shown in this window so a screen recording
            # captures them. 0 returns to the live camera view.
            hud.show_card(chr(key))
        elif key == ord('0'):
            hud.show_card(None)
        elif key == ord('p'):
            lat = metrics.average_latency_by_path()
            brk = metrics.average_cheap_path_breakdown()
            storage = metrics.storage_footprint(
                store.stats(), TIER_DIMS,
                fixed_width_dim=nvidia_nim.BASELINE_EMBED_DIM)
            lines = []
            if brk["search_ms"] is not None:
                lines.append(f"memory lookup      {brk['search_ms']:.3f} ms")
            if lat["escalate_ms"]:
                lines.append(f"escalate to VLM    {lat['escalate_ms']:.0f} ms")
                if brk["search_ms"]:
                    lines.append(f"-> ~{lat['escalate_ms'] / brk['search_ms']:,.0f}x cheaper to remember")
            if storage["savings_factor"]:
                lines.append(f"storage  {storage['tiered_bytes'] / 1024:.0f} KB tiered "
                              f"vs {storage['fixed_width_bytes'] / 1024:.0f} KB fixed-width")
                lines.append(f"-> {storage['savings_factor']:.0f}x less storage")
            lines.append(f"escalation rate    {metrics.escalation_rate() * 100:.0f}%")
            hud.show_metrics(lines)
            metrics.print_report(
                tier_counts=store.stats(),
                tier_dims=TIER_DIMS,
                fixed_width_dim=nvidia_nim.BASELINE_EMBED_DIM,
            )

    builtins.print = _real_print
    source.release()
    cv2.destroyAllWindows()

    print("\nSession ended.")
    metrics.print_report(tier_counts=store.stats(), tier_dims=TIER_DIMS,
                          fixed_width_dim=nvidia_nim.BASELINE_EMBED_DIM)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics.save_json(f"metrics_session_{ts}.json")


if __name__ == "__main__":
    main()
