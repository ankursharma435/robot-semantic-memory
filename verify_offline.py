"""
verify_offline.py

Camera-free, keyboard-free smoke test of the whole stack. Added 2026-09-03
because every other entrypoint needs either camera permission (demo_webcam,
demo_live) or a live mic (riva_voice), which makes "is this box set up
correctly?" annoying to answer — especially right after moving machines.

Run this FIRST on a new machine. It exercises the same code paths the demos
do, in the same order, and tells you exactly which layer is broken.

    python verify_offline.py              # offline only — no keys needed
    python verify_offline.py --live       # also hit NV-CLIP / VLM / Riva
    python verify_offline.py --images DIR # use your own photos, not synthetic ones

Exit code is 0 only if every non-skipped check passed, so this works in CI
or as a pre-demo gate.
"""

from __future__ import annotations
import argparse
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

RESULTS = []


def check(name: str, ok: bool | None, detail: str = ""):
    """ok=True pass, ok=False fail, ok=None skipped."""
    tag = {True: "PASS", False: "FAIL", None: "SKIP"}[ok]
    RESULTS.append((name, ok))
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))


# ---------------------------------------------------------------- synthetic scenes

def make_synthetic_scenes(directory: str) -> list[str]:
    """Four crude but distinguishable scenes, so this script needs no assets.

    These are good enough to prove the PLUMBING works. They are NOT good
    enough to judge retrieval quality from — flat synthetic shapes sit in a
    weird corner of CLIP's input distribution, and on them the correct- and
    incorrect-match score distributions overlap. Use --images with real
    photographs for anything you intend to quote as a number.
    """
    from PIL import Image, ImageDraw

    W = H = 448
    ROOM, FLOOR = (170, 168, 160), (120, 110, 95)

    def base():
        img = Image.new("RGB", (W, H), ROOM)
        d = ImageDraw.Draw(img)
        d.rectangle([0, int(H * 0.68), W, H], fill=FLOOR)
        return img, d

    def red_mug():
        img, d = base()
        d.rounded_rectangle([150, 210, 280, 330], radius=14, fill=(190, 30, 30))
        d.ellipse([265, 235, 330, 300], outline=(190, 30, 30), width=18)
        d.ellipse([150, 198, 280, 226], fill=(215, 60, 60))
        return img

    def blue_book():
        img, d = base()
        d.polygon([(120, 320), (330, 320), (350, 250), (140, 250)], fill=(35, 60, 175))
        d.polygon([(120, 320), (330, 320), (330, 306), (120, 306)], fill=(240, 238, 230))
        return img

    def green_plant():
        img, d = base()
        d.polygon([(190, 330), (270, 330), (258, 250), (202, 250)], fill=(150, 90, 60))
        for dx in (-55, -25, 0, 25, 55):
            d.ellipse([230 + dx - 26, 130, 230 + dx + 26, 258], fill=(40, 130, 55))
        return img

    def yellow_banana():
        img, d = base()
        d.arc([130, 200, 340, 360], start=200, end=340, fill=(225, 200, 40), width=34)
        return img

    os.makedirs(directory, exist_ok=True)
    paths = []
    for name, fn in (("red_mug", red_mug), ("blue_book", blue_book),
                      ("green_plant", green_plant), ("yellow_banana", yellow_banana)):
        p = os.path.join(directory, f"{name}.jpg")
        fn().save(p, quality=92)
        paths.append(p)
    return paths


def label_from_filename(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.replace("_", " ").replace("-", " ")


# ---------------------------------------------------------------- checks

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", help="folder of photos to use instead of synthetic scenes")
    ap.add_argument("--watch", help="live frame file written by camera_bridge.py "
                                     "(e.g. .camera_feed/latest.jpg) — verifies against "
                                     "real camera frames without needing camera permission")
    ap.add_argument("--live", action="store_true",
                    help="also exercise NV-CLIP, the VLM escalation call, and Riva TTS/ASR")
    args = ap.parse_args()

    print("=" * 64)
    print("OFFLINE VERIFICATION")
    print("=" * 64)

    # ---- 1. imports -------------------------------------------------------
    print("\n[1] dependencies")
    try:
        import numpy, cv2, PIL, sentence_transformers  # noqa: F401
        check("core imports (numpy, cv2, PIL, sentence-transformers)", True,
              f"sentence-transformers {sentence_transformers.__version__}")
    except Exception as e:
        check("core imports", False, f"{type(e).__name__}: {e}")
        return finish()

    if tuple(int(x) for x in sentence_transformers.__version__.split(".")[:2]) < (3, 1):
        check("sentence-transformers >= 3.1", False,
              "jina-clip-v2 needs the custom_st loader from 3.x+; "
              f"found {sentence_transformers.__version__}")
    else:
        check("sentence-transformers >= 3.1", True)

    # ---- 2. frame source --------------------------------------------------
    print("\n[2] frame source")
    tmpdir = None
    if args.watch:
        # Grab a burst of live frames from the bridge and treat them as the
        # test set. This is the "real camera without camera permission" path.
        from frame_source import FrameSource as _FS
        tmpdir = tempfile.mkdtemp(prefix="rsm_live_")
        try:
            live = _FS.open(watch=args.watch)
            import cv2 as _cv
            grabbed = 0
            for i in range(4):
                ok, fr = live.read()
                if ok:
                    _cv.imwrite(os.path.join(tmpdir, f"live_{i:02d}.jpg"), fr)
                    grabbed += 1
                time.sleep(0.6)   # let the publisher move on to a new frame
            image_dir = tmpdir
            check("live frames captured via bridge", grabbed > 0,
                  f"{grabbed} frames from {args.watch}")
        except Exception as e:
            check("live frames captured via bridge", False, f"{type(e).__name__}: {e}")
            return finish()
    elif args.images:
        image_dir = args.images
        check("image folder", os.path.isdir(image_dir), image_dir)
    else:
        tmpdir = tempfile.mkdtemp(prefix="rsm_scenes_")
        try:
            make_synthetic_scenes(tmpdir)
            image_dir = tmpdir
            check("synthetic scenes generated", True, f"4 scenes in {tmpdir}")
        except Exception as e:
            check("synthetic scenes generated", False, f"{type(e).__name__}: {e}")
            return finish()

    from frame_source import FrameSource
    try:
        source = FrameSource.open(images=image_dir)
        ok, frame = source.read()
        check("FrameSource reads frames", ok and frame is not None,
              f"{source.description}, first frame {frame.shape if ok else 'n/a'}")
    except Exception as e:
        check("FrameSource reads frames", False, f"{type(e).__name__}: {e}")
        return finish()

    # Camera is reported, never required — it's the one thing this script
    # deliberately does not gate on, since permission is a host-app setting.
    try:
        import cv2 as _cv2
        cap = _cv2.VideoCapture(0)
        cam_ok = cap.isOpened()
        cap.release()
        print(f"  [INFO] camera index 0 {'available' if cam_ok else 'NOT available'}"
              + ("" if cam_ok else " — demos need --images, or grant camera"
                                    " permission to the host app and restart it"))
    except Exception:
        pass

    # ---- 3. encoder -------------------------------------------------------
    print("\n[3] encoder (first run downloads ~1.6GB)")
    try:
        from encoder import embed_image, embed_text, FULL_DIM
        t0 = time.time()
        qv = embed_text("warmup")
        load_s = time.time() - t0
        check("model loads + embeds text", len(qv) == FULL_DIM,
              f"{len(qv)}-dim in {load_s:.1f}s")
    except Exception as e:
        check("model loads + embeds text", False, f"{type(e).__name__}: {e}")
        return finish()

    # ---- 4. memory + gate -------------------------------------------------
    print("\n[4] tiered memory + gate")
    from PIL import Image
    from memory_store import MemoryStore, TIER_DIMS
    from gate import decide, Decision, TIER_CONFIDENCE_THRESHOLDS

    store = MemoryStore(short_ttl_seconds=30.0)
    import glob
    paths = sorted(p for ext in ("jpg", "jpeg", "png")
                    for p in glob.glob(os.path.join(image_dir, f"*.{ext}")))
    try:
        for i, p in enumerate(paths):
            vec = embed_image(Image.open(p).convert("RGB"))
            store.add("short", vec, float(i), 0.0)
            store.add("medium", vec, float(i), 0.0, label=label_from_filename(p))
            store.add("long", vec, float(i), 0.0, label=label_from_filename(p))
        check("memorize into all three tiers", True, f"{store.stats()}")
    except Exception as e:
        check("memorize into all three tiers", False, f"{type(e).__name__}: {e}")
        return finish()

    try:
        store.prune_short_term()
        check("short-term pruning runs", True)
    except Exception as e:
        check("short-term pruning runs", False, f"{type(e).__name__}: {e}")

    # The regression guard for the cross-tier bug: best_match must never
    # answer from the unlabeled short tier just because 64-dim scores are
    # numerically larger.
    try:
        q = embed_text(f"where is the {label_from_filename(paths[0])}")
        per_tier = store.query_all_tiers(q, top_k=1)
        print("       per-tier top-1: " + "  ".join(
            f"{t}({TIER_DIMS[t]}d)={m[0][0]:.3f}" for t, m in per_tier.items() if m))
        tier, score, entry = store.best_match(q, TIER_CONFIDENCE_THRESHOLDS)
        inflated = (per_tier["short"][0][0] > per_tier["long"][0][0])
        check("best_match does not answer from unlabeled short tier",
              entry is not None and entry.label is not None,
              f"answered from {tier} (short-tier score "
              f"{'is' if inflated else 'is not'} inflated above long here)")
    except Exception as e:
        check("best_match does not answer from unlabeled short tier", False,
              f"{type(e).__name__}: {e}")

    try:
        r_cheap = decide(0.99, requires_manipulation=False, tier="long")
        r_manip = decide(0.99, requires_manipulation=True, tier="long")
        r_novel = decide(0.01, requires_manipulation=False, tier="long")
        ok = (r_cheap.decision == Decision.CHEAP
              and r_manip.decision == Decision.ESCALATE
              and r_novel.decision == Decision.ESCALATE)
        check("gate returns CHEAP / ESCALATE correctly", ok)
    except Exception as e:
        check("gate returns CHEAP / ESCALATE correctly", False, f"{type(e).__name__}: {e}")

    # ---- 5. metrics -------------------------------------------------------
    print("\n[5] metrics")
    try:
        from metrics import MetricsLogger
        m = MetricsLogger()
        m.log_decision("q1", "long", 0.5, "cheap", 12.0)
        m.log_decision("q2", "long", 0.1, "escalate", 12.0, escalate_latency_ms=900.0)
        blend = m.blended_cost_vs_always_escalate()
        check("metrics compute blended cost", blend["speedup_factor"] is not None,
              f"{blend['speedup_factor']:.1f}x cheaper on this toy 2-event session")
    except Exception as e:
        check("metrics compute blended cost", False, f"{type(e).__name__}: {e}")

    # ---- 6. live services -------------------------------------------------
    print("\n[6] NVIDIA services" + ("" if args.live else " (skipped — pass --live)"))
    if not args.live:
        for n in ("baseline embedder", "VLM escalation call", "Riva TTS", "Riva ASR"):
            check(n, None)
    else:
        import cv2 as _c
        frame_bytes = _c.imencode(".jpg", _c.imread(paths[0]))[1].tobytes()

        import nvidia_nim
        try:
            vec, ms = nvidia_nim.embed_image_baseline(frame_bytes)
            check("baseline embedder", True,
                  f"{vec.shape[0]}-dim fixed width in {ms:.0f}ms "
                  f"({nvidia_nim.BASELINE_EMBED_MODEL})")
        except Exception as e:
            check("baseline embedder", False, str(e)[:160])

        try:
            text, ms = nvidia_nim.call_vlm_escalation(
                frame_bytes, "Describe what you see in one sentence.")
            check("VLM escalation call", True, f"{ms:.0f}ms — {text.strip()[:70]!r}")
        except Exception as e:
            check("VLM escalation call", False, str(e)[:160])

        import riva_voice
        tts_audio = None
        try:
            tts_audio, ms = riva_voice.speak("Verification test.", play=False)
            check("Riva TTS", True, f"{ms:.0f}ms, {len(tts_audio)} samples")
        except Exception as e:
            check("Riva TTS", False, str(e)[:160])

        # ASR without a mic: transcribe our own TTS output. Round-tripping
        # like this is how the streaming fix was validated in the first place.
        if tts_audio is not None:
            try:
                import numpy as np
                phrase = "Where is the red mug?"
                audio, _ = riva_voice.speak(phrase, play=False)
                text, ms = riva_voice.transcribe(np.asarray(audio, dtype=np.int16))
                check("Riva ASR", bool(text.strip()),
                      f"{ms:.0f}ms — said {phrase!r}, heard {text.strip()!r}")
            except Exception as e:
                check("Riva ASR", False, str(e)[:160])
        else:
            check("Riva ASR", None, "TTS failed, so no audio to round-trip")

    return finish()


def finish():
    passed = sum(1 for _, ok in RESULTS if ok is True)
    failed = [n for n, ok in RESULTS if ok is False]
    skipped = sum(1 for _, ok in RESULTS if ok is None)
    print("\n" + "=" * 64)
    print(f"{passed} passed, {len(failed)} failed, {skipped} skipped")
    if failed:
        print("\nFailed:")
        for n in failed:
            print(f"  - {n}")
    print("=" * 64)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
