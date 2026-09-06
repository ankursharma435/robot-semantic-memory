"""
evaluate_tiers.py

Measures Recall@1 at each memory tier (64 / 256 / 768 dims) using your OWN
photographed objects — this turns the doc's "Mug A vs Mug B" thought
experiment into a real number from your actual test set.

Setup:
  1. Create a folder, e.g. test_objects/, with one photo per object:
       test_objects/mug_plain.jpg
       test_objects/mug_logo.jpg
       test_objects/red_pen.jpg
       ...
  2. Create a manifest.json alongside it:
       {
         "mug_plain.jpg": "mug",
         "mug_logo.jpg": "mug",
         "red_pen.jpg": "pen",
         ...
       }
     Objects sharing a label are the ones you WANT correctly grouped;
     near-duplicate pairs with DIFFERENT labels (e.g. "mug_a" vs "mug_b")
     are what stress-tests the low-dim tiers.

Run:
  python evaluate_tiers.py test_objects/manifest.json

What it does: embeds every image once at full resolution, then for each
tier (64/256/768), treats every image as a query against every OTHER
image and checks whether the nearest neighbor shares its label. Reports
Recall@1 per tier.
"""

from __future__ import annotations
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from PIL import Image
from encoder import embed_image, embed_text, truncate, cosine, TIER_DIMS


def load_manifest(manifest_path: str):
    with open(manifest_path) as f:
        manifest = json.load(f)
    base_dir = os.path.dirname(manifest_path)
    items = []
    for filename, label in manifest.items():
        path = os.path.join(base_dir, filename)
        items.append((path, label))
    return items


def recall_at_1_for_tier(embeddings_full, labels, dim: int) -> float:
    truncated = [truncate(e, dim) for e in embeddings_full]
    n = len(truncated)
    correct = 0
    for i in range(n):
        best_j, best_score = None, -1.0
        for j in range(n):
            if i == j:
                continue
            score = cosine(truncated[i], truncated[j])
            if score > best_score:
                best_score = score
                best_j = j
        if best_j is not None and labels[best_j] == labels[i]:
            correct += 1
    return correct / n if n else 0.0


def calibrate_thresholds(embeddings_full, labels, prompt_template: str,
                          write_to: str | None = None, manifest_path: str = ""):
    """Measure the text->image similarity distributions the gate thresholds
    actually have to separate, per tier.

    Added 2026-09-03. gate.py's thresholds were guesses, and a single
    absolute cutoff cannot work across 64/256/768 dims because truncation
    systematically raises cosine scores. This prints, for each tier, the
    similarity of the CORRECT text->image pair versus the best INCORRECT
    one, so you can set gate.TIER_CONFIDENCE_THRESHOLDS from data.

    Read the 'separation' line first: if the correct-match minimum is below
    the incorrect-match maximum, the distributions OVERLAP and NO absolute
    threshold can separate them on this test set. That is the honest signal
    that you need better/more distinctive test photos (or a margin-based
    decision rule) rather than a cleverer threshold.
    """
    queries = [prompt_template.format(label=lab) for lab in labels]
    print(f"\nEmbedding {len(queries)} text queries "
          f"(template: {prompt_template!r})...")
    q_full = [embed_text(q) for q in queries]

    print("\nTHRESHOLD CALIBRATION — text->image similarity by tier")
    print("(correct = query vs its own image; incorrect = query vs the")
    print(" best-scoring image with a DIFFERENT label)\n")

    measured = {}
    for tier, dim in TIER_DIMS.items():
        imgs = [truncate(e, dim) for e in embeddings_full]
        qs = [truncate(q, dim) for q in q_full]

        correct, incorrect = [], []
        for i in range(len(qs)):
            correct.append(cosine(qs[i], imgs[i]))
            others = [cosine(qs[i], imgs[j]) for j in range(len(imgs))
                       if labels[j] != labels[i]]
            if others:
                incorrect.append(max(others))

        if not correct or not incorrect:
            print(f"  {tier:8s} ({dim:4d} dims): not enough distinct labels to calibrate")
            continue

        c_min, c_max = min(correct), max(correct)
        c_avg = sum(correct) / len(correct)
        i_min, i_max = min(incorrect), max(incorrect)
        i_avg = sum(incorrect) / len(incorrect)

        print(f"  {tier:8s} ({dim:4d} dims)")
        print(f"      correct:   min {c_min:.3f}  avg {c_avg:.3f}  max {c_max:.3f}")
        print(f"      incorrect: min {i_min:.3f}  avg {i_avg:.3f}  max {i_max:.3f}")
        if c_min > i_max:
            suggested = (c_min + i_max) / 2
            print(f"      separation: CLEAN (gap {c_min - i_max:+.3f})")
            print(f"      --> suggested threshold for this tier: {suggested:.3f}")
        else:
            # Distributions overlap, so no cutoff is correct. Fall back to
            # just above the worst wrong match: that keeps known-bad matches
            # out (the failure that looks worst on stage — confidently
            # navigating to something never seen) at the cost of escalating
            # some correct ones, which merely looks slow.
            suggested = i_max + 0.005
            print(f"      separation: OVERLAP (gap {c_min - i_max:+.3f})")
            print(f"      --> no absolute threshold separates these; a cutoff here")
            print(f"          will either admit wrong matches or reject right ones.")
            print(f"      --> using {suggested:.3f} (just above the worst wrong match),")
            print(f"          which escalates some correct matches on purpose.")

        measured[tier] = {
            "confidence": round(suggested, 4),
            # Novelty = "matches nothing at all", so sit it below the
            # confidence cutoff, near the top of the incorrect distribution.
            "novelty": round(min(suggested, i_avg), 4),
            "correct_min": round(c_min, 4), "correct_avg": round(c_avg, 4),
            "incorrect_max": round(i_max, 4), "incorrect_avg": round(i_avg, 4),
            "separation": "clean" if c_min > i_max else "overlap",
        }

    if write_to and measured:
        import datetime
        payload = {
            "source": f"{manifest_path} ({len(labels)} images, "
                       f"{len(set(labels))} labels), prompt {prompt_template!r}, "
                       f"generated {datetime.date.today().isoformat()}",
            "confidence": {t: v["confidence"] for t, v in measured.items()},
            "novelty": {t: v["novelty"] for t, v in measured.items()},
            "detail": measured,
        }
        with open(write_to, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote {write_to} — src/gate.py picks this up automatically "
              f"on the next run.")
        if any(v["separation"] == "overlap" for v in measured.values()):
            print("NOTE: at least one tier had OVERLAPPING distributions, so these")
            print("      thresholds are a compromise, not a clean decision boundary.")
            print("      More distinctive test photos would help more than tuning.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", help="path to manifest.json")
    ap.add_argument("--calibrate", action="store_true",
                    help="also measure text->image score distributions per tier, "
                          "to set gate.TIER_CONFIDENCE_THRESHOLDS from real data")
    ap.add_argument("--prompt", default="a photo of a {label}",
                    help="text query template used by --calibrate "
                          "(default: 'a photo of a {label}')")
    ap.add_argument("--write-thresholds", metavar="PATH", default=None,
                    help="with --calibrate, write measured thresholds to PATH "
                          "(use thresholds.json — src/gate.py loads it automatically)")
    args = ap.parse_args()

    items = load_manifest(args.manifest)
    print(f"Loaded {len(items)} labeled test images.")

    embeddings, labels = [], []
    for path, label in items:
        img = Image.open(path).convert("RGB")
        vec = embed_image(img)
        embeddings.append(vec)
        labels.append(label)
        print(f"  embedded {os.path.basename(path)} -> '{label}'")

    print("\nRecall@1 by tier (higher is better — this is the real, measured")
    print("version of the 'what's lost at each dimension' claim):\n")
    for tier, dim in TIER_DIMS.items():
        r1 = recall_at_1_for_tier(embeddings, labels, dim)
        print(f"  {tier:8s} ({dim:4d} dims): {r1 * 100:5.1f}%")

    if args.calibrate:
        calibrate_thresholds(embeddings, labels, args.prompt,
                              write_to=args.write_thresholds,
                              manifest_path=args.manifest)
    elif args.write_thresholds:
        print("\n--write-thresholds needs --calibrate; nothing written.")


if __name__ == "__main__":
    main()
