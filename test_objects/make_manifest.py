"""
make_manifest.py

Builds manifest.json from filenames, so you never hand-write the JSON.

Naming convention — label first, view number last:

    mug_white_01.jpg   ->  label "mug_white"
    mug_white_02.jpg   ->  label "mug_white"
    mug_logo_01.jpg    ->  label "mug_logo"      <- different label on purpose
    banana_01.jpg      ->  label "banana"

Anything after the final underscore that is a number is treated as the view
index and stripped; everything before it is the label. Photos of the SAME
physical object share a label; near-duplicate objects you want kept apart get
DIFFERENT labels — that pair is the whole point of the test set.

Run:
    python3 test_objects/make_manifest.py                  # writes test_objects/manifest.json
    python3 test_objects/make_manifest.py --holdout shoe   # mark labels never memorized

Holdout labels go into the manifest under "_holdout" so the novelty test knows
which objects the memory was deliberately never shown.
"""

from __future__ import annotations
import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))


def label_for(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0]
    # strip a trailing view index: mug_white_01 -> mug_white
    return re.sub(r"_\d+$", "", stem)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=HERE, help="folder of photos (default: this one)")
    ap.add_argument("--holdout", nargs="*", default=[],
                    help="labels that were NEVER memorized — the novelty controls")
    ap.add_argument("--out", default=None, help="output path (default: <dir>/manifest.json)")
    args = ap.parse_args()

    paths = sorted(p for ext in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG")
                    for p in glob.glob(os.path.join(args.dir, f"*.{ext}")))
    if not paths:
        print(f"No photos found in {args.dir}")
        print("Expected files like: mug_white_01.jpg, mug_white_02.jpg, banana_01.jpg")
        return 1

    manifest = {os.path.basename(p): label_for(p) for p in paths}
    counts = Counter(manifest.values())

    out = args.out or os.path.join(args.dir, "manifest.json")
    payload = dict(manifest)
    if args.holdout:
        payload["_holdout"] = list(args.holdout)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(f"Wrote {out}")
    print(f"  {len(manifest)} photos, {len(counts)} labels\n")

    # The check that actually matters: Recall@1 is meaningless with one photo
    # per label, because the nearest neighbour can never share a label.
    singles = [lab for lab, n in counts.items() if n < 2 and lab not in args.holdout]
    for lab, n in sorted(counts.items()):
        flag = ""
        if lab in args.holdout:
            flag = "  (holdout — never memorized)"
        elif n < 2:
            flag = "  <-- NEEDS A SECOND PHOTO"
        print(f"  {lab:22} {n} photo{'s' if n != 1 else ' '}{flag}")

    if singles:
        print(f"\n{len(singles)} label(s) have only one photo. Recall@1 scores those as")
        print("0% by construction — the nearest neighbour can never share the label.")
        print("Shoot a second view of each before running evaluate_tiers.py.")
    if args.holdout:
        missing = [h for h in args.holdout if h not in counts]
        if missing:
            print(f"\nHoldout labels with no photos: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
