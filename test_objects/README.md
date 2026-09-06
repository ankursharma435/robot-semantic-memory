# test_objects/

Real photographs for `evaluate_tiers.py`. This folder ships empty on
purpose — the whole point is that the numbers come from **your** objects,
photographed with **your** camera, in **your** lighting.

## The one rule that matters

`recall_at_1_for_tier()` asks: *is my nearest neighbour an image with the
same label?* So **every label needs at least two photos.** With one photo
per label the nearest neighbour can never share a label and Recall@1 is
0% by construction, no matter how good the encoder is.

Aim for 2–4 photos per object, from different angles/distances.

## What actually stress-tests the low-dim tiers

Near-duplicates with **different** labels. Two visually similar mugs that
must stay distinct is the case where a 64-dim prefix falls apart, and it's
the interesting result for the submission. A set of four wildly different
objects will score well at every tier and prove nothing about tiering.

## Layout

```
test_objects/
  mug_plain_01.jpg     mug_plain_02.jpg
  mug_logo_01.jpg      mug_logo_02.jpg
  red_pen_01.jpg       red_pen_02.jpg
  manifest.json
```

`manifest.json` maps filename -> label (see `manifest.example.json`):

```json
{
  "mug_plain_01.jpg": "mug_plain",
  "mug_plain_02.jpg": "mug_plain",
  "mug_logo_01.jpg":  "mug_logo",
  "mug_logo_02.jpg":  "mug_logo"
}
```

Note `mug_plain` and `mug_logo` are *different* labels — that pair is the
stress test. Give two photos of the *same* mug the *same* label.

## Run

```bash
python evaluate_tiers.py test_objects/manifest.json
python evaluate_tiers.py test_objects/manifest.json --calibrate
```

`--calibrate` additionally measures the text->image score distributions the
gate has to separate, and prints a suggested per-tier threshold. Use it to
replace the provisional values in `src/gate.py` — those are corrected for
dimensional bias but are **not** tuned to real data.

Read its `separation:` line. `OVERLAP` means no absolute threshold can
separate correct from incorrect matches on this set — get more distinctive
photos rather than hunting for a better number.
