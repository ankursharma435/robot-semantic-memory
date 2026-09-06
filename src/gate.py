"""
gate.py

The decision point from deck slide 12. Given a memory lookup, decide:
  - CHEAP  : confident match, nothing novel being asked -> act on memory
             directly (e.g. navigate to remembered coordinates).
  - ESCALATE: low confidence, novel scene, or the situation needs fine
              visual grounding / manipulation -> this is where you would
              call the actual VLA. Stubbed out here — wire in a real
              model (e.g. an OpenVLA or pi0 checkpoint served locally or
              over the network) once you're past the memory-only phase.

Thresholds below are starting points, not tuned values — expect to
adjust once you're looking at real similarity scores from your own
camera and environment.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass
from enum import Enum


class Decision(Enum):
    CHEAP = "cheap"
    ESCALATE = "escalate"


@dataclass
class GateResult:
    decision: Decision
    confidence: float
    reason: str


# Starting thresholds — tune against your own data.
CONFIDENCE_THRESHOLD = 0.35   # below this, treat the match as unreliable
NOVELTY_THRESHOLD = 0.25      # below this similarity to *anything* stored, treat scene as new

# Per-tier thresholds, added 2026-09-03.
#
# WHY THIS HAS TO BE PER-TIER: a single absolute cosine cutoff cannot serve
# 64/256/768 dims, because truncating an MRL embedding systematically RAISES
# the cosine of any given pair. Measured on one query, identical pair:
#     long (768d) = 0.354   medium (256d) = 0.386   short (64d) = 0.435
# i.e. roughly +0.03 at 256d and +0.08 at 64d relative to 768d. Judging a
# 64-dim score with a threshold calibrated at 768 dims lets weak matches
# through, which is exactly how novelty detection was failing.
#
# The BASE values are still the author's original guesses (unchanged, and
# still unvalidated); the per-tier offsets below only correct the known
# dimensional bias. These are NOT tuned values. On synthetic test scenes the
# correct- and incorrect-match score distributions actually OVERLAP
# (a correct match scored 0.185 while an unrelated object scored 0.208), so
# no absolute threshold can separate them there. Recalibrate against real
# photographs before trusting the CHEAP path — `evaluate_tiers.py --calibrate`
# prints the measured distributions you need to set these from data.
_DIM_BIAS = {"long": 0.0, "medium": 0.03, "short": 0.08}

TIER_CONFIDENCE_THRESHOLDS = {t: CONFIDENCE_THRESHOLD + b for t, b in _DIM_BIAS.items()}
TIER_NOVELTY_THRESHOLDS = {t: NOVELTY_THRESHOLD + b for t, b in _DIM_BIAS.items()}

# ---- calibration override --------------------------------------------------
#
# The values above are still guesses. Measured text->image scores from two
# separate synthetic sets put correct matches at ~0.24-0.39 (768d), which
# means a 0.35 cutoff rejects most CORRECT matches — the demo would escalate
# almost everything. The right fix is to calibrate on real photographs, not
# to hand-pick a number here, so:
#
#     python evaluate_tiers.py test_objects/manifest.json --calibrate \
#         --write-thresholds thresholds.json
#
# writes measured per-tier thresholds, and this block picks them up on the
# next run. Keeping calibration in a data file (rather than edited constants)
# means re-shooting your test photos is a one-command change and the demo's
# behaviour is always traceable to a measurement.
THRESHOLDS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "thresholds.json")

CALIBRATED_FROM = None


def load_calibration(path: str, confidence: dict, novelty: dict):
    """Apply a thresholds.json to the given dicts, in place.

    Returns the calibration's `source` string, or None if there was nothing
    usable. A malformed or unreadable file is deliberately NON-fatal: losing
    calibration mid-demo should degrade to the provisional defaults, not
    crash the demo. Extracted from module scope so it is testable.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            cal = json.load(f)
        conf, nov = cal.get("confidence", {}), cal.get("novelty", {})
        for tier in list(confidence):
            if tier in conf:
                confidence[tier] = float(conf[tier])
            if tier in nov:
                novelty[tier] = float(nov[tier])
        return cal.get("source", path)
    except Exception as e:  # never let a bad file break the demo
        print(f"[gate] ignoring malformed {path}: {e}")
        return None


CALIBRATED_FROM = load_calibration(
    THRESHOLDS_FILE, TIER_CONFIDENCE_THRESHOLDS, TIER_NOVELTY_THRESHOLDS)


def threshold_summary() -> str:
    """One line describing which thresholds are in force — worth printing at
    demo startup so you never present numbers you can't account for."""
    src = f"calibrated from {CALIBRATED_FROM}" if CALIBRATED_FROM else (
        "PROVISIONAL defaults (not calibrated — run evaluate_tiers.py --calibrate)")
    vals = "  ".join(f"{t}={v:.3f}" for t, v in TIER_CONFIDENCE_THRESHOLDS.items())
    return f"gate thresholds [{src}]: {vals}"


def decide(best_similarity: float, requires_manipulation: bool = False,
            tier: str = "long") -> GateResult:
    """
    best_similarity: top cosine similarity returned by MemoryStore.query()
                      against the relevant tier.
    requires_manipulation: set True when the current task is something
                      memory alone can't finish (e.g. "pick up the mug",
                      not just "go to the mug's last known location").
    tier: which tier `best_similarity` was measured at, so the right
                      threshold is applied. Defaults to "long" (the
                      full-width, least-biased tier).
    """
    confidence_threshold = TIER_CONFIDENCE_THRESHOLDS.get(tier, CONFIDENCE_THRESHOLD)
    novelty_threshold = TIER_NOVELTY_THRESHOLDS.get(tier, NOVELTY_THRESHOLD)

    if requires_manipulation:
        return GateResult(
            Decision.ESCALATE, best_similarity,
            "task requires fine-grained perception/action beyond a stored coordinate",
        )
    if best_similarity < novelty_threshold:
        return GateResult(
            Decision.ESCALATE, best_similarity,
            f"scene doesn't match anything in memory closely enough "
            f"(< {novelty_threshold:.2f} at {tier}) — treat as novel",
        )
    if best_similarity < confidence_threshold:
        return GateResult(
            Decision.ESCALATE, best_similarity,
            f"weak match ({best_similarity:.3f} < {confidence_threshold:.2f} at "
            f"{tier}) — not confident enough to act on memory alone",
        )
    return GateResult(
        Decision.CHEAP, best_similarity,
        f"confident match ({best_similarity:.3f} >= {confidence_threshold:.2f} "
        f"at {tier}) — proceed using stored coordinates",
    )


def call_vla_stub(instruction: str):
    """Placeholder for the expensive path. Replace this with an actual
    call to a VLA checkpoint (local inference or a served endpoint) once
    the memory system is working end to end. Keeping it a stub for now
    means you can test and demo the entire gating logic before you've
    even touched the VLA model."""
    print(f"[STUB] would call VLA now for instruction: '{instruction}'")
    return {"action": "noop", "note": "VLA not yet wired in"}
