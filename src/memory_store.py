"""
memory_store.py

A tiered semantic memory: every entry is (embedding, metadata). No faiss —
at robot-demo scale (hundreds to low thousands of entries) a brute-force
numpy dot product against the whole tier is a few hundred microseconds,
and it keeps this dependency-free and portable to Jetson's arm64 without
fighting wheel availability. If you outgrow this (tens of thousands of
entries), swap the linear scan in `query()` for a proper ANN index
(e.g. hnswlib) without changing anything else's interface.
"""

from __future__ import annotations
import time
import uuid
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from encoder import truncate, cosine, TIER_DIMS


@dataclass
class MemoryEntry:
    id: str
    tier: str                # "short" | "medium" | "long"
    embedding: np.ndarray     # already truncated to the tier's dim
    x: float
    y: float
    label: Optional[str] = None
    created_at: float = field(default_factory=time.time)


class MemoryStore:
    def __init__(self, short_ttl_seconds: float = 30.0):
        self._entries: dict[str, list[MemoryEntry]] = {"short": [], "medium": [], "long": []}
        self.short_ttl_seconds = short_ttl_seconds

    def add(self, tier: str, full_embedding: np.ndarray, x: float, y: float,
            label: Optional[str] = None) -> MemoryEntry:
        dim = TIER_DIMS[tier]
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            tier=tier,
            embedding=truncate(full_embedding, dim),
            x=x, y=y, label=label,
        )
        self._entries[tier].append(entry)
        return entry

    def prune_short_term(self):
        """Short-term memory decays fast — call this periodically so it
        doesn't grow without bound."""
        now = time.time()
        self._entries["short"] = [
            e for e in self._entries["short"]
            if now - e.created_at < self.short_ttl_seconds
        ]

    def query(self, full_query_embedding: np.ndarray, tier: str, top_k: int = 1):
        """Compare a query against one tier. Truncates the query to match
        that tier's stored dimension, then returns the top_k closest
        entries with their similarity scores."""
        dim = TIER_DIMS[tier]
        q = truncate(full_query_embedding, dim)
        scored = [(cosine(q, e.embedding), e) for e in self._entries[tier]]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[:top_k]

    def query_all_tiers(self, full_query_embedding: np.ndarray, top_k: int = 1):
        """Convenience: check long-term first (most durable), then
        medium, then short — mirrors how you'd actually want a robot to
        recall ("do I already know this place well?" before "did I just
        see this a moment ago?").

        NOTE: this returns a per-tier dict for INSPECTION. Do not pick a
        winner by max()-ing the scores across tiers — see best_match()
        for why that is wrong. Use best_match() to choose an answer.
        """
        results = {}
        for tier in ("long", "medium", "short"):
            results[tier] = self.query(full_query_embedding, tier, top_k)
        return results

    # Priority order for answering a recall query: most durable first.
    PRIORITY = ("long", "medium", "short")

    def best_match(self, full_query_embedding: np.ndarray, thresholds: dict,
                    priority: tuple = None):
        """Pick one answer across tiers, respecting tier priority.

        Added 2026-09-03 to fix a real bug in both demos. They used to do
        `max()` over the per-tier scores from query_all_tiers(). Cosine
        scores from different MRL truncation widths are NOT comparable:
        a narrow prefix systematically scores HIGHER than the same image/
        text pair does at full width, because the dropped dimensions are
        the discriminative ones. Measured on one query, same pair:

            long (768d) = 0.354   medium (256d) = 0.386   short (64d) = 0.435

        The gap was consistent across every query tried, so max() always
        returned the SHORT tier — which is the always-watching tier that
        is stored without labels and had the worst Recall@1 (0/4 vs 3/4
        at 256d on the same set). That produced two visible symptoms:
        every query printed label='None', and novelty detection broke
        (an object never seen at all still cleared the threshold on an
        inflated 64-dim score).

        So: walk the tiers in priority order and return the first whose
        top-1 clears THAT TIER's own threshold. If nothing clears, return
        the top-1 of the most durable non-empty tier, so the caller's gate
        sees a comparable score and escalates on it.

        Returns (tier, similarity, entry) or (None, -1.0, None) if the
        store is empty.
        """
        priority = priority or self.PRIORITY

        fallback = (None, -1.0, None)
        for tier in priority:
            top = self.query(full_query_embedding, tier, top_k=1)
            if not top:
                continue
            score, entry = top[0]
            if fallback[2] is None:
                fallback = (tier, score, entry)
            if score >= thresholds[tier]:
                return tier, score, entry
        return fallback

    def stats(self):
        return {tier: len(entries) for tier, entries in self._entries.items()}
