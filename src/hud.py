"""
hud.py

An on-frame heads-up display, so a screen recording of the demo window tells
the whole story on its own.

Before this, the interesting state — per-tier similarity, the gate verdict,
which NVIDIA service was called, the latencies — only existed as terminal
scrollback, while the video window showed the camera and one line of text.
That splits a recording across two places and makes the demo hard to read.

Drawn with PIL rather than cv2.putText, because Hershey fonts look like a
debug tool and this is the thing people will actually look at. The panel is
cached and only re-rendered when the state changes, so the camera still
runs at full frame rate.

Usage:
    hud = HUD()
    hud.set_memory(store.stats())
    hud.set_query("where is the red mug", per_tier, tier, decision, reason, ...)
    cv2.imshow("...", hud.compose(frame))
"""

from __future__ import annotations
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PANEL_W = 420
BG = (10, 14, 18)
PANEL_BG = (16, 21, 26)
LINE = (44, 56, 65)
INK = (230, 236, 241)
INK_SOFT = (150, 165, 176)
INK_FAINT = (110, 125, 137)
CHEAP = (79, 190, 168)
CHEAP_BG = (11, 74, 64)
ESC = (224, 154, 78)
ESC_BG = (74, 46, 11)
NV = (118, 185, 0)
BAR_BG = (38, 48, 56)
BAR = (78, 127, 158)
BAR_WIN = (127, 192, 222)

_MONO = ["/System/Library/Fonts/Menlo.ttc",
         "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]
_SANS = ["/System/Library/Fonts/HelveticaNeue.ttc",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]


def _font(paths, size, index=0):
    for p in paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size, index=index)
            except Exception:
                continue
    return ImageFont.load_default()


class HUD:
    def __init__(self, tier_dims: dict | None = None):
        self.tier_dims = tier_dims or {"short": 64, "medium": 256, "long": 768}
        self.f_label = _font(_MONO, 11)
        self.f_mono = _font(_MONO, 13)
        self.f_mono_b = _font(_MONO, 14, 1)
        self.f_body = _font(_SANS, 14)

        self.counts = {"short": 0, "medium": 0, "long": 0}
        self.query = ""
        self.per_tier = {}
        self.tier = None
        self.decision = None      # "cheap" | "escalate" | None
        self.reason = ""
        self.label = None
        self.pos = None
        self.encode_ms = None
        self.search_ms = None
        self.vlm_ms = None
        self.vlm_text = ""
        self.active = set()        # NVIDIA services lit this beat
        self.status = "ready"
        self._panel = None         # cached RGB ndarray
        self._dirty = True

    # ---- state -----------------------------------------------------------

    def _touch(self):
        self._dirty = True

    def set_memory(self, counts: dict):
        if counts != self.counts:
            self.counts = dict(counts)
            self._touch()

    def set_status(self, text: str):
        if text != self.status:
            self.status = text
            self._touch()

    def begin_query(self, query: str, spoken: bool = False):
        self.query = ("\U0001F3A4 " if spoken else "") + query
        self.per_tier, self.tier, self.decision = {}, None, None
        self.reason, self.label, self.pos = "", None, None
        self.encode_ms = self.search_ms = self.vlm_ms = None
        self.vlm_text = ""
        self.active = {"Riva ASR"} if spoken else set()
        self.status = "thinking"
        self._touch()

    def set_result(self, per_tier, tier, decision, reason, label, pos,
                    encode_ms, search_ms):
        self.per_tier, self.tier = dict(per_tier), tier
        self.decision, self.reason = decision, reason
        self.label, self.pos = label, pos
        self.encode_ms, self.search_ms = encode_ms, search_ms
        self.status = "ready"
        self._touch()

    def set_vlm(self, ms: float, text: str):
        self.vlm_ms, self.vlm_text = ms, text
        self.active = set(self.active) | {"NIM VLM"}
        self._touch()

    def light(self, *services):
        self.active = set(self.active) | set(services)
        self._touch()

    # ---- drawing ---------------------------------------------------------

    def _wrap(self, d, s, font, width):
        out, line = [], ""
        for w in s.split():
            t = (line + " " + w).strip()
            if d.textlength(t, font=font) > width and line:
                out.append(line)
                line = w
            else:
                line = t
        if line:
            out.append(line)
        return out

    def _bar(self, d, x, y, w, h, frac, colour):
        d.rectangle([x, y, x + w, y + h], fill=BAR_BG)
        fw = int(w * max(0.0, min(1.0, frac)))
        if fw > 0:
            d.rectangle([x, y, x + fw, y + h], fill=colour)

    def _render_panel(self, height: int) -> np.ndarray:
        im = Image.new("RGB", (PANEL_W, height), PANEL_BG)
        d = ImageDraw.Draw(im)
        x, w = 20, PANEL_W - 40
        y = 20

        d.text((x, y), "MEMORY", font=self.f_label, fill=INK_FAINT)
        y += 20
        for t in ("short", "medium", "long"):
            d.text((x, y), f"{t} {self.tier_dims[t]}d", font=self.f_mono, fill=INK_SOFT)
            d.text((x + w, y), str(self.counts.get(t, 0)), font=self.f_mono,
                   fill=INK, anchor="ra")
            y += 19
        y += 14

        if self.per_tier:
            d.text((x, y), "SIMILARITY BY TIER", font=self.f_label, fill=INK_FAINT)
            y += 22
            for t in ("long", "medium", "short"):
                v = self.per_tier.get(t, 0.0)
                win = (t == self.tier)
                d.text((x, y - 1), f"{t} {self.tier_dims[t]}d", font=self.f_mono,
                       fill=INK if win else (INK_FAINT if t == "short" else INK_SOFT))
                self._bar(d, x + 96, y + 3, w - 156, 8, v / 0.55,
                          BAR_WIN if win else BAR)
                d.text((x + w, y - 1), f"{v:.3f}", font=self.f_mono,
                       fill=INK if win else INK_FAINT, anchor="ra")
                y += 23
            y += 10

        if self.decision:
            cheap = self.decision == "cheap"
            col, bgc = (CHEAP, CHEAP_BG) if cheap else (ESC, ESC_BG)
            lbl = "CHEAP" if cheap else "ESCALATE"
            tw = int(d.textlength(lbl, font=self.f_mono_b)) + 26
            d.rounded_rectangle([x, y, x + tw, y + 28], radius=3, fill=bgc, outline=col)
            d.text((x + 13, y + 7), lbl, font=self.f_mono_b, fill=col)
            y += 38
            for ln in self._wrap(d, self.reason, self.f_mono, w)[:3]:
                d.text((x, y), ln, font=self.f_mono, fill=INK_SOFT)
                y += 18
            y += 6

        if self.vlm_text:
            d.text((x, y), "VLM RESPONSE", font=self.f_label, fill=INK_FAINT)
            y += 20
            for ln in self._wrap(d, self.vlm_text, self.f_mono, w)[:5]:
                d.text((x, y), ln, font=self.f_mono, fill=INK)
                y += 18

        # ---- NVIDIA rail + latency strip, pinned to the bottom
        ry = height - 78
        cx = x
        for c in ("NIM embed", "NIM VLM", "Riva ASR", "Riva TTS"):
            on = c in self.active
            cw = int(d.textlength(c, font=self.f_label)) + 18
            d.rounded_rectangle([cx, ry, cx + cw, ry + 20], radius=3,
                                 fill=(24, 38, 12) if on else None,
                                 outline=NV if on else LINE)
            d.text((cx + 9, ry + 4), c, font=self.f_label, fill=NV if on else INK_FAINT)
            cx += cw + 6

        ly = height - 46
        cells = []
        if self.encode_ms is not None:
            cells.append(("encode", f"{self.encode_ms:.0f} ms"))
        if self.search_ms is not None:
            cells.append(("search", f"{self.search_ms:.3f} ms"))
        cells.append(("VLM", f"{self.vlm_ms:.0f} ms") if self.vlm_ms
                     else ("VLA", "not called"))
        cx = x
        for k, v in cells:
            d.text((cx, ly), k, font=self.f_label, fill=INK_FAINT)
            d.text((cx, ly + 15), v, font=self.f_mono, fill=INK)
            cx += 125

        return np.array(im)

    def compose(self, frame):
        """frame is BGR from the camera; returns a wider BGR canvas."""
        h, fw = frame.shape[:2]
        if self._dirty or self._panel is None or self._panel.shape[0] != h:
            self._panel = self._render_panel(h)
            self._dirty = False
        panel_bgr = self._panel[:, :, ::-1]          # RGB -> BGR
        canvas = np.zeros((h, fw + PANEL_W, 3), dtype=np.uint8)
        canvas[:, :] = BG[::-1]
        canvas[:, :fw] = frame
        canvas[:, fw:] = panel_bgr
        return canvas
