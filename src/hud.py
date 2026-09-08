"""
hud.py

An on-frame heads-up display, so a screen recording of the demo window tells
the whole story on its own.

Before this, the interesting state — per-tier similarity, the gate verdict,
which NVIDIA service was called, the latencies — only existed as terminal
scrollback, while the video window showed the camera and one line of text.
That splits a recording across two places and makes the demo hard to read.

Layout is composed onto a FIXED canvas (1280x720 by default) rather than
sized from the camera frame. Two reasons: the window is then the same size
whatever resolution the webcam gives you, and the panel gets the full canvas
height, which leaves room for text large enough to read in a screen
recording that a viewer may watch scaled down on a phone.

Drawn with PIL rather than cv2.putText, because Hershey fonts look like a
debug tool and this is the thing people actually look at. The panel is
cached and only re-rendered when the state changes, so the camera still
runs at full frame rate.

Usage:
    hud = HUD()
    hud.set_memory(store.stats())
    hud.set_result(...)
    cv2.imshow("...", hud.compose(frame))
"""

from __future__ import annotations
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CANVAS_W, CANVAS_H = 1280, 720
PANEL_W = 560                      # right-hand panel
CAM_W = CANVAS_W - PANEL_W         # 720px for the camera

BG = (10, 14, 18)
PANEL_BG = (16, 21, 26)
LINE = (44, 56, 65)
INK = (233, 239, 244)
INK_SOFT = (158, 173, 184)
INK_FAINT = (116, 131, 143)
CHEAP = (88, 205, 181)
CHEAP_BG = (11, 74, 64)
ESC = (232, 162, 82)
ESC_BG = (74, 46, 11)
NV = (129, 200, 0)
BAR_BG = (38, 48, 56)
BAR = (78, 127, 158)
BAR_WIN = (138, 202, 232)

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
    def __init__(self, tier_dims: dict | None = None, scale: float = 1.0):
        """scale nudges every font size — raise it if you're recording for
        a small screen, lower it if the panel starts overflowing."""
        self.tier_dims = tier_dims or {"short": 64, "medium": 256, "long": 768}
        s = scale
        self.f_label = _font(_MONO, int(15 * s))
        self.f_mono = _font(_MONO, int(20 * s))
        self.f_mono_b = _font(_MONO, int(22 * s), 1)
        self.f_val = _font(_MONO, int(21 * s), 1)
        self.f_badge = _font(_MONO, int(26 * s), 1)
        # Prose (the gate's reason, the VLM's answer) gets a smaller face than
        # the numbers. Keeps the figures legible while leaving room for the
        # VLM response, which is the payoff of the escalation beat.
        self.f_prose = _font(_MONO, int(17 * s))

        self.counts = {"short": 0, "medium": 0, "long": 0}
        self.query = ""
        self.per_tier = {}
        self.tier = None
        self.decision = None
        self.reason = ""
        self.label = None
        self.pos = None
        self.encode_ms = None
        self.search_ms = None
        self.vlm_ms = None
        self.vlm_text = ""
        self.active = set()
        self.status = "ready"
        self._panel = None
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
        self.query = query
        self.spoken = spoken
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

    def _render_panel(self) -> np.ndarray:
        im = Image.new("RGB", (PANEL_W, CANVAS_H), PANEL_BG)
        d = ImageDraw.Draw(im)
        x = 28
        w = PANEL_W - 56
        y = 26

        # ---- query, if one is running
        if self.query:
            d.text((x, y), "QUERY", font=self.f_label, fill=INK_FAINT)
            y += 26
            prefix = "\U0001F3A4 " if getattr(self, "spoken", False) else "> "
            for ln in self._wrap(d, prefix + self.query, self.f_mono, w)[:2]:
                d.text((x, y), ln, font=self.f_mono, fill=INK)
                y += 27
            y += 12

        d.text((x, y), "MEMORY", font=self.f_label, fill=INK_FAINT)
        y += 26
        for t in ("short", "medium", "long"):
            d.text((x, y), f"{t}  {self.tier_dims[t]}d", font=self.f_mono, fill=INK_SOFT)
            d.text((x + w, y), str(self.counts.get(t, 0)), font=self.f_val,
                   fill=INK, anchor="ra")
            y += 28
        y += 14

        if self.per_tier:
            d.text((x, y), "SIMILARITY BY TIER", font=self.f_label, fill=INK_FAINT)
            y += 28
            for t in ("long", "medium", "short"):
                v = self.per_tier.get(t, 0.0)
                win = (t == self.tier)
                d.text((x, y), f"{t}  {self.tier_dims[t]}d", font=self.f_mono,
                       fill=INK if win else (INK_FAINT if t == "short" else INK_SOFT))
                self._bar(d, x + 150, y + 8, w - 260, 12, v / 0.55,
                          BAR_WIN if win else BAR)
                d.text((x + w, y), f"{v:.3f}", font=self.f_val,
                       fill=INK if win else INK_FAINT, anchor="ra")
                y += 32
            y += 14

        if self.decision:
            cheap = self.decision == "cheap"
            col, bgc = (CHEAP, CHEAP_BG) if cheap else (ESC, ESC_BG)
            lbl = "CHEAP" if cheap else "ESCALATE"
            tw = int(d.textlength(lbl, font=self.f_badge)) + 40
            d.rounded_rectangle([x, y, x + tw, y + 44], radius=5, fill=bgc,
                                 outline=col, width=2)
            d.text((x + 20, y + 9), lbl, font=self.f_badge, fill=col)
            y += 54
            for ln in self._wrap(d, self.reason, self.f_prose, w)[:2]:
                d.text((x, y), ln, font=self.f_prose, fill=INK_SOFT)
                y += 23
            y += 12

        # The rail and latency strip are pinned to the bottom, so anything
        # flowing down the panel has to stop short of them or it overlaps.
        FLOW_LIMIT = CANVAS_H - 136

        if self.vlm_text and y < FLOW_LIMIT:
            d.text((x, y), "VLM RESPONSE", font=self.f_label, fill=INK_FAINT)
            y += 26
            lines = self._wrap(d, self.vlm_text, self.f_prose, w)
            for i, ln in enumerate(lines):
                if y + 23 > FLOW_LIMIT:
                    remaining = len(lines) - i
                    if remaining:
                        d.text((x, y), f"... +{remaining} more line"
                                        f"{'s' if remaining > 1 else ''}",
                               font=self.f_label, fill=INK_FAINT)
                    break
                d.text((x, y), ln, font=self.f_prose, fill=INK)
                y += 23

        # ---- NVIDIA rail + latency strip, pinned to the bottom
        ry = CANVAS_H - 118
        cx = x
        for c in ("NIM embed", "NIM VLM", "Riva ASR", "Riva TTS"):
            on = c in self.active
            cw = int(d.textlength(c, font=self.f_label)) + 26
            d.rounded_rectangle([cx, ry, cx + cw, ry + 30], radius=4,
                                 fill=(26, 42, 12) if on else None,
                                 outline=NV if on else LINE, width=2 if on else 1)
            d.text((cx + 13, ry + 7), c, font=self.f_label, fill=NV if on else INK_FAINT)
            cx += cw + 9

        ly = CANVAS_H - 68
        cells = []
        if self.encode_ms is not None:
            cells.append(("encode", f"{self.encode_ms:.0f} ms"))
        if self.search_ms is not None:
            cells.append(("search", f"{self.search_ms:.3f} ms"))
        cells.append(("VLM", f"{self.vlm_ms:.0f} ms") if self.vlm_ms
                     else ("VLA", "not called"))
        cw = w // max(1, len(cells))
        for i, (k, v) in enumerate(cells):
            cx = x + i * cw
            d.text((cx, ly), k, font=self.f_label, fill=INK_FAINT)
            d.text((cx, ly + 22), v, font=self.f_val, fill=INK)

        return np.array(im)

    def compose(self, frame):
        """frame is BGR from the camera; returns a fixed-size BGR canvas."""
        if self._dirty or self._panel is None:
            self._panel = self._render_panel()
            self._dirty = False

        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        canvas[:, :] = BG[::-1]

        # camera, scaled to fit the left column, centred
        fh, fw = frame.shape[:2]
        s = min(CAM_W / fw, CANVAS_H / fh)
        nw, nh = max(1, int(fw * s)), max(1, int(fh * s))
        import cv2
        small = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        y0 = (CANVAS_H - nh) // 2
        x0 = (CAM_W - nw) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = small

        canvas[:, CAM_W:] = self._panel[:, :, ::-1]      # RGB -> BGR
        return canvas
