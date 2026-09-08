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
CAM_W = CANVAS_W - PANEL_W         # 720px for the camera column
CAM_H = 466                        # camera sits above...
LOG_H = CANVAS_H - CAM_H           # ...an embedded log of the demo's output

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
        self._card_img = None
        self.card = None
        self._toast = None          # (text, expires_at)
        self._log = []              # embedded terminal output
        self._log_img = None
        self._log_dirty = True
        self._dyn = None            # dynamically built card
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

    # ---- presentation cards ---------------------------------------------
    #
    # Full-canvas title cards, shown in the demo window itself so a screen
    # recording captures them. Without these the recording is 90 seconds of
    # camera feed with no framing, and the narration has nothing to sit
    # against. Press 1-5 to show one, 0 to return to the live view.

    CARDS = {
        "1": ("THE PROBLEM",
              "A robot sees the same things,\nover and over.",
              ["every question to a large VLM pays full price",
               "for answers it already has"]),
        "2": ("THE INNOVATION",
              "One encode.\nThree memory tiers.",
              ["Matryoshka: any prefix is still a valid embedding",
               "short 64d   what I just saw",
               "medium 256d  what I've learned",
               "long 768d   what I know well"]),
        "3": ("WHY IT MATTERS ON A ROBOT",
              "Cheap memory,\nno network needed.",
              ["an 8-hour shift of memory = 7 MB, not 236 MB",
               "the cheap path needs no network at all",
               "the accelerator stays free for driving and grasping"]),
        "4": ("THE STACK",
              "Open weights,\nNVIDIA microservices.",
              ["jina-clip-v2         local, CC BY-NC, no GPU",
               "llama-3.2-11b-vision  NVIDIA NIM",
               "llama-nemotron-embed-vl  NVIDIA NIM",
               "parakeet ASR + magpie TTS  NVIDIA Riva"]),
        "5": ("SHIPPED",
              "Code, tests,\nand the numbers.",
              ["44 unit tests, every measurement reproducible",
               "github.com/ankursharma435/robot-semantic-memory"]),
    }

    def log(self, text: str, keep: int = 9):
        """Mirror a line of the demo's output into the window, so a screen
        recording of this one window shows everything — no second window to
        frame, and nothing important living only in terminal scrollback."""
        for raw in str(text).rstrip().split("\n"):
            self._log.append(raw)
        self._log = self._log[-keep:]
        self._log_dirty = True

    def toast(self, text: str, seconds: float = 2.5):
        """A banner across the camera view. Confirms an action ON THE FRAME,
        so a recording that never shows the terminal still shows what
        happened — pressing 'm' used to change only a small tier count."""
        import time as _t
        self._toast = (text, _t.time() + seconds)

    def show_metrics(self, measured, projected=None):
        """A card built at runtime from the session's real numbers, so the
        metrics beat is visible without showing terminal output.

        Two columns: what was measured in this run, and what that projects to
        on a fleet. Kept side by side rather than stacked because ten stacked
        rows overflow the canvas, and because the split is the point -- a
        viewer should be able to see at a glance which half is measurement
        and which half is extrapolation.
        """
        self._dyn = (list(measured), list(projected or []))
        self.card = "_dyn"
        self._card_img = None
        self._touch()

    def _render_metrics_card(self) -> np.ndarray:
        measured, projected = self._dyn or ([], [])
        im = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
        d = ImageDraw.Draw(im)
        f_kick = _font(_MONO, 18)
        f_title = _font(_SANS, 54, 1)
        f_row = _font(_MONO, 19)
        f_note = _font(_MONO, 14)

        d.text((80, 96), "WHAT IT COSTS", font=f_kick, fill=NV)
        d.text((80, 132), "Remembering vs asking.", font=f_title, fill=INK)

        colw = (CANVAS_W - 160 - 60) // 2
        for i, (head, rows, col) in enumerate(
                [("MEASURED THIS RUN", measured, CHEAP),
                 ("PROJECTED TO A FLEET", projected, (183, 166, 220))]):
            if not rows:
                continue
            x = 80 + i * (colw + 60)
            d.rounded_rectangle([x, 236, x + colw, 236 + 340], radius=6,
                                 fill=PANEL_BG)
            d.text((x + 26, 262), head, font=f_note, fill=col)
            y = 300
            for r in rows[:8]:
                # Clip rather than overflow the card. A long row used to run
                # past the panel edge and get cut mid-character, which reads
                # as a rendering bug in a recording.
                t = r
                while t and d.textlength(t, font=f_row) > colw - 52:
                    t = t[:-2]
                d.text((x + 26, y), t, font=f_row, fill=INK)
                y += 34

        d.text((80, CANVAS_H - 92),
               "Projections assume 1 frame/sec, 8-hour shifts, 250 shifts/year.",
               font=f_note, fill=INK_FAINT)
        d.text((80, CANVAS_H - 66),
               "Energy would be inferred from compute, not measured.",
               font=f_note, fill=INK_FAINT)
        return np.array(im)

    def show_card(self, key: str | None):
        """key is '1'-'5' to show a card, or None/'0' to go back to live."""
        new = key if (key in self.CARDS or key == "_dyn") else None
        if new != getattr(self, "card", None):
            self.card = new
            self._touch()
        return new is not None

    def _render_card(self, key) -> np.ndarray:
        if key == "_dyn":
            return self._render_metrics_card()
        kicker, title, bullets = self.CARDS[key]
        im = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
        d = ImageDraw.Draw(im)
        f_kick = _font(_MONO, 20)
        f_title = _font(_SANS, 62, 1)
        f_bul = _font(_MONO, 24)
        y = 150
        d.text((90, y), kicker, font=f_kick, fill=NV); y += 52
        for ln in title.split("\n"):
            d.text((90, y), ln, font=f_title, fill=INK); y += 74
        y += 30
        for b in bullets:
            d.text((90, y), "\u2022", font=f_bul, fill=NV)
            d.text((124, y), b, font=f_bul, fill=INK_SOFT); y += 40
        return np.array(im)

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

    def _render_log(self) -> np.ndarray:
        im = Image.new("RGB", (CAM_W, LOG_H), (7, 10, 13))
        d = ImageDraw.Draw(im)
        f = _font(_MONO, 15)
        fl = _font(_MONO, 12)
        d.line([(0, 0), (CAM_W, 0)], fill=LINE)
        d.text((20, 10), "DEMO OUTPUT", font=fl, fill=INK_FAINT)
        y = 32
        for ln in self._log:
            colour = INK_SOFT
            low = ln.lower()
            if "escalat" in low or "vlm" in low:
                colour = ESC
            elif "cheap" in low or "memorized" in low or "memorised" in low:
                colour = CHEAP
            # trim to the panel width rather than wrapping; these are log lines
            while ln and d.textlength(ln, font=f) > CAM_W - 40:
                ln = ln[:-2]
            d.text((20, y), ln, font=f, fill=colour)
            y += 20
            if y > LOG_H - 18:
                break
        return np.array(im)

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

        d.text((x, y), "MATRYOSHKA EMBEDDING BASED SEMANTIC MEMORY",
               font=self.f_label, fill=INK_FAINT)
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
        """frame is BGR from the camera; returns a fixed-size BGR canvas.
        If a presentation card is active, the card replaces the whole view."""
        if getattr(self, "card", None):
            if self._dirty or getattr(self, "_card_img", None) is None:
                self._card_img = self._render_card(self.card)
                self._dirty = False
            return self._card_img[:, :, ::-1].copy()

        if self._dirty or self._panel is None:
            self._panel = self._render_panel()
            self._dirty = False

        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        canvas[:, :] = BG[::-1]

        # camera, scaled into the top-left region
        fh, fw = frame.shape[:2]
        s = min(CAM_W / fw, CAM_H / fh)
        nw, nh = max(1, int(fw * s)), max(1, int(fh * s))
        import cv2
        small = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        y0 = (CAM_H - nh) // 2
        x0 = (CAM_W - nw) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = small

        # embedded log beneath it
        if self._log_dirty or self._log_img is None:
            self._log_img = self._render_log()
            self._log_dirty = False
        canvas[CAM_H:, :CAM_W] = self._log_img[:, :, ::-1]

        canvas[:, CAM_W:] = self._panel[:, :, ::-1]      # RGB -> BGR

        if self._toast:
            import time as _t
            text, expires = self._toast
            if _t.time() < expires:
                canvas = self._draw_toast(canvas, text)
            else:
                self._toast = None
        return canvas

    def _draw_toast(self, canvas, text):
        """Green banner across the camera column."""
        im = Image.fromarray(canvas[:, :, ::-1])
        d = ImageDraw.Draw(im, "RGBA")
        f = _font(_MONO, 30, 1)
        pad, bh = 26, 86
        y0 = CAM_H // 2 - bh // 2
        d.rectangle([0, y0, CAM_W, y0 + bh], fill=(11, 74, 64, 235))
        d.rectangle([0, y0, 8, y0 + bh], fill=CHEAP)
        lines = self._wrap(d, text, f, CAM_W - 2 * pad - 20)[:2]
        ty = y0 + (bh - len(lines) * 36) // 2
        for ln in lines:
            d.text((pad + 14, ty), ln, font=f, fill=(200, 255, 240))
            ty += 36
        return np.array(im)[:, :, ::-1].copy()
