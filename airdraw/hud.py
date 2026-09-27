"""On-screen overlay: FPS, mode, palette, help card, cursor and hand skeleton."""

import time

import cv2
import numpy as np

from . import config
from .tracker import FINGER_TIPS, HAND_BONES, INDEX_TIP
from .ui import (alpha_circle, draw_text, ease_out, fill_rounded, glass_panel,
                 mix, stroke_rounded, text_size)

MODE_LABELS = {"draw": "Drawing", "erase": "Eraser", "idle": "Pen off"}

WHITE = (255, 255, 255)
MUTED = (200, 192, 186)
GOOD = (120, 220, 100)
WARN = (60, 190, 250)
BAD = (90, 90, 245)

HELP_LINES = [
    ("Right pinky + index", "pen on / off"),
    ("Left thumb + index", "next colour"),
    ("Left pinky + index", "eraser"),
]
KEY_HINT = "X settings   Z undo   C clear   S save   H help"


class Hud:
    def __init__(self, width, height):
        self.w, self.h = width, height
        self.help_until = time.perf_counter() + 9.0
        self.help_pinned = False
        self._toast = None           # (text, color, start_time)
        self._pulses = []            # (x, y, color, start_time)
        self._palette_x = None       # animated position of the selection ring

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------
    def toast(self, text, color=WHITE):
        self._toast = (text, color, time.perf_counter())

    def pulse(self, pt, color=WHITE):
        self._pulses.append((float(pt[0]), float(pt[1]), color, time.perf_counter()))

    def toggle_help(self):
        self.help_pinned = not self.help_pinned
        if not self.help_pinned:
            self.help_until = 0

    # ------------------------------------------------------------------
    # Hand overlay
    # ------------------------------------------------------------------
    def draw_skeleton(self, img, hand):
        pts = hand.points
        x0, y0 = np.floor(pts.min(axis=0)).astype(int) - 8
        x1, y1 = np.ceil(pts.max(axis=0)).astype(int) + 8
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, self.w), min(y1, self.h)
        if x1 <= x0 or y1 <= y0:
            return

        # Draw on a copy of the hand's bounding box, then fade it in, so the
        # bones read as a light overlay rather than solid paint.
        roi = img[y0:y1, x0:x1]
        layer = roi.copy()
        local = ((pts - (x0, y0)) * 16).astype(np.int32)
        for a, b in HAND_BONES:
            cv2.line(layer, tuple(local[a]), tuple(local[b]), (235, 228, 222), 2, cv2.LINE_AA, shift=4)
        for i, p in enumerate(local):
            if i == INDEX_TIP:
                continue
            r = 4 if i in FINGER_TIPS else 3
            cv2.circle(layer, tuple(p), r * 16, (255, 255, 255), -1, cv2.LINE_AA, shift=4)
        cv2.addWeighted(layer, 0.6, roi, 0.4, 0, dst=roi)

    def draw_cursor(self, img, pt, mode, color, brush, eraser, suspended):
        if mode == "erase":
            alpha_circle(img, pt, eraser, WHITE, 0.07)
            alpha_circle(img, pt, eraser, WHITE, 0.85, thickness=1)
            alpha_circle(img, pt, 2, WHITE, 0.9)
        elif mode == "draw" and not suspended:
            alpha_circle(img, pt, brush + 9, color, 0.18)
            alpha_circle(img, pt, max(brush // 2 + 2, 4), color, 1.0)
            alpha_circle(img, pt, max(brush // 2 + 2, 4) - 2, mix(color, WHITE, 0.7), 1.0)
        else:
            alpha_circle(img, pt, 11, WHITE, 0.75, thickness=1)
            alpha_circle(img, pt, 3, color, 0.9)

    def draw_pulses(self, img):
        now = time.perf_counter()
        alive = []
        for x, y, color, t0 in self._pulses:
            t = (now - t0) / 0.5
            if t >= 1:
                continue
            e = ease_out(t)
            alpha_circle(img, (x, y), 10 + 34 * e, color, 0.9 * (1 - t), thickness=2)
            alive.append((x, y, color, t0))
        self._pulses = alive

    # ------------------------------------------------------------------
    # Panels
    # ------------------------------------------------------------------
    def draw_fps(self, img, fps):
        label = f"{fps:.0f} FPS"
        tw, _ = text_size(label, 14, "semibold")
        w, h = tw + 40, 32
        x, y = self.w - w - 16, 16
        glass_panel(img, (x, y, w, h), r=h // 2)
        dot = GOOD if fps >= 30 else WARN if fps >= config.FPS_MIN else BAD
        alpha_circle(img, (x + 17, y + h / 2), 4, dot, 1.0)
        draw_text(img, label, (x + 28, y + h // 2), 14, WHITE, "semibold", anchor="lm")

    def draw_mode(self, img, mode, color, color_name, left_seen, right_seen):
        label = MODE_LABELS[mode]
        sub = color_name if mode != "erase" else "rub to clear"
        lw, _ = text_size(label, 15, "semibold")
        sw, _ = text_size(sub, 12)
        w = 16 + 22 + max(lw, sw) + 22 + 46
        h = 48
        x, y = 16, 16
        glass_panel(img, (x, y, w, h), r=16)

        # Mode swatch
        cx, cy = x + 26, y + h / 2
        if mode == "erase":
            alpha_circle(img, (cx, cy), 9, WHITE, 0.9, thickness=2)
        else:
            alpha_circle(img, (cx, cy), 12, color, 0.25 if mode == "draw" else 0.12)
            alpha_circle(img, (cx, cy), 7, color, 1.0 if mode == "draw" else 0.55)

        draw_text(img, label, (x + 46, y + 9), 15, WHITE, "semibold")
        draw_text(img, sub, (x + 46, y + 28), 12, MUTED)

        # Hand presence chips
        bx = x + w - 52
        for i, (tag, seen) in enumerate((("L", left_seen), ("R", right_seen))):
            rx = bx + i * 22
            fill_rounded(img, (rx, y + 14, 18, 20), 6, config.ACCENT if seen else (90, 84, 80),
                         0.9 if seen else 0.5)
            draw_text(img, tag, (rx + 9, y + 24), 11, (20, 16, 14) if seen else MUTED,
                      "semibold", anchor="cm")

    def draw_palette(self, img, index):
        n = len(config.PALETTE)
        gap = 30
        w, h = n * gap + 20, 40
        x, y = (self.w - w) // 2, self.h - h - 16
        glass_panel(img, (x, y, w, h), r=h // 2)

        cy = y + h / 2
        target = x + 10 + gap / 2 + index * gap
        if self._palette_x is None:
            self._palette_x = target
        self._palette_x += (target - self._palette_x) * 0.35

        for i, (_, color) in enumerate(config.PALETTE):
            cx = x + 10 + gap / 2 + i * gap
            alpha_circle(img, (cx, cy), 8 if i == index else 7, color, 1.0)
        alpha_circle(img, (self._palette_x, cy), 12, WHITE, 0.95, thickness=2)

    def draw_help(self, img):
        now = time.perf_counter()
        if self.help_pinned:
            a = 1.0
        else:
            remaining = self.help_until - now
            if remaining <= 0:
                return
            a = min(1.0, remaining / 0.6)

        x, y, w = 16, 76, 286
        h = 30 + len(HELP_LINES) * 24 + 30
        glass_panel(img, (x, y, w, h), r=14, tint_alpha=0.55 * a, border_alpha=0.14 * a, blur=a > 0.5)
        draw_text(img, "GESTURES", (x + 16, y + 12), 11, config.ACCENT, "semibold", alpha=a)
        for i, (gesture, action) in enumerate(HELP_LINES):
            ly = y + 34 + i * 24
            draw_text(img, gesture, (x + 16, ly), 13, WHITE, alpha=a)
            draw_text(img, action, (x + w - 16, ly), 13, MUTED, anchor="rt", alpha=a)
        stroke_rounded(img, (x + 16, y + h - 30, w - 32, 1), 0, WHITE, 0.08 * a)
        draw_text(img, KEY_HINT, (x + 16, y + h - 22), 11, MUTED, alpha=a * 0.9)

    def draw_toast(self, img):
        if self._toast is None:
            return
        text, color, t0 = self._toast
        t = time.perf_counter() - t0
        if t > 1.4:
            self._toast = None
            return
        a = min(1.0, t / 0.12) * (1.0 if t < 1.0 else 1 - (t - 1.0) / 0.4)
        rise = (1 - ease_out(t / 0.25)) * 8
        tw, _ = text_size(text, 15, "semibold")
        w, h = tw + 48, 36
        x, y = (self.w - w) // 2, int(18 + rise)
        glass_panel(img, (x, y, w, h), r=h // 2, tint_alpha=0.6 * a, border_alpha=0.15 * a, blur=a > 0.5)
        alpha_circle(img, (x + 20, y + h / 2), 5, color, a)
        draw_text(img, text, (x + 32, y + h // 2), 15, WHITE, "semibold", alpha=a, anchor="lm")

    def draw_message(self, img, title, detail=""):
        """Centred card used while waiting for the camera."""
        tw, _ = text_size(title, 18, "semibold")
        dw, _ = text_size(detail, 13) if detail else (0, 0)
        w, h = max(tw, dw) + 64, 86 if detail else 60
        x, y = (self.w - w) // 2, (self.h - h) // 2
        glass_panel(img, (x, y, w, h), r=18, blur=False, tint_alpha=0.8)
        draw_text(img, title, (self.w // 2, y + 20), 18, WHITE, "semibold", anchor="ct")
        if detail:
            draw_text(img, detail, (self.w // 2, y + 52), 13, MUTED, anchor="ct")
