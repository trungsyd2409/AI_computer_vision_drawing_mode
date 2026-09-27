"""The drawing surface.

Strokes are rasterised straight into two image layers that stay on screen
until erased:

* ``body``  - the coloured line itself
* ``core``  - a thinner, brighter line on top, which gives the "hot" centre
              you see in neon tubes

The glow is a blurred copy of the body layer added over the camera image.
It is only recomputed when the drawing actually changes.
"""

from collections import deque

import cv2
import numpy as np

from . import config
from .ui import mix


class NeonCanvas:
    def __init__(self, width, height):
        self.w, self.h = width, height
        self.body = np.zeros((height, width, 3), np.uint8)
        self.core = np.zeros((height, width, 3), np.uint8)

        self._undo = deque(maxlen=config.UNDO_LIMIT)
        self._glow = None
        self._glow_strength = None
        self._dirty = True

        self._points = []        # points of the stroke in progress
        self._last_erase = None

        # Soft vignette, built once. Keeps attention on the centre of the view.
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        d = np.sqrt(((xx - width / 2) / (width / 2)) ** 2 + ((yy - height / 2) / (height / 2)) ** 2)
        v = np.clip(1.0 - 0.38 * np.clip(d - 0.35, 0, None) ** 1.6, 0, 1)
        self._vignette = cv2.merge([(v * 255).astype(np.uint8)] * 3)

    # ------------------------------------------------------------------
    # Editing
    # ------------------------------------------------------------------
    def begin_stroke(self):
        """Call once when the pen goes down. Saves an undo snapshot."""
        self._undo.append((self.body.copy(), self.core.copy()))
        self._points = []
        self._last_erase = None

    def end_stroke(self):
        self._points = []
        self._last_erase = None

    def add_point(self, pt, color, size):
        """Extend the current stroke to ``pt``.

        Consecutive points are joined with quadratic curves through their
        midpoints, which turns the jagged fingertip path into a smooth line.
        """
        pt = (float(pt[0]), float(pt[1]))
        pts = self._points
        if pts and np.hypot(pt[0] - pts[-1][0], pt[1] - pts[-1][1]) < 1.5:
            return
        pts.append(pt)

        if len(pts) == 1:
            self._segment([pt, pt], color, size)
            return
        if len(pts) == 2:
            mid = _mid(pts[0], pts[1])
            self._segment([pts[0], mid], color, size)
            return

        p0, p1, p2 = pts[-3], pts[-2], pts[-1]
        a, b = _mid(p0, p1), _mid(p1, p2)
        steps = max(2, int(np.hypot(b[0] - a[0], b[1] - a[1]) / 3))
        t = np.linspace(0.0, 1.0, steps + 1)[:, None]
        curve = (1 - t) ** 2 * np.array(a) + 2 * (1 - t) * t * np.array(p1) + t ** 2 * np.array(b)
        self._segment(curve, color, size)

        # Only the last few points are ever needed.
        if len(pts) > 8:
            del pts[:-3]

    def erase_to(self, pt, radius):
        """Rub out everything under a round eraser moving to ``pt``."""
        p = (int(round(pt[0])), int(round(pt[1])))
        r = int(radius)
        for layer in (self.body, self.core):
            if self._last_erase is not None:
                cv2.line(layer, self._last_erase, p, (0, 0, 0), r * 2, cv2.LINE_AA)
            cv2.circle(layer, p, r, (0, 0, 0), -1, cv2.LINE_AA)
        self._last_erase = p
        self._dirty = True

    def clear(self):
        if self.body.any():
            self._undo.append((self.body.copy(), self.core.copy()))
        self.body[:] = 0
        self.core[:] = 0
        self._dirty = True

    def undo(self):
        if not self._undo:
            return False
        self.body, self.core = self._undo.pop()
        self._dirty = True
        return True

    def _segment(self, pts, color, size):
        poly = np.round(np.asarray(pts) * 16).astype(np.int32).reshape(-1, 1, 2)
        size = int(size)
        core_size = max(1, int(round(size * 0.38)))
        core_color = mix(color, (255, 255, 255), 0.7)
        # shift=4 gives sub-pixel precision (coordinates are in 1/16 px).
        cv2.polylines(self.body, [poly], False, color, size, cv2.LINE_AA, shift=4)
        cv2.polylines(self.core, [poly], False, core_color, core_size, cv2.LINE_AA, shift=4)
        self._dirty = True

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _glow_layer(self, strength):
        if not self._dirty and self._glow is not None and self._glow_strength == strength:
            return self._glow

        w, h = self.w, self.h
        # Two blur radii: a tight halo plus a wide soft bloom.
        half = cv2.resize(self.body, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        near = cv2.GaussianBlur(half, (0, 0), 5)
        quarter = cv2.resize(half, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        far = cv2.GaussianBlur(quarter, (0, 0), 9)

        near = cv2.resize(near, (w, h), interpolation=cv2.INTER_LINEAR)
        far = cv2.resize(far, (w, h), interpolation=cv2.INTER_LINEAR)
        glow = cv2.addWeighted(near, 1.1 * strength, far, 1.6 * strength, 0)

        self._glow = glow
        self._glow_strength = strength
        self._dirty = False
        return glow

    def compose(self, frame, dim=0.55, glow=1.0):
        """Blend the drawing over a camera frame of the same size."""
        # Darker, slightly desaturated camera so the neon lines stand out.
        gray = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        bg = cv2.addWeighted(frame, dim * 0.8, gray, dim * 0.2, 0)
        bg = cv2.multiply(bg, self._vignette, scale=1 / 255.0)

        if glow > 0:
            bg = cv2.add(bg, self._glow_layer(glow))
        out = np.maximum(bg, self.body)
        np.maximum(out, self.core, out=out)
        return out

    @property
    def can_undo(self):
        return bool(self._undo)


def _mid(a, b):
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)
