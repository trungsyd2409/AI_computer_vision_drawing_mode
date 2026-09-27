"""Gesture helpers: fingertip smoothing and "two fingers touch" detection."""

import math

import numpy as np

from . import config


class OneEuroFilter:
    """Adaptive low-pass filter for a 2D point.

    Slow movements are smoothed heavily (no jitter while writing small
    details), fast movements are smoothed lightly (no lag on quick strokes).
    Reference: Casiez et al., "1 Euro Filter", CHI 2012.
    """

    def __init__(self, min_cutoff=config.PEN_MIN_CUTOFF, beta=config.PEN_BETA, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self):
        self._x = None
        self._dx = np.zeros(2, dtype=np.float32)
        self._t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, point, t):
        point = np.asarray(point, dtype=np.float32)
        if self._x is None:
            self._x, self._t = point, t
            return point

        dt = max(t - self._t, 1e-3)
        self._t = t

        dx = (point - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx

        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self._dx))
        a = self._alpha(cutoff, dt)
        self._x = a * point + (1 - a) * self._x
        return self._x


class TouchDetector:
    """Turns a noisy fingertip distance into clean one-shot "touch" events.

    * Hysteresis: the fingers must come closer than ``press`` to count as
      touching, and move further apart than ``release`` before another touch
      can fire. The gap between the two stops flickering at the boundary.
    * Stability: the touch has to last a couple of frames.
    * Cooldown: a minimum time between two events.
    * Locked start: after the hand (re)appears, fingers that are already
      touching are ignored until they separate once.
    """

    def __init__(
        self,
        press=config.TOUCH_PRESS_RATIO,
        release=config.TOUCH_RELEASE_RATIO,
        stable_frames=config.TOUCH_STABLE_FRAMES,
        cooldown=config.TOUCH_COOLDOWN_S,
    ):
        self.press = press
        self.release = release
        self.stable_frames = stable_frames
        self.cooldown = cooldown
        self._last_fire = -1e9
        self.reset()

    def reset(self):
        self.touching = False
        self._locked = True
        self._engaged = False
        self._count = 0

    @property
    def engaged(self):
        """True while the fingers are closer than the release distance."""
        return self._engaged

    def update(self, ratio, now):
        """Feed the latest ratio. Returns True exactly once per touch."""
        self._engaged = ratio < self.release

        if self._locked:
            if ratio > self.release:
                self._locked = False
            return False

        if self.touching:
            if ratio > self.release:
                self.touching = False
                self._count = 0
            return False

        if ratio < self.press:
            self._count += 1
            if self._count >= self.stable_frames and now - self._last_fire > self.cooldown:
                self.touching = True
                self._last_fire = now
                return True
        else:
            self._count = 0
        return False
