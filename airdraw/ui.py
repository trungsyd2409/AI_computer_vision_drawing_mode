"""Small drawing toolkit shared by the HUD and the settings window.

OpenCV's built-in Hershey fonts look dated, so text is rendered once with
Pillow using a proper system font, cached as an alpha mask, and blended onto
frames with numpy. Rounded, translucent "glass" panels are done the same way.
"""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------
# Fonts
# --------------------------------------------------------------------------
_FONT_CANDIDATES = {
    "regular": [
        "C:/Windows/Fonts/segoeui.ttf",
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/inter/Inter-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ],
    "semibold": [
        "C:/Windows/Fonts/seguisb.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/inter/Inter-SemiBold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],
}


@lru_cache(maxsize=32)
def _font(size, weight):
    for path in _FONT_CANDIDATES.get(weight, _FONT_CANDIDATES["regular"]):
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default(size=size)


@lru_cache(maxsize=512)
def text_mask(text, size=14, weight="regular"):
    """Render ``text`` to a uint8 alpha mask. Cached, so repeated labels are free."""
    font = _font(size, weight)
    left, top, right, bottom = font.getbbox(text)
    ascent, descent = font.getmetrics()
    w = max(right - left, 1) + 2
    h = ascent + descent + 2
    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).text((1 - left, 1), text, font=font, fill=255)
    return np.asarray(img)


def text_size(text, size=14, weight="regular"):
    m = text_mask(text, size, weight)
    return m.shape[1], m.shape[0]


def draw_text(img, text, org, size=14, color=(255, 255, 255), weight="regular",
              alpha=1.0, anchor="lt"):
    """Draw text. ``anchor`` is two letters: l/c/r horizontally, t/m/b vertically."""
    if not text or alpha <= 0:
        return
    mask = text_mask(text, size, weight)
    h, w = mask.shape
    x, y = int(org[0]), int(org[1])
    if anchor[0] == "c":
        x -= w // 2
    elif anchor[0] == "r":
        x -= w
    if anchor[1] == "m":
        y -= h // 2
    elif anchor[1] == "b":
        y -= h
    blend_mask(img, mask, (x, y), color, alpha)


def blend_mask(img, mask, xy, color, alpha=1.0):
    """Blend a solid colour into ``img`` through an 8-bit mask, with clipping."""
    x, y = xy
    h, w = mask.shape
    H, W = img.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return
    m = mask[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32) * (alpha / 255.0)
    m = m[..., None]
    roi = img[y0:y1, x0:x1]
    col = np.array(color, dtype=np.float32)
    roi[:] = (roi.astype(np.float32) * (1.0 - m) + col * m).astype(np.uint8)


# --------------------------------------------------------------------------
# Shapes
# --------------------------------------------------------------------------
@lru_cache(maxsize=128)
def rounded_mask(w, h, r):
    r = int(max(0, min(r, w // 2, h // 2)))
    m = np.zeros((h, w), np.uint8)
    if r == 0:
        m[:] = 255
        return m
    cv2.rectangle(m, (r, 0), (w - 1 - r, h - 1), 255, -1)
    cv2.rectangle(m, (0, r), (w - 1, h - 1 - r), 255, -1)
    for cx, cy in ((r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)):
        cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
    return m


@lru_cache(maxsize=128)
def rounded_outline(w, h, r, thickness=1):
    m = rounded_mask(w, h, r)
    inner = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=thickness)
    return cv2.subtract(m, inner)


def fill_rounded(img, rect, r, color, alpha=1.0):
    x, y, w, h = map(int, rect)
    if w <= 0 or h <= 0:
        return
    blend_mask(img, rounded_mask(w, h, r), (x, y), color, alpha)


def stroke_rounded(img, rect, r, color, alpha=1.0, thickness=1):
    x, y, w, h = map(int, rect)
    if w <= 2 or h <= 2:
        return
    blend_mask(img, rounded_outline(w, h, r, thickness), (x, y), color, alpha)


def glass_panel(img, rect, r=14, tint=(22, 18, 16), tint_alpha=0.55,
                border_alpha=0.14, blur=True):
    """Frosted glass card: blur what is behind, darken it, add a hairline edge."""
    x, y, w, h = map(int, rect)
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return
    if blur:
        mask = rounded_mask(w, h, r)[y0 - y:y1 - y, x0 - x:x1 - x]
        roi = img[y0:y1, x0:x1]
        # Blurring a downscaled copy is much cheaper and looks the same.
        small = cv2.resize(roi, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 3)
        blurred = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        m = (mask.astype(np.float32) / 255.0)[..., None]
        roi[:] = (roi * (1 - m) + blurred * m).astype(np.uint8)
    fill_rounded(img, rect, r, tint, tint_alpha)
    stroke_rounded(img, rect, r, (255, 255, 255), border_alpha)


def alpha_circle(img, center, radius, color, alpha=1.0, thickness=-1):
    """Anti-aliased circle with transparency, drawn on a small local patch only."""
    cx, cy = int(round(center[0])), int(round(center[1]))
    radius = int(round(radius))
    if radius <= 0 or alpha <= 0:
        return
    pad = radius + max(thickness, 1) + 2
    size = pad * 2 + 1
    m = np.zeros((size, size), np.uint8)
    cv2.circle(m, (pad, pad), radius, 255, thickness, cv2.LINE_AA)
    blend_mask(img, m, (cx - pad, cy - pad), color, alpha)


def mix(c1, c2, t):
    """Linear blend of two BGR colours."""
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def ease_out(t):
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** 3
