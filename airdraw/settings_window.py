"""A separate window for camera and canvas settings (toggle with the X key).

OpenCV trackbars look like they belong in 2005, so this window draws its own
controls (sliders, switches, segmented buttons) and handles the mouse itself.
Nothing here blocks: the main loop calls ``update()`` once per frame.
"""

import time

import cv2
import numpy as np

from . import config
from .ui import alpha_circle, draw_text, fill_rounded, stroke_rounded

BG = (20, 17, 15)
CARD = (32, 28, 25)
LINE = (58, 52, 48)
TEXT = (245, 242, 240)
MUTED = (160, 152, 146)
ACCENT = config.ACCENT

WIDTH, HEIGHT = 660, 600


# --------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------
class Control:
    enabled = staticmethod(lambda: True)

    def __init__(self, rect):
        self.rect = rect

    def hit(self, x, y):
        rx, ry, rw, rh = self.rect
        return rx <= x < rx + rw and ry <= y < ry + rh

    def draw(self, img):
        raise NotImplementedError

    def press(self, x, y):
        pass

    def drag(self, x, y):
        pass

    def release(self, x, y):
        pass


class Slider(Control):
    def __init__(self, rect, label, lo, hi, step, get, set_, fmt="{:.0f}",
                 on_release=None, enabled=None):
        super().__init__(rect)
        self.label, self.lo, self.hi, self.step = label, lo, hi, step
        self.get, self.set, self.fmt = get, set_, fmt
        self.on_release = on_release
        if enabled:
            self.enabled = enabled
        self.active = False

    def _track(self):
        x, y, w, h = self.rect
        return x, y + h - 12, w

    def _value_at(self, x):
        tx, _, tw = self._track()
        t = min(max((x - tx) / tw, 0.0), 1.0)
        v = self.lo + t * (self.hi - self.lo)
        return round(v / self.step) * self.step

    def press(self, x, y):
        self.active = True
        self.set(self._value_at(x))

    def drag(self, x, y):
        if self.active:
            self.set(self._value_at(x))

    def release(self, x, y):
        if self.active and self.on_release:
            self.on_release()
        self.active = False

    def draw(self, img):
        on = self.enabled()
        x, y, w, h = self.rect
        value = self.get()
        draw_text(img, self.label, (x, y), 13, TEXT if on else MUTED)
        draw_text(img, self.fmt.format(value), (x + w, y), 13, ACCENT if on else MUTED,
                  "semibold", anchor="rt")

        tx, ty, tw = self._track()
        t = (value - self.lo) / (self.hi - self.lo)
        fill_rounded(img, (tx, ty - 2, tw, 4), 2, LINE)
        if on:
            fill_rounded(img, (tx, ty - 2, max(int(tw * t), 4), 4), 2, ACCENT)
        kx = tx + tw * t
        alpha_circle(img, (kx, ty), 10 if self.active else 8, ACCENT, 0.25 if on else 0.0)
        alpha_circle(img, (kx, ty), 7, TEXT if on else MUTED, 1.0)


class Switch(Control):
    def __init__(self, rect, label, get, set_):
        super().__init__(rect)
        self.label, self.get, self.set = label, get, set_

    def press(self, x, y):
        self.set(not self.get())

    def draw(self, img):
        x, y, w, h = self.rect
        on = self.get()
        draw_text(img, self.label, (x, y + h // 2), 13, TEXT, anchor="lm")
        sw, sh = 38, 22
        sx, sy = x + w - sw, y + (h - sh) // 2
        fill_rounded(img, (sx, sy, sw, sh), sh // 2, ACCENT if on else LINE)
        kx = sx + sw - sh // 2 if on else sx + sh // 2
        alpha_circle(img, (kx, sy + sh / 2), 8, TEXT, 1.0)


class Segmented(Control):
    def __init__(self, rect, label, options, get, set_):
        super().__init__(rect)
        self.label, self.options, self.get, self.set = label, options, get, set_

    def _cells(self):
        x, y, w, h = self.rect
        n = len(self.options)
        cw = w / n
        return [(int(x + i * cw), y + 22, int(cw), h - 22) for i in range(n)]

    def press(self, x, y):
        for (cx, cy, cw, ch), (_, value) in zip(self._cells(), self.options):
            if cx <= x < cx + cw and cy <= y < cy + ch:
                if value != self.get():
                    self.set(value)
                return

    def draw(self, img):
        x, y, w, h = self.rect
        draw_text(img, self.label, (x, y), 13, TEXT)
        fill_rounded(img, (x, y + 22, w, h - 22), 9, LINE, 0.6)
        current = self.get()
        for (cx, cy, cw, ch), (text, value) in zip(self._cells(), self.options):
            if value == current:
                fill_rounded(img, (cx + 3, cy + 3, cw - 6, ch - 6), 7, ACCENT)
                color = BG
            else:
                color = MUTED
            draw_text(img, text, (cx + cw // 2, cy + ch // 2), 12, color, "semibold", anchor="cm")


class Button(Control):
    def __init__(self, rect, label, on_click, primary=False):
        super().__init__(rect)
        self.label, self.on_click, self.primary = label, on_click, primary
        self.down = False

    def press(self, x, y):
        self.down = True

    def release(self, x, y):
        if self.down and self.hit(x, y):
            self.on_click()
        self.down = False

    def draw(self, img):
        x, y, w, h = self.rect
        if self.primary:
            fill_rounded(img, self.rect, 10, ACCENT, 0.8 if self.down else 1.0)
            color = BG
        else:
            fill_rounded(img, self.rect, 10, LINE, 0.9 if self.down else 0.55)
            color = TEXT
        draw_text(img, self.label, (x + w // 2, y + h // 2), 13, color, "semibold", anchor="cm")


# --------------------------------------------------------------------------
# Window
# --------------------------------------------------------------------------
class SettingsWindow:
    def __init__(self, settings, camera):
        self.s = settings
        self.camera = camera
        self.visible = False
        self._controls = []
        self._grabbed = None
        self._dirty = True
        self._last_info = 0.0
        self._img = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        self._build()

    # Layout -----------------------------------------------------------------
    def _build(self):
        s, cam = self.s, self.camera
        pad, gutter = 28, 36
        col_w = (WIDTH - pad * 2 - gutter) // 2
        lx, rx = pad, pad + col_w + gutter
        c = []

        def setter(name, after=None, cast=None):
            def _set(v):
                setattr(s, name, cast(v) if cast else v)
                if after:
                    after()
            return _set

        # Left column: camera
        y = 104
        c.append(Segmented((lx, y, col_w, 58), "Capture resolution",
                           [(f"{w}×{h}", (w, h)) for w, h in config.RESOLUTIONS],
                           lambda: s.resolution, setter("resolution", cam.reopen)))
        y += 76
        c.append(Slider((lx, y, col_w, 40), "Target FPS", config.FPS_MIN, config.FPS_MAX, 5,
                        lambda: s.target_fps, setter("target_fps", cast=int),
                        on_release=cam.reopen))
        y += 56
        c.append(Switch((lx, y, col_w, 30), "Auto exposure",
                        lambda: s.auto_exposure, setter("auto_exposure", cam.apply_exposure)))
        y += 44
        c.append(Slider((lx, y, col_w, 40), "Exposure", -11, -2, 1,
                        lambda: s.exposure, setter("exposure", cast=int),
                        on_release=cam.apply_exposure, enabled=lambda: not s.auto_exposure))
        y += 56
        c.append(Switch((lx, y, col_w, 30), "Mirror view",
                        lambda: s.mirror, setter("mirror")))

        # Right column: image + canvas
        y = 104
        c.append(Slider((rx, y, col_w, 40), "Brightness", -60, 60, 2,
                        lambda: s.brightness, setter("brightness", cast=int), fmt="{:+.0f}"))
        y += 52
        c.append(Slider((rx, y, col_w, 40), "Contrast", 0.6, 1.6, 0.05,
                        lambda: s.contrast, setter("contrast"), fmt="{:.2f}"))
        y += 52
        c.append(Slider((rx, y, col_w, 40), "Saturation", 0.0, 1.6, 0.05,
                        lambda: s.saturation, setter("saturation"), fmt="{:.2f}"))

        y += 84
        c.append(Slider((rx, y, col_w, 40), "Brush size", 2, 20, 1,
                        lambda: s.brush_size, setter("brush_size", cast=int)))
        y += 52
        c.append(Slider((rx, y, col_w, 40), "Eraser size", 12, 80, 2,
                        lambda: s.eraser_size, setter("eraser_size", cast=int)))
        y += 52
        c.append(Slider((rx, y, col_w, 40), "Glow", 0.0, 2.0, 0.1,
                        lambda: s.glow, setter("glow"), fmt="{:.1f}"))
        y += 52
        c.append(Slider((rx, y, col_w, 40), "Background", 0.2, 1.0, 0.05,
                        lambda: s.background_dim, setter("background_dim"),
                        fmt="{:.0%}"))

        # Footer buttons
        by = HEIGHT - 56
        c.append(Button((lx, by, 128, 36), "Driver panel", cam.open_driver_panel))
        c.append(Button((lx + 138, by, 90, 36), "Reset", self._reset))
        c.append(Button((WIDTH - pad - 96, by, 96, 36), "Done", self.hide, primary=True))

        self._controls = c
        self._cols = (lx, rx, col_w)

    def _reset(self):
        defaults = config.Settings()
        needs_reopen = (defaults.resolution != self.s.resolution
                        or defaults.target_fps != self.s.target_fps)
        for key, value in vars(defaults).items():
            if key != "camera_index":
                setattr(self.s, key, value)
        if needs_reopen:
            self.camera.reopen()
        self.camera.apply_exposure()

    # Visibility -------------------------------------------------------------
    def toggle(self):
        self.hide() if self.visible else self.show()

    def show(self):
        if self.visible:
            return
        cv2.namedWindow(config.SETTINGS_TITLE, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(config.SETTINGS_TITLE, self._on_mouse)
        self.visible = True
        self._dirty = True

    def hide(self):
        if not self.visible:
            return
        self.visible = False
        self._grabbed = None
        try:
            cv2.destroyWindow(config.SETTINGS_TITLE)
        except cv2.error:
            pass

    # Events -----------------------------------------------------------------
    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for ctl in self._controls:
                if ctl.hit(x, y) and ctl.enabled():
                    self._grabbed = ctl
                    ctl.press(x, y)
                    break
        elif event == cv2.EVENT_MOUSEMOVE and self._grabbed is not None:
            self._grabbed.drag(x, y)
        elif event == cv2.EVENT_LBUTTONUP and self._grabbed is not None:
            self._grabbed.release(x, y)
            self._grabbed = None
        else:
            return
        self._dirty = True

    # Rendering --------------------------------------------------------------
    def update(self):
        if not self.visible:
            return
        # The user closed the window with its title bar button.
        try:
            if cv2.getWindowProperty(config.SETTINGS_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                self.visible = False
                self._grabbed = None
                return
        except cv2.error:
            self.visible = False
            return

        now = time.perf_counter()
        if self._dirty or now - self._last_info > 0.5:
            self._render()
            self._last_info = now
            self._dirty = False
            cv2.imshow(config.SETTINGS_TITLE, self._img)

    def render_image(self):
        """Render and return the panel (also used for screenshots/tests)."""
        self._render()
        return self._img

    def _render(self):
        img = self._img
        img[:] = BG
        lx, rx, col_w = self._cols

        draw_text(img, "Settings", (lx, 22), 22, TEXT, "semibold")
        draw_text(img, "Changes apply instantly. Press X to close.", (lx, 54), 13, MUTED)

        # Section cards
        for x, y, h in ((lx, 78, 318), (rx, 78, 164), (rx, 262, 250)):
            fill_rounded(img, (x - 12, y, col_w + 24, h), 14, CARD)
        draw_text(img, "CAMERA", (lx, 86), 11, ACCENT, "semibold")
        draw_text(img, "IMAGE", (rx, 86), 11, ACCENT, "semibold")
        draw_text(img, "CANVAS", (rx, 270), 11, ACCENT, "semibold")

        for ctl in self._controls:
            ctl.draw(img)

        # Live camera info
        cam = self.camera
        w, h = cam.actual_size
        info = [
            ("Device", f"#{self.s.camera_index}  ({cam.backend})"),
            ("Delivered", f"{w} x {h}"),
            ("Camera rate", f"{cam.measured_fps:.1f} fps"),
        ]
        iy = 78 + 318 + 14
        for i, (k, v) in enumerate(info):
            draw_text(img, k, (lx, iy + i * 20), 12, MUTED)
            draw_text(img, v, (lx + col_w, iy + i * 20), 12, TEXT, anchor="rt")

        if cam.measured_fps and cam.measured_fps < config.FPS_MIN:
            # Webcams stretch exposure time in dim rooms, which lowers the rate.
            draw_text(img, "Camera is slow. Add light or", (lx, iy + 70), 12, (90, 170, 250))
            draw_text(img, "turn off auto exposure.", (lx, iy + 88), 12, (90, 170, 250))
        stroke_rounded(img, (0, HEIGHT - 76, WIDTH, 1), 0, (255, 255, 255), 0.06)
