"""Main loop: camera -> hand tracking -> gestures -> canvas -> screen."""

import time
from datetime import datetime

import cv2
import numpy as np

from . import config
from .camera import Camera
from .canvas import NeonCanvas
from .gestures import OneEuroFilter, TouchDetector
from .hud import Hud
from .settings_window import SettingsWindow
from .tracker import INDEX_TIP, PINKY_TIP, THUMB_TIP, HandTracker

CAPTURE_DIR = config.PROJECT_ROOT / "captures"


def fit_to_view(frame, width=config.VIEW_W, height=config.VIEW_H):
    """Centre-crop to the view's aspect ratio, then resize. Never stretches."""
    h, w = frame.shape[:2]
    target = width / height
    if w / h > target:
        new_w = int(h * target)
        x0 = (w - new_w) // 2
        frame = frame[:, x0:x0 + new_w]
    elif w / h < target:
        new_h = int(w / target)
        y0 = (h - new_h) // 2
        frame = frame[y0:y0 + new_h]
    interp = cv2.INTER_AREA if frame.shape[1] > width else cv2.INTER_LINEAR
    return cv2.resize(frame, (width, height), interpolation=interp)


def adjust_image(frame, s):
    """Software brightness / contrast / saturation (works on every webcam)."""
    if s.contrast != 1.0 or s.brightness != 0:
        # Contrast pivots around mid grey so it does not also brighten the image.
        beta = s.brightness + 128 * (1 - s.contrast)
        frame = cv2.convertScaleAbs(frame, alpha=s.contrast, beta=beta)
    if abs(s.saturation - 1.0) > 1e-3:
        gray = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        frame = cv2.addWeighted(frame, s.saturation, gray, 1 - s.saturation, 0)
    return frame


class AirDraw:
    def __init__(self, camera_index=0):
        self.s = config.Settings(camera_index=camera_index)
        self.camera = Camera(self.s)
        self.tracker = HandTracker()
        self.canvas = NeonCanvas(config.VIEW_W, config.VIEW_H)
        self.hud = Hud(config.VIEW_W, config.VIEW_H)
        self.settings_ui = SettingsWindow(self.s, self.camera)

        # One detector per gesture so they never interfere with each other.
        self.pen_toggle = TouchDetector()      # right pinky + index
        self.next_color = TouchDetector()      # left thumb + index
        self.eraser_toggle = TouchDetector()   # left pinky + index

        self.pen_filter = OneEuroFilter()
        self.mode = "idle"                     # idle | draw | erase
        self.color_index = 0
        self.stroke_open = False
        self.right_last_seen = 0.0

        self.fps = 0.0
        self._last_frame_time = None
        self._last_output = None

    # ------------------------------------------------------------------
    @property
    def color(self):
        return config.PALETTE[self.color_index][1]

    @property
    def color_name(self):
        return config.PALETTE[self.color_index][0]

    def _end_stroke(self):
        if self.stroke_open:
            self.canvas.end_stroke()
            self.stroke_open = False

    def _set_mode(self, mode, where=None):
        self._end_stroke()
        self.mode = mode
        labels = {"draw": ("Pen on", self.color), "idle": ("Pen off", (160, 152, 146)),
                  "erase": ("Eraser", (255, 255, 255))}
        text, color = labels[mode]
        self.hud.toast(text, color)
        if where is not None:
            self.hud.pulse(where, color)

    # ------------------------------------------------------------------
    # Gestures
    # ------------------------------------------------------------------
    def _handle_gestures(self, hands, now):
        left, right = hands.get("Left"), hands.get("Right")

        if left is not None:
            if self.next_color.update(left.touch_ratio(THUMB_TIP, INDEX_TIP), now):
                self.color_index = (self.color_index + 1) % len(config.PALETTE)
                where = (left.tip(THUMB_TIP) + left.tip(INDEX_TIP)) / 2
                self.hud.pulse(where, self.color)
                self.hud.toast(self.color_name, self.color)

            if self.eraser_toggle.update(left.touch_ratio(PINKY_TIP, INDEX_TIP), now):
                where = (left.tip(PINKY_TIP) + left.tip(INDEX_TIP)) / 2
                self._set_mode("draw" if self.mode == "erase" else "erase", where)
        else:
            self.next_color.reset()
            self.eraser_toggle.reset()

        if right is None:
            # Short dropouts happen; only end the stroke after a real gap.
            if now - self.right_last_seen > config.STROKE_GAP_S:
                self._end_stroke()
                self.pen_filter.reset()
                self.pen_toggle.reset()
            return None, False

        self.right_last_seen = now
        if self.pen_toggle.update(right.touch_ratio(PINKY_TIP, INDEX_TIP), now):
            where = (right.tip(PINKY_TIP) + right.tip(INDEX_TIP)) / 2
            self._set_mode("idle" if self.mode == "draw" else "draw", where)

        pen = self.pen_filter(right.tip(INDEX_TIP), now)

        # While the pinky is reaching for the index finger the pen tip moves
        # on its own, so drawing pauses to avoid a stray line.
        suspended = self.pen_toggle.engaged

        if self.mode in ("draw", "erase") and not suspended:
            if not self.stroke_open:
                self.canvas.begin_stroke()
                self.stroke_open = True
            if self.mode == "draw":
                self.canvas.add_point(pen, self.color, self.s.brush_size)
            else:
                self.canvas.erase_to(pen, self.s.eraser_size)
        else:
            self._end_stroke()

        return pen, suspended

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------
    def _handle_key(self, key):
        if key in (ord("q"), ord("Q"), 27):
            return False
        if key in (ord("x"), ord("X")):
            self.settings_ui.toggle()
        elif key in (ord("c"), ord("C")):
            self._end_stroke()
            self.canvas.clear()
            self.hud.toast("Canvas cleared")
        elif key in (ord("z"), ord("Z")):
            self._end_stroke()
            self.hud.toast("Undo" if self.canvas.undo() else "Nothing to undo")
        elif key in (ord("s"), ord("S")) and self._last_output is not None:
            CAPTURE_DIR.mkdir(exist_ok=True)
            path = CAPTURE_DIR / f"airdraw_{datetime.now():%Y%m%d_%H%M%S}.png"
            cv2.imwrite(str(path), self._last_output)
            self.hud.toast(f"Saved {path.name}")
        elif key in (ord("h"), ord("H")):
            self.hud.toggle_help()
        return True

    # ------------------------------------------------------------------
    # Frame
    # ------------------------------------------------------------------
    def render(self, view, hands, pen, suspended):
        s = self.s
        out = self.canvas.compose(view, dim=s.background_dim, glow=s.glow)

        right = hands.get("Right")
        if right is not None:
            self.hud.draw_skeleton(out, right)
        self.hud.draw_pulses(out)
        if pen is not None:
            self.hud.draw_cursor(out, pen, self.mode, self.color, s.brush_size,
                                 s.eraser_size, suspended)

        self.hud.draw_mode(out, self.mode, self.color, self.color_name,
                           "Left" in hands, right is not None)
        self.hud.draw_help(out)
        self.hud.draw_palette(out, self.color_index)
        self.hud.draw_fps(out, self.fps)
        self.hud.draw_toast(out)
        return out

    def _tick_fps(self, now):
        if self._last_frame_time is not None:
            dt = now - self._last_frame_time
            if dt > 0:
                inst = 1.0 / dt
                self.fps = inst if self.fps == 0 else self.fps * 0.9 + inst * 0.1
        self._last_frame_time = now

    def _waiting_screen(self, title, detail):
        img = np.full((config.VIEW_H, config.VIEW_W, 3), (18, 15, 13), np.uint8)
        self.hud.draw_message(img, title, detail)
        return img

    # ------------------------------------------------------------------
    def run(self):
        # GUI_NORMAL hides the Qt toolbar on Linux builds; Windows ignores it.
        cv2.namedWindow(config.APP_TITLE, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
        cv2.imshow(config.APP_TITLE, self._waiting_screen("Starting camera", "One moment..."))
        cv2.waitKey(1)

        if not self.camera.start():
            cv2.imshow(config.APP_TITLE, self._waiting_screen(
                "Camera not available", f"{self.camera.error}. Press any key to exit."))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return

        last_id = 0
        running = True
        try:
            while running:
                loop_start = time.perf_counter()
                frame, frame_id = self.camera.read(last_id, timeout=0.25)

                if frame is None:
                    cv2.imshow(config.APP_TITLE, self._waiting_screen(
                        "Waiting for camera", "Check that no other app is using it."))
                elif frame_id != last_id:
                    last_id = frame_id
                    now = time.perf_counter()

                    view = fit_to_view(frame)
                    if self.s.mirror:
                        view = cv2.flip(view, 1)
                    view = adjust_image(view, self.s)

                    hands = self.tracker.detect(view, mirrored=self.s.mirror)
                    pen, suspended = self._handle_gestures(hands, now)

                    self._tick_fps(now)
                    out = self.render(view, hands, pen, suspended)
                    self._last_output = out
                    cv2.imshow(config.APP_TITLE, out)

                self.settings_ui.update()

                key = cv2.waitKey(1) & 0xFF
                if key != 0xFF:
                    running = self._handle_key(key)

                # Closing the main window with its title bar button quits.
                if cv2.getWindowProperty(config.APP_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                    running = False

                # Respect the FPS cap from the settings window.
                budget = 1.0 / max(self.s.target_fps, config.FPS_MIN)
                spare = budget - (time.perf_counter() - loop_start)
                if spare > 0.001:
                    time.sleep(spare)
        finally:
            self.camera.stop()
            self.tracker.close()
            cv2.destroyAllWindows()
