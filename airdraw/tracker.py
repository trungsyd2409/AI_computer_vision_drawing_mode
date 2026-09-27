"""Hand landmark detection with MediaPipe.

MediaPipe returns 21 points per hand. The ones this app cares about:

    0  wrist          4  thumb tip      8  index tip
    9  middle knuckle 20 pinky tip

Points are converted to pixel coordinates of the 800x600 view so the rest of
the app never has to think about normalised values.
"""

import sys
import time
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from . import config

WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP, PINKY_TIP = 0, 4, 8, 9, 20

# Bone list for drawing the skeleton (pairs of landmark indices).
HAND_BONES = (
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # pinky + palm edge
)
FINGER_TIPS = (4, 8, 12, 16, 20)


@dataclass
class Hand:
    side: str               # "Left" or "Right" from the user's point of view
    points: np.ndarray      # (21, 2) float32, pixel coordinates in the view
    score: float

    @property
    def palm_size(self):
        return max(float(np.linalg.norm(self.points[WRIST] - self.points[MIDDLE_MCP])), 1.0)

    def touch_ratio(self, a, b):
        """Distance between two landmarks relative to the palm size."""
        return float(np.linalg.norm(self.points[a] - self.points[b])) / self.palm_size

    def tip(self, idx):
        return self.points[idx]


def ensure_model(path=config.MODEL_PATH, url=config.MODEL_URL):
    """Download the model bundle on first run (about 7.5 MB)."""
    if path.exists() and path.stat().st_size > 0:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    print(f"Downloading hand model to {path} ...")

    def progress(blocks, block_size, total):
        if total > 0:
            done = min(blocks * block_size / total, 1.0)
            sys.stdout.write(f"\r  {done * 100:5.1f}%")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, tmp, progress)
    except Exception as exc:  # network errors vary a lot between platforms
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            "Could not download the hand model. Download it manually from\n"
            f"  {url}\nand save it as\n  {path}"
        ) from exc

    tmp.replace(path)
    print("\n  done.")
    return path


class HandTracker:
    def __init__(self, model_path=None, max_hands=2):
        model_path = model_path or ensure_model()
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def close(self):
        self._landmarker.close()

    def detect(self, frame_bgr, mirrored=True):
        """Return a dict ``{"Left": Hand, "Right": Hand}`` with the hands found."""
        h, w = frame_bgr.shape[:2]

        # Downscale for speed. Landmarks come back normalised (0..1), so the
        # detection size does not affect where they land on the full view.
        if w > config.DETECT_WIDTH:
            dh = int(h * config.DETECT_WIDTH / w)
            small = cv2.resize(frame_bgr, (config.DETECT_WIDTH, dh), interpolation=cv2.INTER_AREA)
        else:
            small = frame_bgr
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode requires strictly increasing timestamps.
        ts = int(time.perf_counter() * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts

        result = self._landmarker.detect_for_video(image, ts)
        if not result.hand_landmarks:
            return {}

        found = []
        for lms, handed in zip(result.hand_landmarks, result.handedness):
            pts = np.array([(p.x * w, p.y * h) for p in lms], dtype=np.float32)
            label = handed[0].category_name
            # MediaPipe labels hands assuming a mirrored selfie image. When the
            # view is not mirrored, the labels have to be swapped.
            if not mirrored:
                label = "Left" if label == "Right" else "Right"
            found.append(Hand(label, pts, float(handed[0].score)))

        # Occasionally both hands get the same label. Fall back to screen
        # position: in a mirrored view the right hand sits on the right side.
        if len(found) == 2 and found[0].side == found[1].side:
            found.sort(key=lambda hand: hand.points[WRIST][0])
            left_first = mirrored
            found[0].side = "Left" if left_first else "Right"
            found[1].side = "Right" if left_first else "Left"

        hands = {}
        for hand in found:
            # Keep the most confident hand if a label is still duplicated.
            if hand.side not in hands or hand.score > hands[hand.side].score:
                hands[hand.side] = hand
        return hands
