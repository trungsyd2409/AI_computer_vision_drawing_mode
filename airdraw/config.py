"""Central place for tunable values.

Everything that someone might want to tweak without reading the whole code base
lives here: window size, gesture thresholds, colours and default camera values.
"""

from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------
# Window
# --------------------------------------------------------------------------
APP_TITLE = "Air Draw"
SETTINGS_TITLE = "Camera Settings"

# The output is always rendered at 4:3. Camera frames with a different aspect
# ratio (for example 16:9) are centre-cropped first so nothing looks stretched.
VIEW_W, VIEW_H = 800, 600

# --------------------------------------------------------------------------
# Hand tracking model
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# Frames are resized to this width before being sent to the model. Smaller is
# faster; 640 keeps fingertip accuracy good enough for drawing.
DETECT_WIDTH = 640

# --------------------------------------------------------------------------
# Gestures
# --------------------------------------------------------------------------
# A "touch" is measured as fingertip distance divided by palm size, so it works
# the same whether the hand is close to the camera or far away.
TOUCH_PRESS_RATIO = 0.30     # below this the fingers count as touching
TOUCH_RELEASE_RATIO = 0.45   # must open past this before the next touch counts
TOUCH_STABLE_FRAMES = 2      # frames a touch must be held to fire (kills flicker)
TOUCH_COOLDOWN_S = 0.45      # minimum time between two triggers of one gesture

# If the right hand disappears for longer than this, the current stroke ends.
STROKE_GAP_S = 0.15

# One Euro filter for the pen tip (see gestures.OneEuroFilter).
PEN_MIN_CUTOFF = 1.2
PEN_BETA = 0.012

# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
# Neon palette, stored as (name, BGR). The first entry is the default colour.
PALETTE = [
    ("White", (250, 246, 244)),
    ("Cyan", (238, 211, 34)),
    ("Sky", (250, 165, 96)),
    ("Violet", (250, 139, 167)),
    ("Pink", (182, 114, 244)),
    ("Coral", (133, 113, 251)),
    ("Amber", (36, 191, 251)),
    ("Lime", (53, 230, 163)),
]

UNDO_LIMIT = 25

# UI accent colour (BGR) used for highlights in the HUD and settings window.
ACCENT = (238, 211, 34)


@dataclass
class Settings:
    """Values that can be changed live from the settings window."""

    # Camera
    camera_index: int = 0
    resolution: tuple = (1280, 720)
    target_fps: int = 60
    auto_exposure: bool = True
    exposure: int = -6          # DirectShow style: log2(seconds), e.g. -6 = 1/64 s
    brightness: int = 0         # software adjustment, -60 .. 60
    contrast: float = 1.0       # software adjustment, 0.6 .. 1.6
    saturation: float = 1.0     # software adjustment, 0.0 .. 1.6
    mirror: bool = True

    # Canvas look
    brush_size: int = 6
    eraser_size: int = 34
    glow: float = 1.0
    background_dim: float = 0.55


RESOLUTIONS = [(640, 480), (1280, 720), (1920, 1080)]
FPS_MIN, FPS_MAX = 15, 60
