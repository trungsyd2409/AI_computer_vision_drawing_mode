# Air Draw

Draw glowing neon lines in the air with your fingertip. A webcam tracks both of
your hands: the right index finger is the pen, and simple finger touches switch
the pen on and off, change the colour and turn on the eraser. Everything you draw
stays on screen until you erase it.

---

## Quick start

### Requirements

- Python 3.10 – 3.13 (64-bit)
- A webcam
- Windows 10/11, macOS or Linux

### Install

```bash
# 1. Go to the project folder
cd AI_computer_vision_drawing_mode

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. Install the dependencies
pip install -r requirements.txt
```

### Run

```bash
python main.py
```

On the first run the app downloads the MediaPipe hand model (about 7.5 MB) into
`models/hand_landmarker.task`. If your network blocks the download, get the file
manually from the link below and put it in the `models` folder:

```
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

If you have more than one camera, choose one with `--camera`:

```bash
python main.py --camera 1
```

---

## Controls

### Gestures

| Hand  | Touch these fingertips together | What happens                         |
|-------|---------------------------------|--------------------------------------|
| Right | Pinky + index                   | Pen on / off                         |
| Left  | Thumb + index                   | Next colour (8 neon colours)         |
| Left  | Pinky + index                   | Eraser on / back to drawing          |

- The **right index fingertip** is always the pen (or the eraser).
- The pen starts **off** and the default colour is **white**.
- Each touch counts once. Separate the fingers before touching again.

### Keyboard

| Key         | Action                           |
|-------------|----------------------------------|
| `X`         | Open / close the settings window |
| `Z`         | Undo the last stroke             |
| `C`         | Clear the canvas                 |
| `S`         | Save a screenshot to `captures/` |
| `H`         | Show / hide the gesture guide    |
| `Q` / `Esc` | Quit                             |

### Settings window (`X`)

| Section | Options                                                                  |
|---------|--------------------------------------------------------------------------|
| Camera  | Capture resolution, target FPS (15–60), auto/manual exposure, mirror view |
| Image   | Brightness, contrast, saturation                                         |
| Canvas  | Brush size, eraser size, glow strength, background brightness            |

It also shows the camera's real frame rate and the resolution it actually
delivers. On Windows, **Driver panel** opens the webcam's own settings dialog.

---

## Tech stack

| Tool                                    | What it is used for                                      |
|-----------------------------------------|----------------------------------------------------------|
| [Python](https://www.python.org/)       | Language for the whole app                               |
| [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) | Finds 21 key points on each hand and says whether it is a left or right hand |
| [OpenCV](https://opencv.org/)           | Webcam capture, image processing, windows and keyboard/mouse input |
| [NumPy](https://numpy.org/)             | Fast maths on images and point coordinates               |
| [Pillow](https://python-pillow.org/)    | Renders clean, modern text for the on-screen interface   |

---

## How it works (for everyone)

Think of the app as a short production line that runs about 30 to 60 times every
second:

1. **Look.** The webcam takes a picture. The app flips it like a mirror so moving
   your hand to the right moves it to the right on screen.
2. **Find the hands.** A small AI model, trained by Google on many hand photos,
   marks 21 points on each hand: the wrist, every knuckle and every fingertip. It
   also tells which hand is the left one and which is the right one.
3. **Read the gestures.** The app measures the distance between particular
   fingertips. When two fingertips come very close, it counts as a "touch", like
   pressing a button. Because the distance is compared to the size of your palm,
   this works whether your hand is near the camera or far from it.
4. **Draw.** While the pen is on, the app follows your right index fingertip and
   paints a line along its path. The path is smoothed first, so natural hand
   shake does not show up as a wobbly line.
5. **Make it glow.** The lines are painted on a separate transparent layer. A
   blurred copy of that layer is added on top of a slightly darkened camera
   image, which creates the neon light effect.
6. **Show it.** The finished picture, with the status, colour palette and frame
   rate on top, appears in the window.

Apart from downloading the hand model once, nothing is sent over the internet.
All processing happens on your computer.

---

## How it works (technical)

### Pipeline

```
Camera thread ──► latest frame
                      │
Main loop:            ▼
  fit_to_view (centre crop to 4:3, resize to 800x600) ─► mirror ─► image adjust
      │
      ▼
  HandTracker.detect ─► {"Left": Hand, "Right": Hand}   (21 landmarks each, view pixels)
      │
      ▼
  Gesture layer: TouchDetector x3 + One Euro filter on the right index tip
      │
      ▼
  NeonCanvas (persistent body/core layers, undo stack) ─► compose with camera
      │
      ▼
  HUD (FPS, mode, palette, guide, cursor, skeleton) ─► cv2.imshow
```

### Modules

| File                          | Responsibility |
|-------------------------------|----------------|
| `main.py`                     | Command-line entry point |
| `airdraw/config.py`           | All tunable constants and the live `Settings` dataclass |
| `airdraw/camera.py`           | Threaded capture, MJPG/FPS/exposure setup, re-open on setting changes |
| `airdraw/tracker.py`          | MediaPipe Hand Landmarker wrapper, model download, handedness correction |
| `airdraw/gestures.py`         | `TouchDetector` (hysteresis + debounce) and `OneEuroFilter` |
| `airdraw/canvas.py`           | Stroke rasterisation, eraser, undo, neon glow compositing |
| `airdraw/hud.py`              | On-screen overlay |
| `airdraw/settings_window.py`  | Custom-drawn settings window with its own sliders and switches |
| `airdraw/ui.py`               | Text rendering with system fonts, rounded/frosted panels, alpha blending |
| `airdraw/app.py`              | Main loop, mode state machine and keyboard handling |

### Hand tracking and handedness

`HandLandmarker` runs in `VIDEO` mode, which reuses the previous frame's hand
position instead of running the palm detector on every frame. Frames are
downscaled to 640 px wide for inference. The returned landmarks are normalised,
so they are mapped straight back to the 800x600 view.

MediaPipe labels hands assuming a mirrored selfie image. When mirroring is turned
off in the settings, the labels are swapped. If both hands receive the same label
(it happens during fast motion), the tie is broken by horizontal position.

### Gesture detection

For each gesture the app computes

```
ratio = distance(fingertip A, fingertip B) / distance(wrist, middle-finger knuckle)
```

and feeds it to a `TouchDetector`:

- **Hysteresis:** a touch starts below `0.30` and ends above `0.45`, so the state
  does not flicker around one threshold.
- **Stability:** the touch must last 2 consecutive frames.
- **Cooldown:** at least 0.45 s between two events of the same gesture.
- **Locked start:** when a hand appears with fingers already touching, nothing
  fires until they separate once.

Mode logic is a small state machine with three states: `idle` (pen off), `draw`
and `erase`. The right-hand gesture toggles `draw` ↔ `idle`; the left pinky
gesture toggles `erase` ↔ `draw`. While the right pinky is moving towards the
index finger (ratio below the release threshold), drawing is paused so the
gesture itself does not leave a mark.

### Smoothing and stroke rendering

The pen position goes through a **One Euro filter** (Casiez et al., 2012). Its
cut-off frequency rises with speed: slow movements get strong smoothing (no
jitter on details), fast movements get little (no lag).

Strokes are drawn as **quadratic Bézier segments between midpoints** of
consecutive samples, which removes the corners of a plain polyline. Lines are
drawn with sub-pixel precision (`shift=4`) and anti-aliasing.

### Neon compositing

Two persistent `uint8` layers hold the drawing:

- `body` – the coloured line
- `core` – a thinner line mixed 70% towards white, drawn on top

The glow is built from `body` at half and quarter resolution with two Gaussian
blurs (a tight halo and a wide bloom), upscaled and added. It is cached and only
recomputed when the drawing changes. The final frame is:

```
background = vignette * dim * (80% colour + 20% greyscale camera)
output     = max(max(background + glow, body), core)
```

The eraser draws black circles (and connecting thick lines, so fast movement
leaves no gaps) into both layers. Undo keeps up to 25 layer snapshots, one per
stroke.

### Performance

- Capture runs on its own thread with a buffer size of 1, so the main loop always
  gets the newest frame and never waits on the camera driver.
- The camera is asked for MJPG, which most USB webcams need to reach 60 FPS.
- Text is rendered once per distinct string and cached; panels blur a 1/4-scale
  copy of the region behind them.
- In testing, compositing and the overlay take about 10 ms per frame at 800x600.
  Hand tracking usually takes 10–25 ms on a modern laptop CPU.

The FPS counter (top right) is green at 30 FPS or more, amber from 15 to 30 and
red below 15.

### Tuning

Gesture thresholds, smoothing, palette colours and default settings are in
`airdraw/config.py`. The most useful values:

| Constant              | Effect |
|-----------------------|--------|
| `TOUCH_PRESS_RATIO`   | Higher = touches trigger more easily |
| `TOUCH_RELEASE_RATIO` | Higher = fingers must open wider before the next touch |
| `PEN_MIN_CUTOFF`      | Lower = smoother but more lag when moving slowly |
| `PEN_BETA`            | Higher = less lag on fast strokes |
| `DETECT_WIDTH`        | Lower = faster tracking, slightly less precise |

---

## Project structure

```
AI_computer_vision_drawing_mode/
├── main.py
├── requirements.txt
├── README.md
├── airdraw/
│   ├── __init__.py
│   ├── app.py
│   ├── camera.py
│   ├── canvas.py
│   ├── config.py
│   ├── gestures.py
│   ├── hud.py
│   ├── settings_window.py
│   ├── tracker.py
│   └── ui.py
├── models/        # hand model, downloaded on first run
└── captures/      # screenshots saved with S
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Camera not available" | Close other apps using the webcam (Zoom, Teams, browser tabs) or try `--camera 1`. |
| FPS stays around 15 | Webcams slow down in dim light. Add light, or turn off **Auto exposure** and lower **Exposure** in the settings. |
| Left and right hands are swapped | Make sure **Mirror view** is on. |
| Gestures trigger by accident | Lower `TOUCH_PRESS_RATIO` in `airdraw/config.py`. |
| Gestures are hard to trigger | Raise `TOUCH_PRESS_RATIO`, and keep your hand fully in view with good lighting. |
| `pip install mediapipe` fails | Check that you are on 64-bit Python 3.10–3.13 and that `pip` is up to date (`python -m pip install -U pip`). |
