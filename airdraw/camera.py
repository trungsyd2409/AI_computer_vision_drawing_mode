"""Webcam capture on a background thread.

Reading from a webcam blocks until the next frame arrives. Doing that on the
main thread would tie the render loop to the camera's timing, so a small worker
thread keeps grabbing frames and the main loop simply picks up the newest one.

OpenCV's VideoCapture is not thread safe, which is why every property change
requested from the UI is queued and applied by the worker itself.
"""

import platform
import queue
import threading
import time

import cv2

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"


class Camera:
    def __init__(self, settings):
        self.settings = settings

        self._cond = threading.Condition()
        self._frame = None
        self._frame_id = 0

        self._jobs = queue.SimpleQueue()
        self._thread = None
        self._running = False
        self._cap = None

        # Read-only info for the settings window.
        self.actual_size = (0, 0)
        self.measured_fps = 0.0
        self.backend = "-"
        self.error = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self):
        """Open the device and start the capture thread. Returns False on failure."""
        self._cap = self._open()
        if self._cap is None:
            return False
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.5)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read(self, last_id, timeout=0.1):
        """Wait for a frame newer than ``last_id``.

        Returns ``(frame, frame_id)``. If nothing new arrives before the
        timeout, the previous frame is returned again with the same id.
        """
        with self._cond:
            self._cond.wait_for(lambda: self._frame_id != last_id, timeout=timeout)
            return self._frame, self._frame_id

    def reopen(self):
        """Re-open the device, used after a resolution or FPS change."""
        self._jobs.put(("reopen", None))

    def apply_exposure(self):
        self._jobs.put(("exposure", None))

    def open_driver_panel(self):
        """Show the driver's own property dialog (DirectShow on Windows only)."""
        self._jobs.put(("driver_panel", None))

    # ------------------------------------------------------------------
    # Worker side
    # ------------------------------------------------------------------
    def _open(self):
        s = self.settings
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if IS_WINDOWS else [cv2.CAP_ANY]

        cap = None
        for api in backends:
            cap = cv2.VideoCapture(s.camera_index, api)
            if cap.isOpened():
                break
            cap.release()
            cap = None

        if cap is None:
            self.error = f"Could not open camera #{s.camera_index}"
            return None

        # MJPG lets most USB webcams reach 60 FPS at 720p; raw YUY2 usually
        # cannot because of USB bandwidth.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, s.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, s.resolution[1])
        cap.set(cv2.CAP_PROP_FPS, s.target_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._set_exposure(cap)

        self.actual_size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        try:
            self.backend = cap.getBackendName()
        except cv2.error:
            self.backend = "unknown"
        self.error = None
        return cap

    def _set_exposure(self, cap):
        # Exposure control is driver specific. These are the values that the
        # common backends understand; unsupported calls are simply ignored.
        s = self.settings
        if IS_LINUX:
            auto_on, auto_off = 3, 1
        else:
            auto_on, auto_off = 0.75, 0.25

        if s.auto_exposure:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_on)
        else:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_off)
            cap.set(cv2.CAP_PROP_EXPOSURE, s.exposure)

    def _run_jobs(self):
        while True:
            try:
                job, _ = self._jobs.get_nowait()
            except queue.Empty:
                return
            if job == "reopen":
                self._cap.release()
                cap = self._open()
                # Keep the old settings alive if the new ones are rejected.
                if cap is None:
                    time.sleep(0.5)
                    cap = self._open()
                self._cap = cap
                if cap is None:
                    self._running = False
                    return
            elif job == "exposure":
                self._set_exposure(self._cap)
            elif job == "driver_panel":
                self._cap.set(cv2.CAP_PROP_SETTINGS, 1)

    def _loop(self):
        misses = 0
        last = time.perf_counter()

        while self._running:
            self._run_jobs()
            if self._cap is None:
                break

            ok, frame = self._cap.read()
            if not ok or frame is None:
                misses += 1
                if misses > 90:
                    # The device dropped out (cable, sleep, another app).
                    self._jobs.put(("reopen", None))
                    misses = 0
                time.sleep(0.005)
                continue
            misses = 0

            now = time.perf_counter()
            dt = now - last
            last = now
            if dt > 0:
                inst = 1.0 / dt
                self.measured_fps = inst if self.measured_fps == 0 else self.measured_fps * 0.9 + inst * 0.1

            with self._cond:
                self._frame = frame
                self._frame_id += 1
                self._cond.notify_all()
