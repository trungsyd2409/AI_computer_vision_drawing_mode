"""Air Draw - draw in the air with your fingertip.

Usage:
    python main.py                # default webcam
    python main.py --camera 1     # pick another webcam
"""

import argparse
import os

# Quiet the TensorFlow Lite / MediaPipe start-up logs.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from airdraw.app import AirDraw  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Draw on the webcam view with hand gestures.")
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    AirDraw(camera_index=args.camera).run()
