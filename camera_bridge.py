"""
camera_bridge.py

Publishes the camera as a plain image file so a process WITHOUT camera
permission can consume live frames.

Why this exists: macOS gates the camera per process (TCC), and the
permission belongs to the *responsible app* — the thing that launched the
process. Terminal.app gets prompted and can be granted; an editor or agent
running in its own app bundle may have no way to be granted at all (if its
bundle doesn't declare NSCameraUsageDescription, no prompt appears and no
System Settings entry is ever created).

So: YOU run this from Terminal.app, where the camera can be granted. It
writes the newest frame to one file, over and over. Anything else on the
machine can then read that file and see through the camera, with no
permission of its own — which is how the demo gets driven end to end by a
process that cannot open /dev/video itself.

    # in Terminal.app (grant the camera prompt on first run):
    python3 camera_bridge.py

    # then, from anywhere:
    python3 demo_webcam.py --watch .camera_feed/latest.jpg
    python3 verify_offline.py --watch .camera_feed/latest.jpg

Ctrl-C to stop. The frame file is deleted on exit.

This is a deliberate, visible, revocable hole: it runs only while you leave
it running, in a terminal you can see, writing to a path you chose. Stop it
and the access is gone. It is NOT a permission bypass — it needs you to
grant Terminal the camera first.
"""

from __future__ import annotations
import argparse
import os
import signal
import sys
import time

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

DEFAULT_DIR = ".camera_feed"
DEFAULT_NAME = "latest.jpg"

_stop = False


def _handle_sigint(signum, frame):
    global _stop
    _stop = True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0,
                    help="camera index (default 0). List them with: "
                          "ffmpeg -f avfoundation -list_devices true -i ''")
    ap.add_argument("--out", default=os.path.join(DEFAULT_DIR, DEFAULT_NAME),
                    help=f"where to write the newest frame (default {DEFAULT_DIR}/{DEFAULT_NAME})")
    ap.add_argument("--fps", type=float, default=5.0,
                    help="how often to publish a frame (default 5). Low is fine — "
                          "the consumer only needs a current frame, not video.")
    ap.add_argument("--width", type=int, default=640,
                    help="downscale width before writing (default 640, 0 = native)")
    ap.add_argument("--keep", action="store_true",
                    help="leave the frame file behind on exit instead of deleting it")
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    tmp_path = args.out + ".tmp"

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Could not open camera index {args.camera}.")
        print()
        print("If you are running this from Terminal.app and saw no permission")
        print("prompt, check System Settings -> Privacy & Security -> Camera and")
        print("enable Terminal, then restart Terminal (the grant is only picked")
        print("up on a fresh start).")
        print()
        print("List available cameras with:")
        print("  ffmpeg -f avfoundation -list_devices true -i \"\"")
        return 1

    ok, frame = cap.read()
    if not ok or frame is None:
        print(f"Camera index {args.camera} opened but returned no frame.")
        cap.release()
        return 1

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    period = 1.0 / args.fps if args.fps > 0 else 0.2
    print(f"Publishing camera {args.camera} -> {args.out} at ~{args.fps:g} fps")
    print(f"Native frame size: {frame.shape[1]}x{frame.shape[0]}"
          + (f", downscaling to width {args.width}" if args.width else ""))
    print("Leave this running. Ctrl-C to stop and revoke access.")

    n = 0
    try:
        while not _stop:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(period)
                continue

            if args.width and frame.shape[1] > args.width:
                scale = args.width / frame.shape[1]
                frame = cv2.resize(
                    frame, (args.width, int(frame.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA)

            # Write to a temp file and rename: os.replace is atomic, so a
            # reader never catches a half-written JPEG.
            if cv2.imwrite(tmp_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85]):
                os.replace(tmp_path, args.out)
                n += 1
                if n % 25 == 0:
                    print(f"  {n} frames published", end="\r", flush=True)
            time.sleep(period)
    finally:
        cap.release()
        for p in (tmp_path, None if args.keep else args.out):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        print(f"\nStopped after {n} frames. "
              + ("Frame file kept." if args.keep else "Frame file removed."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
