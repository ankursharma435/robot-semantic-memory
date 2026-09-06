"""
frame_source.py

Where demo frames come from. Added 2026-09-03 for two reasons found while
getting the demo running on this MacBook:

  1. This machine has TWO cameras (a built-in FaceTime HD and an S65VC UVC
     webcam in the USB monitor), so hardcoding VideoCapture(0) is a coin
     flip as to which one you get. You can now choose.

  2. macOS gates camera access per-process (TCC). When the host process
     hasn't been granted it, EVERY index fails with
     "OpenCV: not authorized to capture video (status 0), requesting..."
     — it is not a camera-specific or index-specific problem, and no code
     change fixes it. The old error message ("Check your camera /
     permissions") didn't say what to actually do; now it does.

     That is also why a directory of still images is supported as a frame
     source: it keeps the whole demo runnable with no camera at all, which
     is a useful safety net on demo day and the only way to smoke-test the
     loop on a box without camera permission.

Usage:
    src = FrameSource.open(camera=1)              # specific camera index
    src = FrameSource.open(images="test_objects") # no camera needed
    src = FrameSource.open()                      # camera from env, else 0
"""

from __future__ import annotations
import glob
import os
import time

import cv2

CAMERA_ENV_VAR = "ROBOT_CAMERA_INDEX"

PERMISSION_HELP = """
Could not open any camera.

On macOS this is almost always the per-process camera permission, not the
camera itself. If you saw "OpenCV: not authorized to capture video" above,
that is exactly what happened.

  Fix: System Settings -> Privacy & Security -> Camera -> enable the app
       you launched this from (Terminal / iTerm / your IDE / Claude), then
       RESTART that app. The permission is only picked up on a fresh start.

  Which camera: `system_profiler SPCameraDataType` lists them. Pick one
       with --camera N (or {env}=N). Index order usually follows that list.

  No camera at all: run with --images DIR to drive the demo from a folder
       of still photographs instead. Everything except live capture works.
""".format(env=CAMERA_ENV_VAR)


class FrameSource:
    """Yields BGR frames (same format cv2.VideoCapture.read() returns)."""

    def __init__(self, cap=None, paths=None, description=""):
        self._cap = cap
        self._paths = paths or []
        self._watch_path = None
        self._i = 0
        self.description = description

    # ---- construction ----

    @classmethod
    def open(cls, camera: int | None = None, images: str | None = None,
              watch: str | None = None) -> "FrameSource":
        if watch:
            return cls._open_watch(watch)
        if images:
            return cls._open_images(images)
        if camera is None:
            env = os.environ.get(CAMERA_ENV_VAR, "").strip()
            camera = int(env) if env else 0
        return cls._open_camera(camera)

    @classmethod
    def _open_watch(cls, path: str, timeout_s: float = 15.0) -> "FrameSource":
        """Live frames from a single file that another process keeps
        rewriting — see camera_bridge.py.

        This is how a process with no camera permission of its own gets
        live camera frames: something that DOES have permission (started
        from Terminal.app) republishes the newest frame to `path`, and we
        re-read it. Waits for the file to appear so the two can be started
        in either order.
        """
        deadline = time.time() + timeout_s
        while not os.path.exists(path):
            if time.time() > deadline:
                raise RuntimeError(
                    f"No frame file at {path!r} after {timeout_s:g}s.\n\n"
                    "Start the publisher first, from Terminal.app (which can "
                    "be granted camera access):\n"
                    f"    python3 camera_bridge.py --out {path}\n\n"
                    "Leave it running, then re-run this."
                )
            time.sleep(0.25)
        src = cls(description=f"live frames watched at {path}")
        src._watch_path = path
        ok, _ = src.read()
        if not ok:
            raise RuntimeError(f"Frame file {path!r} exists but could not be decoded.")
        return src

    @classmethod
    def _open_camera(cls, index: int) -> "FrameSource":
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"Camera index {index} would not open.\n{PERMISSION_HELP}"
            )
        ok, _ = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(
                f"Camera index {index} opened but returned no frame.\n{PERMISSION_HELP}"
            )
        return cls(cap=cap, description=f"camera index {index}")

    @classmethod
    def _open_images(cls, directory: str) -> "FrameSource":
        paths = sorted(
            p for ext in ("jpg", "jpeg", "png")
            for p in glob.glob(os.path.join(directory, f"*.{ext}"))
        )
        if not paths:
            raise RuntimeError(f"No .jpg/.jpeg/.png files found in {directory!r}")
        return cls(paths=paths, description=f"{len(paths)} still images from {directory}")

    # ---- reading ----

    def read(self):
        """Returns (ok, frame). The image source loops forever so the demo's
        `while True: read()` loop behaves the same as with a live camera."""
        if self._cap is not None:
            return self._cap.read()
        if self._watch_path is not None:
            # Re-read the same path every time; the publisher renames a temp
            # file over it atomically, so we never see a partial JPEG. A read
            # can still fail if it lands exactly on the rename — retry once.
            for _ in range(3):
                frame = cv2.imread(self._watch_path)
                if frame is not None:
                    self._i += 1
                    return True, frame
                time.sleep(0.05)
            return False, None
        path = self._paths[self._i % len(self._paths)]
        self._i += 1
        frame = cv2.imread(path)
        return frame is not None, frame

    def current_name(self) -> str:
        """Filename of the frame just served (image source only) — handy for
        labelling when memorizing from a folder."""
        if self._cap is not None or not self._paths:
            return ""
        return os.path.basename(self._paths[(self._i - 1) % len(self._paths)])

    @property
    def is_live(self) -> bool:
        """True if frames change over time (real camera or watched file)."""
        return self._cap is not None or self._watch_path is not None

    def release(self):
        if self._cap is not None:
            self._cap.release()


def add_source_args(parser):
    """Shared CLI flags for both demos."""
    parser.add_argument("--camera", type=int, default=None,
                        help=f"camera index (default: ${CAMERA_ENV_VAR} or 0). "
                             "Run `system_profiler SPCameraDataType` to list cameras.")
    parser.add_argument("--images", type=str, default=None,
                        help="drive from a folder of still images instead of a camera")
    parser.add_argument("--watch", type=str, default=None,
                        help="drive from a single live-updating frame file, as written "
                              "by camera_bridge.py (e.g. .camera_feed/latest.jpg). Lets a "
                              "process without camera permission use the camera.")
    return parser
