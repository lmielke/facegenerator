# facegen/drawing.py
"""
OpenCV renderer for 2D outlines (single + timed sequence).
Single class: Renderer
"""

from __future__ import annotations
import math, time
from typing import Iterable, Sequence
import cv2
import numpy as np
import cupy as cp


class Renderer:
    """OpenCV window + drawing utilities."""

    def __init__(self, *args, name: str = "facegen: view",
                 w: int = 1000, h: int = 1000, pos: Iterable[int] = (50, 50),
                 **kwargs) -> None:
        self.name, self.w, self.h = name, int(w), int(h)
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.name, self.w, self.h)
        try:
            cv2.moveWindow(self.name, *pos)
        except Exception:
            pass

    # ---- public API (kept exactly for live.py) ----
    def render(self, *args, outline: np.ndarray, normalize: bool,
               params: object | None = None, **kwargs):
        """
        Draw one outline (Nx3 or Nx2). If normalize=True, fit to window.
        Else compute pixels/unit from params (expects A0 and amplitude_sum()).
        """
        pts = self._to_cpu_2d(*args, arr=outline, **kwargs)
        if normalize or params is None or not hasattr(params, "A0"):
            scale = self._scale_fit(*args, pts=pts, **kwargs)
        else:
            # px per world unit using A0 + amplitude_sum()
            A0 = float(getattr(params, "A0", 1.0))
            amp = float(getattr(params, "amplitude_sum", lambda *a, **k: 0.0)())
            r_exp = max(1e-6, A0 + amp)
            scale = 0.95 * min(self.w, self.h) / (2.0 * r_exp)
        img = self._draw_polyline(*args, pts=pts, scale=scale, **kwargs)
        cv2.imshow(self.name, img)
        return img

    def render_sequence(self, arr_list, *args, seconds_per_item: float = 10.0, **kwargs):
        """Show multiple outlines; any key advances early."""
        for arr in arr_list:
            self.render(*args, outline=arr, normalize=True, **kwargs)
            t0 = time.time()
            while True:
                if cv2.waitKey(40) >= 0:
                    break
                if time.time() - t0 >= seconds_per_item:
                    break

    def teardown(self, *args, **kwargs) -> None:
        """Close the window (name kept for sketch.py)."""
        try:
            cv2.destroyWindow(self.name)
        except Exception:
            pass

    # ---- private helpers ----
    def _to_cpu_2d(self, *args, arr, **kwargs) -> np.ndarray:
        try:
            if isinstance(arr, cp.ndarray):
                arr = cp.asnumpy(arr)
        except Exception:
            pass
        a = np.asarray(arr, dtype=np.float32)
        return a[:, :2] if a.ndim == 2 and a.shape[1] >= 2 else a.reshape(-1, 2)

    def _scale_fit(self, *args, pts: np.ndarray, **kwargs) -> float:
        r = float(np.linalg.norm(pts, axis=1).max() or 1e-6)
        return 0.95 * min(self.w, self.h) / (2.0 * r)

    def _draw_polyline(self, *args, pts: np.ndarray, scale: float,
                       color=(0, 255, 0), thickness: int = 2, **kwargs) -> np.ndarray:
        h, w = self.h, self.w
        img = np.zeros((h, w, 3), dtype=np.uint8)
        ctr = np.array([w / 2, h / 2], dtype=np.float32)
        pix = (pts * float(scale) + ctr).round().astype(np.int32)
        cv2.polylines(img, [pix], isClosed=True, color=color,
                      thickness=int(thickness), lineType=cv2.LINE_AA)
        return img


# ============================== View: Controls ===============================
class SliderPanel:
    """Owns the control window + sliders (short ASCII labels)."""

    def __init__(self, *args, name: str = "facegen: controls", w: int = 380, h: int = 420,
                 pos: Iterable[int] | None = None, **kwargs):
        self.name = name
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.name, w, h)
        try:
            if pos: cv2.moveWindow(self.name, *pos)
        except Exception:
            pass
        self._build(*args, **kwargs)

    def _build(self, *args, **kwargs):
        cv2.createTrackbar("P",      self.name, 5,   24,  lambda v: None)
        cv2.createTrackbar("P amp",  self.name, 22, 60,  lambda v: None)
        cv2.createTrackbar("2P amp", self.name, 10, 60,  lambda v: None)
        cv2.createTrackbar("4P amp", self.name, 5,  60,  lambda v: None)
        cv2.createTrackbar("P phase",  self.name, 0,   360, lambda v: None)
        cv2.createTrackbar("2P phase", self.name, 180, 360, lambda v: None)
        cv2.createTrackbar("4P phase", self.name, 0,   360, lambda v: None)
        # NEW: orbit controls (degrees)
        cv2.createTrackbar("Yaw°",   self.name, 0, 360, lambda v: None)
        cv2.createTrackbar("Pitch°", self.name, 0, 180, lambda v: None)
        cv2.createTrackbar("Roll°",  self.name, 0, 360, lambda v: None)

    def read(self, *args, **kwargs) -> dict:
        P    = max(3, min(24, cv2.getTrackbarPos("P", self.name)))
        A_P  = 0.01 * cv2.getTrackbarPos("P amp",  self.name)
        A_2P = 0.01 * cv2.getTrackbarPos("2P amp", self.name)
        A_4P = 0.01 * cv2.getTrackbarPos("4P amp", self.name)
        phi_P  = math.radians(cv2.getTrackbarPos("P phase",  self.name))
        phi_2P = math.radians(cv2.getTrackbarPos("2P phase", self.name))
        phi_4P = math.radians(cv2.getTrackbarPos("4P phase", self.name))

        # NEW: angles in radians
        yaw   = math.radians(cv2.getTrackbarPos("Yaw°",   self.name))
        pitch = math.radians(cv2.getTrackbarPos("Pitch°", self.name))
        roll  = math.radians(cv2.getTrackbarPos("Roll°",  self.name))

        return {
            "P": P,
            "A_P": A_P, "A_2P": A_2P, "A_4P": A_4P,
            "phi_P": phi_P, "phi_2P": phi_2P, "phi_4P": phi_4P,
            "yaw": yaw, "pitch": pitch, "roll": roll,     # ← add these
        }



    def teardown(self, *args, **kwargs):
        try:
            cv2.destroyWindow(self.name)
        except Exception:
            pass



# ============================== Timing/Keys ==================================
class Pacer:
    def __init__(self, *args, fps: int = 60, **kwargs):
        self.dt_ms = max(1, int(1000 / max(1, fps)))
        self.last = time.time()

    def poll_key(self, *args, **kwargs) -> int:
        k = cv2.waitKey(self.dt_ms)
        now = time.time()
        sleep_ms = self.dt_ms - int((now - self.last) * 1000)
        if sleep_ms > 1:
            cv2.waitKey(sleep_ms)
        self.last = now
        return k

