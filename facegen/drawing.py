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

from facegen.ipc_shared import ShmArray, ShmSpec
from facegen.param_morpher import ParamMorpher, UiBusSpec
from facegen.rotations import rotate3d  # NEW: for simple 3D view rotation



class Renderer:
    """OpenCV window + drawing utilities."""

    def __init__(self, *args, name: str = "facegen: view",
                 w: int = 1000, h: int = 1000, pos: Iterable[int] = (50, 50),
                 # + NEW:
                 shm_name: str | None = None, shm_source: str | None = None, shm_points: int | None = None,
                 **kwargs) -> None:
        self.name, self.w, self.h = name, int(w), int(h)
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.name, self.w, self.h)
        try:
            cv2.moveWindow(self.name, *pos)
        except Exception:
            pass
        # + NEW:
        self._shm = None
        if shm_name and shm_source and shm_points:
            self._attach_shm(*args, name=shm_name, source=shm_source, n=int(shm_points), **kwargs)

    # + NEW (place inside class Renderer):
    def _attach_shm(self, *args, name: str, source: str, n: int, **kwargs) -> None:
        spec = ShmSpec(f"{name}:r", (n, 3), np.float32) if source == "rot" else ShmSpec(f"{name}:v", (n, 2), np.float32)
        try:
            self._shm = ShmArray(spec=spec, create=False)
        except Exception:
            self._shm = None  # fall back to local mode if not present


    # change the signature to allow outline=None
    def render(self, *args, outline: np.ndarray | None, normalize: bool,
               params: object | None = None, **kwargs):
        """
        Draw one outline (Nx3 or Nx2). If normalize=True, fit to window.
        Else compute pixels/unit from params (expects A0 and amplitude_sum()).
        """
        # + NEW: source selection
        if outline is None and getattr(self, "_shm", None) is not None:
            pts = self._to_cpu_2d(*args, arr=self._shm.a.copy(), **kwargs)  # snapshot to avoid tearing
        else:
            pts = self._to_cpu_2d(*args, arr=outline, **kwargs)

        if normalize or params is None or not hasattr(params, "A0"):
            scale = self._scale_fit(*args, pts=pts, **kwargs)
        else:
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
        try:
            cv2.destroyWindow(self.name)
        except Exception:
            pass
        # + NEW:
        try:
            if getattr(self, "_shm", None) is not None:
                self._shm.close()
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

    def render_grid(self, *args, grid: np.ndarray, normalize: bool,
                    yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0,
                    stride_u: int = 12, stride_v: int = 1,
                    color=(180, 180, 180), thickness: int = 1, **kwargs):
        """
        Draw a wireframe of a (M,N,3) grid. Renders layer rings (V) and vertical
        seams (U). Orientation set by yaw/pitch/roll; orthographic projection.
        """
        g = np.asarray(grid, dtype=np.float32)
        M, N = int(g.shape[0]), int(g.shape[1])
        rot = rotate3d(g.reshape(-1, 3), (float(yaw), float(pitch), float(roll))).reshape(M, N, 3)
        pts2 = rot[:, :, :2].reshape(-1, 2)
        scale = self._scale_fit(*args, pts=pts2, **kwargs) if normalize else 1.0
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        ctr = np.array([self.w / 2, self.h / 2], dtype=np.float32)

        # layer rings (V direction)
        for i in range(0, M, max(1, int(stride_v))):
            pix = (rot[i, :, :2] * scale + ctr).round().astype(np.int32)
            cv2.polylines(img, [pix], isClosed=True, color=color, thickness=int(thickness), lineType=cv2.LINE_AA)

        # vertical seams (U direction)
        for j in range(0, N, max(1, int(stride_u))):
            line = rot[:, j, :2]
            pix = (line * scale + ctr).round().astype(np.int32)
            cv2.polylines(img, [pix], isClosed=False, color=color, thickness=int(thickness), lineType=cv2.LINE_AA)

        cv2.imshow(self.name, img)
        return img


class SliderPanel(ParamMorpher):
    """Controls window + sliders; UI dict lives in ParamMorpher."""

    def __init__(self, *args, name: str = "facegen: controls", w: int = 380, h: int = 420,
                 pos: Iterable[int] | None = None, ui_shm_name: str = "face_ui", **kwargs):
        super().__init__(*args, ui_spec=UiBusSpec(ui_shm_name), create=True, **kwargs)
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
        cv2.createTrackbar("P amp",  self.name, 22,  60,  lambda v: None)
        cv2.createTrackbar("2P amp", self.name, 10,  60,  lambda v: None)
        cv2.createTrackbar("4P amp", self.name, 5,   60,  lambda v: None)
        cv2.createTrackbar("P phase",  self.name, 0,   360, lambda v: None)
        cv2.createTrackbar("2P phase", self.name, 180, 360, lambda v: None)
        cv2.createTrackbar("4P phase", self.name, 0,   360, lambda v: None)
        cv2.createTrackbar("Yaw°",   self.name, 0, 360, lambda v: None)
        cv2.createTrackbar("Pitch°", self.name, 0, 180, lambda v: None)
        cv2.createTrackbar("Roll°",  self.name, 0, 360, lambda v: None)

    def read(self, *args, **kwargs) -> dict:
        # Child sets fields directly on self.ui[...] then parent publishes+returns.
        P = max(3, min(24, cv2.getTrackbarPos("P", self.name)))
        self.ui["P"] = P
        self.ui["A_P"]  = 0.01 * cv2.getTrackbarPos("P amp",  self.name)
        self.ui["A_2P"] = 0.01 * cv2.getTrackbarPos("2P amp", self.name)
        self.ui["A_4P"] = 0.01 * cv2.getTrackbarPos("4P amp", self.name)
        self.ui["phi_P"]  = math.radians(cv2.getTrackbarPos("P phase",  self.name))
        self.ui["phi_2P"] = math.radians(cv2.getTrackbarPos("2P phase", self.name))
        self.ui["phi_4P"] = math.radians(cv2.getTrackbarPos("4P phase", self.name))
        self.ui["yaw"]   = math.radians(cv2.getTrackbarPos("Yaw°",   self.name))
        self.ui["pitch"] = math.radians(cv2.getTrackbarPos("Pitch°", self.name))
        self.ui["roll"]  = math.radians(cv2.getTrackbarPos("Roll°",  self.name))
        return super().read(*args, **kwargs)

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

