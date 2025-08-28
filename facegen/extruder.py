# facegen/extruder.py
"""
First draft:
- Extruder: copies a single outline along +Z (stack). Optionally applies a
  per-layer geometric morph (yaw/pitch/roll deltas) so copy i is rotated by
  i * delta (e.g. Roll += i * π/12).
- ExtrusionMorpher: OpenCV panel analogous to SliderPanel, but its sliders
  mean **per-layer increments** (the “applied each copy” amount). It publishes
  those deltas to the shared UI bus (via ParamMorpher). Fourier deltas (A_P,
  phi_2P, …) are exposed for later use when we enable per-layer re-evaluation
  of the outline; today the Extruder only uses yaw/pitch/roll deltas (geo).

Notes:
- Keep methods short (≤6 lines) and *args/**kwargs-friendly.
- Outline is (N,2|3) float32, grid is (M,N,3), indices are (T,3) int32.
"""

from __future__ import annotations
import math, numpy as np, cv2
from dataclasses import dataclass
from typing import Iterable, Optional

from facegen.rotations import rotate3d
from facegen.param_morpher import ParamMorpher, UiBusSpec
from facegen.ipc_shared import ShmArray, ShmSpec


# ============================== Extruder =====================================

@dataclass(frozen=True)
class StackSpec:
    M: int = 120                  # number of copies/layers
    dz: float = 0.6               # Z step per layer
    wrap_u: bool = True           # close the ring in U for indices


class Extruder:
    """Outline (N,2|3) → stacked grid (M,N,3)."""

    def __init__(self, *args,
                 outline: Optional[np.ndarray] = None,
                 shm_name: Optional[str] = None, shm_source: str = "rot",
                 shm_points: Optional[int] = None,
                 **kwargs) -> None:
        self.base = self._to_xyz(*args, v=outline, **kwargs) if outline is not None else None
        self.grid: Optional[np.ndarray] = None; self.idx: Optional[np.ndarray] = None
        self._shm = None
        if self.base is None and shm_name and shm_points:
            name = f"{shm_name}:r" if shm_source == "rot" else f"{shm_name}:v"
            shp = (int(shm_points), 3) if shm_source == "rot" else (int(shm_points), 2)
            try: self._shm = ShmArray(spec=ShmSpec(name, shp, np.float32), create=False)
            except Exception: self._shm = None  # fall back requires outline param

    def stack(self, *args, spec: StackSpec, morph: "ExtrusionMorpher|None" = None, **kwargs) -> np.ndarray:
        base = self._get_outline(*args, **kwargs)
        M, N = int(spec.M), base.shape[0]
        g = np.repeat(base[None, :, :], M, axis=0).astype(np.float32)
        g[:, :, 2] += np.arange(M, dtype=np.float32)[:, None] * float(spec.dz)
        if morph is not None:
            deltas = morph.read(*args, **kwargs)  # per-layer increments (radians/amps)
            self._apply_geo_deltas(*args, g=g, deltas=deltas, **kwargs)
        self.grid = g
        self.idx = self._build_indices(*args, M=M, N=N, wrap_u=bool(spec.wrap_u), **kwargs)
        return self.grid

    # ---------------- helpers ----------------
    def _get_outline(self, *args, **kwargs) -> np.ndarray:
        # Always promote to (N,3) xyz inside the extruder
        if self.base is not None:
            return self._to_xyz(*args, v=self.base, **kwargs)
        if self._shm is not None:
            a = self._shm.a.copy()
            return a if a.shape[1] == 3 else np.c_[a, np.zeros(a.shape[0], np.float32)]
        raise ValueError("Extruder needs an outline or a valid SHM attachment.")

    def _to_xyz(self, *args, v: np.ndarray, **kwargs) -> np.ndarray:
        a = np.asarray(v, dtype=np.float32)
        if a.ndim != 2 or a.shape[1] not in (2, 3):
            raise ValueError("outline must be (N,2) or (N,3)")
        return a if a.shape[1] == 3 else np.c_[a, np.zeros(a.shape[0], np.float32)]

    def _apply_geo_deltas(self, *args, g: np.ndarray, deltas: dict, **kwargs) -> None:
        dy = float(deltas.get("yaw")  or 0.0); dp = float(deltas.get("pitch") or 0.0); dr = float(deltas.get("roll") or 0.0)
        if dy == dp == dr == 0.0: return
        M = g.shape[0];  a = np.arange(M, dtype=np.float32)
        for i in range(M):
            g[i] = rotate3d(g[i], (a[i]*dy, a[i]*dp, a[i]*dr))

    def _build_indices(self, *args, M: int, N: int, wrap_u: bool, **kwargs) -> np.ndarray:
        quads = []
        for i in range(M - 1):
            for j in range(N):
                jn = (j + 1) % N if wrap_u else min(j + 1, N - 1)
                a, b, c, d = i*N + j, i*N + jn, (i+1)*N + j, (i+1)*N + jn
                quads += [(a, c, b), (b, c, d)]
        return np.asarray(quads, dtype=np.int32)


# =========================== ExtrusionMorpher ================================

class ExtrusionMorpher(ParamMorpher):
    """
    Like SliderPanel, but sliders are **per-layer increments**:
    - Amplitudes (A_P, A_2P, A_4P): delta amplitude per layer (not applied yet; kept for parity).
    - Phases (phi_*): delta radians per layer (not applied yet; kept for parity).
    - Orbits (yaw/pitch/roll): delta radians per layer (used by Extruder today).
    """

    def __init__(self, *args, name: str = "facegen: extrude morph", w: int = 380, h: int = 360,
                 pos: Iterable[int] | None = None, ui_shm_name: str = "face_ui", **kwargs):
        super().__init__(*args, ui_spec=UiBusSpec(ui_shm_name), create=True, **kwargs)
        self.name = name
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL); cv2.resizeWindow(self.name, w, h)
        try:
            if pos: cv2.moveWindow(self.name, *pos)
        except Exception: pass
        self._build(*args, **kwargs)

    # sliders express **increments per layer**
    def _build(self, *args, **kwargs):
        cv2.createTrackbar("Δ P amp",   self.name, 0, 40,  lambda v: None)   # 0..0.40 per-layer
        cv2.createTrackbar("Δ 2P amp",  self.name, 0, 40,  lambda v: None)
        cv2.createTrackbar("Δ 4P amp",  self.name, 0, 40,  lambda v: None)
        cv2.createTrackbar("Δ P phase°",  self.name, 0, 180, lambda v: None) # degrees per-layer
        cv2.createTrackbar("Δ 2P phase°", self.name, 0, 180, lambda v: None)
        cv2.createTrackbar("Δ 4P phase°", self.name, 0, 180, lambda v: None)
        cv2.createTrackbar("Δ Yaw°",    self.name, 0,  45,  lambda v: None)
        cv2.createTrackbar("Δ Pitch°",  self.name, 0,  45,  lambda v: None)
        cv2.createTrackbar("Δ Roll°",   self.name, 0,  90,  lambda v: None)

    def read(self, *args, **kwargs) -> dict:
        # child writes into self.ui[...] as per-layer deltas, then parent publishes+returns
        self.ui["A_P"]   = 0.01 * cv2.getTrackbarPos("Δ P amp",   self.name)
        self.ui["A_2P"]  = 0.01 * cv2.getTrackbarPos("Δ 2P amp",  self.name)
        self.ui["A_4P"]  = 0.01 * cv2.getTrackbarPos("Δ 4P amp",  self.name)
        self.ui["phi_P"]  = math.radians(cv2.getTrackbarPos("Δ P phase°",   self.name))
        self.ui["phi_2P"] = math.radians(cv2.getTrackbarPos("Δ 2P phase°",  self.name))
        self.ui["phi_4P"] = math.radians(cv2.getTrackbarPos("Δ 4P phase°",  self.name))
        self.ui["yaw"]    = math.radians(cv2.getTrackbarPos("Δ Yaw°",   self.name))
        self.ui["pitch"]  = math.radians(cv2.getTrackbarPos("Δ Pitch°", self.name))
        self.ui["roll"]   = math.radians(cv2.getTrackbarPos("Δ Roll°",  self.name))
        # P itself doesn’t have a meaningful linear “delta per layer”; leave None unless you want drift.
        return super().read(*args, **kwargs)

    # Utility: build a per-layer UI dict from a base (for future Fourier re-eval)
    def apply_to_base_ui(self, *args, base_ui: dict, i: int, **kwargs) -> dict:
        ui = dict(base_ui)
        for k in ("A_P","A_2P","A_4P","phi_P","phi_2P","phi_4P","yaw","pitch","roll"):
            if ui.get(k) is not None and self.ui.get(k) is not None:
                ui[k] = float(ui[k]) + float(self.ui[k]) * int(i)
        return ui
