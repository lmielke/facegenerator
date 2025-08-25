# sketcher.py
"""
Fourier-based sketch generator (GPU / CuPy).
- Named preset: loads ~/.facegen/sketches/<name>.yml
- Random preset: "--sketch random" uses Randomizer._randomize_params() (no YAML)
"""

from __future__ import annotations
import os, json, yaml
from dataclasses import dataclass, replace
from typing import Any, Dict, Tuple

import cupy as cp
import numpy as np

from facegen.random_params import Randomizer
from facegen.rotations import rotate3d


# =============================== Params ======================================

@dataclass(frozen=True)
class Harm:
    """
    Represents a single harmonic component (winding vector) in the Fourier radial series.
    Each term defines a cosine wave around the circle with a frequency,
    amplitude, and phase offset. Multiple terms combine with the base radius
    to form the final outline shape.
    """
    k: int     # frequency multiplier (cycles around the circle)
    A: float   # amplitude (strength of this harmonic, radius offset)
    phi: float # phase angle in radians (starting angle / rotation)

@dataclass(frozen=True)
class Orbits:
    """Angles (radians) for orbit controls."""
    yaw: float   # rotate around +Z in screen terms (or choose your convention)
    pitch: float # around +Y
    roll: float  # around +X

    @staticmethod
    def from_ui(ui: dict) -> "Orbits":
        # ui already returns radians from your sliders
        return Orbits(yaw=float(ui["yaw"]),
                      pitch=float(ui["pitch"]),
                      roll=float(ui["roll"]))

@dataclass(frozen=True)
class FourierParams:
    """Container for all Fourier parameters that define a sketch outline.

    Holds the base radius, the fundamental symmetry order, and the list of
    harmonic terms. Together these parameters generate r(θ), the polar
    radius function for the shape.
    """
    A0: float             # base radius (average circle size)
    P: int                # fundamental symmetry order (e.g. petals/lobes)
    terms: Tuple[Harm, ...]  # list of harmonic terms that shape the outline


    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "FourierParams":
        ts = tuple(Harm(**t) for t in d.get("terms", []))
        A0 = float(d.get("A0", 1.0))
        P = int(d.get("P", 1) or 1)
        return FourierParams(A0=A0, P=P, terms=ts)

    def to_json(self, *args, **kwargs) -> str:
        d = {"A0": self.A0, "P": self.P,
             "terms": [{"k": t.k, "A": t.A, "phi": t.phi} for t in self.terms]}
        return json.dumps(d, indent=2)

    def amplitude_sum(self, *args, **kwargs) -> float:
        return float(sum(abs(t.A) for t in self.terms))

    def with_caps(self, *args, fundamental_cap: float = 0.90,
                  ripple_cap: float = 1.50, **kwargs) -> "FourierParams":
        """Return a capped copy: limit fundamental and total ripple vs A0."""
        ts = list(self.terms)
        if ts and ts[0].k == self.P:
            a0_cap = fundamental_cap * float(self.A0)
            if ts[0].A > a0_cap:
                ts[0] = Harm(k=ts[0].k, A=a0_cap, phi=ts[0].phi)

        ripple = sum(t.A for t in ts)
        cap = ripple_cap * float(self.A0)
        if ripple > cap and ripple > 1e-6:
            s = cap / ripple
            ts = [Harm(k=t.k, A=t.A * s, phi=t.phi) for t in ts]
        return FourierParams(A0=float(self.A0), P=int(self.P), terms=tuple(ts))

    @staticmethod
    def from_ui(ui: dict, apply_caps: bool = False, **kwargs) -> "FourierParams":
        P = int(ui["P"])   # 👈 must be defined first
        terms = (
            Harm(k=P,   A=ui["A_P"],  phi=ui["phi_P"]),
            Harm(k=2*P, A=ui["A_2P"], phi=ui["phi_2P"]),
            Harm(k=4*P, A=ui["A_4P"], phi=ui["phi_4P"]),
        )
        params = FourierParams(A0=float(ui.get("A0", 1.20)), P=P, terms=terms)
        return params.with_caps(**kwargs) if apply_caps else params


# ============================== Sketcher =====================================

class FourierSketcher:
    """Generates arc-length–resampled outlines from Fourier parameters."""

    def __init__(self, *args, sketch_name: str | None = None, **kwargs):
        self.sketch_name = sketch_name
        self.oversample = int(kwargs.get("oversample", 100))
        self.n = int(kwargs.get("num_objects", 1))
        self.x = int(kwargs.get("num_points", 255))
        raw = self._load_or_randomize(*args, **kwargs)
        self.params = FourierParams.from_dict(raw)
        kwargs.update({
            "A0": self.params.A0,
            "P": self.params.P,
            "terms": [t.__dict__ for t in self.params.terms],
        })
        self.kwargs = kwargs

    # --- configuration -------------------------------------------------------

    def _load_or_randomize(self, *args, **kwargs) -> Dict[str, Any]:
        name = (self.sketch_name or "").lower()
        if name == "random":
            return Randomizer._randomize_params()
        if not self.sketch_name:
            return {}
        path = os.path.join(os.path.expanduser("~/.facegen/sketches"),
                            f"{self.sketch_name}.yml")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Preset not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    # --- core math -----------------------------------------------------------

    def _radius(self, theta: cp.ndarray, *args,
                params: FourierParams | None = None, **kwargs) -> cp.ndarray:
        p = params or self.params
        r = cp.full(theta.shape, p.A0, dtype=cp.float32)
        for t in p.terms:
            r += t.A * cp.cos(t.k * theta + t.phi)
        return r

    def outline_single(self, *args, **kwargs) -> cp.ndarray:
        X = int(self.x) * int(self.oversample)
        theta = cp.linspace(0, 2 * cp.pi, X, endpoint=False, dtype=cp.float32)
        r = self._radius(theta, *args, **kwargs)
        neg = r < 0
        if cp.any(neg):
            theta = cp.where(neg, theta + cp.pi, theta)
            r = cp.abs(r)
        xy = cp.stack([r * cp.cos(theta), r * cp.sin(theta)], axis=1)
        xy_even = self._resample_arclength(cp.asnumpy(xy).astype(np.float32),
                                           num_sample_points=self.x)
        xy_even_cp = cp.asarray(xy_even)
        z = cp.zeros((self.x, 1), dtype=xy_even_cp.dtype)
        return cp.concatenate([xy_even_cp, z], axis=1)

    def batch_outlines(self, n: int, *args, **kwargs) -> cp.ndarray:
        one = self.outline_single(*args, **kwargs)     # (X,3)
        return cp.tile(one[None, :, :], (int(n), 1, 1))

    # --- utilities -----------------------------------------------------------

    def _resample_arclength(self, xy: np.ndarray, *args,
                            num_sample_points: int, **kwargs) -> np.ndarray:
        seg = xy[1:] - xy[:-1]
        d = np.sqrt((seg * seg).sum(axis=1))
        d = np.concatenate([d, [np.linalg.norm(xy[0] - xy[-1])]])
        s = np.concatenate([[0.0], np.cumsum(d)])
        total = float(s[-1])
        if total <= 1e-9:
            return np.repeat(xy[:1], num_sample_points, axis=0).astype(np.float32)
        t = np.linspace(0.0, total, num_sample_points, endpoint=False, dtype=np.float32)
        xy_closed = np.vstack([xy, xy[0]])
        x = np.interp(t, s, xy_closed[:, 0]); y = np.interp(t, s, xy_closed[:, 1])
        return np.stack([x, y], axis=1).astype(np.float32)


# ============================== Rules ========================================
class Stabilizer:
    """Mild guards; allows negative r while preventing extreme blowups."""

    def __init__(self, *args, fundamental_cap: float = 0.90, ripple_cap: float = 1.50, **kwargs):
        self.fund_cap = float(fundamental_cap)  # as multiple of A0
        self.ripple_cap = float(ripple_cap)     # as multiple of A0

    def apply(self, *args, params: FourierParams, **kwargs) -> FourierParams:
        A0 = params.A0
        ts = list(params.terms)
        # assume ts[0] is k=P
        ts[0] = replace(ts[0], A=min(ts[0].A, self.fund_cap * A0))
        ripple = sum(t.A for t in ts)
        cap = self.ripple_cap * A0
        if ripple > cap and ripple > 1e-6:
            s = cap / ripple
            ts = [replace(t, A=t.A * s) for t in ts]
        return FourierParams(A0=A0, P=params.P, terms=tuple(ts))

# ============================== Model ========================================
class OutlineModel:
    """Pure math for outline generation (NumPy, cached theta/trig, in-place ops)."""

    def __init__(self, *args, num_points:int, **kwargs):
        self.num_points = num_points
        self._theta_cache: dict[tuple[int, int], np.ndarray] = {}
        self._trig_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._r_buf: np.ndarray | None = None
        self._xy_buf: np.ndarray | None = None
        # resample scratch/output (sized on demand)
        self._d: np.ndarray | None = None
        self._s: np.ndarray | None = None
        self._xext: np.ndarray | None = None
        self._yext: np.ndarray | None = None
        self._out: np.ndarray | None = None
        self.vtxs = np.empty((num_points, 2), dtype=np.float32)  # (x,y) pairs for resampling
        self.rotateds = np.empty((num_points, 3), dtype=np.float32)  # (x,y) pairs for rotation
        self.cumRads = np.zeros((3), dtype=np.float32)  # cumulative radii for rotation
        self.oldRads = np.zeros((3), dtype=np.float32)  # for detecting change in rotation

    # ---------- cached param grids ----------
    def _theta(self, points: int, oversample: int) -> np.ndarray:
        key = (int(points), int(oversample))
        th = self._theta_cache.get(key)
        if th is None:
            th = np.linspace(0.0, 2.0*np.pi, key[0]*key[1], endpoint=False, dtype=np.float32)
            self._theta_cache[key] = th
        return th

    def _trig(self, points: int, oversample: int) -> tuple[np.ndarray, np.ndarray]:
        key = (int(points), int(oversample))
        cs = self._trig_cache.get(key)
        if cs is None:
            th = self._theta(*key)
            c, s = np.cos(th, dtype=np.float32), np.sin(th, dtype=np.float32)
            self._trig_cache[key] = (c, s)
            return c, s
        return cs

    # ---------- buffers ----------
    def _ensure_r(self, n: int) -> np.ndarray:
        if self._r_buf is None or self._r_buf.shape[0] != n:
            self._r_buf = np.empty(n, dtype=np.float32)
        return self._r_buf

    def _ensure_xy(self, n: int) -> np.ndarray:
        if self._xy_buf is None or self._xy_buf.shape[0] != n:
            self._xy_buf = np.empty((n, 2), dtype=np.float32)
        return self._xy_buf

    def _ensure_resample_buffers(self, n: int, out_points: int) -> None:
        if self._d is None or self._d.shape[0] != n:
            self._d = np.empty(n, dtype=np.float32)
            self._s = np.empty(n + 1, dtype=np.float32)
            self._xext = np.empty(n + 1, dtype=np.float32)
            self._yext = np.empty(n + 1, dtype=np.float32)
        if self._out is None or self._out.shape[0] != out_points:
            self._out = np.empty((int(out_points), 2), dtype=np.float32)

    # ---------- core math ----------
    def eval_radius(self, *args, theta: np.ndarray, params: FourierParams, **kwargs) -> np.ndarray:
        r = self._ensure_r(theta.shape[0])
        r.fill(np.float32(params.A0))
        for t in params.terms:
            np.add(r, np.float32(t.A) * np.cos(t.k * theta + t.phi), out=r)
        return r  # returning a reference (no copy)

    # NOTE: negative-r remap is unnecessary for XY (it cancels out in trig).
    # Keep only if you need explicit (r,theta) post-mapping.

    def resample_arclength(self, *args, xy: np.ndarray, out_points: int, **kwargs) -> np.ndarray:
        n = int(xy.shape[0])
        if n == 0:
            return np.empty((0, 2), dtype=np.float32)
        self._ensure_resample_buffers(n, int(out_points))

        # segment lengths with wrap: d[i] = |xy[i+1]-xy[i]|, xy[n]≡xy[0]
        dx = np.empty(n, dtype=np.float32); dy = np.empty(n, dtype=np.float32)
        np.subtract(np.roll(xy[:, 0], -1), xy[:, 0], out=dx)
        np.subtract(np.roll(xy[:, 1], -1), xy[:, 1], out=dy)
        np.hypot(dx, dy, out=self._d)

        # cumulative arc-length s[0]=0, s[-1]=total
        self._s[0] = 0.0
        np.cumsum(self._d, out=self._s[1:])
        total = float(self._s[-1])
        if total <= 1e-9:
            self._out[:] = xy[0:1]
            return self._out

        # targets
        t = np.linspace(0.0, total, int(out_points), endpoint=False, dtype=np.float32)

        # x/y extended (close loop) to match s length
        self._xext[:-1] = xy[:, 0]; self._xext[-1] = xy[0, 0]
        self._yext[:-1] = xy[:, 1]; self._yext[-1] = xy[0, 1]

        # interpolate → write into out buffer
        self._out[:, 0] = np.interp(t, self._s, self._xext).astype(np.float32, copy=False)
        self._out[:, 1] = np.interp(t, self._s, self._yext).astype(np.float32, copy=False)
        return self._out

    def update_orbit(self, orbits: Orbits) -> None:
        # store angles in the model; compute() will use them
        self.cumRads[:] = (orbits.yaw, orbits.pitch, orbits.roll)

    # ---------- public API ----------
    def compute(self, *args, params: FourierParams, oversample: int = 120, **kwargs) -> np.ndarray:
        n_os = int(self.num_points) * int(oversample)
        cos_th, sin_th = self._trig(points=int(self.num_points), oversample=int(oversample))
        r = self.eval_radius(*args, theta=self._theta(points=int(self.num_points), oversample=int(oversample)),
                             params=params, **kwargs)
        xy = self._ensure_xy(n_os)
        np.multiply(r, cos_th, out=xy[:, 0])
        np.multiply(r, sin_th, out=xy[:, 1])
        self.vtxs[:] = self.resample_arclength(*args, xy=xy, out_points=int(self.num_points), **kwargs)
        self.rotateds[:] = rotate3d(self.vtxs, tuple(self.cumRads))


