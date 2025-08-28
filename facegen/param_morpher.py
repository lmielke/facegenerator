# facegen/param_morpher.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from facegen.ipc_shared import ShmArray, ShmSpec
from facegen.sketcher import FourierParams, Orbits

@dataclass(frozen=True)
class UiBusSpec:
    name: str = "face_ui"

class ParamMorpher:
    """
    Holds the canonical UI dict + a shared-memory bus.
    Child classes set fields in self.ui[...] and then call super().read().
    """
    def __init__(self, *args, ui_spec: UiBusSpec = UiBusSpec(),
                 create: bool = True, **kwargs):
        self.ui: dict[str, float | None] = self._init_ui(*args, **kwargs)
        self._keys = tuple(self.ui.keys())
        n = len(self._keys)
        self._vec = ShmArray(spec=ShmSpec(f"{ui_spec.name}:u", (n,), np.float32), create=create)
        self._stp = ShmArray(spec=ShmSpec(f"{ui_spec.name}:s", (1,),  np.int32),  create=create)

    # ---- public API ----
    def read(self, *args, **kwargs) -> dict:
        """Publish current self.ui to SHM (None→NaN) and return self.ui."""
        v = self._to_vec(self.ui)
        s = int(self._stp.a[0]); self._stp.a[0] = s + 1
        self._vec.a[...] = v
        self._stp.a[0] = s + 2
        return self.ui

    def snapshot(self, *args, **kwargs) -> dict:
        """Load from SHM into self.ui (NaN→None), return self.ui."""
        s1 = int(self._stp.a[0]); v = self._vec.a.copy(); s2 = int(self._stp.a[0])
        if s1 != s2 or (s2 % 2) != 0: v = self._vec.a.copy()  # best-effort
        self.ui = self._from_vec(v)
        return self.ui

    def close(self, *args, **kwargs) -> None:
        try: self._vec.close()
        except Exception: pass
        try: self._stp.close()
        except Exception: pass

    def _init_ui(self, *args, **kwargs) -> dict[str, float | None]:
        # Single source of truth for field names:
        ui = {}
        ui.update(FourierParams.ui_template(include_A0=False))
        ui.update(Orbits.ui_template())
        return ui

    def _to_vec(self, ui: dict) -> np.ndarray:
        v = np.empty(len(self._keys), dtype=np.float32)
        for i, k in enumerate(self._keys):
            x = ui.get(k, None)
            v[i] = np.nan if x is None else float(x)
        return v

    def _from_vec(self, v: np.ndarray) -> dict:
        d: dict[str, float | None] = {}
        for i, k in enumerate(self._keys):
            x = float(v[i])
            d[k] = None if np.isnan(x) else x
        if d.get("P") is not None: d["P"] = int(round(d["P"]))
        return d
