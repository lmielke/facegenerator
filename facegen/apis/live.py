# facegen/apis/live.py
"""
Live Fourier sketch editor.
Run:  face live  [-x <points>] [-z <layers>|None]
Keys: ESC = quit, S = print current params (JSON)

Publishes vtxs/rotateds to SHM ("face_vtx") if OutlineModel.enable_shared exists.
"""

from __future__ import annotations
import cv2
import math
from facegen.drawing import Renderer, SliderPanel, Pacer
from facegen.sketcher import FourierParams, OutlineModel, Orbits
from facegen.extruder import Extruder, StackSpec, ExtrusionMorpher


class LiveApp:
    """Read UI → compute sketch → render 2D → (optional) extrude + render 3D) → keys."""

    # ---------- declarative window config ----------
    # Logical key → params for title, position, size, companions, and multi-monitor placement
    window_params: dict[str, dict] = {
        "sketch_view": {
            "title": "facegen: view",
            "pos":   (50, 50),
            "size":  None,  # (w,h) or None to use Renderer defaults
            "move":  [("controls", (50 + 1000 + 20, 50))],   # companions
            # no cross-display behavior here (main view defines the anchor)
        },
        "extrude_view": {
            "title": "facegen: extrude",
            "pos":   (50, 50 + 1020),  # will be relocated to other display (same local pos)
            "size":  (800, 480),       # smaller fits most setups
            "move":  [("morph", (50 + 1000 + 20, 50 + 460))],
            "same_pos_on_other_display_as": "sketch_view",
        },
        "controls": {"title": "facegen: controls"},
        "morph":    {"title": "facegen: morph"},
    }
    # -----------------------------------------------

    def __init__(self, *args, num_layers: int | None = None, **kwargs) -> None:
        # timing + controls
        self.pacer = Pacer(*args, **kwargs)              # Pacer owns fps default
        self.controls = SliderPanel(*args, **kwargs)

        # model
        self.model = OutlineModel(*args, **kwargs)

        # SHM publish (best-effort)
        self.setup_shm(*args, **kwargs)

        # renderers
        self.renderer: Renderer | None = None
        self.z_renderer: Renderer | None = None

        # extrusion setup
        self.num_layers = num_layers
        self.extruder: Extruder | None = None
        self.stack_spec: StackSpec | None = None
        self.morph: ExtrusionMorpher | None = None
        self._last_grid_shape: tuple[int, int, int] | None = None
        self.anim_enabled: bool = True
        self._spin = [0.0, 0.0, 0.0]           # yaw, pitch, roll offsets
        self._dspin = [0.01, 0.02, 0.03]       # per-frame deltas (rad)

        # create windows (2D always; 3D only when extruding)
        self.renderer = self.setup_renderer(*args, name="sketch_view", **kwargs)
        self.initialize(*args, **kwargs)

    # ---------- initialization ----------
    def initialize(self, *args, morph_ui: bool = False, **kwargs) -> None:
        if not self.num_layers:
            return
        # Create morph first so placement mapping can move it
        if morph_ui:
            try:
                self.morph = ExtrusionMorpher(*args, name=self._title_for("morph"), **kwargs)
            except Exception:
                self.morph = None

        # Create extrude window and place it (potentially on other display)
        self.z_renderer = self.setup_renderer(*args, name="extrude_view", **kwargs)

        self.extruder = Extruder(*args, outline=None, **kwargs)
        self.stack_spec = StackSpec(M=int(self.num_layers))

    def setup_shm(self, *args, shm_name: str = 'face_vtx', **kwargs) -> None:
        if hasattr(self.model, "enable_shared"):
            try:
                self.model.enable_shared(*args, name=shm_name, **kwargs)
            except Exception:
                pass

    # ---------- unified renderer setup ----------
    def setup_renderer(self, *args, name: str, **kwargs) -> Renderer:
        """
        Use window_params[name] to create/move a window, move its companions,
        and (optionally) place it on another display at the same local pos.
        """
        cfg = dict(self.window_params.get(name, {}))
        title = cfg.get("title", name)
        pos   = cfg.get("pos", (50, 50))
        size  = cfg.get("size", None)
        move  = cfg.get("move", [])
        other_anchor = cfg.get("same_pos_on_other_display_as", None)

        # Instantiate renderer; pass size if provided
        if size and isinstance(size, tuple) and len(size) == 2:
            r = Renderer(*args, name=title, w=int(size[0]), h=int(size[1]), **kwargs)
        else:
            r = Renderer(*args, name=title, **kwargs)

        # Place the window
        try:
            cv2.moveWindow(title, *pos)
        except Exception:
            pass

        # If configured, mirror position to "other" display based on anchor window
        if other_anchor:
            anchor_title = self._title_for(other_anchor)
            self._move_same_spot_to_other_display(src_window=anchor_title, dst_window=title)

        # Move companions
        for logical, (x, y) in move:
            win_title = self._title_for(logical)
            # dynamic instances: controls/morph have real names
            if logical == "controls":
                try: cv2.moveWindow(self.controls.name, x, y)
                except Exception: pass
            elif logical == "morph" and self.morph is not None:
                try: cv2.moveWindow(self.morph.name, x, y)
                except Exception: pass
            else:
                try: cv2.moveWindow(win_title, x, y)
                except Exception: pass

        # Register renderer handle on the instance for known views
        if name == "sketch_view":
            self.renderer = r
        elif name == "extrude_view":
            self.z_renderer = r
        else:
            setattr(self, f"{name}_renderer", r)

        return r

    # ---------- per-frame rendering ----------
    def _render_outline_frame(self, *args, params: FourierParams, **kwargs) -> None:
        if not self.renderer:
            return
        self.renderer.render(*args, outline=self.model.rotateds,
                             normalize=True, params=params, **kwargs)

    def _render_extrusion_frame(self, *args, **kwargs) -> None:
        if not (self.num_layers and self.extruder and self.stack_spec and self.z_renderer):
            return
        # feed unrotated 2D outline; view rotation is applied in the renderer
        self.extruder.base = self.model.vtxs.copy()
        grid = self.extruder.stack(*args, spec=self.stack_spec, morph=self.morph, **kwargs)
        yaw, pitch, roll = map(float, self.model.cumRads)
        # before calling render_grid
        base_yaw, base_pitch, base_roll = map(float, self.model.cumRads)
        yaw   = base_yaw   + (self._spin[0] if self.anim_enabled else 0.0)
        pitch = base_pitch + (self._spin[1] if self.anim_enabled else 0.0)
        roll  = base_roll  + (self._spin[2] if self.anim_enabled else 0.0)

        self.z_renderer.render_grid(*args, grid=grid, normalize=True, yaw=yaw, pitch=pitch, roll=roll, **kwargs )
        shp = tuple(grid.shape)
        if shp != self._last_grid_shape:
            self._last_grid_shape = shp

    # ---------- loop ----------
    def run(self, *args, **kwargs) -> None:
        while True:
            ui = self.controls.read(*args, **kwargs)
            self.model.update_orbit(Orbits.from_ui(ui))
            params = FourierParams.from_ui(self.controls.ui, *args, **kwargs)
            self.model.compute(*args, params=params, oversample=120, **kwargs)

            self._render_outline_frame(*args, params=params, **kwargs)
            if self.num_layers:
                self._render_extrusion_frame(*args, **kwargs)

            k = self.pacer.poll_key(*args, **kwargs)
            if k == 27:
                break
            if k in (ord('s'), ord('S')):
                print(params.to_json(*args, **kwargs))
            elif k in (ord('a'), ord('A')):        # toggle animation
                self.anim_enabled = not self.anim_enabled
            elif k in (ord('r'), ord('R')):        # reset spin
                self._spin[:] = [0.0, 0.0, 0.0]


    def teardown(self, *args, **kwargs) -> None:
        try:
            if self.renderer: self.renderer.teardown(*args, **kwargs)
        except Exception:
            pass
        try:
            if self.z_renderer: self.z_renderer.teardown(*args, **kwargs)
        except Exception:
            pass
        try:
            self.controls.teardown(*args, **kwargs)
        except Exception:
            pass
        try:
            if self.morph is not None:
                cv2.destroyWindow(self.morph.name)
        except Exception:
            pass

    # ---------- helpers ----------
    def _title_for(self, logical: str) -> str:
        """Resolve a logical key to the actual OpenCV window title."""
        cfg = self.window_params.get(logical, {})
        # dynamic instances override the static title
        if logical == "controls":
            return getattr(self.controls, "name", cfg.get("title", logical))
        if logical == "morph" and self.morph is not None:
            return getattr(self.morph, "name", cfg.get("title", logical))
        return cfg.get("title", logical)

    # Multi-monitor (Windows; best-effort/no-op elsewhere)
    def _enum_monitors_windows(self):
        try:
            import ctypes
            user32 = ctypes.windll.user32

            class RECT(ctypes.Structure):
                _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                            ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

            class MONITORINFOEX(ctypes.Structure):
                _fields_ = [
                    ("cbSize", ctypes.c_ulong),
                    ("rcMonitor", RECT),
                    ("rcWork", RECT),
                    ("dwFlags", ctypes.c_ulong),
                    ("szDevice", ctypes.c_wchar * 32),
                ]

            MONITORENUMPROC = ctypes.WINFUNCTYPE(
                ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.POINTER(RECT), ctypes.c_double
            )

            mons = []

            def _cb(hMon, hdc, lprc, data):
                mi = MONITORINFOEX()
                mi.cbSize = ctypes.sizeof(MONITORINFOEX)
                user32.GetMonitorInfoW(hMon, ctypes.byref(mi))
                r = mi.rcMonitor
                mons.append({
                    "left":   int(r.left),  "top":    int(r.top),
                    "right":  int(r.right), "bottom": int(r.bottom),
                    "width":  int(r.right - r.left),
                    "height": int(r.bottom - r.top),
                    "hMon":   int(hMon),
                })
                return 1

            user32.EnumDisplayMonitors(0, 0, MONITORENUMPROC(_cb), 0)
            return mons
        except Exception:
            return []

    def _move_same_spot_to_other_display(self, *, src_window: str, dst_window: str) -> None:
        """
        Move dst_window to the same local (x,y) position but on the 'other' monitor
        relative to src_window's monitor. If only one monitor is found, no-op.
        """
        try:
            sx, sy, sw, sh = cv2.getWindowImageRect(src_window)
            mons = self._enum_monitors_windows()
            if len(mons) < 2:
                return

            def contains(m, x, y):
                return (m["left"] <= x < m["right"]) and (m["top"] <= y < m["bottom"])

            src_mon = next((m for m in mons if contains(m, sx, sy)), mons[0])
            other = next((m for m in mons if m is not src_mon), mons[-1])

            lx = sx - src_mon["left"]
            ly = sy - src_mon["top"]
            dx = other["left"] + lx
            dy = other["top"] + ly
            cv2.moveWindow(dst_window, int(dx), int(dy))
        except Exception:
            pass

    def _advance_spin(self) -> None:
        """Accumulate small spin offsets (kept within 2π)."""
        if not self.anim_enabled:
            return
        twopi = 2.0 * math.pi
        for i in range(3):
            self._spin[i] = (self._spin[i] + self._dspin[i]) % twopi


def main(*args, **kwargs):
    app = LiveApp(*args, **kwargs)
    try:
        app.run(*args, **kwargs)
    finally:
        app.teardown(*args, **kwargs)
    return "live: closed"
