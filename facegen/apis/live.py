# facegen/apis/live.py
"""
Live Fourier sketch editor (class-based, *args/**kwargs-friendly).
Run:  face live  [-x <points>] [--fps 60]
Keys: ESC = quit, S = print current params (JSON)
"""

import json, math, time
import cv2
import numpy as np
from typing import Iterable

from facegen.drawing import Renderer, SliderPanel, Pacer
from facegen.sketcher import FourierParams, Harm, Stabilizer, OutlineModel, Orbits


# ============================== App ==========================================
class LiveApp:
    """Orchestrator: read UI → build params → stabilize → compute → render → keys."""
    fps = 60

    def __init__(self, *args, **kwargs):
        self.pacer = Pacer(*args, fps=self.fps, **kwargs)
        self.controls = SliderPanel(*args, **kwargs)
        # place controls to the right of drawing window:
        try:
            cv2.moveWindow(self.controls.name, 50 + 1000 + 20, 50)
        except Exception:
            pass
        self.renderer = Renderer(*args, **kwargs)
        self.model = OutlineModel(*args, **kwargs)

    def run(self, *args, **kwargs):
        while True:
            ui = self.controls.read(*args, **kwargs)
            self.model.update_orbit(Orbits.from_ui(ui))
            params = FourierParams.from_ui(ui, *args, **kwargs)
            self.model.compute(*args, params=params, oversample=120, **kwargs)
            print(f"{self.model.rotateds.shape = }")
            self.renderer.render(*args, outline=self.model.rotateds, normalize=True, params=params, **kwargs)
            k = self.pacer.poll_key(*args, **kwargs)
            if k == 27:  # ESC
                break
            if k in (ord('s'), ord('S')):
                print(params.to_json(*args, **kwargs))

    def teardown(self, *args, **kwargs):
        self.renderer.teardown(*args, **kwargs)
        self.controls.teardown(*args, **kwargs)


# ============================== Entrypoints ==================================
def main(*args, **kwargs):
    app = LiveApp(*args, **kwargs)
    try:
        app.run(*args, **kwargs)
    finally:
        app.teardown(*args, **kwargs)
    return "live: closed"
