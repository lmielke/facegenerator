# sketch.py
"""
API endpoint: face sketch
- Named preset: stack n copies
- Random preset: generate n independent outlines (fresh params per item)
Example:
    face sketch -s random -n 3 -x 1023
"""

import cupy as cp
import numpy as np
from facegen.sketcher import FourierSketcher
from facegen.drawing import Renderer


def _mk_random_batch(n: int, x: int, *args, **kwargs) -> cp.ndarray:
    """Build (n, x, 3) where each item is randomized independently."""
    arrs = []
    for _ in range(n):
        sk = FourierSketcher(*args, sketch_name="random", num_objects=1, num_points=x, **kwargs)
        arrs.append(sk.outline_single())
    return cp.stack(arrs, axis=0)


def run_sketch(*args, sketch_name: str = "flower", num_objects: int = 1, num_points: int = 255, **kwargs):
    """
    Returns summary string and shows a timed sequence (10s each).
    """
    n, x = int(num_objects), int(num_points)

    if (sketch_name or "").lower() == "random":
        batch = _mk_random_batch(n, x, *args, **kwargs)                   # (n, x, 3)
    else:
        sk = FourierSketcher(*args, sketch_name=sketch_name, num_objects=n, num_points=x, **kwargs)
        batch = sk.batch_outlines(n)                                      # (n, x, 3)

    # visualize n outlines for 10s each
    renderer = Renderer(title=f"facegen: {sketch_name}")
    cpu_list = [cp.asnumpy(batch[i]) for i in range(batch.shape[0])]
    renderer.render_sequence(cpu_list, seconds_per_item=3.0)
    renderer.close()

    # compact summary
    sample = np.asarray(cpu_list[0][:5])
    return f"Sketch '{sketch_name}' → batch {batch.shape}\nSample[0,:5]:\n{sample}"


def main(*args, **kwargs):
    return run_sketch(*args, **kwargs)
