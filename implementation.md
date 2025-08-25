
Functions remain **short (≤6 lines)** and typed where possible.

---

## gpu/sketcher.py  (merged Fourier + Envelope)

**Module purpose (docstring-ready, 7 lines):**  
Implements a GPU **FourierSketcher** that evaluates circular‑biased Fourier profiles
and derives a **robust outer outline** suitable for extrusion. Coefficients are
user‑controlled (`A0, m, {k,A,phi}`) and evaluated with **CuPy**. The
outer envelope is computed with a **ray–segment** CUDA kernel: one thread per
angle bin scans all poly segments and selects the farthest valid hit, guaranteeing
a gap‑free, angle‑ordered outline—even for self‑overlapping (“S‑curve”) profiles.
The API returns contiguous `(N,2)` `float32` on GPU (and NumPy on request).

**Classes (docstring bullets):**
- `Harm`: tiny dataclass holding one harmonic `{k:int, A:float, phi:float}`.
- `AngleBins`: precomputes `B` evenly spaced angles and their `(cos,sin)` on GPU.
- `GPUEnvelopeKernel`: owns the compiled `RawKernel` for ray–segment envelope.
- `FourierEval`: static helpers to evaluate `r(θ)` and XY via CuPy.
- `FourierSketcher`: main façade; exposes `outline(psi=0.0)` → `(B,2)` outer edge.

**FourierSketcher responsibilities:**
1) Validate sizes (Nyquist: `N ≥ 2*m*k_max`), clamp ripple ≤ 0.3·`A0` (optional).  
2) Allocate GPU buffers once: profile XY `(N,2)`, outline `(B,2)`.  
3) Evaluate Fourier XY (vectorized CuPy).  
4) Run `GPUEnvelopeKernel` to get outer outline (angle‑ordered).  
5) Return GPU/CPU arrays as needed (for streaming or debug).

---

## gpu/extrude.py

**Module purpose (docstring-ready, ~6 lines):**  
Transforms a 2D outline into a `(M,N,3)` surface grid by **stacking** along Z or
**partial revolving** around Y. Includes in‑place **twist**, **profile spin**, and
**scale‑along** kernels parameterized by layer index. Designed for fixed sizes and
resident GPU buffers; indices are built once on CPU. Produces positions (and UV)
ready for normals and rendering.

**Classes:**
- `Extruder`: builds `(M,N,3)` grid from outline; owns twist/scale ops.  
- `IndexStrip`: CPU helper to build/hold reusable triangle indices.

---

## gpu/normals.py

**Module purpose (docstring-ready):**  
Computes per‑vertex **normals** and optional **metric** (`t_u,t_v`) via central
differences on the `(M,N,3)` grid, with wrap in `u` only. Results support lighting,
feature placement, and curvature‑aware rules. Implemented as CuPy elementwise
kernels; outputs `(M,N,3)` `float32`.

**Classes:**  
- `GridNormals`: methods `compute_normals()`, `compute_metric()`.

---

## gpu/dynamics.py

**Module purpose (docstring-ready):**  
Simulates UV motif centers with **grid‑hash neighbor search** for O(F) steps.
Forces: gravity (v‑down), magnetism (poles), spacing (capped inverse‑square),
and linear drag. Integrator: **semi‑implicit Euler** with wrap/clamp of `(u,v)`.
Designed to run per‑frame on GPU; emits updated centers and optional preview mask.

**Classes:**  
- `CellHash`: GPU binning of feature centers.  
- `UVDynamics`: `step(dt)` applies forces + integration.

---

## debug/cv_vis.py  (OpenCV debug & tests)

**Module purpose (docstring-ready):**  
Lightweight **OpenCV** utilities to visualize backend geometry without the browser.
Renders: Fourier profile, outer outline overlay, UV masks, and Z‑slices of the 3D
grid. Includes mouse interactivity (zoom/pan) and FPS text to support profiling.
These tools are for development and unit‑style visual checks.

**Functions (examples):**
- `show_xy(profile_xy, outline_xy)` → window with overlay + zoom/pan.  
- `show_uv_mask(mask)` → inspect feature stamping.  
- `show_slice(grid_xyz, i)` → any Z‑layer slice view.

---

## Backend: WebSocket API (Flask‑Sock)

Messages (JSON initially; Phase 2: binary frames):
- **Params →** `{type:"params", data:{A0,m,terms,H,M,twist,...}}`
- **Mesh ←** `{type:"mesh", pos_b64:"...", idx_b64:"...", n:int, m:int}`
- **Sketch2D ←** `{type:"sketch2d", xy_b64:"...", n:int}`
- **Freeze →** `{type:"action", data:"freeze"}` → writes OBJ/STL + TOML → `{type:"saved", files:[...]}`

---

## Frontend (three.js SPA)

Three panes: Fourier (2D), Extrude (2D), 3D Mesh (OrbitControls).  
UI: harmonics, twist/scale, feature tools, dynamics sliders; buttons: Shuffle / Start‑Pause / Freeze.

---

## Milestones (unchanged, condensed)

M0 CuPy OK + WS echo → M1 Fourier→outline → M2 Extrude→mesh →  
M3 Twist/Scale/Spin → M4 UV features (preview) → M5 Dynamics → M6 Freeze/export → M7 binary WS & polish.

---

## Notes

- Fixed sizes at startup; restart to change them.  
- GTX 1660 Ti is sufficient for preview; export may run heavier CPU CSG.  
- Keep methods ≤6 lines where possible; prefer pure, preallocated kernels.
"""

Path("/mnt/data/implementation.md").write_text(content, encoding="utf-8")
print("implementation.md written")
