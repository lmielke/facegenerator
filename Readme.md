# facegenerator (GPU edition)
Facegenerator is a tool to generated 3d surfaces based on Fourier series sketches, with a focus on procedural generation and dynamic UV stamping. The resulting surfaces can be exported as 3d CAD files to 
be used in a CAD workflow or for 3D printing.
For example we want to 3d print a star shaped vase with a spiral pattern on it, or a wavy surface with a floral pattern on it. The surface might feature star shaped cutouts or holes as decorative elements.

This tool allowes you to 
1. generate base sketches using parametrized fourier series (e.g. a star, wave, spiral, etc.)
2. decide upon a 3d shaped extrude path (i.e. spiral, wave, etc.)
3. create a 3d surface based on the sketch and the extrude path
4. generate a mesh from the resulting surface
5. apply a UV stamping pattern to the mesh
6. View the resulting mesh in a live preview in the browser (zoom, rotate, pan)
7. export the mesh as a 3D CAD file (OBJ/STL)



# Goal
- Procedural 2D Fourier sketch → 3D surface (stack/revolve) on **GPU (CuPy)**
- UV feature stamping + simple dynamics (gravity/magnetism/spacing)
- Live preview in browser (three.js), export OBJ/STL on freeze

## Architecture
Backend (Python/CuPy/Flask)
- Fourier eval (CuPy), outer envelope (GPU ray–segment), extrude/warp (CuPy)
- UV dynamics (CuPy), preview UV mask
- WebSocket API → stream buffers (positions/indices/normals)
- Freeze → high-res pass → (optional CSG on CPU) → OBJ/STL

Frontend (three.js)
- 3 panes: XY sketch, Z extrude, 3D surface (OrbitControls)
- Controls: harmonics/sliders, twist/scale, feature tools, dynamics knobs
- Buttons: Shuffle, Start/Pause, Freeze & Save

## Data flow
UI params → WS → Python → CuPy recompute → buffers → WS → three.js render

## Fixed sizes (set at startup)
N_profile, M_layers, F_count, F_res. Change → restart.

## Run
- Backend: `pipenv run python -m facegen.apis.server`
- Frontend: `npm run dev` (opens `http://localhost:5173`)

## Export
- `freeze` → writes `./dist/<stamp>.{obj,stl}` + `<stamp>.toml` (coeffs + dims)

## Notes
- GPU: NVIDIA GTX 1660 Ti (CUDA 11.x). Intel iGPU unused.
