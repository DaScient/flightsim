# Flight Simulation & Guidance Library Specification

## Overview

`flight` is a Python library that provides a modular, extensible toolkit for rigid-body dynamics, orbital mechanics, guidance, navigation and control (GNC), and hardware-in-the-loop experimentation. It is designed for researchers, engineers and students working on spacecraft, drones or other vehicles that require high-fidelity simulation and control. The library is portable across CPU and GPU environments, with optional real-time visualization.

In addition to the Python library, the repo hosts a **lightweight, browser-ready 3D flight simulator** written in a single HTML file. This lets users experiment with 6-DOF flight dynamics instantly on GitHub Pages without installing anything.

---

## Features

### Core Dynamics (Python)
- **Six Degree-of-Freedom Integrator** with optional environmental forces and torques.
- **Orbital propagators** (two-body, J2).
- **Control & Estimation**: PID, LQR, B-dot detumble, MEKF.
- **GPU Acceleration** via PyTorch (CUDA or Apple MPS).
- **Streamlit Web UI** for interactive simulation and parameter sweeps.

### Web Flight Simulator (HTML/JS)
- Runs directly in the browser — no install required.
- **Physics-based 6-DOF aircraft dynamics** (quaternion attitude, gravity, aero lift/drag).
- **Controls**:  
  - Mouse/finger drag for pitch/roll/yaw.  
  - Keys for throttle, trim, camera, and reset.  
- **Rendering**: Raw WebGL with chase and free-orbit camera modes, infinite ground grid, HUD overlay.
- **Deployment**: Served via GitHub Pages under `/labs/flight3d/`.

---

## Quick Start (Web Flight Simulator)

### Run locally
1. Save `flight3d/index.html` to your repo (or your desktop).  
2. Open it directly in any modern browser:
   ```
   file:///path/to/flight3d/index.html
   ```

### Run on GitHub Pages
1. Copy the file into your repo:
   ```bash
   mkdir -p flight3d
   cp flight3d.html flight3d/index.html
   git add flight3d/index.html
   git commit -m "Add single-file Flight3D simulator"
   git push
   ```
2. Enable **Pages** in your repo settings (branch: `main`, folder: `/`).  
3. Visit:
   ```
   https://dascient.github.io/labs/flight3d/
   ```

---

## Examples and Tests (Python)

- `examples/run_6dof_demo.py` – Propagate a spacecraft in orbit.
- `examples/run_detumble.py` – Simulate magnetic detumble.
- `examples/run_mekf_demo.py` – Attitude estimation with MEKF.
- `examples/run_gpu_6dof_demo.py` – Run on GPU.
- `examples/run_streamlit_app.py` – Launch the Streamlit web UI.

Unit tests are provided under `tests/` for quaternion math, rigid-body integration, GPU backends, and the MEKF.

---

## Development Notes

* Written for Python 3.9+ using NumPy, Matplotlib, PyTorch, fastplotlib, and Streamlit.  
* GPU integrator automatically selects `cuda`, `mps` or `cpu`.  
* Streamlit app uses fastplotlib if available; otherwise falls back to Matplotlib 4.0.  
* Unit tests ensure numerical stability and correctness. CI can be configured for CPU and GPU.

---

Copyright (c) 2025 DaScient, LLC | flightsim Contributors
