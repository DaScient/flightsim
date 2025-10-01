# Flight Simulation & Guidance Library Specification

## Overview

`flight` is a Python library that provides a modular, extensible toolkit for rigid‑body dynamics, orbital mechanics, guidance, navigation and control (GNC), and hardware‑in‑the‑loop experimentation.  It is designed for researchers, engineers and students working on spacecraft, drones or other vehicles that require high‑fidelity simulation and control.  The library is designed to be easy to install, easy to integrate with external tools, and portable across CPU and GPU environments.

Unlike many academic codebases, `flight` treats simulation, control and estimation as **first‑class modules**.  The library exposes a clear API for each subsystem and comes with a suite of examples that run out of the box.  Optional GPU acceleration (via PyTorch and Apple’s Metal Performance Shaders) and an interactive Streamlit application allow users to explore 6‑DOF dynamics and controller behaviour in real time.

## [Use Chrome to Use This Simulator Test Flight Now!](https://dascient.github.io/flightsim/flight3d/)
## Features

### Core Dynamics

- **Six Degree‑of‑Freedom Integrator** – A coupled rigid‑body translational and rotational dynamic model that propagates position, velocity, quaternion attitude and body‑rate state.  A fourth‑order Runge–Kutta (RK4) integrator updates the full state at each time‑step.  Optional environmental forces and torques include:
  - **Two‑body gravitation** (central gravity).
  - **J2 Earth gravity perturbation** for higher‑fidelity orbit propagation.
  - **Atmospheric drag** via a cannonball model with configurable drag coefficient, reference area and density models.
  - **Solar radiation pressure** (SRP) with specifiable reflectivity coefficient and cross‑section.
  - **Gravity‑gradient torque**, **magnetic torque**, **aerodynamic torque** and **SRP torque** based on vehicle geometry and orientation.
  - User‑specified external forces and torques to model thruster impulses, reaction wheel disturbance or other control inputs.

### Guidance, Navigation & Control

- **Control modules**:
  - **PID** and **LQR** controllers for linearised attitude or relative motion problems.
  - Ready‑to‑use detumble (B‑dot) controller stub that demonstrates magnetic torque rod control.
  - Reaction wheel and magnetorquer models can be easily added by users.
- **Estimation modules**:
  - A minimal **Multiplicative Extended Kalman Filter (MEKF)** for attitude determination using gyroscopes and vector measurements (e.g. star tracker, magnetometer).  Users can supply measurement noise covariances and initial covariance.
  - Support for gyro bias estimation and covariance propagation.
- **Orbit & optimisation tools**:
  - Two‑body and J2 orbit propagators with state transition matrices.
  - **Clohessy–Wiltshire (Hill) equations** for relative motion and rendezvous guidance.
  - Templates for linear quadratic regulator (LQR) design for formation keeping.

### GPU Acceleration

- The `flight.gpu` subpackage leverages **PyTorch** to accelerate 6‑DOF dynamics on available GPUs.  On Apple Silicon devices, the library automatically selects the **Metal Performance Shaders (MPS)** backend.  On systems with CUDA, it will use CUDA.  A fallback to CPU is provided.  The GPU integrator shares the same API as the CPU version but operates on PyTorch tensors.

### Interactive Web Application

- A **Streamlit** application (`flight.web.app`) allows users to run simulations from a browser.  The app provides forms for selecting vehicle parameters, time‑step, environmental forces and controller settings.  Results are visualised using either Matplotlib or **fastplotlib**, a modern GPU‑accelerated plotting library built on the pygfx renderer【†L56-L64】【†L297-L334】.  The Streamlit app demonstrates real‑time simulation and plotting capabilities, making it ideal for education and prototyping.

### Examples and Tests

- The `examples/` directory contains ready‑to‑run scripts:
  - `run_6dof_demo.py` – Propagates a spacecraft in a circular Low Earth Orbit with optional environmental forces and prints altitude over time.
  - `run_detumble.py` – Demonstrates a B‑dot detumble algorithm using a simple magnetic field and magnetorquer control law.
  - `run_mekf_demo.py` – Simulates an inertial attitude drift with gyro bias and uses the MEKF to estimate the quaternion attitude from vector measurements.
    - `run_gpu_6dof_demo.py` – Runs the 6‑DOF simulation on a GPU (if available) and plots altitude versus time.
    - `run_streamlit_app.py` – Launches the Streamlit web UI by invoking the packaged app with the `streamlit` command.  This convenience wrapper ensures the correct Python interpreter is used and locates the `app.py` entry point via `importlib.resources`.
- The `tests/` directory includes unit tests for quaternion arithmetic, rigid‑body integration, and the GPU backend.  Tests skip gracefully if PyTorch is not installed.

## Package Structure

```
flight/
├── README.md               # Project introduction and installation instructions
├── spec.md                 # This specification document
├── pyproject.toml          # Package metadata and dependencies
├── examples/               # Example scripts for demos
├── src/flight/             # Source package
│   ├── __init__.py
│   ├── control/
│   │   ├── __init__.py
│   │   ├── pid.py          # PID controller implementation
│   │   └── lqr.py          # LQR regulator helper
│   ├── dynamics/
│   │   ├── __init__.py
│   │   └── rigid_body.py   # 6‑DOF integrator with optional forces/torques
│   ├── estimation/
│   │   ├── __init__.py
│   │   └── mekf.py         # Multiplicative extended Kalman filter for attitude
│   ├── mathutils/
│   │   ├── __init__.py
│   │   ├── quaternion.py   # Quaternion operations (norm, multiplication, rotation)
│   │   └── skew.py         # Skew‑symmetric matrix helper
│   ├── orbit/
│   │   ├── __init__.py
│   │   └── propagators.py  # Two‑body and J2 accelerations and state transition
│   ├── optimization/
│   │   ├── __init__.py
│   │   └── cw.py           # Clohessy–Wiltshire helpers for rendezvous
│   ├── gpu/
│   │   ├── __init__.py
│   │   └── torch_dynamics.py  # GPU‑accelerated 6‑DOF integrator using PyTorch
│   └── web/
│       ├── __init__.py
│       └── app.py          # Streamlit application entry point
└── tests/
    ├── test_quaternion.py
    ├── test_rigid_body.py
    ├── test_gpu.py
    └── test_mekf.py
```

## Installation

1. **Create a virtual environment** and activate it:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. **Install the package in editable mode** and its dependencies:

```bash
pip install -e .[dev]
```

3. **Install optional GPU support** (requires PyTorch):
   - On Apple Silicon (M‑series), install a build with Metal Performance Shaders:

     ```bash
     pip install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cpu
     ```

   - On systems with CUDA GPUs, install the CUDA‑enabled build from the official PyTorch index.

4. **Optional fastplotlib** (for interactive GPU plotting):

```bash
pip install "fastplotlib[notebook,imgui]"
```

5. **Run examples**:

```bash
python examples/run_6dof_demo.py
python examples/run_detumble.py
python examples/run_mekf_demo.py
python examples/run_gpu_6dof_demo.py
```

6. **Launch the Streamlit application**:

```bash
streamlit run src/flight/web/app.py
```

## Development Notes

* The code is written for Python 3.9+ and uses only the standard library plus optional third‑party dependencies (NumPy, Matplotlib, PyTorch, fastplotlib, and Streamlit).  NumPy is a required dependency; others are optional but highly recommended.
* The GPU integrator is designed to work with PyTorch’s device abstraction, automatically selecting `cuda`, `mps` or `cpu` devices.  Users must ensure that PyTorch is installed with the appropriate backend.
* The Streamlit application attempts to use fastplotlib for GPU rendering if available; otherwise it falls back to Matplotlib 4.0’s GPU‑accelerated mode【†L98-L100】.
* Unit tests are provided to ensure numerical stability and correctness.  Continuous integration can be configured to run on CPU and GPU environments.

## Citations

This library references open research and tools on GPU‑accelerated plotting and training.  Notable sources include:

- The `fastplotlib` documentation, which describes the library as a next‑generation plotting library built on pygfx and using Vulkan, DirectX 12 or Metal via WGPU.
- The PyTorch blog on accelerated training on Mac, which introduces the Metal Performance Shaders backend enabling GPU acceleration on Apple Silicon devices.
- The Matplotlib 4.0 release notes highlighting built‑in GPU rendering and a default dark theme.


Copyright (c) 2025 DaScient, LLC | flightsim Contributors
