# flight

A modular, extensible library for six‑degree‑of‑freedom spacecraft simulation, guidance, navigation and control.

Flight provides a Python package for propagating rigid‑body dynamics with orbital perturbations, designing attitude controllers, running filters for state estimation, and even leveraging GPUs for acceleration.  In addition to the simulation library, it includes a simple web application (Streamlit) to visualise results interactively.

## Features

- **6‑DOF propagation** – Numerically integrate the translational and rotational motion of a rigid body subject to two‑body gravity, J2 perturbations, drag, solar radiation pressure and gravity‑gradient torques.
- **Attitude control** – Tune and apply PID and LQR controllers, as well as detumbling logic for magnetic torque rods.
- **State estimation** – A multiplicative extended Kalman filter (MEKF) skeleton for attitude determination with gyro bias tracking.
- **Orbit & relative motion tools** – Helpers for two‑body propagation, J2 acceleration, and Clohessy–Wiltshire relative motion for formation flying and station‑keeping.
- **GPU acceleration (optional)** – Offload the dynamics integration to Apple Silicon or CUDA devices using PyTorch.  The GPU integrator selects the best device (`mps`/`cuda`/`cpu`) automatically.
- **Streamlit front‑end** – A simple web UI lets you configure initial conditions, run a simulation and plot altitude vs time in your browser.  Great for demonstrations and quick experiments.

## Installation

### Prerequisites

* Python ≥ 3.8
* [NumPy](https://numpy.org/) for vector mathematics
* [Matplotlib](https://matplotlib.org/) for plotting
* [Streamlit](https://streamlit.io/) (optional) for the web UI
* [PyTorch](https://pytorch.org/) with MPS or CUDA support (optional) for GPU acceleration

### Create a virtual environment

We recommend using conda or Python’s built‑in venv.  The commands below show how to set up an environment and install flight in editable mode:

```bash
conda create -n flight python=3.11
conda activate flight
# install the package and its core dependencies
pip install -e .
# install Streamlit for the web app
pip install streamlit
# install PyTorch with MPS (for Apple Silicon) or CUDA (for NVIDIA)
pip install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cpu
# optionally install fastplotlib for GPU‑accelerated plotting
pip install "fastplotlib[notebook,imgui]"
```

## Running examples

Run the provided example scripts via the `PYTHONPATH=src` mechanism or install the package and call them directly:

```bash
PYTHONPATH=src python examples/run_6dof_demo.py          # Basic 6‑DOF simulation
PYTHONPATH=src python examples/run_detumble.py           # Detumble control example
PYTHONPATH=src python examples/run_mekf_demo.py          # MEKF convergence
PYTHONPATH=src python examples/run_gpu_6dof_demo.py      # GPU‑accelerated 6‑DOF demo
PYTHONPATH=src python examples/run_streamlit_app.py      # Launch the Streamlit UI
```

On macOS with Apple silicon, the GPU demo will detect and use the MPS backend automatically【329595555539941†L66-L83】.  If no GPU is available or PyTorch is not installed, it will fall back to CPU execution.  For interactive, high‑frame‑rate visualisation you can experiment with fastplotlib; this library uses the pygfx render engine and WGPU to achieve cross‑platform GPU acceleration【659221846865534†L56-L64】【273943724016156†L297-L334】.

## Project layout

* **src/flight** – The core Python package.  Subpackages include `dynamics`, `control`, `estimation`, `orbit`, `optimization`, `mathutils`, `gpu` and `web`.
* **examples/** – Stand‑alone scripts demonstrating various aspects of the library.
* **tests/** – A small suite of unit tests to exercise key functions.  Many tests are skipped gracefully if optional dependencies are missing.
* **docs/spec.md** – A comprehensive specification and design document for the library.

## License

This project is released under the MIT License; see the LICENSE file for details.
