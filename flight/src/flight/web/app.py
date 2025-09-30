"""
Streamlit application for the flight library.

Run this app via:

```
streamlit run flight/web/app.py
```

The app allows users to configure a six‑degree‑of‑freedom simulation,
select CPU or GPU integration, and visualise the altitude over time.
It demonstrates how to build interactive dashboards on top of the
``flight`` library without writing any GUI code yourself.
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from flight.dynamics.rigid_body import rigid_body_6dof_step, MU_EARTH, R_EARTH, J2_EARTH

try:
    from ..gpu import rigid_body_6dof_step_torch, device_for_acceleration
    import torch  # type: ignore
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False


def run_simulation_cpu(params: dict, state: tuple, dt: float, t_final: float) -> tuple:
    times = np.arange(0.0, t_final + dt, dt)
    r_hist = []
    for _ in times:
        state = rigid_body_6dof_step(state, np.zeros(3), params, dt)
        r_hist.append(state[0])
    return times, np.array(r_hist)


def run_simulation_gpu(params: dict, state: tuple, dt: float, t_final: float) -> tuple:
    # Convert numpy arrays to torch tensors on the appropriate device
    device = device_for_acceleration()
    r, v, q, w = state
    r_t = torch.tensor(r, dtype=torch.float64, device=device)
    v_t = torch.tensor(v, dtype=torch.float64, device=device)
    q_t = torch.tensor(q, dtype=torch.float64, device=device)
    w_t = torch.tensor(w, dtype=torch.float64, device=device)
    state_t = (r_t, v_t, q_t, w_t)
    # Convert inertia matrix
    params_t = params.copy()
    params_t["J"] = torch.tensor(params["J"], dtype=torch.float64, device=device)
    times = np.arange(0.0, t_final + dt, dt)
    r_hist = []
    for _ in times:
        state_t = rigid_body_6dof_step_torch(state_t, torch.zeros(3, dtype=torch.float64, device=device), params_t, dt)
        r_hist.append(state_t[0].cpu().numpy())
    return times, np.array(r_hist)


def main() -> None:
    st.set_page_config(page_title="Flight 6‑DOF Simulator", layout="wide")
    st.title("Flight 6‑DOF Simulator")

    st.sidebar.header("Simulation Parameters")
    dt = st.sidebar.number_input("Time step (s)", 0.01, 10.0, 0.1, step=0.01)
    t_final = st.sidebar.number_input("Simulation duration (s)", 1.0, 3600.0, 300.0, step=1.0)
    mass = st.sidebar.number_input("Mass (kg)", 0.1, 1000.0, 10.0, step=0.1)
    Jx = st.sidebar.number_input("Inertia Jx (kg·m²)", 0.001, 10.0, 0.12, step=0.001)
    Jy = st.sidebar.number_input("Inertia Jy (kg·m²)", 0.001, 10.0, 0.10, step=0.001)
    Jz = st.sidebar.number_input("Inertia Jz (kg·m²)", 0.001, 10.0, 0.20, step=0.001)
    # Flags
    use_j2 = st.sidebar.checkbox("Enable J2 perturbation", value=True)
    use_drag = st.sidebar.checkbox("Enable atmospheric drag", value=False)
    use_srp = st.sidebar.checkbox("Enable solar radiation pressure", value=False)
    use_gpu = st.sidebar.checkbox("Use GPU (if available)", value=GPU_AVAILABLE)

    st.sidebar.header("Initial State")
    r0_km = st.sidebar.number_input("Initial altitude above Earth surface (km)", 100.0, 35786.0, 500.0, step=10.0)
    inclination = st.sidebar.slider("Orbital inclination (deg)", 0.0, 90.0, 0.0, step=0.1)
    # Compute initial state
    # Circular orbit with altitude r0_km and inclination
    a = (R_EARTH + r0_km * 1000.0)
    # initial position and velocity in ECI
    r0 = np.array([a, 0.0, 0.0])
    v0_mag = np.sqrt(MU_EARTH / a)
    v0 = np.array([0.0, v0_mag, 0.0])
    # rotate position and velocity by inclination about x-axis
    inc_rad = np.radians(inclination)
    R_inc = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(inc_rad), -np.sin(inc_rad)],
        [0.0, np.sin(inc_rad), np.cos(inc_rad)],
    ])
    r0 = R_inc @ r0
    v0 = R_inc @ v0
    # Attitude quaternion identity
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    w0 = np.zeros(3)
    state0 = (r0, v0, q0, w0)

    # Build params dict
    params = {
        "mass": mass,
        "J": np.diag([Jx, Jy, Jz]),
        "mu": MU_EARTH,
        "J2": J2_EARTH if use_j2 else 0.0,
        "R_earth": R_EARTH,
        # Optional drag parameters
        "Cd": 2.2,
        "A_ref": 0.0,
        "rho": 0.0,
        # Optional SRP parameters
        "Cr": 1.8,
        "A_srp": 0.0,
    }

    if st.sidebar.button("Run Simulation"):
        if use_gpu and GPU_AVAILABLE:
            times, r_hist = run_simulation_gpu(params, state0, dt, t_final)
        else:
            times, r_hist = run_simulation_cpu(params, state0, dt, t_final)
        # Compute altitude (distance from Earth's centre minus radius)
        alt_km = np.linalg.norm(r_hist, axis=1) / 1000.0 - (R_EARTH / 1000.0)
        st.subheader("Altitude over time")
        fig, ax = plt.subplots()
        ax.plot(times / 60.0, alt_km)
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Altitude (km)")
        ax.grid(True)
        st.pyplot(fig)


if __name__ == "__main__":
    main()
