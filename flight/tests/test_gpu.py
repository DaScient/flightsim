"""
Simple tests for the GPU integrator.

These tests do not assert physical correctness but check that the
GPU functions can be imported and invoked, that they return the
expected shapes, and that the output stays finite.  They are
automatically skipped if PyTorch is not installed, making the suite
safe to run on CI without GPUs.
"""

import pytest

try:
    import torch
except ImportError:
    torch = None  # type: ignore


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_device_selection():
    from flight.gpu import device_for_acceleration

    dev = device_for_acceleration()
    assert isinstance(dev, torch.device)


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_step_shapes():
    from flight.gpu import rigid_body_6dof_step_torch, device_for_acceleration

    device = device_for_acceleration()
    # build a simple state on selected device
    r = torch.tensor([7.0e6, 0.0, 0.0], dtype=torch.float64, device=device)
    v = torch.tensor([0.0, 7.5e3, 0.0], dtype=torch.float64, device=device)
    q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64, device=device)
    w = torch.tensor([0.0, 0.0, 0.01], dtype=torch.float64, device=device)
    state = (r, v, q, w)
    # inertia matrix and params
    J = torch.diag(torch.tensor([0.12, 0.10, 0.20], dtype=torch.float64, device=device))
    params = {"mass": 10.0, "J": J, "mu": 3.986004418e14}
    u = torch.zeros(3, dtype=torch.float64, device=device)
    new_state = rigid_body_6dof_step_torch(state, u, params, dt=1.0, device=device)
    assert isinstance(new_state, tuple)
    assert len(new_state) == 4
    r_new, v_new, q_new, w_new = new_state
    # check shapes
    assert r_new.shape == (3,)
    assert v_new.shape == (3,)
    assert q_new.shape == (4,)
    assert w_new.shape == (3,)
    # check finiteness
    for tensor in new_state:
        assert torch.isfinite(tensor).all()