"""Tests for FlowZero conditional flow matching."""

from __future__ import annotations

import numpy as np

from flowzero import (
    CFMPath,
    FlowMatchingEngine,
    MLPVectorField,
    cfm_loss,
    odeint_euler,
    odeint_midpoint,
    sample_gaussian_mixture,
    train_cfm_step,
)


def test_ot_path_endpoints():
    path = CFMPath()
    x0 = np.zeros((4, 2))
    x1 = np.ones((4, 2))
    t0 = np.zeros((4, 1))
    t1 = np.ones((4, 1))
    xt0, ut = path.sample_xt(x0, x1, t0)
    xt1, _ = path.sample_xt(x0, x1, t1)
    assert np.allclose(xt0, x0)
    assert np.allclose(xt1, x1)
    assert np.allclose(ut, x1 - x0)


def test_mlp_shape():
    m = MLPVectorField(dim=3, hidden=16)
    x = np.random.randn(5, 3)
    t = np.random.rand(5, 1)
    y = m(t, x)
    assert y.shape == (5, 3)


def test_cfm_loss_decreases():
    rng = np.random.default_rng(0)
    dim = 2
    model = MLPVectorField(dim, hidden=32, seed=1)
    data = sample_gaussian_mixture(128, dim=dim, seed=2)
    losses = []
    for _ in range(30):
        idx = rng.integers(0, len(data), size=32)
        x1 = data[idx]
        x0 = rng.normal(size=x1.shape)
        losses.append(train_cfm_step(model, x0, x1, lr=3e-2, rng=rng))
    assert losses[-1] < losses[0] * 0.9 or losses[-1] < losses[0]


def test_ode_integrators_constant_field():
    def v(t, x):
        return np.ones_like(x)

    x0 = np.zeros((3, 2))
    xe = odeint_euler(v, x0, 0.0, 1.0, n_steps=20)
    xm = odeint_midpoint(v, x0, 0.0, 1.0, n_steps=20)
    assert np.allclose(xe, 1.0, atol=1e-6)
    assert np.allclose(xm, 1.0, atol=1e-6)


def test_engine_fit_sample():
    eng = FlowMatchingEngine(dim=2, hidden=32, seed=0)
    data = sample_gaussian_mixture(200, dim=2, seed=3)
    losses = eng.fit(data, steps=40, batch_size=32, lr=2e-2)
    samples = eng.sample(n=32, n_steps=30)
    assert samples.shape == (32, 2)
    assert np.all(np.isfinite(samples))
    assert np.isfinite(losses[-1])


def test_gaussian_path_shapes():
    path = CFMPath()
    x1 = np.random.randn(10, 4)
    t = np.random.rand(10)
    xt, ut, x0 = path.sample_xt_gaussian(x1, t, rng=np.random.default_rng(0))
    assert xt.shape == x1.shape
    assert ut.shape == x1.shape
    assert x0.shape == x1.shape
