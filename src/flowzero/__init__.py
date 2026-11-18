"""FlowZero: continuous normalizing flows via conditional flow matching (CFM).

Implements:
  - Gaussian conditional probability paths (Lipman et al. / Tong et al. CFM)
  - Exact OT displacement interpolation path
  - MLP vector field regression target
  - Euler / midpoint ODE integration for sampling
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np


@dataclass
class CFMPath:
    """Conditional flow-matching path between x0 ~ p0 and x1 ~ p1."""

    sigma_min: float = 1e-4

    def sample_xt(
        self, x0: np.ndarray, x1: np.ndarray, t: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """OT path: x_t = (1-t) x0 + t x1; u_t = x1 - x0."""
        t = np.asarray(t, dtype=float)
        if t.ndim == 1:
            t = t.reshape(-1, *([1] * (x0.ndim - 1)))
        xt = (1.0 - t) * x0 + t * x1
        ut = x1 - x0
        return xt, ut

    def sample_xt_gaussian(
        self, x1: np.ndarray, t: np.ndarray, rng: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gaussian conditional path toward data x1 from N(0,I)."""
        rng = rng or np.random.default_rng()
        x0 = rng.normal(size=x1.shape)
        t = np.asarray(t, dtype=float).reshape(-1, *([1] * (x1.ndim - 1)))
        mu = t * x1
        sigma = 1.0 - (1.0 - self.sigma_min) * t
        xt = mu + sigma * x0
        # target vector field for CFM
        ut = (x1 - (1.0 - self.sigma_min) * xt) / (sigma + 1e-12)
        return xt, ut, x0


class MLPVectorField:
    """Tiny 2-layer MLP v_theta(t, x) with tanh activations (numpy)."""

    def __init__(self, dim: int, hidden: int = 64, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.dim = dim
        # input: x (dim) + t (1)
        self.W1 = rng.normal(0, 0.1, size=(hidden, dim + 1))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 0.1, size=(dim, hidden))
        self.b2 = np.zeros(dim)

    def __call__(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        x = np.asarray(x, dtype=float)
        if t.ndim == 0:
            t = np.full((x.shape[0], 1), float(t))
        elif t.ndim == 1:
            t = t.reshape(-1, 1)
        h = np.tanh(x @ self.W1[:, : self.dim].T + t @ self.W1[:, self.dim :].T + self.b1)
        return h @ self.W2.T + self.b2

    def parameters(self) -> list[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2]


def cfm_loss(
    model: MLPVectorField,
    x0: np.ndarray,
    x1: np.ndarray,
    path: Optional[CFMPath] = None,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Mean squared error between model vector field and OT target u_t."""
    path = path or CFMPath()
    rng = rng or np.random.default_rng()
    n = x0.shape[0]
    t = rng.uniform(0.0, 1.0, size=(n, 1))
    xt, ut = path.sample_xt(x0, x1, t)
    pred = model(t, xt)
    return float(np.mean((pred - ut) ** 2))


def train_cfm_step(
    model: MLPVectorField,
    x0: np.ndarray,
    x1: np.ndarray,
    lr: float = 1e-2,
    path: Optional[CFMPath] = None,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """One SGD step on CFM loss via finite-difference / analytic small MLP grads."""
    path = path or CFMPath()
    rng = rng or np.random.default_rng()
    n = x0.shape[0]
    t = rng.uniform(0.0, 1.0, size=(n, 1))
    xt, ut = path.sample_xt(x0, x1, t)

    # Forward
    tin = np.hstack([xt, t])
    z1 = tin @ model.W1.T + model.b1
    h = np.tanh(z1)
    pred = h @ model.W2.T + model.b2
    err = pred - ut
    loss = float(np.mean(err**2))

    # Backward
    dpred = (2.0 / err.size) * err  # (n, dim)
    dW2 = dpred.T @ h
    db2 = dpred.sum(axis=0)
    dh = dpred @ model.W2
    dz1 = dh * (1.0 - h**2)
    dW1 = dz1.T @ tin
    db1 = dz1.sum(axis=0)

    model.W2 -= lr * dW2
    model.b2 -= lr * db2
    model.W1 -= lr * dW1
    model.b1 -= lr * db1
    return loss


def odeint_euler(
    v: Callable[[float, np.ndarray], np.ndarray],
    x0: np.ndarray,
    t0: float = 0.0,
    t1: float = 1.0,
    n_steps: int = 50,
) -> np.ndarray:
    x = np.asarray(x0, dtype=float).copy()
    dt = (t1 - t0) / n_steps
    t = t0
    for _ in range(n_steps):
        x = x + dt * v(t, x)
        t += dt
    return x


def odeint_midpoint(
    v: Callable[[float, np.ndarray], np.ndarray],
    x0: np.ndarray,
    t0: float = 0.0,
    t1: float = 1.0,
    n_steps: int = 50,
) -> np.ndarray:
    x = np.asarray(x0, dtype=float).copy()
    dt = (t1 - t0) / n_steps
    t = t0
    for _ in range(n_steps):
        k1 = v(t, x)
        k2 = v(t + 0.5 * dt, x + 0.5 * dt * k1)
        x = x + dt * k2
        t += dt
    return x


def sample_gaussian_mixture(
    n: int, dim: int = 2, seed: int = 0
) -> np.ndarray:
    """Two-mode Gaussian mixture target."""
    rng = np.random.default_rng(seed)
    half = n // 2
    a = rng.normal(size=(half, dim)) * 0.4 + np.array([1.5] + [0.0] * (dim - 1))
    b = rng.normal(size=(n - half, dim)) * 0.4 + np.array([-1.5] + [0.0] * (dim - 1))
    return np.vstack([a, b])


class FlowMatchingEngine:
    """High-level CFM train + sample API."""

    def __init__(self, dim: int = 2, hidden: int = 64, seed: int = 0):
        self.dim = dim
        self.model = MLPVectorField(dim, hidden=hidden, seed=seed)
        self.path = CFMPath()
        self.rng = np.random.default_rng(seed)

    def fit(
        self,
        data: np.ndarray,
        steps: int = 200,
        batch_size: int = 64,
        lr: float = 2e-2,
    ) -> list[float]:
        data = np.asarray(data, dtype=float)
        losses = []
        n = data.shape[0]
        for _ in range(steps):
            idx = self.rng.integers(0, n, size=batch_size)
            x1 = data[idx]
            x0 = self.rng.normal(size=x1.shape)
            losses.append(train_cfm_step(self.model, x0, x1, lr=lr, path=self.path, rng=self.rng))
        return losses

    def sample(self, n: int = 64, n_steps: int = 40, method: str = "midpoint") -> np.ndarray:
        x0 = self.rng.normal(size=(n, self.dim))

        def v(t, x):
            tt = np.full((x.shape[0], 1), float(t))
            return self.model(tt, x)

        if method == "euler":
            return odeint_euler(v, x0, 0.0, 1.0, n_steps)
        return odeint_midpoint(v, x0, 0.0, 1.0, n_steps)
