"""
sa.py
Simulated Annealing for swarm-fire problem.
Optimizes BOTH the control tensor U (T,N,2) and the time step Δt (sim.dt_var).

Key ideas:
- Decision vector z = [vec(U), dt_var]
- Neighbor: add Gaussian noise to a random subset of velocities AND small noise to dt_var
- Projection: enforce ||u|| <= vmax and clamp dt_var to [dt_min, dt_max]
- Cooling: geometric T = T0 * alpha^iter

All functions are NumPy-only so they run anywhere.
"""
from __future__ import annotations
import numpy as np
from typing import Callable, Dict, Tuple

Array = np.ndarray

def project_speed(U: Array, vmax: float) -> Array:
    norms = np.linalg.norm(U, axis=2, keepdims=True) + 1e-12
    scale = np.minimum(1.0, vmax / norms)
    return U * scale

def pack_U(U: Array) -> Array:
    return U.ravel()

def unpack_U(x: Array, T: int, N: int) -> Array:
    return x.reshape(T, N, 2)

def make_neighbor(
    U: Array,
    dt_var: float,
    sigma_u: float,
    frac: float,
    sigma_dt: float,
    dt_min: float,
    dt_max: float,
    vmax: float,
    rng: np.random.Generator
) -> Tuple[Array, float]:
    """
    Perturb a fraction of (k,i) velocity vectors with N(0, sigma_u^2 I2)
    and dt_var with N(0, sigma_dt^2). Then project to constraints.
    """
    T, N, _ = U.shape
    U_new = U.copy()
    K = max(1, int(frac * T * N))
    ks = rng.integers(0, T, size=K)
    isel = rng.integers(0, N, size=K)
    noise = rng.normal(scale=sigma_u, size=(K, 2))
    U_new[ks, isel, :] += noise
    U_new = project_speed(U_new, vmax)

    dt_new = dt_var + rng.normal(scale=sigma_dt)
    # clamp dt into bounds
    dt_new = float(np.clip(dt_new, dt_min, dt_max))
    return U_new, dt_new

def simulated_annealing(
    cost_fn: Callable[[Array, float], float],
    U0: Array,
    dt0: float,
    vmax: float,
    dt_min: float,
    dt_max: float,
    *,
    iters: int = 600,
    T0: float = 5.0,
    alpha: float = 0.96,
    frac: float = 0.07,
    sigma_u: float = 0.25,
    sigma_dt: float = 0.03,
    stall_check: int = 200,
    seed: int = 0
) -> Tuple[Array, float, Dict[str, float]]:
    """
    Run SA and return (U_best, dt_best, log).

    cost_fn(U, dt) must return a scalar fitness (lower is better).
    """
    rng = np.random.default_rng(seed)

    U_cur = project_speed(U0, vmax)
    dt_cur = float(np.clip(dt0, dt_min, dt_max))
    J_cur = float(cost_fn(U_cur, dt_cur))

    U_best, dt_best, J_best = U_cur.copy(), dt_cur, J_cur

    Ttemp = float(T0)
    accepts = 0
    improves = 0
    since_improve = 0

    for t in range(iters):
        # Propose neighbor
        U_prop, dt_prop = make_neighbor(
            U_cur, dt_cur, sigma_u, frac, sigma_dt, dt_min, dt_max, vmax, rng
        )
        J_prop = float(cost_fn(U_prop, dt_prop))
        dJ = J_prop - J_cur

        # Metropolis acceptance
        if dJ <= 0.0 or rng.random() < np.exp(-dJ / max(1e-12, Ttemp)):
            U_cur, dt_cur, J_cur = U_prop, dt_prop, J_prop
            accepts += 1
            if J_cur < J_best:
                U_best, dt_best, J_best = U_cur.copy(), dt_cur, J_cur
                improves += 1
                since_improve = 0
            else:
                since_improve += 1
        else:
            since_improve += 1

        # cool down
        Ttemp *= alpha

        if stall_check and since_improve >= stall_check:
            break

    log = {
        "best_cost": float(J_best),
        "iters": t + 1,
        "accepts": accepts,
        "improves": improves
    }
    return U_best, dt_best, log
