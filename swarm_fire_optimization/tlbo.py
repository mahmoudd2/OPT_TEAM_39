"""
tlbo.py
-------
Teaching–Learning Based Optimization (TLBO)
for swarm-fire problem.

Optimizes:
 - control tensor U (T, N, 2)
 - time step dt_var

This implementation includes a small exploration term to
avoid premature convergence in continuous search spaces.
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Dict, Tuple

Array = np.ndarray


# -------------------- Helper Functions -------------------- #

def project_speed(U: Array, vmax: float) -> Array:
    """Project velocities to satisfy ||u|| <= vmax."""
    norms = np.linalg.norm(U, axis=2, keepdims=True) + 1e-12
    scale = np.minimum(1.0, vmax / norms)
    return U * scale


def clamp_dt(dt: float, dt_min: float, dt_max: float) -> float:
    """Clamp dt into [dt_min, dt_max]."""
    return float(np.clip(dt, dt_min, dt_max))


# -------------------- TLBO Main Function -------------------- #

def teaching_learning_based_optimization(
    cost_fn: Callable[[Array, float], float],
    U0: Array,
    dt0: float,
    vmax: float,
    dt_min: float,
    dt_max: float,
    *,
    pop_size: int = 30,
    iters: int = 120,
    seed: int = 0,
) -> Tuple[Array, float, Dict[str, Array | float | int]]:
    """
    Run TLBO and return (U_best, dt_best, log).
    """

    rng = np.random.default_rng(seed)
    T, N, _ = U0.shape

    # ----------- Initialize Population ----------- #
    pop_U = []
    pop_dt = []

    for _ in range(pop_size):
        U = U0 + rng.normal(scale=0.1, size=U0.shape)
        U = project_speed(U, 0.8 * vmax)   # slightly conservative start
        dt = clamp_dt(dt0 + rng.normal(scale=0.05), dt_min, dt_max)
        pop_U.append(U)
        pop_dt.append(dt)

    fitness = np.array([cost_fn(U, dt) for U, dt in zip(pop_U, pop_dt)])

    best_idx = int(np.argmin(fitness))
    U_best = pop_U[best_idx].copy()
    dt_best = float(pop_dt[best_idx])
    best_cost = float(fitness[best_idx])

    history = [best_cost]

    # ---------------- TLBO Iterations ---------------- #
    for it in range(iters):

        # ================= Teacher Phase ================= #
        teacher_idx = int(np.argmin(fitness))
        teacher_U = pop_U[teacher_idx]
        teacher_dt = pop_dt[teacher_idx]

        mean_U = np.mean(np.stack(pop_U), axis=0)
        mean_dt = float(np.mean(pop_dt))

        TF = rng.integers(1, 3)  # Teaching factor ∈ {1, 2}

        for i in range(pop_size):
            r = rng.random(size=U0.shape)

            # Core TLBO update + exploration
            U_new = (
                pop_U[i]
                + r * (teacher_U - TF * mean_U)
                + 0.15 * rng.normal(size=U0.shape)
            )

            dt_new = pop_dt[i] + rng.random() * (teacher_dt - TF * mean_dt)

            U_new = project_speed(U_new, vmax)
            dt_new = clamp_dt(dt_new, dt_min, dt_max)

            J_new = float(cost_fn(U_new, dt_new))

            if J_new < fitness[i]:
                pop_U[i] = U_new
                pop_dt[i] = dt_new
                fitness[i] = J_new

        # ================= Learner Phase ================= #
        for i in range(pop_size):
            j = rng.integers(0, pop_size)

            if fitness[i] < fitness[j]:
                better, worse = i, j
            else:
                better, worse = j, i

            r = rng.random(size=U0.shape)

            # Learner interaction + exploration
            U_new = (
                pop_U[worse]
                + r * (pop_U[better] - pop_U[worse])
                + 0.15 * rng.normal(size=U0.shape)
            )

            dt_new = pop_dt[worse] + rng.random() * (pop_dt[better] - pop_dt[worse])

            U_new = project_speed(U_new, vmax)
            dt_new = clamp_dt(dt_new, dt_min, dt_max)

            J_new = float(cost_fn(U_new, dt_new))

            if J_new < fitness[worse]:
                pop_U[worse] = U_new
                pop_dt[worse] = dt_new
                fitness[worse] = J_new

        # ================= Update Global Best ================= #
        idx = int(np.argmin(fitness))
        if fitness[idx] < best_cost:
            best_cost = float(fitness[idx])
            U_best = pop_U[idx].copy()
            dt_best = float(pop_dt[idx])

        history.append(best_cost)

    log = {
        "best_cost": best_cost,
        "iters": iters,
        "history": np.array(history, dtype=float),
    }

    return U_best, dt_best, log
