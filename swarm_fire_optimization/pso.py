"""
pso.py
------
Particle Swarm Optimization (PSO) for the swarm-fire problem.

Optimizes BOTH:
  - the control tensor U (T, N, 2)
  - the time step variable dt_var   (SimulationParams.dt_var)

Interface is compatible with SA / GA:
  particle_swarm_optimization(
      cost_fn, U0, dt0, vmax, dt_min, dt_max, ...
  ) -> (U_best, dt_best, log)

where:
  cost_fn(U, dt) -> scalar fitness (lower is better).
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Dict, Tuple

Array = np.ndarray


# ----------------------- Helper Functions ---------------------------- #

def project_speed(U: Array, vmax: float) -> Array:
    """
    Project all velocity vectors in U to satisfy ||u|| <= vmax.
    U: (T, N, 2)
    """
    norms = np.linalg.norm(U, axis=2, keepdims=True) + 1e-12
    scale = np.minimum(1.0, vmax / norms)
    return U * scale


def clamp_dt(dt: float, dt_min: float, dt_max: float) -> float:
    """Clamp dt into [dt_min, dt_max]."""
    return float(np.clip(dt, dt_min, dt_max))


# ----------------------- PSO Main Function ---------------------------- #

def particle_swarm_optimization(
    cost_fn: Callable[[Array, float], float],
    U0: Array,
    dt0: float,
    vmax: float,
    dt_min: float,
    dt_max: float,
    *,
    swarm_size: int = 25,
    iters: int = 80,
    w_inertia: float = 0.72,
    c1: float = 1.49,   # cognitive
    c2: float = 1.49,   # social
    vel_scale_u: float = 0.4,
    vel_scale_dt: float = 0.05,
    seed: int = 0
) -> Tuple[Array, float, Dict[str, Array | float | int]]:
    """
    Run PSO and return (U_best, dt_best, log).

    cost_fn(U, dt) must return a scalar fitness (lower is better).
    U0 : (T, N, 2) initial guess for control tensor
    dt0: initial guess for dt_var
    vmax: robot max speed -> used for projection
    dt_min, dt_max: bounds for dt_var

    PSO details:
      - Each particle has position (U_p, dt_p) and velocity (V_U_p, V_dt_p).
      - Standard PSO update:
          v <- w * v + c1 * r1 * (p_best - x) + c2 * r2 * (g_best - x)
          x <- x + v
      - After update, we project U to speed <= vmax and clamp dt into bounds.
    """
    rng = np.random.default_rng(seed)
    T, N, _ = U0.shape

    # ----------- Initialization ----------- #
    # Particle positions
    swarm_U = []
    swarm_dt = []

    # Particle velocities
    swarm_VU = []
    swarm_Vdt = []

    # Initialize around U0, dt0 with small random perturbations
    for _ in range(swarm_size):
        # position
        U_p = U0 + rng.normal(scale=0.1, size=U0.shape)
        U_p = project_speed(U_p, vmax)

        dt_p = clamp_dt(dt0 + rng.normal(scale=0.05), dt_min, dt_max)

        swarm_U.append(U_p)
        swarm_dt.append(dt_p)

        # velocities
        V_U_p = rng.normal(scale=vel_scale_u, size=U0.shape)
        V_dt_p = rng.normal(scale=vel_scale_dt)

        swarm_VU.append(V_U_p)
        swarm_Vdt.append(V_dt_p)

    # Convert to lists of arrays (we'll keep them as lists to avoid confusion)
    # Personal bests
    pbest_U = [u.copy() for u in swarm_U]
    pbest_dt = [float(dt) for dt in swarm_dt]
    pbest_cost = np.zeros(swarm_size, dtype=float)

    # Evaluate initial swarm & set pbest, gbest
    for i in range(swarm_size):
        pbest_cost[i] = float(cost_fn(pbest_U[i], pbest_dt[i]))

    gbest_index = int(np.argmin(pbest_cost))
    gbest_U = pbest_U[gbest_index].copy()
    gbest_dt = float(pbest_dt[gbest_index])
    gbest_cost = float(pbest_cost[gbest_index])

    # Logging (global best over iterations)
    gbest_history = [gbest_cost]

    # --------------- PSO Loop --------------- #
    for it in range(iters):
        for i in range(swarm_size):
            U_p = swarm_U[i]
            dt_p = swarm_dt[i]
            V_U_p = swarm_VU[i]
            V_dt_p = swarm_Vdt[i]

            # --- Velocity update --- #
            # Random factors for cognitive & social components
            r1_U = rng.random(size=U0.shape)
            r2_U = rng.random(size=U0.shape)

            # For dt, we use scalars
            r1_dt = rng.random()
            r2_dt = rng.random()

            # Position differences
            diff_pbest_U = pbest_U[i] - U_p
            diff_gbest_U = gbest_U - U_p

            diff_pbest_dt = pbest_dt[i] - dt_p
            diff_gbest_dt = gbest_dt - dt_p

            # Update velocities
            V_U_p = (
                w_inertia * V_U_p
                + c1 * r1_U * diff_pbest_U
                + c2 * r2_U * diff_gbest_U
            )
            V_dt_p = (
                w_inertia * V_dt_p
                + c1 * r1_dt * diff_pbest_dt
                + c2 * r2_dt * diff_gbest_dt
            )

            # Optional: clip velocity magnitudes for stability (in U-space)
            # This is just a safety clamp, not the physical vmax.
            max_vel_u = 2.0 * vmax
            speed_V = np.linalg.norm(V_U_p, axis=2, keepdims=True) + 1e-12
            scale_V = np.minimum(1.0, max_vel_u / speed_V)
            V_U_p = V_U_p * scale_V

            # dt velocity clamp (optional)
            V_dt_p = float(np.clip(V_dt_p, -vel_scale_dt * 3.0, vel_scale_dt * 3.0))

            # --- Position update --- #
            U_p = U_p + V_U_p
            dt_p = dt_p + V_dt_p

            # Enforce constraints
            U_p = project_speed(U_p, vmax)
            dt_p = clamp_dt(dt_p, dt_min, dt_max)

            # Save updated positions and velocities
            swarm_U[i] = U_p
            swarm_dt[i] = dt_p
            swarm_VU[i] = V_U_p
            swarm_Vdt[i] = V_dt_p

            # --- Evaluate new position --- #
            J_new = float(cost_fn(U_p, dt_p))

            # Update personal best
            if J_new < pbest_cost[i]:
                pbest_cost[i] = J_new
                pbest_U[i] = U_p.copy()
                pbest_dt[i] = float(dt_p)

        # --- Update global best --- #
        best_i = int(np.argmin(pbest_cost))
        if pbest_cost[best_i] < gbest_cost:
            gbest_cost = float(pbest_cost[best_i])
            gbest_U = pbest_U[best_i].copy()
            gbest_dt = float(pbest_dt[best_i])

        gbest_history.append(gbest_cost)

    log: Dict[str, Array | float | int] = {
        "best_cost": float(gbest_cost),
        "iters": iters,
        "gbest_history": np.array(gbest_history, dtype=float),
    }
    return gbest_U, gbest_dt, log
