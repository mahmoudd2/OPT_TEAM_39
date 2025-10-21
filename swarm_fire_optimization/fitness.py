"""
fitness.py
----------
Extended fitness function with time as a decision variable.
J = α1*J_cov + α2*J_dist + α3*J_bal + α4*J_time + β·penalties
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Sequence
from .models import Workspace, RobotParams, SimulationParams, evolve_fire_front_time, propagate_dynamics
from .objectives import total_coverage_cost, travel_distance_cost, energy_balance_cost
from .constraints import penalty_terms

Array = np.ndarray

def compute_fitness(
    X0: Array,
    U: Array,
    W: Workspace,
    robot: RobotParams,
    sim: SimulationParams,
    alpha: Sequence[float] = (1.0, 0.1, 0.05, 0.3),
    penalty_weights: Dict[str, float] | None = None,
    wk_list: Sequence[np.ndarray] | None = None,
) -> Dict[str, float]:
    if penalty_weights is None:
        penalty_weights = {"P_speed": 1e4, "P_dist": 1e4, "P_conn": 5e3, "P_energy": 1e4, "P_ws": 1e4}

    dt_used = sim.dt if sim.dt_var is None else sim.dt_var

    # 1) Dynamics
    X = propagate_dynamics(X0, U, sim, dt_override=dt_used)

    # 2) Fire front (time-based)
    S_list = [evolve_fire_front_time(k * dt_used, sim) for k in range(sim.T + 1)]

    # 3) Objectives
    time_weights = [dt_used] * (sim.T + 1)
    J_cov = total_coverage_cost(X, S_list, robot.sigma, wk_list, time_weights)
    J_dist = travel_distance_cost(X)
    J_bal = energy_balance_cost(U)
    J_time = sim.T * dt_used

    # 4) Constraints (energy scales with dt_used)
    P = penalty_terms(X, U, robot.vmax, robot.dmin, robot.rc, robot.Emax, dt_used, W)

    # 5) Aggregate
    a1, a2, a3, a4 = alpha if len(alpha) == 4 else (*alpha, 0.0)
    J_obj = a1 * J_cov + a2 * J_dist + a3 * J_bal + a4 * J_time
    J_pen = sum(penalty_weights[k] * P[k] for k in P)
    J_total = J_obj + J_pen

    out = dict(
        J_total=float(J_total),
        J_cov=float(J_cov),
        J_dist=float(J_dist),
        J_bal=float(J_bal),
        J_time=float(J_time),
    )
    out.update({k: float(P[k]) for k in P})
    out["X"] = X
    return out

def generate_random_controls(sim: SimulationParams, robot: RobotParams, seed: int | None = 0) -> Array:
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(sim.T, sim.N, 2))
    norms = np.linalg.norm(U, axis=2, keepdims=True) + 1e-12
    U = U / norms
    mags = rng.uniform(0.0, robot.vmax, size=(sim.T, sim.N, 1))
    return U * mags
