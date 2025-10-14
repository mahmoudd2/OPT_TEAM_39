"""
fitness.py
----------
Top-level fitness function:
    J = α1 * J_cov + α2 * J_dist + α3 * J_bal  +  β · penalties

Given decision variables (U), it computes X from dynamics, samples Γ_k, then evaluates J.
This is the function to plug into PSO/GA/SA.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Sequence
from .models import Workspace, RobotParams, SimulationParams, evolve_fire_front, propagate_dynamics
from .objectives import total_coverage_cost, travel_distance_cost, energy_balance_cost
from .constraints import penalty_terms

Array = np.ndarray

def compute_fitness(
    X0: Array,
    U: Array,
    W: Workspace,
    robot: RobotParams,
    sim: SimulationParams,
    alpha: Sequence[float] = (1.0, 0.1, 0.05),
    penalty_weights: Dict[str, float] | None = None,
    wk_list: Sequence[np.ndarray] | None = None,
) -> Dict[str, float]:
    """
    Compute the multi-objective weighted sum with soft penalties.
    Returns:
        dict with 'J_total', 'J_cov', 'J_dist', 'J_bal', each penalty term, and 'X' (positions).
    """
    if penalty_weights is None:
        penalty_weights = {"P_speed": 1e4, "P_dist": 1e4, "P_conn": 5e3, "P_energy": 1e4, "P_ws": 1e4}

    # 1) Unroll dynamics to get X
    X = propagate_dynamics(X0, U, sim)  # (T+1,N,2)

    # 2) Sample dynamic fire boundary Γ_k at each step
    S_list = [evolve_fire_front(k, sim) for k in range(sim.T+1)]  # each (M,2)

    # 3) Objectives
    J_cov = total_coverage_cost(X, S_list, robot.sigma, wk_list)
    J_dist = travel_distance_cost(X)
    J_bal = energy_balance_cost(U)

    # 4) Constraint penalties
    P = penalty_terms(X, U, robot.vmax, robot.dmin, robot.rc, robot.Emax, sim.dt, W)

    # 5) Aggregate cost
    a1, a2, a3 = alpha
    J_obj = a1 * J_cov + a2 * J_dist + a3 * J_bal
    J_pen = sum(penalty_weights[k] * P[k] for k in P)
    J_total = float(J_obj + J_pen)

    out = dict(J_total=J_total, J_cov=float(J_cov), J_dist=float(J_dist), J_bal=float(J_bal))
    out.update({k: float(P[k]) for k in P})
    out["X"] = X  # keep positions for downstream visualization if needed
    return out

def generate_random_controls(sim: SimulationParams, robot: RobotParams, seed: int | None = 0) -> Array:
    """
    Utility to generate a feasible-ish random control tensor U of shape (T,N,2) with ||u|| <= vmax.
    """
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(sim.T, sim.N, 2))
    norms = np.linalg.norm(U, axis=2, keepdims=True) + 1e-12
    U = U / norms  # unit directions
    mags = rng.uniform(0.0, robot.vmax, size=(sim.T, sim.N, 1))
    return U * mags
