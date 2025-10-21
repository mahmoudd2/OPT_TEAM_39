"""
objectives.py
-------------
Implements the three objectives:

J_cov  : uncovered fire boundary measure
J_dist : total travel distance of all robots
J_bal  : variance of per-robot energy usage
"""
from __future__ import annotations
import numpy as np
from typing import Sequence

Array = np.ndarray

def coverage_objective(Xk: Array, Sk: Array, sigma: float, wk: Array | None = None) -> float:
    """
    One-timestep coverage cost J_cov,k = sum_j w_{k,j} [1 - C_{k,j}]
    where C_{k,j} = 1 - Π_i (1 - p_{k,i,j}), with p_{k,i,j} = exp(-||s_{k,j}-x_{k,i}||^2 / (2σ^2)).
    Inputs:
        Xk : (N,2) robot positions at step k
        Sk : (M,2) fire boundary samples at step k
        sigma : coverage decay parameter σ
        wk : (M,) optional weights; if None, use uniform weights 1/M
    Returns:
        scalar uncovered cost at step k.
    """
    M = Sk.shape[0]
    if wk is None:
        wk = np.ones(M, dtype=float) / M
    d2 = ((Xk[:, None, :] - Sk[None, :, :])**2).sum(axis=2)  # (N,M)
    p = np.exp(-d2 / (2.0 * sigma**2))                       # (N,M)
    not_covered = np.prod(1.0 - p, axis=0)                   # Π_i (1 - p_{i,j})
    C = 1.0 - not_covered
    return float(np.sum(wk * (1.0 - C)))

def total_coverage_cost(
    X: Array,
    S_list: Sequence[Array],
    sigma: float,
    w_list: Sequence[Array] | None = None,
    time_weights: Sequence[float] | None = None,
) -> float:
    """
    J_cov = sum_k Δt_k * sum_j w_{k,j} [1 - C_{k,j}]
    If time_weights is None, each step is equally weighted.
    """
    T = X.shape[0] - 1
    J = 0.0
    for k in range(T + 1):
        wk = None if (w_list is None) else w_list[k]
        tau = 1.0 if (time_weights is None) else float(time_weights[k])
        J += tau * coverage_objective(X[k], S_list[k], sigma, wk)
    return float(J)


def travel_distance_cost(X: Array) -> float:
    """
    J_dist = sum_i sum_{k=0}^{T-1} ||x_{k+1,i} - x_{k,i}||.
    """
    diffs = X[1:] - X[:-1]                 # (T,N,2)
    step_dist = np.linalg.norm(diffs, axis=2)  # (T,N)
    return float(step_dist.sum())

def energy_balance_cost(U: Array) -> float:
    """
    J_bal = Var( sum_{k=0}^{T-1} ||u_{k,i}||^2 ).
    """
    per_robot_energy = (np.linalg.norm(U, axis=2)**2).sum(axis=0)  # (N,)
    return float(per_robot_energy.var())
