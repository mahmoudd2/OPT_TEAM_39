"""
constraints.py
---------------
Constraint checks and soft penalty construction.

We expose two levels:
1) hard_checks(...)    -> boolean flags for feasibility diagnostics
2) penalty_terms(...)  -> continuous penalties added to the objective with large multipliers
"""
from __future__ import annotations
import numpy as np
from typing import Dict
from .models import Workspace

Array = np.ndarray

def hard_checks(X: Array, U: Array, vmax: float, dmin: float, rc: float, Emax: float, dt: float, W: Workspace) -> Dict[str, bool]:
    """
    Return feasibility booleans for each constraint (True means satisfied everywhere).
    """
    T, N = U.shape[0], U.shape[1]
    speeds_ok = (np.linalg.norm(U, axis=2) <= vmax + 1e-9).all()

    # Inter-robot distance (k = 0..T)
    dists_ok = True
    for k in range(T+1):
        Xk = X[k]  # (N,2)
        D = np.linalg.norm(Xk[:, None, :] - Xk[None, :, :], axis=2)  # (N,N)
        mask = ~np.eye(N, dtype=bool)
        if not (D[mask] >= dmin - 1e-9).all():
            dists_ok = False
            break

    # Connectivity (each robot must have at least one neighbor within rc)
    conn_ok = True
    for k in range(T+1):
        Xk = X[k]
        D = np.linalg.norm(Xk[:, None, :] - Xk[None, :, :], axis=2)
        np.fill_diagonal(D, np.inf)
        if not ((D <= rc).any(axis=1)).all():
            conn_ok = False
            break

    energies = (np.linalg.norm(U, axis=2)**2).sum(axis=0) * dt
    energy_ok = (energies <= Emax + 1e-9).all()

    ws_ok = True
    for k in range(T+1):
        for i in range(N):
            if not W.in_free_space(X[k, i]):
                ws_ok = False
                break
        if not ws_ok:
            break

    return dict(speed=speeds_ok, distances=dists_ok, connectivity=conn_ok, energy=energy_ok, workspace=ws_ok)

def penalty_terms(X: Array, U: Array, vmax: float, dmin: float, rc: float, Emax: float, dt: float, W: Workspace) -> Dict[str, float]:
    """
    Soft penalties (≥0) for constraint violations to be added to the objective with large multipliers.
    They are zero if constraints are satisfied.
    """
    T, N = U.shape[0], U.shape[1]

    # Speed penalty
    speed_violation = np.maximum(0.0, np.linalg.norm(U, axis=2) - vmax)
    P_speed = float((speed_violation**2).sum())

    # Distance penalty
    P_dist = 0.0
    for k in range(T+1):
        Xk = X[k]
        for i in range(N):
            for l in range(i+1, N):
                d = np.linalg.norm(Xk[i] - Xk[l])
                P_dist += max(0.0, (dmin - d))**2

    # Connectivity penalty: if a robot's nearest neighbor > rc
    P_conn = 0.0
    for k in range(T+1):
        Xk = X[k]
        D = np.linalg.norm(Xk[:, None, :] - Xk[None, :, :], axis=2)
        np.fill_diagonal(D, np.inf)
        min_nn = D.min(axis=1)  # (N,)
        too_far = np.maximum(0.0, min_nn - rc)
        P_conn += float((too_far**2).sum())

    # Energy penalty
    energies = (np.linalg.norm(U, axis=2)**2).sum(axis=0) * dt
    P_energy = float((np.maximum(0.0, energies - Emax)**2).sum())

    # Workspace penalty: box + rectangular obstacles
    P_ws = 0.0
    for k in range(T+1):
        for i in range(N):
            x, y = X[k, i]
            P_ws += max(0.0, W.xmin - x)**2
            P_ws += max(0.0, x - W.xmax)**2
            P_ws += max(0.0, W.ymin - y)**2
            P_ws += max(0.0, y - W.ymax)**2
            if W.obstacles:
                for (ox1, oy1, ox2, oy2) in W.obstacles:
                    if (ox1 <= x <= ox2) and (oy1 <= y <= oy2):
                        cx, cy = 0.5*(ox1+ox2), 0.5*(oy1+oy2)
                        P_ws += ((x - cx)**2 + (y - cy)**2 + 1.0)  # +1 ensures > 0

    return dict(P_speed=P_speed, P_dist=P_dist, P_conn=P_conn, P_energy=P_energy, P_ws=P_ws)
