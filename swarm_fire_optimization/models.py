"""
models.py
----------
Environment, robot kinematics, and a simple dynamic fire-front model with boundary sampling.

- Single-integrator discrete kinematics: x_{k+1,i} = x_{k,i} + Δt * u_{k,i}
- Fire boundary Γ(t) sampled into M_k points s_{k,j}
- Coverage kernel p_{k,i,j} = exp(-||s_{k,j} - x_{k,i}||^2 / (2σ^2))
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

Array = np.ndarray

@dataclass
class Workspace:
    """2D rectangular workspace without internal holes. Obstacles are axis-aligned rectangles."""
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    # Obstacles: list of (xmin, ymin, xmax, ymax)
    obstacles: list | None = None

    def in_free_space(self, x: Array) -> bool:
        """Return True if a point x = [x, y] is inside W and not inside any obstacle rectangle."""
        if not (self.xmin <= x[0] <= self.xmax and self.ymin <= x[1] <= self.ymax):
            return False
        if self.obstacles:
            for (ox1, oy1, ox2, oy2) in self.obstacles:
                if (ox1 <= x[0] <= ox2) and (oy1 <= x[1] <= oy2):
                    return False
        return True

@dataclass
class RobotParams:
    vmax: float          # max speed
    rs: float            # sensing radius (used indirectly via kernel σ)
    rc: float            # communication radius
    Emax: float          # per-robot energy budget
    sigma: float         # coverage decay parameter σ
    dmin: float          # minimum inter-robot distance

@dataclass
class SimulationParams:
    dt: float            # base Δt
    T: int               # number of time steps
    N: int               # number of robots
    Mk: int              # number of boundary samples per step
    dt_var: float | None = None  # variable time step (decision variable)
    
def sample_fire_boundary(center: Array, a: float, b: float, angle: float, Mk: int) -> Array:
    """
    Sample Mk points from a rotated ellipse boundary used as a proxy for Γ(t).
    """
    t = np.linspace(0, 2*np.pi, Mk, endpoint=False)
    pts = np.stack([a*np.cos(t), b*np.sin(t)], axis=1)  # (Mk,2)
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    pts = pts @ R.T
    pts += np.asarray(center)[None, :]
    return pts

def evolve_fire_front_time(t: float, sim: SimulationParams) -> Array:
    """Time-dependent fire boundary Γ(t)."""
    a = 1.5 + 0.04 * t
    b = 1.0 + 0.03 * t
    angle = 0.1 * t
    center = np.array([5.0 + 0.04 * t, 5.0])
    return sample_fire_boundary(center, a, b, angle, sim.Mk)

def propagate_dynamics(X0: Array, U: Array, sim: SimulationParams, dt_override: float | None = None) -> Array:
    """Propagate x_{k+1,i} = x_{k,i} + Δt * u_{k,i}."""
    dt_used = sim.dt if dt_override is None else dt_override
    N = sim.N
    assert U.shape == (sim.T, N, 2)
    X = np.zeros((sim.T + 1, N, 2))
    X[0] = X0
    for k in range(sim.T):
        X[k + 1] = X[k] + dt_used * U[k]
    return X