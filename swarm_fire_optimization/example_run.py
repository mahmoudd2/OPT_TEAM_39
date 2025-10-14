"""
example_run.py
--------------
Example on how to call the fitness with random decision variables U.
This prints a cost breakdown and saves a small NPZ (X, U).
"""
from __future__ import annotations
import numpy as np
from swarm_fire_optimization.models import Workspace, RobotParams, SimulationParams
from swarm_fire_optimization.fitness import compute_fitness, generate_random_controls

def main():
    # --- Parameters ---
    W = Workspace(xmin=0.0, xmax=12.0, ymin=0.0, ymax=10.0, obstacles=[(7.5, 3.0, 8.5, 6.0)])
    robot = RobotParams(
        vmax=1.2,   # max speed
        rs=2.0,
        rc=3.0,     # comm radius
        Emax=25.0,  # energy budget per robot
        sigma=1.0,  # coverage decay σ
        dmin=0.8    # min spacing
    )
    sim = SimulationParams(
        dt=0.5,   # Δt
        T=25,     # steps
        N=5,      # robots
        Mk=64     # Γ_k samples
    )

    # Initial positions (inside W)
    X0 = np.array([[2.0, 2.0],
                   [2.5, 7.5],
                   [4.0, 4.0],
                   [3.0, 6.0],
                   [1.5, 4.5]], dtype=float)

    # Decision variables: velocities U (T,N,2)
    U = generate_random_controls(sim, robot, seed=42)

    # Objective weights
    alpha = (1.0, 0.1, 0.05)

    # Penalty weights
    penalty_weights = {"P_speed": 1e4, "P_dist": 1e4, "P_conn": 5e3, "P_energy": 1e4, "P_ws": 1e4}

    out = compute_fitness(X0, U, W, robot, sim, alpha=alpha, penalty_weights=penalty_weights)

    print("=== Fitness Breakdown ===")
    for k in ["J_total", "J_cov", "J_dist", "J_bal", "P_speed", "P_dist", "P_conn", "P_energy", "P_ws"]:
        print(f"{k:>10s}: {out[k]:.6f}")

    np.savez("milestone2_demo_outputs.npz", X=out["X"], U=U, params=dict(alpha=alpha))
    print("\nSaved: milestone2_demo_outputs.npz")

if __name__ == "__main__":
    main()
