from __future__ import annotations
import numpy as np
from swarm_fire_optimization.models import Workspace, RobotParams, SimulationParams
from swarm_fire_optimization.fitness import compute_fitness, generate_random_controls

def main():
    W = Workspace(xmin=0.0, xmax=12.0, ymin=0.0, ymax=10.0, obstacles=[(7.5, 3.0, 8.5, 6.0)])
    robot = RobotParams(vmax=1.5, rs=2.0, rc=3.0, Emax=25.0, sigma=1.0, dmin=0.8)

    # dt_var = variable time step to be optimized
    sim = SimulationParams(dt=0.5, T=25, N=5, Mk=64, dt_var=0.35)

    X0 = np.array([[2.0, 2.0],
                   [2.5, 7.5],
                   [4.0, 4.0],
                   [3.0, 6.0],
                   [1.5, 4.5]])

    U = generate_random_controls(sim, robot, seed=42)

    alpha = (1.0, 0.1, 0.05, 0.4)
    penalty_weights = {"P_speed": 1e4, "P_dist": 1e4, "P_conn": 5e3, "P_energy": 1e4, "P_ws": 1e4}

    out = compute_fitness(X0, U, W, robot, sim, alpha=alpha, penalty_weights=penalty_weights)

    print("=== Fitness Breakdown (with time variable) ===")
    for key in ["J_total", "J_cov", "J_dist", "J_bal", "J_time", "P_speed", "P_dist", "P_conn", "P_energy", "P_ws"]:
        print(f"{key:>10s}: {out[key]:.6f}")

if __name__ == "__main__":
    main()
