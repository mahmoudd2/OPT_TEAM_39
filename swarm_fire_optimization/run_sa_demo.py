"""
run_sa_demo.py
SA on Team 39 swarm-fire problem, optimizing U and Δt together.

Run from project root:
  python -m swarm_fire_optimization.run_sa_demo
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from swarm_fire_optimization.models import Workspace, RobotParams, SimulationParams
from swarm_fire_optimization.fitness import compute_fitness, generate_random_controls
from swarm_fire_optimization.sa import simulated_annealing
from swarm_fire_optimization.plotting import plot_paths_with_fire

def make_problem_small():
    W = Workspace(0.0, 12.0, 0.0, 10.0, obstacles=[(7.5, 3.0, 8.5, 6.0)])
    robot = RobotParams(vmax=1.6, rs=2.0, rc=3.0, Emax=25.0, sigma=1.0, dmin=0.8)
    sim = SimulationParams(dt=0.5, T=12, N=3, Mk=64, dt_var=None)
    X0 = np.array([[2.0, 2.0], [3.0, 6.0], [1.5, 4.5]], dtype=float)
    return W, robot, sim, X0

def make_problem_full():
    W = Workspace(0.0, 12.0, 0.0, 10.0, obstacles=[(7.5, 3.0, 8.5, 6.0)])
    robot = RobotParams(vmax=1.6, rs=2.0, rc=3.0, Emax=25.0, sigma=1.0, dmin=0.8)
    sim = SimulationParams(dt=0.5, T=25, N=5, Mk=64, dt_var=None)
    X0 = np.array([[2.0, 2.0], [2.5, 7.5], [4.0, 4.0], [3.0, 6.0], [1.5, 4.5]], dtype=float)
    return W, robot, sim, X0

def run_case(label, W, robot, sim, X0, dt_bounds=(0.20, 0.80), seed=0):
    # Initial guess
    U0 = generate_random_controls(sim, robot, seed=42)
    dt0 = float(np.mean(dt_bounds))

    alpha = (1.0, 0.1, 0.05, 0.4)  # [J_cov, J_dist, J_bal, J_time]
    penalty_weights = {"P_speed": 1e4, "P_dist": 1e4, "P_conn": 5e3, "P_energy": 1e4, "P_ws": 1e4}

    # Histories of evaluation-level values (for smoother convergence plots)
    eval_obj_hist = []
    eval_pen_hist = []
    eval_total_hist = []
    
    # Cost wrapper for SA (sets sim.dt_var each eval)
    def cost(U, dt_var):
        sim.dt_var = float(dt_var)
        out = compute_fitness(X0, U, W, robot, sim, alpha=alpha, penalty_weights=penalty_weights)
        eval_obj_hist.append(out["J_obj"])
        eval_pen_hist.append(out["J_pen"])
        eval_total_hist.append(out["J_total"])
        return out["J_total"]


    # Optimize
    U_best, dt_best, log = simulated_annealing(
        cost_fn=cost, U0=U0, dt0=dt0, vmax=robot.vmax,
        dt_min=dt_bounds[0], dt_max=dt_bounds[1],
        iters=500, T0=6.0, alpha=0.96, frac=0.07,
        sigma_u=0.22, sigma_dt=0.03, stall_check=150, seed=seed
    )
    # Final evaluation
    sim.dt_var = float(dt_best)
    out = compute_fitness(X0, U_best, W, robot, sim, alpha=alpha, penalty_weights=penalty_weights)
    X = out["X"]

    print(f"\n[{label}] SA finished: best J={out['J_total']:.6f}, dt*={dt_best:.3f}s, iters={log['iters']}, accepts={log['accepts']}, improves={log['improves']}")
    keys = ["J_cov","J_dist","J_bal","J_time","P_speed","P_dist","P_conn","P_energy","P_ws"]
    for k in keys:
        print(f"  {k:>10s}: {out[k]:.6f}")

    # Plot
    fig, ax = plot_paths_with_fire(X, sim, W, title=f"SA Result – {label} (dt={dt_best:.2f}s)", steps=(0, sim.T//2, sim.T))
    png = f"sa_{label}.png"
    fig.savefig(png, dpi=150)
    print(f"[{label}] Saved figure: {png}")

    # Save artifacts
    np.savez(
        f"sa_{label}.npz",
        X=X,
        U=U_best,
        dt=dt_best,
        alpha=np.array(alpha, dtype=float),
        pen=np.array(list(penalty_weights.items()), dtype=object),
        log=log,
        eval_obj_hist=np.array(eval_obj_hist, dtype=float),
        eval_pen_hist=np.array(eval_pen_hist, dtype=float),
        eval_total_hist=np.array(eval_total_hist, dtype=float),
    )
    return out, dt_best

def main():
    # Case A (small)
    outA, dtA = run_case("caseA_small", *make_problem_small(), seed=0)
    # Case B (full)
    outB, dtB = run_case("caseB_full", *make_problem_full(), seed=1)

if __name__ == "__main__":
    main()
