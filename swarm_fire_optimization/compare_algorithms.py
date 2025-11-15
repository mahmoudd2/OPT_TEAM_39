import os
import numpy as np
import matplotlib.pyplot as plt
from swarm_fire_optimization.plotting import plot_paths_with_fire
from swarm_fire_optimization.models import Workspace, RobotParams, SimulationParams
from swarm_fire_optimization.fitness import compute_fitness

# ------------------------------------------
# Utility functions
# ------------------------------------------

def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_results(case):
    sa = np.load(f"sa_{case}.npz", allow_pickle=True)
    ga = np.load(f"ga_{case}.npz", allow_pickle=True)
    return sa, ga


def compute_metrics(sa, ga, W, robot, sim):
    alpha = (1.0, 0.1, 0.05, 0.4)
    penalty_weights = {"P_speed":1e4,"P_dist":1e4,"P_conn":5e3,"P_energy":1e4,"P_ws":1e4}

    # SA evaluate
    sim.dt_var = float(sa["dt"])
    out_sa = compute_fitness(sa["X"][0], sa["U"], W, robot, sim, alpha=alpha, penalty_weights=penalty_weights)

    # GA evaluate
    sim.dt_var = float(ga["dt"])
    out_ga = compute_fitness(ga["X"][0], ga["U"], W, robot, sim, alpha=alpha, penalty_weights=penalty_weights)

    return out_sa, out_ga


def plot_convergence(sa_log, ga_log, out_folder):
    plt.figure(figsize=(8,5))
    plt.plot(ga_log["history"], label="GA", linewidth=2)
    plt.axhline(sa_log["best_cost"], color="orange", linestyle="--", label="SA Best")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness J")
    plt.title("Convergence Curve – Algorithms Comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "convergence_curve.png"), dpi=150)
    plt.close()


def plot_objectives(out_sa, out_ga, out_folder):
    metrics = ["J_cov", "J_dist", "J_bal", "J_time"]
    sa_vals = [out_sa[m] for m in metrics]
    ga_vals = [out_ga[m] for m in metrics]

    plt.figure(figsize=(8,5))
    x = np.arange(len(metrics))
    plt.bar(x - 0.2, sa_vals, width=0.4, label="SA")
    plt.bar(x + 0.2, ga_vals, width=0.4, label="GA")
    plt.xticks(x, metrics)
    plt.ylabel("Cost")
    plt.title("Objective Components Comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "objective_comparison.png"), dpi=150)
    plt.close()


def plot_penalties(out_sa, out_ga, out_folder):
    penalties = ["P_speed", "P_dist", "P_conn", "P_energy", "P_ws"]
    sa_vals = [out_sa[p] for p in penalties]
    ga_vals = [out_ga[p] for p in penalties]

    plt.figure(figsize=(9,5))
    x = np.arange(len(penalties))
    plt.bar(x - 0.2, sa_vals, width=0.4, label="SA")
    plt.bar(x + 0.2, ga_vals, width=0.4, label="GA")
    plt.xticks(x, penalties)
    plt.ylabel("Penalty")
    plt.title("Constraint Penalties Comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "penalty_comparison.png"), dpi=150)
    plt.close()


def plot_trajectories(sa, ga, case, W, sim, out_folder):
    # SA
    fig, ax = plot_paths_with_fire(
        sa["X"], sim, W,
        title=f"Trajectory – SA – {case}",
        steps=(0, sim.T//2, sim.T)
    )
    fig.savefig(os.path.join(out_folder, "trajectory_sa.png"), dpi=150)
    plt.close()

    # GA
    fig, ax = plot_paths_with_fire(
        ga["X"], sim, W,
        title=f"Trajectory – GA – {case}",
        steps=(0, sim.T//2, sim.T)
    )
    fig.savefig(os.path.join(out_folder, "trajectory_ga.png"), dpi=150)
    plt.close()


# ------------------------------------------
# Main comparison function
# ------------------------------------------

def compare_case(case_name, T, N):
    print(f"\n----- Comparing Algorithms for {case_name} -----")

    # Workspace, robot, sim identical to both GA/SA runs
    W = Workspace(0,12,0,10, obstacles=[(7.5,3,8.5,6)])
    robot = RobotParams(vmax=1.6, rs=2.0, rc=3.0, Emax=25.0, sigma=1.0, dmin=0.8)
    sim = SimulationParams(dt=0.5, T=T, N=N, Mk=64, dt_var=None)

    sa, ga = load_results(case_name)

    out_sa, out_ga = compute_metrics(sa, ga, W, robot, sim)

    # Folder: comparison_graphs/<case_name>/
    out_folder = os.path.join("comparison_graphs", case_name)
    ensure_folder(out_folder)

    # 1. Convergence
    plot_convergence(sa["log"].item(), ga["log"].item(), out_folder)

    # 2. Objective components
    plot_objectives(out_sa, out_ga, out_folder)

    # 3. Penalties
    plot_penalties(out_sa, out_ga, out_folder)

    # 4. Trajectories
    plot_trajectories(sa, ga, case_name, W, sim, out_folder)

    print(f"Graphs saved to {out_folder}")


# ------------------------------------------
# Run all cases
# ------------------------------------------

def main():
    compare_case("caseA_small", T=12, N=3)
    compare_case("caseB_full", T=25, N=5)

if __name__ == "__main__":
    main()
