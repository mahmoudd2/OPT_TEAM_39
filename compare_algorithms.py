import os
import numpy as np
import matplotlib.pyplot as plt
from swarm_fire_optimization.plotting import plot_paths_with_fire
from swarm_fire_optimization.models import Workspace, RobotParams, SimulationParams
from swarm_fire_optimization.fitness import compute_fitness

# ------------------------------------------
# Utility
# ------------------------------------------

def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_results(case):
    sa = np.load(f"sa_{case}.npz", allow_pickle=True)
    ga = np.load(f"ga_{case}.npz", allow_pickle=True)
    pso = np.load(f"pso_{case}.npz", allow_pickle=True)
    tlbo = np.load(f"tlbo_{case}.npz", allow_pickle=True)   # NEW
    return sa, ga, pso, tlbo


def compute_metrics(result, W, robot, sim):
    alpha = (1.0, 0.1, 0.05, 0.4)
    penalty_weights = {
        "P_speed":1e4,
        "P_dist":1e4,
        "P_conn":5e3,
        "P_energy":1e4,
        "P_ws":1e4
    }

    sim.dt_var = float(result["dt"])
    out = compute_fitness(
        result["X"][0],
        result["U"],
        W,
        robot,
        sim,
        alpha=alpha,
        penalty_weights=penalty_weights
    )
    return out

# ------------------------------------------
# 1. Convergence plots
# ------------------------------------------

def plot_convergence(sa_log, ga_log, pso_log, tlbo_log, out_folder):
    plt.figure(figsize=(8,5))
    plt.plot(ga_log["history"], label="GA", linewidth=2)
    plt.plot(pso_log["gbest_history"], label="PSO", linewidth=2)
    plt.plot(tlbo_log["history"], label="TLBO", linewidth=2)
    plt.axhline(sa_log["best_cost"], color="orange", linestyle="--", label="SA Best")

    plt.xlabel("Iteration")
    plt.ylabel("Fitness J")
    plt.title("Convergence Curve – SA vs GA vs PSO vs TLBO")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "convergence_curve.png"), dpi=150)
    plt.close()


def plot_objective_convergence(sa, ga, pso, tlbo, out_folder):
    sa_obj = sa["eval_obj_hist"]
    ga_obj = ga["eval_obj_hist"]
    pso_obj = pso["eval_obj_hist"]
    tlbo_obj = tlbo["eval_obj_hist"]

    plt.figure(figsize=(8,5))
    plt.plot(sa_obj, label="SA", linewidth=1.8)
    plt.plot(ga_obj, label="GA", linewidth=1.8)
    plt.plot(pso_obj, label="PSO", linewidth=1.8)
    plt.plot(tlbo_obj, label="TLBO", linewidth=1.8)

    plt.xlabel("Evaluation index")
    plt.ylabel("Objective J_obj")
    plt.title("Objective-only Convergence (No Penalties)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "objective_convergence.png"), dpi=150)
    plt.close()

# ------------------------------------------
# 2. Objective components
# ------------------------------------------

def plot_objectives(out_sa, out_ga, out_pso, out_tlbo, out_folder):
    metrics = ["J_cov", "J_dist", "J_bal", "J_time"]
    sa_vals = [out_sa[m] for m in metrics]
    ga_vals = [out_ga[m] for m in metrics]
    pso_vals = [out_pso[m] for m in metrics]
    tlbo_vals = [out_tlbo[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.2

    plt.figure(figsize=(9,5))
    plt.bar(x - 1.5*width, sa_vals, width=width, label="SA")
    plt.bar(x - 0.5*width, ga_vals, width=width, label="GA")
    plt.bar(x + 0.5*width, pso_vals, width=width, label="PSO")
    plt.bar(x + 1.5*width, tlbo_vals, width=width, label="TLBO")

    plt.xticks(x, metrics)
    plt.ylabel("Cost")
    plt.title("Objective Components Comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "objective_comparison.png"), dpi=150)
    plt.close()

# ------------------------------------------
# 3. Penalties
# ------------------------------------------

def plot_penalties(out_sa, out_ga, out_pso, out_tlbo, out_folder):
    penalties = ["P_speed", "P_dist", "P_conn", "P_energy", "P_ws"]
    sa_vals = [out_sa[p] for p in penalties]
    ga_vals = [out_ga[p] for p in penalties]
    pso_vals = [out_pso[p] for p in penalties]
    tlbo_vals = [out_tlbo[p] for p in penalties]

    x = np.arange(len(penalties))
    width = 0.2

    plt.figure(figsize=(10,5))
    plt.bar(x - 1.5*width, sa_vals, width=width, label="SA")
    plt.bar(x - 0.5*width, ga_vals, width=width, label="GA")
    plt.bar(x + 0.5*width, pso_vals, width=width, label="PSO")
    plt.bar(x + 1.5*width, tlbo_vals, width=width, label="TLBO")

    plt.xticks(x, penalties)
    plt.ylabel("Penalty Value")
    plt.title("Constraint Penalties Comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "penalty_comparison.png"), dpi=150)
    plt.close()

# ------------------------------------------
# 4. Trajectory plots
# ------------------------------------------

def plot_trajectories(sa, ga, pso, tlbo, case, W, sim, out_folder):

    fig, _ = plot_paths_with_fire(
        sa["X"], sim, W,
        title=f"Trajectory – SA – {case}",
        steps=(0, sim.T//2, sim.T)
    )
    fig.savefig(os.path.join(out_folder, "trajectory_sa.png"), dpi=150)
    plt.close()

    fig, _ = plot_paths_with_fire(
        ga["X"], sim, W,
        title=f"Trajectory – GA – {case}",
        steps=(0, sim.T//2, sim.T)
    )
    fig.savefig(os.path.join(out_folder, "trajectory_ga.png"), dpi=150)
    plt.close()

    fig, _ = plot_paths_with_fire(
        pso["X"], sim, W,
        title=f"Trajectory – PSO – {case}",
        steps=(0, sim.T//2, sim.T)
    )
    fig.savefig(os.path.join(out_folder, "trajectory_pso.png"), dpi=150)
    plt.close()

    fig, _ = plot_paths_with_fire(
        tlbo["X"], sim, W,
        title=f"Trajectory – TLBO – {case}",
        steps=(0, sim.T//2, sim.T)
    )
    fig.savefig(os.path.join(out_folder, "trajectory_tlbo.png"), dpi=150)
    plt.close()

# ------------------------------------------
# Main comparison
# ------------------------------------------

def compare_case(case_name, T, N):
    print(f"\n----- Comparing Algorithms for {case_name} -----")

    W = Workspace(0,12,0,10, obstacles=[(7.5,3,8.5,6)])
    robot = RobotParams(vmax=1.6, rs=2.0, rc=3.0, Emax=25.0, sigma=1.0, dmin=0.8)
    sim = SimulationParams(dt=0.5, T=T, N=N, Mk=64, dt_var=None)

    sa, ga, pso, tlbo = load_results(case_name)

    out_sa = compute_metrics(sa, W, robot, sim)
    out_ga = compute_metrics(ga, W, robot, sim)
    out_pso = compute_metrics(pso, W, robot, sim)
    out_tlbo = compute_metrics(tlbo, W, robot, sim)

    out_folder = os.path.join("comparison_graphs", case_name)
    ensure_folder(out_folder)

    plot_convergence(
        sa["log"].item(),
        ga["log"].item(),
        pso["log"].item(),
        tlbo["log"].item(),
        out_folder
    )

    plot_objective_convergence(sa, ga, pso, tlbo, out_folder)
    plot_objectives(out_sa, out_ga, out_pso, out_tlbo, out_folder)
    plot_penalties(out_sa, out_ga, out_pso, out_tlbo, out_folder)
    plot_trajectories(sa, ga, pso, tlbo, case_name, W, sim, out_folder)

    print(f"Graphs saved to {out_folder}")

# ------------------------------------------

def main():
    compare_case("caseA_small", T=12, N=3)
    compare_case("caseB_full", T=25, N=5)

if __name__ == "__main__":
    main()
