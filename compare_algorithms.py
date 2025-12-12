import os
import time
import numpy as np
import matplotlib.pyplot as plt

from swarm_fire_optimization.plotting import plot_paths_with_fire
from swarm_fire_optimization.models import Workspace, RobotParams, SimulationParams
from swarm_fire_optimization.fitness import compute_fitness
from swarm_fire_optimization.ml_controller import SimpleNN
from swarm_fire_optimization.generate_ml_dataset import load_dataset

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
    tlbo = np.load(f"tlbo_{case}.npz", allow_pickle=True)
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
    return compute_fitness(
        result["X"][0],
        result["U"],
        W,
        robot,
        sim,
        alpha=alpha,
        penalty_weights=penalty_weights
    )

# ------------------------------------------
# ML evaluation (Option B)
# ------------------------------------------

def evaluate_ml():
    """
    Evaluate ML controller as a single-shot method.
    Objective is approximated by MSE to TLBO controls.
    """
    X, y = load_dataset("ml_dataset.npz")

    model = SimpleNN(in_dim=5, hidden_dim=16, out_dim=2)
    model.train(X, y)

    start = time.time()
    preds = model.forward(X)
    inference_time = time.time() - start

    mse = ((preds - y) ** 2).mean()

    return {
        "J_cov": mse,
        "J_dist": mse,
        "J_bal": mse,
        "J_time": inference_time
    }

# ------------------------------------------
# 1. Convergence plots (NO ML here on purpose)
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
    plt.figure(figsize=(8,5))
    plt.plot(sa["eval_obj_hist"], label="SA", linewidth=1.8)
    plt.plot(ga["eval_obj_hist"], label="GA", linewidth=1.8)
    plt.plot(pso["eval_obj_hist"], label="PSO", linewidth=1.8)
    plt.plot(tlbo["eval_obj_hist"], label="TLBO", linewidth=1.8)

    plt.xlabel("Evaluation index")
    plt.ylabel("Objective J_obj")
    plt.title("Objective-only Convergence (No Penalties)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "objective_convergence.png"), dpi=150)
    plt.close()

# ------------------------------------------
# 2. Objective components (WITH ML)
# ------------------------------------------

def plot_objectives(out_sa, out_ga, out_pso, out_tlbo, out_ml, out_folder):
    metrics = ["J_cov", "J_dist", "J_bal", "J_time"]
    x = np.arange(len(metrics))
    w = 0.18

    plt.figure(figsize=(10,5))
    plt.bar(x - 2*w, [out_sa[m] for m in metrics], w, label="SA")
    plt.bar(x - w,   [out_ga[m] for m in metrics], w, label="GA")
    plt.bar(x,       [out_pso[m] for m in metrics], w, label="PSO")
    plt.bar(x + w,   [out_tlbo[m] for m in metrics], w, label="TLBO")
    plt.bar(x + 2*w, [out_ml[m] for m in metrics], w, label="ML")

    plt.xticks(x, metrics)
    plt.ylabel("Cost / Time")
    plt.title("Objective Components Comparison (Including ML)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "objective_comparison_with_ml.png"), dpi=150)
    plt.close()

# ------------------------------------------
# 3. Penalties (NO ML)
# ------------------------------------------

def plot_penalties(out_sa, out_ga, out_pso, out_tlbo, out_folder):
    penalties = ["P_speed", "P_dist", "P_conn", "P_energy", "P_ws"]
    x = np.arange(len(penalties))
    w = 0.2

    plt.figure(figsize=(10,5))
    plt.bar(x - 1.5*w, [out_sa[p] for p in penalties], w, label="SA")
    plt.bar(x - 0.5*w, [out_ga[p] for p in penalties], w, label="GA")
    plt.bar(x + 0.5*w, [out_pso[p] for p in penalties], w, label="PSO")
    plt.bar(x + 1.5*w, [out_tlbo[p] for p in penalties], w, label="TLBO")

    plt.xticks(x, penalties)
    plt.ylabel("Penalty Value")
    plt.title("Constraint Penalties Comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, "penalty_comparison.png"), dpi=150)
    plt.close()

# ------------------------------------------
# 4. Trajectory plots (NO ML here – ML has its own figures)
# ------------------------------------------

def plot_trajectories(sa, ga, pso, tlbo, case, W, sim, out_folder):

    for name, res in [("SA", sa), ("GA", ga), ("PSO", pso), ("TLBO", tlbo)]:
        fig, _ = plot_paths_with_fire(
            res["X"], sim, W,
            title=f"Trajectory – {name} – {case}",
            steps=(0, sim.T//2, sim.T)
        )
        fig.savefig(os.path.join(out_folder, f"trajectory_{name.lower()}.png"), dpi=150)
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
    out_ml = evaluate_ml()

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
    plot_objectives(out_sa, out_ga, out_pso, out_tlbo, out_ml, out_folder)
    plot_penalties(out_sa, out_ga, out_pso, out_tlbo, out_folder)
    plot_trajectories(sa, ga, pso, tlbo, case_name, W, sim, out_folder)

    print(f"Graphs saved to {out_folder}")

# ------------------------------------------

def main():
    compare_case("caseA_small", T=12, N=3)
    compare_case("caseB_full", T=25, N=5)

if __name__ == "__main__":
    main()
