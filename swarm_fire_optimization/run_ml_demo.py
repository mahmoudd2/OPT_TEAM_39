import numpy as np
import matplotlib.pyplot as plt

from swarm_fire_optimization.ml_controller import SimpleNN
from swarm_fire_optimization.generate_ml_dataset import load_dataset
from swarm_fire_optimization.plotting import plot_paths_with_fire
from swarm_fire_optimization.models import Workspace, RobotParams, SimulationParams
from swarm_fire_optimization.fitness import compute_fitness


def run_ml_simulation(model, case_name):
    if case_name == "caseA_small":
        T, N = 12, 3
        X0 = np.array([[2.0, 2.0], [3.0, 6.0], [1.5, 4.5]])
    else:
        T, N = 25, 5
        X0 = np.array([[2.0, 2.0], [2.5, 7.5], [4.0, 4.0], [3.0, 6.0], [1.5, 4.5]])

    W = Workspace(0, 12, 0, 10, obstacles=[(7.5, 3, 8.5, 6)])
    robot = RobotParams(vmax=1.6, rs=2.0, rc=3.0, Emax=25.0, sigma=1.0, dmin=0.8)
    sim = SimulationParams(dt=0.5, T=T, N=N, Mk=64, dt_var=0.2)

    X = np.zeros((T + 1, N, 2))
    X[0] = X0

    for t in range(T):
        fire_center = np.mean(X[t], axis=0)
        for i in range(N):
            xi, yi = X[t, i]
            inp = np.array([[xi, yi, fire_center[0], fire_center[1], t]])
            u = model.forward(inp)[0]
            X[t + 1, i] = X[t, i] + u * sim.dt_var

    return X, sim, W


def main():
    # Load dataset
    X_train, y_train = load_dataset("ml_dataset.npz")

    # Train ML model
    model = SimpleNN(in_dim=5, hidden_dim=16, out_dim=2)
    model.train(X_train, y_train)
    print("ML model training finished.")

    # Run & plot for both cases
    for case in ["caseA_small", "caseB_full"]:
        X_ml, sim, W = run_ml_simulation(model, case)

        fig, _ = plot_paths_with_fire(
            X_ml, sim, W,
            title=f"ML Result – {case}",
            steps=(0, sim.T // 2, sim.T)
        )
        fig.savefig(f"ml_{case}.png", dpi=150)
        plt.close()

        np.savez(f"ml_{case}.npz", X=X_ml)
        print(f"Saved ML results for {case}")


if __name__ == "__main__":
    main()
