"""
plotting.py
Matplotlib helpers to visualize robot trajectories and sampled fire boundary.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence
from .models import SimulationParams, evolve_fire_front_time, Workspace

Array = np.ndarray

def plot_paths_with_fire(X: Array, sim: SimulationParams, W: Workspace, title: str, steps: Sequence[int]):
    T, N = X.shape[0] - 1, X.shape[1]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    # robot paths
    for i in range(N):
        ax.plot(X[:, i, 0], X[:, i, 1], marker=".", linewidth=1.0)

    # draw fire boundary at selected steps
    for k in steps:
        t = k * (sim.dt_var if sim.dt_var is not None else sim.dt)
        S = evolve_fire_front_time(t, sim)
        ax.plot(S[:, 0], S[:, 1], "--", alpha=0.6)

    # workspace + obstacles
    ax.add_patch(plt.Rectangle((W.xmin, W.ymin), W.xmax - W.xmin, W.ymax - W.ymin, fill=False, linewidth=1.0))
    if W.obstacles:
        for (ox1, oy1, ox2, oy2) in W.obstacles:
            ax.add_patch(plt.Rectangle((ox1, oy1), ox2 - ox1, oy2 - oy1, color="black", alpha=0.15))

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig, ax
