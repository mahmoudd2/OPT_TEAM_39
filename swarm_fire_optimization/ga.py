"""
ga.py
Genetic Algorithm for swarm-fire optimization.
Optimizes BOTH the control tensor U (T,N,2) and dt_var.

Population-based evolutionary search:
 - Each individual: {U, dt_var}
 - Fitness: compute_fitness(...)
 - Selection: tournament selection
 - Crossover: blend crossover for velocities + arithmetic crossover for dt
 - Mutation: add gaussian noise to U and dt
 - Constraint handling:
       - Project robot speeds to vmax
       - Clamp dt_var into [dt_min, dt_max]
"""

from __future__ import annotations
import numpy as np
from typing import Callable, Tuple, Dict

Array = np.ndarray


# ----------------------- Helper Functions ---------------------------- #

def project_speed(U: Array, vmax: float) -> Array:
    norms = np.linalg.norm(U, axis=2, keepdims=True) + 1e-12
    scale = np.minimum(1.0, vmax / norms)
    return U * scale


def clamp_dt(dt, dt_min, dt_max):
    return float(np.clip(dt, dt_min, dt_max))


def mutate(U: Array, dt: float, sigma_u: float, sigma_dt: float,
           mutation_rate: float, vmax: float,
           dt_min: float, dt_max: float,
           rng: np.random.Generator):
    """Gaussian mutation on subset of genes."""
    T, N, _ = U.shape
    U_new = U.copy()

    # mutate velocities
    mask = rng.random((T, N)) < mutation_rate
    noise = rng.normal(scale=sigma_u, size=(T, N, 2))
    U_new[mask] += noise[mask]
    U_new = project_speed(U_new, vmax)

    # mutate dt
    dt_new = dt + rng.normal(scale=sigma_dt) if rng.random() < mutation_rate else dt
    dt_new = clamp_dt(dt_new, dt_min, dt_max)

    return U_new, dt_new


def crossover(U1: Array, U2: Array, dt1: float, dt2: float, rng):
    """Blend crossover (BLX-alpha) for velocities, arithmetic crossover for dt."""
    alpha = 0.5
    T, N, _ = U1.shape

    # Velocity crossover
    lam = rng.uniform(-alpha, 1 + alpha, size=(T, N, 1))
    Uc1 = U1 + lam * (U2 - U1)
    Uc2 = U2 + lam * (U1 - U2)

    # dt crossover
    lam_dt = rng.uniform(0, 1)
    dtc1 = lam_dt * dt1 + (1 - lam_dt) * dt2
    dtc2 = lam_dt * dt2 + (1 - lam_dt) * dt1

    return Uc1, dtc1, Uc2, dtc2


def tournament_selection(pop, fitness, k, rng):
    """Select best among k random individuals."""
    idx = rng.choice(len(pop), size=k, replace=False)
    best = idx[0]
    for i in idx:
        if fitness[i] < fitness[best]:
            best = i
    return best


# ----------------------- GA Main Function ---------------------------- #

def genetic_algorithm(
        cost_fn: Callable[[Array, float], float],
        U0: Array,
        dt0: float,
        vmax: float,
        dt_min: float,
        dt_max: float,
        *,
        pop_size: int = 20,
        elite_size: int = 2,
        iters: int = 80,
        tournament_k: int = 3,
        mutation_rate: float = 0.15,
        sigma_u: float = 0.25,
        sigma_dt: float = 0.02,
        seed: int = 0
) -> Tuple[Array, float, Dict[str, float]]:
    """
    Run GA and return (U_best, dt_best, log).
    cost_fn(U, dt) must return scalar fitness (lower is better).
    """
    rng = np.random.default_rng(seed)
    T, N, _ = U0.shape

    # ----- Initialize population -----
    pop_U = []
    pop_dt = []
    for _ in range(pop_size):
        U = U0 + rng.normal(scale=0.1, size=U0.shape)
        U = project_speed(U, vmax)
        dt = clamp_dt(dt0 + rng.normal(scale=0.05), dt_min, dt_max)
        pop_U.append(U)
        pop_dt.append(dt)

    # ----- Evaluate fitness -----
    fitness = np.array([cost_fn(U, dt) for U, dt in zip(pop_U, pop_dt)])
    best_idx = np.argmin(fitness)

    # Logging
    best_cost = float(fitness[best_idx])
    U_best = pop_U[best_idx].copy()
    dt_best = float(pop_dt[best_idx])

    history = [best_cost]

    # -------------------- Evolution Loop -------------------- #
    for gen in range(iters):

        # ----- Elitism -----
        elite_idx = np.argsort(fitness)[:elite_size]
        new_pop_U = [pop_U[i].copy() for i in elite_idx]
        new_pop_dt = [float(pop_dt[i]) for i in elite_idx]

        # ----- Reproduction -----
        while len(new_pop_U) < pop_size:
            p1 = tournament_selection(pop_U, fitness, tournament_k, rng)
            p2 = tournament_selection(pop_U, fitness, tournament_k, rng)

            U1, dt1 = pop_U[p1], pop_dt[p1]
            U2, dt2 = pop_U[p2], pop_dt[p2]

            # Crossover
            Uc1, dtc1, Uc2, dtc2 = crossover(U1, U2, dt1, dt2, rng)

            # Mutation
            Uc1, dtc1 = mutate(Uc1, dtc1, sigma_u, sigma_dt,
                               mutation_rate, vmax, dt_min, dt_max, rng)
            Uc2, dtc2 = mutate(Uc2, dtc2, sigma_u, sigma_dt,
                               mutation_rate, vmax, dt_min, dt_max, rng)

            new_pop_U.append(Uc1)
            new_pop_dt.append(dtc1)
            if len(new_pop_U) < pop_size:
                new_pop_U.append(Uc2)
                new_pop_dt.append(dtc2)

        pop_U = new_pop_U
        pop_dt = new_pop_dt
        fitness = np.array([cost_fn(U, dt) for U, dt in zip(pop_U, pop_dt)])

        # Update best
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_cost:
            U_best = pop_U[best_idx].copy()
            dt_best = float(pop_dt[best_idx])
            best_cost = float(fitness[best_idx])

        history.append(best_cost)

    return U_best, dt_best, {
        "best_cost": best_cost,
        "iters": iters,
        "history": np.array(history)
    }
