"""
mario_neuroevolution.py
=======================
Evolutionary Algorithm (EA) with Elitism for the Mario MLP agent.

Algorithm
---------
1. Initialise a random population of N weight vectors.
2. Evaluate every individual by running the MarioAI simulator.
3. Sort individuals by reward (descending).
4. Carry the top-K elites unchanged into the next generation.
5. Fill the rest with tournament-selected parents + Gaussian mutation.
6. Repeat for G generations.

Usage
-----
    python mario_neuroevolution.py <seed>
    python mario_neuroevolution.py <seed> --population 20 --generations 50 \
                                          --sigma 0.1 --elite_frac 0.1 \
                                          --tournament 3

The seed is positional (kept compatible with the existing random_search
convention of `python script.py <seed>`).
"""

import argparse
import csv
import pickle as pkl
import sys
import time
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend; safe inside multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import torch

from agents.mlp_agent import MLPAgent
from evaluation import evaluate_population

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_plot(best_rewards, mean_rewards, title="EA", path="EA.png"):
    """Save a generation-vs-reward plot to disk."""
    fig, ax = plt.subplots(figsize=(9, 5))
    gens = range(1, len(best_rewards) + 1)
    ax.plot(gens, best_rewards, label="Best Reward",  color="#2196F3", linewidth=2)
    ax.plot(gens, mean_rewards, label="Mean Reward",  color="#FF9800", linewidth=2,
            linestyle="--", alpha=0.8)
    ax.fill_between(gens, mean_rewards, best_rewards, alpha=0.15, color="#2196F3")
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Reward",     fontsize=12)
    ax.set_title(title,         fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[EA] Plot saved → {path}")


# ---------------------------------------------------------------------------
# EA operators
# ---------------------------------------------------------------------------

def _tournament_select(population, fitnesses, k):
    """
    Select one individual via k-way tournament selection.

    Parameters
    ----------
    population : list[np.ndarray]
    fitnesses  : np.ndarray  shape (N,)
    k          : int          tournament size

    Returns
    -------
    np.ndarray  — copy of the winning weight vector
    """
    N = len(population)
    indices  = np.random.choice(N, size=k, replace=False)
    winner   = indices[np.argmax(fitnesses[indices])]
    return deepcopy(population[winner])


def _mutate(parent, sigma):
    """
    Gaussian (isotropic) mutation.

    Returns a new weight vector: parent + σ * N(0, I)
    """
    noise = sigma * np.random.randn(len(parent))
    return parent + noise


# ---------------------------------------------------------------------------
# Main EA loop
# ---------------------------------------------------------------------------

def evolutionary_algorithm(
    population_size: int = 20,
    generations:     int = 50,
    sigma:           float = 0.1,
    elite_frac:      float = 0.1,
    tournament_size: int = 3,
    seed:            int = 42,
    starting_difficulty: int = 0,
):
    """
    Evolve the MLPAgent weight vector.

    Parameters
    ----------
    population_size : number of individuals per generation
    generations     : number of EA generations
    sigma           : Gaussian mutation standard deviation
    elite_frac      : fraction of top individuals copied verbatim (elitism)
    tournament_size : tournament size for parent selection
    seed            : RNG seed (applied before this function is called)

    Returns
    -------
    best_params : np.ndarray — weight vector of the all-time best individual
    """

    # Derive elite count (minimum 1 so we always preserve the best)
    n_elites = max(1, int(round(population_size * elite_frac)))

    # Reference agent to determine the parameter dimension
    ref_agent  = MLPAgent()
    num_params = len(ref_agent.get_param_vector())

    print(f"[EA] Input dim      : {ref_agent.input_dim}")
    print(f"[EA] Parameter count: {num_params}")
    print(f"[EA] Population     : {population_size}")
    print(f"[EA] Elites         : {n_elites}  ({elite_frac*100:.0f}%)")
    print(f"[EA] Tournament size: {tournament_size}")
    print(f"[EA] Sigma          : {sigma}")
    print(f"[EA] Generations    : {generations}")
    print(f"[EA] Start difficulty: {starting_difficulty}")
    print()

    # ---- Initialise population ----
    population = [np.random.randn(num_params) * sigma for _ in range(population_size)]

    # Tracking
    all_time_best_reward = -np.inf
    all_time_best_params = None
    best_rewards = []
    mean_rewards = []

    # Persistence paths
    save_dir = Path("data/mlp_best_agents")
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / f"ea_log_seed{seed}.csv"

    # Open CSV log
    csv_file   = log_path.open("w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["generation", "best_reward", "mean_reward",
                         "min_reward", "std_reward", "n_elites"])

    t_start = time.perf_counter()

    for gen in range(generations):
        t_gen = time.perf_counter()
        print(f"--- Generation {gen+1}/{generations} ---")

        # ---- Evaluate ----
        rewards = evaluate_population(MLPAgent, population,
                                      starting_difficulty=starting_difficulty)
        rewards = np.array(rewards, dtype=np.float64)

        gen_best   = float(rewards.max())
        gen_mean   = float(rewards.mean())
        gen_min    = float(rewards.min())
        gen_std    = float(rewards.std())

        best_rewards.append(gen_best)
        mean_rewards.append(gen_mean)

        # Sort descending by reward
        sorted_idx  = np.argsort(rewards)[::-1]
        pop_sorted  = [population[i] for i in sorted_idx]
        fit_sorted  = rewards[sorted_idx]

        # Track all-time best
        if gen_best > all_time_best_reward:
            all_time_best_reward = gen_best
            all_time_best_params = deepcopy(pop_sorted[0])
            ckpt_path = save_dir / f"ea_gen{gen+1:03d}_seed{seed}_{gen_best:.3f}.pkl"
            with ckpt_path.open("wb") as f:
                pkl.dump(all_time_best_params, f)
            print(f"  *** New best! Reward = {gen_best:.3f}  → saved {ckpt_path.name}")

        elapsed = time.perf_counter() - t_gen
        print(f"  Best={gen_best:.3f}  Mean={gen_mean:.3f}  "
              f"Min={gen_min:.3f}  Std={gen_std:.3f}  "
              f"[{elapsed:.1f}s]")

        # CSV log
        csv_writer.writerow([gen+1, gen_best, gen_mean, gen_min, gen_std, n_elites])
        csv_file.flush()

        # ---- Build next generation ----
        # 1. Elites — copy top-K verbatim
        next_population = [deepcopy(pop_sorted[i]) for i in range(n_elites)]

        # 2. Fill remaining slots with tournament selection + mutation
        while len(next_population) < population_size:
            parent = _tournament_select(pop_sorted, fit_sorted, tournament_size)
            child  = _mutate(parent, sigma)
            next_population.append(child)

        population = next_population

    csv_file.close()
    total_time = time.perf_counter() - t_start

    print(f"\n[EA] Finished {generations} generations in {total_time:.1f}s")
    print(f"[EA] All-time best reward: {all_time_best_reward:.3f}")

    # Final plot
    save_plot(best_rewards, mean_rewards,
              title=f"Mario EA  (pop={population_size}, σ={sigma}, elites={n_elites})",
              path="EA.png")

    # Save final best
    final_path = save_dir / f"ea_FINAL_seed{seed}_{all_time_best_reward:.3f}.pkl"
    with final_path.open("wb") as f:
        pkl.dump(all_time_best_params, f)
    print(f"[EA] Final best saved → {final_path}")

    return all_time_best_params


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Neuroevolution (EA with Elitism) for Mario MLP agent"
    )
    parser.add_argument("seed", type=int,
                        help="RNG seed (positional, for compatibility with existing scripts)")
    parser.add_argument("--population",  type=int,   default=20,
                        help="Population size (default: 20)")
    parser.add_argument("--generations", type=int,   default=50,
                        help="Number of generations (default: 50)")
    parser.add_argument("--sigma",       type=float, default=0.1,
                        help="Gaussian mutation σ (default: 0.1)")
    parser.add_argument("--elite_frac",  type=float, default=0.1,
                        help="Elite fraction [0,1] (default: 0.1 = 10%%)")
    parser.add_argument("--tournament",  type=int,   default=3,
                        help="Tournament size for parent selection (default: 3)")
    parser.add_argument("--difficulty",  type=int,   default=0,
                        help="Starting level difficulty for each evaluation (default: 0)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Seed all RNGs before anything else
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("  Mario Neuroevolution – EA with Elitism")
    print("=" * 60)

    best = evolutionary_algorithm(
        population_size     = args.population,
        generations         = args.generations,
        sigma               = args.sigma,
        elite_frac          = args.elite_frac,
        tournament_size     = args.tournament,
        seed                = args.seed,
        starting_difficulty = args.difficulty,
    )
