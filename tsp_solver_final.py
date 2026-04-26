"""
tsp_solver_final.py
Anjali Polasani — CS 57200: Heuristic Problem Solving
Final Project: Heuristic Optimization for the Traveling Salesman Problem

Algorithms implemented:
  Exact:          Held-Karp dynamic programming (n <= 15)
  Baseline:       Nearest Neighbor (NN), Greedy Insertion
  Enhancement 1:  2-opt Local Search
  Enhancement 2:  Simulated Annealing (SA)

No external dependencies — only Python standard library.
Run: python tsp_solver_final.py
"""

import math
import random
import time
import statistics
from typing import List, Tuple


# ─────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────

def euclidean_distance(c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
    """Euclidean distance between two 2D points."""
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def build_distance_matrix(cities: List[Tuple[float, float]]) -> List[List[float]]:
    """Build full n×n symmetric distance matrix. O(n^2)."""
    n = len(cities)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i][j] = euclidean_distance(cities[i], cities[j])
    return dist


def tour_length(tour: List[int], dist: List[List[float]]) -> float:
    """Compute total tour length (including closing edge). O(n)."""
    n = len(tour)
    return sum(dist[tour[i]][tour[(i + 1) % n]] for i in range(n))


def generate_random_cities(n: int, seed: int = 42) -> List[Tuple[float, float]]:
    """Generate n cities uniformly at random in [0,100]^2."""
    random.seed(seed)
    return [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n)]


# ─────────────────────────────────────────
# EXACT: HELD-KARP (n <= 15)
# ─────────────────────────────────────────

def held_karp(dist: List[List[float]]) -> Tuple[float, List[int]]:
    """
    Held-Karp exact DP TSP solver.
    Time: O(n^2 * 2^n)   Space: O(n * 2^n)
    Optimal for n <= 15; memory-prohibitive for n > 20.
    """
    n = len(dist)
    INF = float('inf')
    size = 1 << n

    dp = [[INF] * n for _ in range(size)]
    parent = [[-1] * n for _ in range(size)]
    dp[1][0] = 0.0

    for S in range(1, size):
        if not (S & 1):
            continue
        for u in range(n):
            if not (S >> u & 1) or dp[S][u] == INF:
                continue
            for v in range(n):
                if S >> v & 1:
                    continue
                nS = S | (1 << v)
                cost = dp[S][u] + dist[u][v]
                if cost < dp[nS][v]:
                    dp[nS][v] = cost
                    parent[nS][v] = u

    full = size - 1
    best_cost, last = INF, -1
    for u in range(1, n):
        cost = dp[full][u] + dist[u][0]
        if cost < best_cost:
            best_cost, last = cost, u

    tour, S, u = [], full, last
    while u != -1:
        tour.append(u)
        pu = parent[S][u]
        S ^= (1 << u)
        u = pu
    tour.reverse()
    return best_cost, tour


# ─────────────────────────────────────────
# BASELINE 1: NEAREST NEIGHBOR
# ─────────────────────────────────────────

def nearest_neighbor(dist: List[List[float]], start: int = 0) -> Tuple[float, List[int]]:
    """
    Nearest Neighbor greedy construction from a single start city.
    Time: O(n^2). Typical quality: 20-25% above optimal.
    """
    n = len(dist)
    visited = [False] * n
    tour = [start]
    visited[start] = True

    for _ in range(n - 1):
        cur = tour[-1]
        best_next, best_d = -1, float('inf')
        for j in range(n):
            if not visited[j] and dist[cur][j] < best_d:
                best_d, best_next = dist[cur][j], j
        tour.append(best_next)
        visited[best_next] = True

    return tour_length(tour, dist), tour


def nearest_neighbor_best_of_all(dist: List[List[float]]) -> Tuple[float, List[int]]:
    """
    Run NN from every starting city; return the best result.
    Time: O(n^3). Recommended for n <= 500.
    """
    best_len, best_tour = float('inf'), []
    for s in range(len(dist)):
        length, tour = nearest_neighbor(dist, s)
        if length < best_len:
            best_len, best_tour = length, tour
    return best_len, best_tour


# ─────────────────────────────────────────
# BASELINE 2: GREEDY INSERTION
# ─────────────────────────────────────────

def greedy_insertion(dist: List[List[float]]) -> Tuple[float, List[int]]:
    """
    Greedy (Cheapest) Insertion constructive heuristic.
    Inserts each remaining city at the cheapest insertion point.
    Time: O(n^3). Typical quality: 15-20% above optimal.
    """
    n = len(dist)
    best_d, ci, cj = float('inf'), 0, 1
    for i in range(n):
        for j in range(i + 1, n):
            if dist[i][j] < best_d:
                best_d, ci, cj = dist[i][j], i, j
    tour = [ci, cj]
    remaining = set(range(n)) - {ci, cj}

    while remaining:
        best_inc, best_city, best_pos = float('inf'), -1, -1
        for city in remaining:
            for pos in range(len(tour)):
                a = tour[pos]
                b = tour[(pos + 1) % len(tour)]
                inc = dist[a][city] + dist[city][b] - dist[a][b]
                if inc < best_inc:
                    best_inc, best_city, best_pos = inc, city, pos
        tour.insert(best_pos + 1, best_city)
        remaining.remove(best_city)

    return tour_length(tour, dist), tour


# ─────────────────────────────────────────
# ENHANCEMENT 1: 2-OPT LOCAL SEARCH
# ─────────────────────────────────────────

def two_opt(tour: List[int], dist: List[List[float]]) -> Tuple[float, List[int]]:
    """
    2-opt local search improvement.
    Iteratively reverses segments to eliminate crossing edges.
    Time: O(n^2) per pass; typically converges in 3-8 passes.
    Improvement over NN: typically 5-12%.
    """
    best = tour[:]
    improved = True
    while improved:
        improved = False
        n = len(best)
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b = best[i - 1], best[i]
                c, d = best[j], best[(j + 1) % n]
                if dist[a][b] + dist[c][d] > dist[a][c] + dist[b][d] + 1e-10:
                    best[i:j + 1] = best[i:j + 1][::-1]
                    improved = True
    return tour_length(best, dist), best


# ─────────────────────────────────────────
# ENHANCEMENT 2: SIMULATED ANNEALING
# ─────────────────────────────────────────

def simulated_annealing(
    dist: List[List[float]],
    initial_tour: List[int],
    T_start: float = 1000.0,
    T_min: float = 1e-3,
    alpha: float = 0.995,
    iterations_per_temp: int = 100,
    seed: int = 0
) -> Tuple[float, List[int]]:
    """
    Simulated Annealing with 2-opt move operator.
    Accepts worse solutions with probability exp(-delta/T).
    Temperature decays geometrically: T <- alpha * T each level.
    Returns the best tour found during the entire run.

    Parameters:
        T_start             = 1000.0  (high enough to accept most moves initially)
        T_min               = 0.001   (termination criterion)
        alpha               = 0.995   (slow cooling for thorough exploration)
        iterations_per_temp = 100     (moves evaluated per temperature level)
    """
    random.seed(seed)
    n = len(dist)
    current = initial_tour[:]
    current_len = tour_length(current, dist)
    best = current[:]
    best_len = current_len
    T = T_start

    while T > T_min:
        for _ in range(iterations_per_temp):
            i = random.randint(0, n - 2)
            j = random.randint(i + 1, n - 1)
            a = current[i - 1]
            b = current[i]
            c = current[j]
            d = current[(j + 1) % n]
            delta = (dist[a][c] + dist[b][d]) - (dist[a][b] + dist[c][d])
            if delta < 0 or random.random() < math.exp(-delta / T):
                current[i:j + 1] = current[i:j + 1][::-1]
                current_len = tour_length(current, dist)
                if current_len < best_len:
                    best_len = current_len
                    best = current[:]
        T *= alpha

    return best_len, best


# ─────────────────────────────────────────
# EXPERIMENT RUNNER
# ─────────────────────────────────────────

def run_experiment(n: int, num_trials: int = 5, seed_base: int = 0) -> dict:
    """
    Run all algorithms on random instances of size n.
    Returns summary dict with mean/stdev of tour lengths and mean runtimes.
    """
    keys = ["nn", "greedy", "two_opt", "sa"]
    results = {k: [] for k in keys}
    times   = {k: [] for k in keys}
    if n <= 15:
        results["hk"] = []
        times["hk"]   = []

    for trial in range(num_trials):
        cities = generate_random_cities(n, seed=seed_base + trial * 100)
        dist   = build_distance_matrix(cities)

        if n <= 15:
            t0 = time.perf_counter()
            hk_len, _ = held_karp(dist)
            results["hk"].append(hk_len)
            times["hk"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        nn_len, nn_tour = nearest_neighbor_best_of_all(dist)
        results["nn"].append(nn_len)
        times["nn"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        gi_len, _ = greedy_insertion(dist)
        results["greedy"].append(gi_len)
        times["greedy"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        two_len, _ = two_opt(nn_tour, dist)
        results["two_opt"].append(two_len)
        times["two_opt"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        sa_len, _ = simulated_annealing(dist, nn_tour, seed=trial)
        results["sa"].append(sa_len)
        times["sa"].append(time.perf_counter() - t0)

    def summarize(vals):
        mean = statistics.mean(vals)
        std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return mean, std

    summary = {"n": n}
    for k in keys:
        m, s = summarize(results[k])
        summary[f"{k}_mean"] = m
        summary[f"{k}_std"]  = s
        summary[f"{k}_time"] = statistics.mean(times[k])
    if n <= 15:
        m, s = summarize(results["hk"])
        summary["hk_mean"] = m
        summary["hk_std"]  = s
        summary["hk_time"] = statistics.mean(times["hk"])

    return summary


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    SEP = "=" * 65

    print(SEP)
    print("TSP Solver — Anjali Polasani — CS 57200 Final Project")
    print("Algorithms: Held-Karp | NN | Greedy | 2-opt | SA")
    print(SEP)

    # ── Test Case 1: n=5 known instance ──────────────────────────────────
    print("\n[Test 1] n=5 known instance")
    cities5 = [(0,0),(1,0),(1,1),(0,1),(0.5,0.5)]
    d5 = build_distance_matrix(cities5)
    hk_len5, hk_tour5 = held_karp(d5)
    nn_len5, nn_tour5 = nearest_neighbor(d5, 0)
    print(f"  Held-Karp (exact): {hk_len5:.4f}  tour={hk_tour5}")
    print(f"  Nearest Neighbor:  {nn_len5:.4f}  tour={nn_tour5}")

    # ── Test Case 2: n=10 random ─────────────────────────────────────────
    print("\n[Test 2] n=10 random instance (seed=42)")
    cities10 = generate_random_cities(10, seed=42)
    d10 = build_distance_matrix(cities10)
    hk10, _      = held_karp(d10)
    nn10, nn_t10 = nearest_neighbor_best_of_all(d10)
    gi10, _      = greedy_insertion(d10)
    t2_10, _     = two_opt(nn_t10, d10)
    sa10, _      = simulated_annealing(d10, nn_t10, seed=0)
    print(f"  Held-Karp (exact):   {hk10:.4f}  (reference)")
    for label, val in [("Nearest Neighbor", nn10), ("Greedy Insertion", gi10),
                        ("NN + 2-opt", t2_10), ("NN + SA", sa10)]:
        pct = (val / hk10 - 1) * 100
        print(f"  {label:<20s}: {val:.4f}  ({pct:+.1f}% vs optimal)")

    # ── Test Case 3: n=20 ────────────────────────────────────────────────
    print("\n[Test 3] n=20 random instance (seed=7)")
    cities20 = generate_random_cities(20, seed=7)
    d20 = build_distance_matrix(cities20)
    nn20, nn_t20 = nearest_neighbor_best_of_all(d20)
    gi20, _      = greedy_insertion(d20)
    t2_20, _     = two_opt(nn_t20, d20)
    sa20, _      = simulated_annealing(d20, nn_t20, seed=0)
    print(f"  Nearest Neighbor:    {nn20:.4f}  (baseline)")
    for label, val in [("Greedy Insertion", gi20), ("NN + 2-opt", t2_20),
                        ("NN + SA", sa20)]:
        pct = (1 - val / nn20) * 100
        print(f"  {label:<20s}: {val:.4f}  ({pct:+.1f}% vs NN baseline)")

    # ── Experiment A: Scaling ─────────────────────────────────────────────
    print("\n" + SEP)
    print("[Experiment A] Scaling — tour quality and runtime vs. n (5 trials each)")
    print(SEP)
    print(f"{'n':>5}  {'NN':>9}  {'2-opt':>9}  {'SA':>9}  {'SA vs NN':>10}  {'SA time':>9}")
    print("-" * 55)
    for n in [10, 20, 50, 100, 200]:
        r = run_experiment(n, num_trials=5)
        pct = (1 - r["sa_mean"] / r["nn_mean"]) * 100
        print(f"{n:>5}  {r['nn_mean']:>9.2f}  {r['two_opt_mean']:>9.2f}  "
              f"{r['sa_mean']:>9.2f}  {pct:>9.1f}%  {r['sa_time']:>9.4f}s")

    # ── Experiment B: Baseline vs. Enhancements, n=50 ────────────────────
    print("\n" + SEP)
    print("[Experiment B] Baseline vs. Enhancements — n=50, 5 trials")
    print(SEP)
    print(f"{'Trial':>6}  {'NN':>9}  {'2-opt':>9}  {'SA':>9}")
    print("-" * 40)
    for trial in range(5):
        cities = generate_random_cities(50, seed=trial * 77)
        d = build_distance_matrix(cities)
        nn_l, nn_t = nearest_neighbor_best_of_all(d)
        t2_l, _    = two_opt(nn_t, d)
        sa_l, _    = simulated_annealing(d, nn_t, seed=trial)
        print(f"{trial+1:>6}  {nn_l:>9.2f}  {t2_l:>9.2f}  {sa_l:>9.2f}")

    # ── Experiment C: Random agent comparison ─────────────────────────────
    print("\n" + SEP)
    print("[Experiment C] NN vs. Random Agent — n=50, 5 trials")
    print(SEP)
    print(f"{'Trial':>6}  {'Random':>10}  {'NN':>10}")
    print("-" * 30)
    for trial in range(5):
        cities = generate_random_cities(50, seed=trial * 77)
        d = build_distance_matrix(cities)
        rng = random.Random(trial)
        rand_tour = list(range(50))
        rng.shuffle(rand_tour)
        rand_l = tour_length(rand_tour, d)
        nn_l, _ = nearest_neighbor_best_of_all(d)
        print(f"{trial+1:>6}  {rand_l:>10.2f}  {nn_l:>10.2f}")

    print(f"\nAll tests complete.")


if __name__ == "__main__":
    main()
