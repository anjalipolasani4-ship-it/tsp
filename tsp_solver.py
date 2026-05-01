import math
import random
import time
import statistics
from typing import List, Tuple


# ─────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────

def euclidean_distance(c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def build_distance_matrix(cities: List[Tuple[float, float]]) -> List[List[float]]:
    n = len(cities)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i][j] = euclidean_distance(cities[i], cities[j])
    return dist


def tour_length(tour: List[int], dist: List[List[float]]) -> float:
    n = len(tour)
    return sum(dist[tour[i]][tour[(i + 1) % n]] for i in range(n))


def generate_random_cities(n: int, seed: int = 42) -> List[Tuple[float, float]]:
    random.seed(seed)
    return [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n)]


# ─────────────────────────────────────────
# BASELINE: NEAREST NEIGHBOR HEURISTIC
# ─────────────────────────────────────────

def nearest_neighbor(dist: List[List[float]], start: int = 0) -> Tuple[float, List[int]]:
    """
    Nearest Neighbor constructive heuristic.
    Greedily appends the closest unvisited city.
    Time: O(n^2). Typically 20-25% above optimal.
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

    length = tour_length(tour, dist)
    return length, tour


def nearest_neighbor_best_of_all(dist: List[List[float]]) -> Tuple[float, List[int]]:
    """Run NN from every starting city and return the best result."""
    best_len, best_tour = float('inf'), []
    for s in range(len(dist)):
        length, tour = nearest_neighbor(dist, s)
        if length < best_len:
            best_len, best_tour = length, tour
    return best_len, best_tour


# ─────────────────────────────────────────
# GREEDY INSERTION (additional baseline)
# ─────────────────────────────────────────

def greedy_insertion(dist: List[List[float]]) -> Tuple[float, List[int]]:
    """
    Greedy Insertion constructive heuristic.
    Inserts each remaining city at the position causing minimum cost increase.
    Typically 15-20% above optimal.
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
# MAIN
# ─────────────────────────────────────────

def main():
    print("=" * 65)
    print("TSP Solver - Baseline Methods Only")
    print("=" * 65)

    # --- Test Case 1: Tiny known instance (n=5) ---
    print("\n[Test 1] n=5 known instance")
    cities5 = [(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)]
    d5 = build_distance_matrix(cities5)
    nn_len, nn_tour = nearest_neighbor(d5, 0)
    gi_len, gi_tour = greedy_insertion(d5)
    print(f"  Nearest Neighbor:  {nn_len:.4f}  tour={nn_tour}")
    print(f"  Greedy Insertion:  {gi_len:.4f}  tour={gi_tour}")

    # --- Test Case 2: Small random instance (n=10) ---
    print("\n[Test 2] n=10 random instance (seed=42)")
    cities10 = generate_random_cities(10, seed=42)
    d10 = build_distance_matrix(cities10)
    nn_len10, nn_tour10 = nearest_neighbor_best_of_all(d10)
    gi_len10, _ = greedy_insertion(d10)
    print(f"  Nearest Neighbor (best-of-all starts): {nn_len10:.4f}")
    print(f"  Greedy Insertion:                      {gi_len10:.4f}")

    # --- Test Case 3: Medium instance (n=20) ---
    print("\n[Test 3] n=20 random instance (seed=7)")
    cities20 = generate_random_cities(20, seed=7)
    d20 = build_distance_matrix(cities20)
    nn_len20, _ = nearest_neighbor_best_of_all(d20)
    gi_len20, _ = greedy_insertion(d20)
    print(f"  Nearest Neighbor: {nn_len20:.4f}")
    print(f"  Greedy Insertion: {gi_len20:.4f}")

    # --- Scaling Experiment ---
    print("\n[Experiment] Scaling: baseline tour quality vs. n")
    print(f"{'n':>5}  {'NN mean':>12}  {'NN time(s)':>12}  {'GI mean':>12}  {'GI time(s)':>12}")
    print("-" * 60)

    for n in [10, 20, 50, 100, 200]:
        nn_lengths, nn_times, gi_lengths, gi_times = [], [], [], []
        for trial in range(5):
            cities = generate_random_cities(n, seed=trial * 100)
            dist = build_distance_matrix(cities)

            t0 = time.perf_counter()
            nn_l, _ = nearest_neighbor_best_of_all(dist)
            nn_times.append(time.perf_counter() - t0)
            nn_lengths.append(nn_l)

            t0 = time.perf_counter()
            gi_l, _ = greedy_insertion(dist)
            gi_times.append(time.perf_counter() - t0)
            gi_lengths.append(gi_l)

        print(f"{n:>5}  {statistics.mean(nn_lengths):>12.2f}  {statistics.mean(nn_times):>12.4f}"
              f"  {statistics.mean(gi_lengths):>12.2f}  {statistics.mean(gi_times):>12.4f}")

    print("\nAll tests complete.")


if __name__ == "__main__":
    main()
