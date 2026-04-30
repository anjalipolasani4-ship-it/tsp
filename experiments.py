import time
import random
import math
import statistics
import itertools

# -----------------------------
# Utils
# -----------------------------
def generate_random_cities(n, seed=None):
    if seed is not None:
        random.seed(seed)
    return [(random.random(), random.random()) for _ in range(n)]

def distance(a, b):
    return math.dist(a, b)

def build_distance_matrix(cities):
    n = len(cities)
    return [[distance(cities[i], cities[j]) for j in range(n)] for i in range(n)]

def tour_length(tour, dist):
    return sum(dist[tour[i]][tour[(i+1) % len(tour)]] for i in range(len(tour)))

# -----------------------------
# Nearest Neighbor
# -----------------------------
def nearest_neighbor(start, dist):
    n = len(dist)
    visited = [False]*n
    tour = [start]
    visited[start] = True

    for _ in range(n-1):
        last = tour[-1]
        next_city = min(
            [(dist[last][j], j) for j in range(n) if not visited[j]]
        )[1]
        tour.append(next_city)
        visited[next_city] = True

    return tour_length(tour, dist), tour

def nearest_neighbor_best_of_all(dist):
    best_len = float('inf')
    best_tour = None

    for i in range(len(dist)):
        length, tour = nearest_neighbor(i, dist)
        if length < best_len:
            best_len = length
            best_tour = tour

    return best_len, best_tour

# -----------------------------
# Greedy Insertion
# -----------------------------
def greedy_insertion(dist):
    n = len(dist)
    tour = [0, 1]

    for k in range(2, n):
        best_pos = None
        best_increase = float('inf')

        for i in range(len(tour)):
            j = (i + 1) % len(tour)
            increase = (
                dist[tour[i]][k]
                + dist[k][tour[j]]
                - dist[tour[i]][tour[j]]
            )
            if increase < best_increase:
                best_increase = increase
                best_pos = j

        tour.insert(best_pos, k)

    return tour_length(tour, dist), tour

# -----------------------------
# 2-opt
# -----------------------------
def two_opt(tour, dist):
    best = tour
    improved = True

    while improved:
        improved = False
        for i in range(1, len(tour) - 2):
            for j in range(i + 1, len(tour)):
                if j - i == 1:
                    continue
                new_tour = best[:]
                new_tour[i:j] = best[j-1:i-1:-1]

                if tour_length(new_tour, dist) < tour_length(best, dist):
                    best = new_tour
                    improved = True
        tour = best

    return tour_length(best, dist), best

# -----------------------------
# Simulated Annealing
# -----------------------------
def simulated_annealing(dist, initial_tour, seed=0):
    random.seed(seed)
    current = initial_tour[:]
    current_len = tour_length(current, dist)

    T = 1000
    cooling = 0.995

    for _ in range(1000):
        i, j = sorted(random.sample(range(len(current)), 2))
        new = current[:]
        new[i:j] = reversed(new[i:j])

        new_len = tour_length(new, dist)
        delta = new_len - current_len

        if delta < 0 or random.random() < math.exp(-delta / T):
            current = new
            current_len = new_len

        T *= cooling

    return current_len, current

# -----------------------------
# Held-Karp (Exact)
# -----------------------------
def held_karp(dist):
    n = len(dist)
    C = {}

    for k in range(1, n):
        C[(1 << k, k)] = (dist[0][k], 0)

    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
            bits = sum(1 << k for k in subset)

            for k in subset:
                prev = bits & ~(1 << k)
                res = []

                for m in subset:
                    if m == k:
                        continue
                    res.append((C[(prev, m)][0] + dist[m][k], m))

                C[(bits, k)] = min(res)

    bits = (2**n - 1) - 1
    res = [(C[(bits, k)][0] + dist[k][0], k) for k in range(1, n)]
    opt, parent = min(res)

    return opt, []

# -----------------------------
# Experiment Runner
# -----------------------------
def run_experiment(n, num_trials=5):
    keys = ["nn", "greedy", "two_opt", "sa"]

    results = {k: [] for k in keys}
    times = {k: [] for k in keys}

    if n <= 12:  # keep small for speed
        results["hk"] = []
        times["hk"] = []

    for trial in range(num_trials):
        cities = generate_random_cities(n, seed=trial)
        dist = build_distance_matrix(cities)

        if n <= 12:
            t0 = time.perf_counter()
            hk_len, _ = held_karp(dist)
            times["hk"].append(time.perf_counter() - t0)
            results["hk"].append(hk_len)

        t0 = time.perf_counter()
        nn_len, nn_tour = nearest_neighbor_best_of_all(dist)
        times["nn"].append(time.perf_counter() - t0)
        results["nn"].append(nn_len)

        t0 = time.perf_counter()
        gi_len, _ = greedy_insertion(dist)
        times["greedy"].append(time.perf_counter() - t0)
        results["greedy"].append(gi_len)

        t0 = time.perf_counter()
        two_len, _ = two_opt(nn_tour, dist)
        times["two_opt"].append(time.perf_counter() - t0)
        results["two_opt"].append(two_len)

        t0 = time.perf_counter()
        sa_len, _ = simulated_annealing(dist, nn_tour)
        times["sa"].append(time.perf_counter() - t0)
        results["sa"].append(sa_len)

    print("\nResults for n =", n)
    for k in results:
        print(
            f"{k}: mean={statistics.mean(results[k]):.3f}, "
            f"time={statistics.mean(times[k]):.4f}s"
        )

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    run_experiment(n=10, num_trials=5)
