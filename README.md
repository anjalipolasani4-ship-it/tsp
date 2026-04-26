# Heuristic Optimization for the Traveling Salesman Problem
**Anjali Polasani — CS 57200: Heuristic Problem Solving — Purdue University Fort Wayne**

---

## Project Overview
This project implements and compares heuristic algorithms for solving the
Traveling Salesman Problem (TSP). Starting from a greedy baseline (Nearest Neighbor),
two enhancements (2-opt Local Search and Simulated Annealing) are applied and evaluated
across random Euclidean instances of sizes n = 10 to n = 200.

---

## Files
| File | Description |
|---|---|
| `tsp_solver_final.py` | All TSP algorithm implementations |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

---

## Algorithms Implemented
| Algorithm | Type | Notes |
|---|---|---|
| Held-Karp | Exact Reference | Optimal solution, feasible for n ≤ 15 |
| Nearest Neighbor (NN) | Baseline | Greedy construction, O(n²) |
| Greedy Insertion | Baseline | Cheapest insertion, O(n³) |
| 2-opt Local Search | Enhancement 1 | Iterative edge reversal, O(n²) per pass |
| Simulated Annealing (SA) | Enhancement 2 | Metaheuristic, escapes local optima |

---

## Setup & Installation

### 1. Install Python
Make sure Python 3.8+ is installed:
```bash
python --version
```
Download from https://www.python.org if needed.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run

```bash
python tsp_solver_final.py
```

Results will print directly to the terminal.

---

## Output
Running `tsp_solver_final.py` produces terminal output covering:
- Test Case 1: n=5 known instance (verified against exact optimal)
- Test Case 2: n=10 random instance (all algorithms vs Held-Karp exact)
- Test Case 3: n=20 random instance
- Scaling experiment: n=10, 20, 50, 100, 200 (5 trials each)
- Baseline vs Enhancement comparison at n=50

---

## Reproducing Results
All random instances use fixed seeds for reproducibility:
- n=50 trials use seeds: 0, 77, 154, 231, 308
- Scaling experiments use seeds: 0, 100, 200, 300, 400

To reproduce all results exactly:
```bash
python tsp_solver_final.py
```

---

## Requirements
- Python 3.8+
- No external dependencies — only Python standard library (`math`, `random`, `time`, `statistics`)

---

## References
1. Dantzig, Fulkerson, Johnson — "Solution of a Large-Scale TSP" (1954)
2. Held, Karp — "The TSP and Minimum Spanning Trees" (1970)
3. Lin, Kernighan — "An Effective Heuristic for the TSP" (1973)
4. Kirkpatrick, Gelatt, Vecchi — "Optimization by Simulated Annealing" (1983)
5. Dorigo, Gambardella — "Ant Colony System" (1997)
6. Russell, Norvig — "Artificial Intelligence: A Modern Approach" (2020)
