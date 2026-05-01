 Heuristic Optimization for the Traveling Salesman Problem

 Project Description

This project explores heuristic approaches for solving the **Traveling Salesman Problem (TSP)**, a classic NP-hard optimization problem. The goal of TSP is to find the shortest possible tour that visits each city exactly once and returns to the starting point.

Since exact solutions become computationally infeasible for larger problem sizes, this project focuses on efficient heuristic and metaheuristic algorithms that produce high-quality approximate solutions within reasonable time.

---

 Objectives

* Implement baseline and advanced heuristic algorithms for TSP
* Compare solution quality across different methods
* Analyze scalability as problem size increases
* Evaluate improvements over a simple greedy approach
* Visualize results for better interpretation

---

## Algorithms Implemented

The project includes both exact and heuristic methods:

* **Held-Karp Algorithm**
  An exact dynamic programming solution used as a reference for small instances (n ≤ 15).

* **Nearest Neighbor (NN)**
  A simple greedy heuristic that constructs a tour by repeatedly visiting the nearest unvisited city.

* **Greedy Insertion**
  Builds a tour incrementally by inserting cities at positions that minimize cost increase.

* **2-opt Local Search**
  Improves an existing tour by iteratively swapping edges to reduce total distance.

* **Simulated Annealing (SA)**
  A probabilistic metaheuristic that explores the solution space and avoids local optima using controlled randomness.

---

## Experimental Setup

The algorithms are evaluated on randomly generated Euclidean TSP instances with sizes ranging from:

* **n = 10 to n = 200**

Experiments include:

* Multiple trials with fixed random seeds for reproducibility
* Comparison of baseline vs improved solutions
* Scaling analysis to observe performance trends
* Evaluation of percentage improvement and absolute savings

---

## Visual Analysis

A separate visualization module generates graphs to support analysis, including:

* Tour route comparisons
* Algorithm performance across trials
* Scaling behavior with increasing problem size
* Percentage improvement over baseline
* Comparison with random solutions

All graphs are saved as PNG files for reporting and interpretation.

---

## Files

| File                    | Description                                                                                                                                                                       |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tsp_solver_final.py`   | Core implementation of all TSP algorithms, including city generation, distance calculations, and execution of experiments. Produces numerical outputs for performance comparison. |
| `tsp_visualizations.py` | Runs experiments using the implemented algorithms and generates visualizations such as plots and charts to analyze algorithm performance.                                         |
| `requirements.txt`      | Specifies required external libraries (`matplotlib`, `numpy`) for running visualization scripts.                                                                                  |
| `README.md`             | Documentation explaining the project, algorithms, setup, and usage instructions.                                                                                                  |

---

## How to Run

### Run Core Experiments

```bash
python tsp_solver_final.py
```

### Run Visualizations

```bash
python tsp_visualizations.py
```

---

## Requirements

* Python 3.8+
* Standard libraries: `math`, `random`, `time`, `statistics`
* Additional libraries (for visualization):

  * `matplotlib`
  * `numpy`

---

## Key Insights

* Greedy methods like Nearest Neighbor are fast but often suboptimal
* Local search (2-opt) significantly improves solution quality
* Simulated Annealing can escape local optima and provide further improvements
* Performance differences become more pronounced as problem size increases

---

## Conclusion

This project demonstrates how heuristic and metaheuristic techniques can effectively approximate solutions to complex optimization problems like TSP. By combining simple heuristics with improvement strategies, it is possible to achieve high-quality solutions with reasonable computational effort.

---
