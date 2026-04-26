"""
tsp_visualizations.py
Anjali Polasani — CS 57200: Heuristic Problem Solving

Runs tsp_solver_final.py algorithms LIVE and generates all visualizations.
Place this file in the SAME folder as tsp_solver_final.py

Requires: pip install matplotlib numpy
Run:      python tsp_visualizations.py
"""

import sys
import os
import math
import random
import time
import statistics
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

# ── Import solver functions from tsp_solver_final.py ─────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsp_solver_final import (
    generate_random_cities,
    build_distance_matrix,
    tour_length,
    nearest_neighbor,
    nearest_neighbor_best_of_all,
    greedy_insertion,
    two_opt,
    simulated_annealing,
    held_karp,
)

# ── Style ─────────────────────────────────────────────────────────────────────
C_NN  = '#888780'
C_GI  = '#e0a840'
C_2OP = '#4a90d9'
C_SA  = '#2ecc8f'
C_RND = '#e05c5c'
C_HK  = '#c97ddb'

plt.rcParams.update({
    'figure.facecolor':  '#0d0d0d',
    'axes.facecolor':    '#161616',
    'axes.edgecolor':    '#2a2a2a',
    'axes.labelcolor':   '#aaaaaa',
    'axes.titlecolor':   '#e8e8e8',
    'axes.titlesize':    11,
    'axes.titleweight':  'bold',
    'axes.grid':         True,
    'grid.color':        '#2a2a2a',
    'grid.linewidth':    0.6,
    'xtick.color':       '#888888',
    'ytick.color':       '#888888',
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'legend.fontsize':   8,
    'legend.framealpha': 0.2,
    'legend.edgecolor':  '#444444',
    'legend.facecolor':  '#1f1f1f',
    'text.color':        '#e8e8e8',
    'font.family':       'monospace',
})

# ─────────────────────────────────────────
# RUN EXPERIMENTS LIVE
# ─────────────────────────────────────────

print("=" * 60)
print("TSP Visualizations — Running experiments live...")
print("=" * 60)

# --- Experiment 1: n=50, 5 trials ---
print("\n[1/4] Running n=50 trials...")
nn_50, opt_50, sa_50, gi_50, rnd_50 = [], [], [], [], []
for trial in range(5):
    seed = trial * 77
    cities = generate_random_cities(50, seed=seed)
    dist = build_distance_matrix(cities)

    rng2 = random.Random(seed + 1)
    rt = list(range(50)); rng2.shuffle(rt)
    rnd_50.append(tour_length(rt, dist))

    nn_l, nn_t = nearest_neighbor_best_of_all(dist)
    nn_50.append(nn_l)
    gi_l, _ = greedy_insertion(dist)
    gi_50.append(gi_l)
    t2_l, _ = two_opt(nn_t, dist)
    opt_50.append(t2_l)
    sa_l, _ = simulated_annealing(dist, nn_t, seed=trial)
    sa_50.append(sa_l)
    print(f"  Trial {trial+1}: NN={nn_l:.1f}  GI={gi_l:.1f}  2-opt={t2_l:.1f}  SA={sa_l:.1f}")

# --- Experiment 2: Scaling ---
print("\n[2/4] Running scaling experiment...")
sizes = [10, 20, 50, 100, 200]
nn_sc, gi_sc, opt_sc, sa_sc = [], [], [], []
for n in sizes:
    nn_t2, gi_t2, opt_t2, sa_t2 = [], [], [], []
    for trial in range(5):
        cities = generate_random_cities(n, seed=trial * 100)
        dist = build_distance_matrix(cities)
        nn_l, nn_t = nearest_neighbor_best_of_all(dist)
        gi_l, _    = greedy_insertion(dist)
        t2_l, _    = two_opt(nn_t, dist)
        sa_l, _    = simulated_annealing(dist, nn_t, seed=trial)
        nn_t2.append(nn_l); gi_t2.append(gi_l)
        opt_t2.append(t2_l); sa_t2.append(sa_l)
    nn_sc.append(statistics.mean(nn_t2))
    gi_sc.append(statistics.mean(gi_t2))
    opt_sc.append(statistics.mean(opt_t2))
    sa_sc.append(statistics.mean(sa_t2))
    print(f"  n={n}: NN={nn_sc[-1]:.1f}  GI={gi_sc[-1]:.1f}  2-opt={opt_sc[-1]:.1f}  SA={sa_sc[-1]:.1f}")

# --- Experiment 3: tour routes for visualization ---
print("\n[3/4] Generating tour route data...")
cities_vis = generate_random_cities(20, seed=42)
dist_vis = build_distance_matrix(cities_vis)
nn_l_vis, nn_t_vis   = nearest_neighbor_best_of_all(dist_vis)
t2_l_vis, t2_t_vis   = two_opt(nn_t_vis, dist_vis)
sa_l_vis, sa_t_vis   = simulated_annealing(dist_vis, nn_t_vis, seed=0)

# --- Derived metrics ---
print("\n[4/4] Computing metrics...")
pct_2opt = [(1 - opt_sc[i]/nn_sc[i])*100 for i in range(len(sizes))]
pct_sa   = [(1 - sa_sc[i]/nn_sc[i])*100  for i in range(len(sizes))]
savings_opt = [nn_50[i] - opt_50[i] for i in range(5)]
savings_sa  = [nn_50[i] - sa_50[i]  for i in range(5)]
trial_labels = [f'Trial {i+1}' for i in range(5)]
x  = np.arange(5)
x2 = np.arange(len(sizes))

# ─────────────────────────────────────────
# HELPER: draw a tour on an axis
# ─────────────────────────────────────────

def draw_tour(ax, cities, tour, color, title, length):
    xs = [cities[i][0] for i in tour] + [cities[tour[0]][0]]
    ys = [cities[i][1] for i in tour] + [cities[tour[0]][1]]
    ax.plot(xs, ys, '-', color=color, linewidth=1.2, alpha=0.8)
    ax.scatter([c[0] for c in cities], [c[1] for c in cities],
               color='white', s=25, zorder=5)
    ax.scatter(cities[tour[0]][0], cities[tour[0]][1],
               color=C_SA, s=60, zorder=6, label='Start')
    ax.set_title(f'{title}\nLength: {length:.1f}', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

# ─────────────────────────────────────────
# GRAPH 1: Tour Route Comparison (n=20)
# ─────────────────────────────────────────

print("\nPlotting Graph 1 — Tour Route Comparison...")
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('Graph 1 — Tour Route Comparison  |  n=20 cities',
             fontsize=12, fontweight='bold', color='#f0c040')
draw_tour(axes[0], cities_vis, nn_t_vis,  C_NN,  'NN Baseline',   nn_l_vis)
draw_tour(axes[1], cities_vis, t2_t_vis,  C_2OP, 'NN + 2-opt',    t2_l_vis)
draw_tour(axes[2], cities_vis, sa_t_vis,  C_SA,  'NN + SA',       sa_l_vis)
fig.tight_layout()
fig.savefig('graph1_tour_routes.png', dpi=150, facecolor=fig.get_facecolor())
print("  Saved: graph1_tour_routes.png")
plt.show()

# ─────────────────────────────────────────
# GRAPH 2: Trial Comparison n=50
# ─────────────────────────────────────────

print("Plotting Graph 2 — Trial Comparison...")
fig, ax = plt.subplots(figsize=(10, 5))
w = 0.2
ax.bar(x - 1.5*w, nn_50,  width=w, label='NN baseline',     color=C_NN,  alpha=0.9)
ax.bar(x - 0.5*w, gi_50,  width=w, label='Greedy Insertion', color=C_GI,  alpha=0.9)
ax.bar(x + 0.5*w, opt_50, width=w, label='NN + 2-opt',       color=C_2OP, alpha=0.9)
ax.bar(x + 1.5*w, sa_50,  width=w, label='NN + SA',          color=C_SA,  alpha=0.9)
ax.set_xticks(x); ax.set_xticklabels(trial_labels)
ax.set_ylabel('Tour Length')
ax.set_title('Graph 2 — All Algorithms per Trial  |  n=50, 5 Trials')
ax.legend()
fig.tight_layout()
fig.savefig('graph2_trial_comparison.png', dpi=150, facecolor=fig.get_facecolor())
print("  Saved: graph2_trial_comparison.png")
plt.show()

# ─────────────────────────────────────────
# GRAPH 3: Scaling Line Chart
# ─────────────────────────────────────────

print("Plotting Graph 3 — Scaling...")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(sizes, nn_sc,  marker='o', label='NN baseline',      color=C_NN,  linewidth=2)
ax.plot(sizes, gi_sc,  marker='D', label='Greedy Insertion', color=C_GI,  linewidth=2, linestyle='-.')
ax.plot(sizes, opt_sc, marker='s', label='NN + 2-opt',       color=C_2OP, linewidth=2, linestyle='--')
ax.plot(sizes, sa_sc,  marker='^', label='NN + SA',          color=C_SA,  linewidth=2, linestyle=':')
ax.set_xlabel('Problem Size (n)')
ax.set_ylabel('Mean Tour Length')
ax.set_title('Graph 3 — Scaling  |  Mean Tour Length vs Problem Size n')
ax.set_xticks(sizes); ax.legend()
fig.tight_layout()
fig.savefig('graph3_scaling.png', dpi=150, facecolor=fig.get_facecolor())
print("  Saved: graph3_scaling.png")
plt.show()

# ─────────────────────────────────────────
# GRAPH 4: % Improvement over NN
# ─────────────────────────────────────────

print("Plotting Graph 4 — % Improvement...")
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x2 - 0.2, pct_2opt, width=0.35, label='2-opt % gain', color=C_2OP, alpha=0.9)
ax.bar(x2 + 0.2, pct_sa,   width=0.35, label='SA % gain',    color=C_SA,  alpha=0.9)
for i, (v1, v2) in enumerate(zip(pct_2opt, pct_sa)):
    ax.text(i - 0.2, v1 + 0.1, f'{v1:.1f}%', ha='center', fontsize=7, color='#e8e8e8')
    ax.text(i + 0.2, v2 + 0.1, f'{v2:.1f}%', ha='center', fontsize=7, color='#e8e8e8')
ax.set_xticks(x2); ax.set_xticklabels([f'n={n}' for n in sizes])
ax.set_ylabel('% Improvement over NN')
ax.set_title('Graph 4 — % Improvement over NN Baseline  |  by Problem Size')
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
ax.legend()
fig.tight_layout()
fig.savefig('graph4_pct_improvement.png', dpi=150, facecolor=fig.get_facecolor())
print("  Saved: graph4_pct_improvement.png")
plt.show()

# ─────────────────────────────────────────
# GRAPH 5: Absolute Savings
# ─────────────────────────────────────────

print("Plotting Graph 5 — Absolute Savings...")
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - 0.2, savings_opt, width=0.35, label='2-opt savings', color=C_2OP, alpha=0.9)
ax.bar(x + 0.2, savings_sa,  width=0.35, label='SA savings',    color=C_SA,  alpha=0.9)
for i, (v1, v2) in enumerate(zip(savings_opt, savings_sa)):
    ax.text(i - 0.2, v1 + 0.3, f'{v1:.1f}', ha='center', fontsize=7, color='#e8e8e8')
    ax.text(i + 0.2, v2 + 0.3, f'{v2:.1f}', ha='center', fontsize=7, color='#e8e8e8')
ax.set_xticks(x); ax.set_xticklabels(trial_labels)
ax.set_ylabel('Units Saved vs NN Baseline')
ax.set_title('Graph 5 — Absolute Savings vs NN Baseline  |  n=50, 5 Trials')
ax.legend()
fig.tight_layout()
fig.savefig('graph5_savings.png', dpi=150, facecolor=fig.get_facecolor())
print("  Saved: graph5_savings.png")
plt.show()

# ─────────────────────────────────────────
# GRAPH 6: NN vs Random Agent
# ─────────────────────────────────────────

print("Plotting Graph 6 — NN vs Random...")
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - 0.2, rnd_50, width=0.35, label='Random tour', color=C_RND, alpha=0.9)
ax.bar(x + 0.2, nn_50,  width=0.35, label='NN baseline', color=C_NN,  alpha=0.9)
for i, (r, n) in enumerate(zip(rnd_50, nn_50)):
    pct = (1 - n/r)*100
    ax.text(i, max(r, n) + 30, f'{pct:.0f}% better', ha='center', fontsize=7, color=C_SA)
ax.set_xticks(x); ax.set_xticklabels(trial_labels)
ax.set_ylabel('Tour Length')
ax.set_title('Graph 6 — NN Baseline vs Random Agent  |  n=50, 5 Trials')
ax.legend()
fig.tight_layout()
fig.savefig('graph6_random_vs_nn.png', dpi=150, facecolor=fig.get_facecolor())
print("  Saved: graph6_random_vs_nn.png")
plt.show()

# ─────────────────────────────────────────
# GRAPH 7: Overall Mean Horizontal Bar
# ─────────────────────────────────────────

print("Plotting Graph 7 — Overall Mean...")
fig, ax = plt.subplots(figsize=(9, 4))
labels  = ['NN baseline', 'Greedy Insertion', 'NN + 2-opt', 'NN + SA']
overall = [np.mean(nn_sc), np.mean(gi_sc), np.mean(opt_sc), np.mean(sa_sc)]
colors  = [C_NN, C_GI, C_2OP, C_SA]
bars = ax.barh(labels, overall, color=colors, alpha=0.9)
for bar, val in zip(bars, overall):
    ax.text(val + 5, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}', va='center', fontsize=9, color='#e8e8e8')
ax.set_xlabel('Mean Tour Length (averaged across all sizes)')
ax.set_title('Graph 7 — Overall Mean Tour Length  |  All Algorithms')
fig.tight_layout()
fig.savefig('graph7_overall_mean.png', dpi=150, facecolor=fig.get_facecolor())
print("  Saved: graph7_overall_mean.png")
plt.show()

# ─────────────────────────────────────────
# GRAPH 8: All Combined
# ─────────────────────────────────────────

print("Plotting Graph 8 — All Combined...")
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle('TSP Algorithm Comparison — Anjali Polasani | CS 57200',
             fontsize=13, fontweight='bold', color='#f0c040')

# 1 - tour routes
draw_tour(axes[0][0], cities_vis, nn_t_vis, C_NN, f'NN Baseline  ({nn_l_vis:.1f})', nn_l_vis)

# 2 - trial comparison
ax = axes[0][1]
ax.bar(x-1.5*0.18, nn_50,  width=0.18, color=C_NN,  alpha=0.9, label='NN')
ax.bar(x-0.5*0.18, gi_50,  width=0.18, color=C_GI,  alpha=0.9, label='GI')
ax.bar(x+0.5*0.18, opt_50, width=0.18, color=C_2OP, alpha=0.9, label='2-opt')
ax.bar(x+1.5*0.18, sa_50,  width=0.18, color=C_SA,  alpha=0.9, label='SA')
ax.set_title('Tour Length per Trial (n=50)')
ax.set_xticks(x); ax.set_xticklabels(['T1','T2','T3','T4','T5'])
ax.legend(fontsize=7)

# 3 - scaling
ax = axes[0][2]
ax.plot(sizes, nn_sc,  marker='o', color=C_NN,  linewidth=2, label='NN')
ax.plot(sizes, gi_sc,  marker='D', color=C_GI,  linewidth=2, linestyle='-.', label='GI')
ax.plot(sizes, opt_sc, marker='s', color=C_2OP, linewidth=2, linestyle='--', label='2-opt')
ax.plot(sizes, sa_sc,  marker='^', color=C_SA,  linewidth=2, linestyle=':', label='SA')
ax.set_title('Scaling — Tour Length vs n')
ax.set_xticks(sizes); ax.legend(fontsize=7)

# 4 - % improvement
ax = axes[1][0]
ax.bar(x2-0.2, pct_2opt, width=0.35, color=C_2OP, alpha=0.9, label='2-opt')
ax.bar(x2+0.2, pct_sa,   width=0.35, color=C_SA,  alpha=0.9, label='SA')
ax.set_title('% Improvement over NN')
ax.set_xticks(x2); ax.set_xticklabels([f'n={n}' for n in sizes])
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
ax.legend(fontsize=7)

# 5 - savings
ax = axes[1][1]
ax.bar(x-0.2, savings_opt, width=0.35, color=C_2OP, alpha=0.9, label='2-opt')
ax.bar(x+0.2, savings_sa,  width=0.35, color=C_SA,  alpha=0.9, label='SA')
ax.set_title('Absolute Savings vs NN (n=50)')
ax.set_xticks(x); ax.set_xticklabels(['T1','T2','T3','T4','T5'])
ax.legend(fontsize=7)

# 6 - random vs nn
ax = axes[1][2]
ax.bar(x-0.2, rnd_50, width=0.35, color=C_RND, alpha=0.9, label='Random')
ax.bar(x+0.2, nn_50,  width=0.35, color=C_NN,  alpha=0.9, label='NN')
ax.set_title('NN vs Random Agent (n=50)')
ax.set_xticks(x); ax.set_xticklabels(['T1','T2','T3','T4','T5'])
ax.legend(fontsize=7)

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig('graph8_all_combined.png', dpi=150, facecolor=fig.get_facecolor())
print("  Saved: graph8_all_combined.png")
plt.show()

print("\n" + "="*60)
print("All 8 graphs done! PNG files saved in the same folder.")
print("="*60)
