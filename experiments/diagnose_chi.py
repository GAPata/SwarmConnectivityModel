"""Diagnostic: chi_hat per robot vs distance from centre of mass at a fixed timestep.

Run:  python experiments/diagnose_chi.py
Out:  results/diagnosis_chi_vs_distance.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.arena import Arena
from core.graph import build_rgg
from core.swarm import init_swarm
from metrics.chi import chi_true, chi_hat

# ------------------------------------------------------------------
# Simulation parameters
# ------------------------------------------------------------------
N          = 100
L          = 20.0
R          = 1.0
step_size  = 0.3
T_steps    = 200
seed       = 42
t_snapshot = 60
ETA        = N * np.pi * R**2 / L**2

rng   = np.random.default_rng(seed)
arena = Arena(L, N, rng=rng)

positions = init_swarm(N, L, mode="uniform", rng=rng)

# ------------------------------------------------------------------
# Advance to t_snapshot using aggregation dynamics
# ------------------------------------------------------------------
for t in range(T_steps):
    if t == t_snapshot:
        snap_positions = positions.copy()
        snap_G, snap_k_i = build_rgg(snap_positions, R)
        snap_chi_hat  = chi_hat(snap_k_i, ETA, N)
        snap_chi_true = chi_true(snap_positions, L)
        break
    positions = arena.aggregation_step(positions, step_size)

# ------------------------------------------------------------------
# Distance of each robot from the global centre of mass
# ------------------------------------------------------------------
com      = snap_positions.mean(axis=0)
dist_com = np.linalg.norm(snap_positions - com, axis=1)   # (N,)

# Exclude saturated robots (chi_hat = +inf) from the plot
finite   = np.isfinite(snap_chi_hat)
n_finite = finite.sum()
n_inf    = (~finite).sum()

# ------------------------------------------------------------------
# Scatter plot
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

sc = ax.scatter(
    dist_com[finite], snap_chi_hat[finite],
    c=snap_k_i[finite], cmap="viridis",
    s=60, edgecolors="white", linewidths=0.4, zorder=3,
)
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("degree $k_i$", fontsize=11)

ax.axhline(snap_chi_true, color="red", linewidth=1.8, linestyle="--",
           label=rf"$\chi_{{\rm true}} = {snap_chi_true:.2f}$")

ax.set_xlabel("Distance from centre of mass  [a.u.]", fontsize=12)
ax.set_ylabel(r"$\hat{\chi}_i$  (per-robot estimate)", fontsize=12)
ax.set_title(
    rf"$\hat{{\chi}}$ vs CoM distance — aggregation, $t={t_snapshot}$"
    rf"  ($N={N},\ R={R},\ \eta={ETA:.2f}$)",
    fontsize=12,
)

note = f"{n_finite} robots shown"
if n_inf:
    note += f"  |  {n_inf} saturated ($k_i = N-1$, $\\hat{{\\chi}} = +\\infty$) excluded"
ax.text(0.02, 0.97, note, transform=ax.transAxes,
        fontsize=9, va="top", color="grey")

ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()

out_path = Path("results/diagnosis_chi_vs_distance.png")
out_path.parent.mkdir(exist_ok=True)
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"Saved → {out_path}")
print(f"t={t_snapshot}: chi_true={snap_chi_true:.3f}  "
      f"chi_hat mean (finite)={snap_chi_hat[finite].mean():.3f}  "
      f"k_i mean={snap_k_i.mean():.1f}  saturated={n_inf}/{N}")
