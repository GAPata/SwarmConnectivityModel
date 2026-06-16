from pathlib import Path

from simulations.runner import SimRunner
from visualizations.plots import plot_chi_tracking, plot_SI_curve
from visualizations.snapshot import make_gif

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

# Common base shared by every example below. R is the only knob that sets
# density (eta = N * pi * R^2 / L^2 with N=100, L=20.0):
#   R=1.6 -> eta~2.0 (sparse)   R=2.4 -> eta~4.5 (mid)   R=3.2 -> eta~8.0 (dense)
BASE = dict(N=100, L=20.0, step_size=1.0, T_steps=500, seed=42, snapshot_every=50)

CONFIGS = {
    # Isotropic random walk, sparse density, no SI diffusion.
    "random_walk_sparse": dict(
        **BASE, R=1.6, init_mode="uniform", mobility_mode="random_walk",
    ),

    # Random walk starting from a Gaussian blob, dense, with SI diffusion
    # triggered at t_star=100.
    "random_walk_dense_SI": dict(
        **BASE, R=3.2, init_mode="gaussian", sigma_init=3.0,
        mobility_mode="random_walk", t_star=100,
    ),

    # Aggregation: robots drift toward their neighbours' centre of mass.
    "aggregation": dict(
        **BASE, R=2.4, init_mode="uniform", mobility_mode="aggregation",
    ),

    # Flocking with the library defaults (looser group, more noise).
    "flocking_default": dict(
        **BASE, R=2.4, init_mode="gaussian", sigma_init=3.0,
        mobility_mode="flocking", t_star=100,
    ),

    # Flocking tuned for a tight, fast-converging, low-noise group.
    "flocking_tight": dict(
        **BASE, R=2.0, init_mode="gaussian", sigma_init=3.0,
        mobility_mode="flocking", t_star=100,
        R_coh=3.0,       # gruppo più stretto
        W_coh=2.0,       # converge più aggressivamente
        flock_noise=0.1, # movimento più rettilineo
    ),
}

# Pick which example to run:
RUN = "flocking_tight"
gif = True

runner = SimRunner(**CONFIGS[RUN])

print(f"Running simulation [{RUN}] ...")
logger = runner.run()
print(f"Done — {len(logger)} timesteps logged.")

logger.save(str(RESULTS / "run_summary.csv"), summary=True)
print("Summary saved.")

plot_chi_tracking(logger, str(RESULTS / "chi_tracking.png"))
print("Chi tracking plot saved.")

plot_SI_curve(logger, str(RESULTS / "SI_curve.png"))
print("SI curve saved.")

if runner.snapshot_every > 0 and gif:
    make_gif("results/snapshots", str(RESULTS / "simulation.gif"))
    print("GIF saved.")
