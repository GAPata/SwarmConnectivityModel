from pathlib import Path
import numpy as np

from core.arena import Arena
from core.swarm import init_swarm
from core.graph import build_rgg
from metrics.chi import chi_true, chi_hat
from data.logger import SimLogger
from visualizations.snapshot import plot_snapshot

_SNAPSHOT_DIR = Path("results/snapshots")


class SimRunner:
    """End-to-end simulation runner for the fixed-step RGG swarm model.

    Parameters
    ----------
    N             : number of robots.
    L             : arena side length.
    R             : interaction radius for the RGG.
    step_size     : fixed displacement magnitude per timestep.
    T_steps       : number of timesteps to simulate.
    init_mode     : 'uniform' or 'gaussian'.
    sigma_init    : initial cluster std-dev (required when init_mode='gaussian').
    seed          : random seed for reproducibility (optional).
    snapshot_every: save a frame snapshot every this many timesteps.
                    0 (default) disables snapshots.
    t_star        : timestep at which a random robot is seeded as informed.
                    None disables SI diffusion (I_t = 0 throughout).
    beta          : transmission probability per contact (reserved; currently
                    deterministic β=1 is used regardless of this value).
    mobility_mode : 'random_walk' — isotropic fixed-step random walk (default).
                    'aggregation' — global 1/d² attraction + short-range
                    repulsion (no RGG). Tuned by agg_d_min, agg_R_personal,
                    agg_W_attract, agg_W_repel, agg_noise.
                    'flocking' — Reynolds flocking on a toroidal arena (see
                    Arena.flocking_step). Tuned by R_sep/R_ali/R_coh,
                    W_sep/W_ali/W_coh, flock_noise, dt, max_force, min_speed.
    """

    _MOBILITY_MODES = {"random_walk", "aggregation", "flocking"}

    def __init__(
        self,
        N: int,
        L: float,
        R: float,
        step_size: float,
        T_steps: int,
        init_mode: str = "uniform",
        sigma_init: float | None = None,
        seed: int | None = None,
        snapshot_every: int = 0,
        t_star: int | None = None,
        beta: float = 1.0,
        mobility_mode: str = "random_walk",
        # aggregation params
        agg_d_min: float = 0.3,
        agg_R_personal: float = 0.5,
        agg_W_attract: float = 1.0,
        agg_W_repel: float = 2.0,
        agg_noise: float = 0.05,
        # flocking params
        R_sep: float = 1.0,
        R_ali: float = 3.0,
        R_coh: float = 5.0,
        W_sep: float = 2.0,
        W_ali: float = 1.0,
        W_coh: float = 1.0,
        flock_noise: float = 0.3,
        dt: float = 0.1,
        max_force: float = 0.5,
        min_speed: float = 0.1,
    ) -> None:
        if init_mode == "gaussian" and sigma_init is None:
            raise ValueError("sigma_init is required when init_mode='gaussian'")
        if mobility_mode not in self._MOBILITY_MODES:
            raise ValueError(
                f"Unknown mobility_mode '{mobility_mode}'. "
                f"Choose one of {sorted(self._MOBILITY_MODES)}."
            )

        self.N              = N
        self.L              = L
        self.R              = R
        self.step_size      = step_size
        self.T_steps        = T_steps
        self.init_mode      = init_mode
        self.sigma_init     = sigma_init
        self.eta            = N * np.pi * R ** 2 / L ** 2
        self.snapshot_every = int(snapshot_every)
        self.t_star         = t_star
        self.beta           = beta
        self.mobility_mode  = mobility_mode
        self.agg_d_min      = agg_d_min
        self.agg_R_personal = agg_R_personal
        self.agg_W_attract  = agg_W_attract
        self.agg_W_repel    = agg_W_repel
        self.agg_noise      = agg_noise
        self.R_sep          = R_sep
        self.R_ali          = R_ali
        self.R_coh          = R_coh
        self.W_sep          = W_sep
        self.W_ali          = W_ali
        self.W_coh          = W_coh
        self.flock_noise    = flock_noise
        self.dt             = dt
        self.max_force      = max_force
        self.min_speed      = min_speed
        self._rng           = np.random.default_rng(seed)

    def run(self) -> SimLogger:
        """Run the simulation and return the populated SimLogger."""
        if self.snapshot_every > 0:
            _SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

        arena  = Arena(self.L, self.N, rng=self._rng)
        logger = SimLogger()

        positions = init_swarm(
            self.N,
            self.L,
            mode=self.init_mode,
            sigma_init=self.sigma_init,
            rng=self._rng,
        )

        informed:   set[int] = set()
        velocities: np.ndarray | None = None

        rgg_L = self.L if self.mobility_mode == "flocking" else None

        for t in range(self.T_steps):
            G, k_i = build_rgg(positions, self.R, L=rgg_L)

            # --- SI diffusion ---
            if self.t_star is not None:
                if t == self.t_star:
                    informed = {int(self._rng.integers(self.N))}
                elif t > self.t_star:
                    new_informed = set(informed)
                    for i in informed:
                        new_informed.update(G.neighbors(i))
                    informed = new_informed

            I_t = len(informed) / self.N

            ct = chi_true(positions, self.L)
            ch = chi_hat(k_i, self.eta, self.N)

            logger.log(t, ct, ch, k_i, I_t)

            if self.snapshot_every > 0 and t % self.snapshot_every == 0:
                informed_mask = np.zeros(self.N, dtype=bool)
                informed_mask[list(informed)] = True
                plot_snapshot(
                    positions, k_i, ch, ct,
                    R=self.R, L=self.L, t=t,
                    output_path=str(_SNAPSHOT_DIR / f"t_{t:04d}.png"),
                    informed=informed_mask,
                )

            if self.mobility_mode == "random_walk":
                positions = arena.random_walk_step(positions, self.step_size)
            elif self.mobility_mode == "aggregation":
                positions = arena.aggregation_step(
                    positions, self.step_size,
                    d_min=self.agg_d_min, R_personal=self.agg_R_personal,
                    W_attract=self.agg_W_attract, W_repel=self.agg_W_repel,
                    noise=self.agg_noise,
                )
            elif self.mobility_mode == "flocking":
                positions, velocities = arena.flocking_step(
                    positions, velocities, self.step_size,
                    R_sep=self.R_sep, R_ali=self.R_ali, R_coh=self.R_coh,
                    W_sep=self.W_sep, W_ali=self.W_ali, W_coh=self.W_coh,
                    noise=self.flock_noise, dt=self.dt,
                    max_force=self.max_force, min_speed=self.min_speed,
                )

        return logger
