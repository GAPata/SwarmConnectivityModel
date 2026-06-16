import numpy as np
from typing import Literal


def init_swarm(
    N: int,
    L: float,
    mode: Literal["uniform", "gaussian"] = "uniform",
    center: tuple[float, float] | None = None,
    sigma_init: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Initialise a swarm of N robots in a square arena of side L.

    Parameters
    ----------
    N          : number of robots.
    L          : arena side length.
    mode       : 'uniform'  — robots drawn from U([0, L]^2).
                 'gaussian' — robots drawn from N(center, sigma_init^2 * I),
                              clipped to [0, L]^2.
    center     : (x, y) centre for Gaussian mode. Defaults to (L/2, L/2).
    sigma_init : std-dev for Gaussian mode. Required when mode='gaussian'.
    rng        : numpy random Generator. A fresh one is created if not given.

    Returns
    -------
    positions : (N, 2) float64 array with coordinates in [0, L]^2.
    """
    if rng is None:
        rng = np.random.default_rng()

    if mode == "uniform":
        return rng.uniform(0.0, L, size=(N, 2))

    if mode == "gaussian":
        if sigma_init is None:
            raise ValueError("sigma_init is required for mode='gaussian'")
        cx, cy = (L / 2, L / 2) if center is None else center
        positions = rng.normal(loc=[cx, cy], scale=sigma_init, size=(N, 2))
        return np.clip(positions, 0.0, L)

    raise ValueError(f"Unknown mode '{mode}'. Choose 'uniform' or 'gaussian'.")
