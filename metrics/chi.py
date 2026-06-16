import numpy as np

# Fitted coefficient of the saturating degree model: k = (N-1)*(1-exp(-η(1+c·χ)/(N-1)))
# Empirically calibrated on the MUC experiments.
C_SAT = 0.911


def chi_true(positions: np.ndarray, L: float) -> float:
    """Global concentration χ from the spatial dispersion of the swarm.

    Definition
    ----------
    χ = (s_uni / s₀)² − 1

    where
      s_uni = L / √12   (std of a uniform distribution on [0, L])
      s₀    = mean(std_x, std_y)  (empirical spatial spread)

    Properties
    ----------
    χ = 0   : uniform distribution  (no concentration)
    χ > 0   : more concentrated than uniform
    χ → ∞  : all robots at the same point

    Parameters
    ----------
    positions : (N, 2) array of robot coordinates.
    L         : arena side length.

    Returns
    -------
    chi : float  (≥ 0; may be slightly negative due to finite-N fluctuations)
    """
    s_uni = L / np.sqrt(12.0)
    s0 = float(np.mean(positions.std(axis=0)))
    if s0 <= 0.0:
        return float("inf")
    return (s_uni / s0) ** 2 - 1.0


def chi_hat(
    k_i: np.ndarray,
    eta: float,
    N: int,
    c_sat: float = C_SAT,
) -> np.ndarray:
    """Per-robot χ estimate obtained by inverting the saturating degree model.

    Forward model (saturating form)
    --------------------------------
    E[k_i] = (N−1) · (1 − exp(−η · (1 + c · χ) / (N−1)))

    Inversion (exact)
    -----------------
    χ̂_i = [ (N−1)/η · (−ln(1 − k_i/(N−1))) − 1 ] / c

    Edge cases
    ----------
    k_i = 0        → χ̂ = −1/c  (robot fully isolated, lower bound)
    k_i ≥ N−1     → χ̂ = +∞   (saturated, cannot invert)

    Parameters
    ----------
    k_i   : (N,) int array of observed per-robot degrees.
    eta   : dimensionless density η = N · π · R² / L².
    N     : number of robots.
    c_sat : saturation coefficient (default C_SAT = 0.911).

    Returns
    -------
    chi_hat : (N,) float array of per-robot χ estimates.
              Isolated robots yield −1/c_sat; saturated robots yield +inf.
    """
    k = np.asarray(k_i, dtype=float)
    chi = np.empty_like(k)

    isolated = k <= 0
    saturated = k >= N - 1
    valid = ~isolated & ~saturated

    chi[isolated] = -1.0 / c_sat

    chi[saturated] = np.inf

    inner = 1.0 - k[valid] / (N - 1)
    chi[valid] = ((N - 1) / eta * (-np.log(inner)) - 1.0) / c_sat

    return chi
