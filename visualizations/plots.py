import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

from data.logger import SimLogger


def plot_chi_tracking(logger: SimLogger, output_path: str) -> None:
    """Plot chi_true(t) vs per-robot chi_hat(t) over the simulation.

    - Black thick line   : chi_true  (global ground truth)
    - Thin grey lines    : chi_hat_i (per-robot local estimate), semi-transparent
    - Red dashed line    : mean of chi_hat across robots (finite values only)

    Parameters
    ----------
    logger      : populated SimLogger instance.
    output_path : file path for the saved figure (e.g. "chi_tracking.png").
    """
    df = logger.to_dataframe()

    t          = df["t"].values
    chi_true   = df["chi_true"].values
    chi_hat_cols = [c for c in df.columns if c.startswith("chi_hat_")]
    chi_hat_mat  = df[chi_hat_cols].values          # (T, N)

    # mean over robots, ignoring inf/nan
    with np.errstate(invalid="ignore"):
        chi_hat_mean = np.nanmean(
            np.where(np.isfinite(chi_hat_mat), chi_hat_mat, np.nan),
            axis=1,
        )

    fig, ax = plt.subplots(figsize=(9, 5))

    # Per-robot lines
    for col in chi_hat_mat.T:
        finite = np.isfinite(col)
        ax.plot(t[finite], col[finite],
                color="steelblue", linewidth=0.6, alpha=0.25)

    # Mean chi_hat
    ax.plot(t, chi_hat_mean,
            color="red", linewidth=1.8, linestyle="--",
            label=r"$\langle\hat{\chi}\rangle$ (robot mean)")

    # chi_true
    ax.plot(t, chi_true,
            color="black", linewidth=2.5,
            label=r"$\chi_{\rm true}$")

    ax.set_xlabel("Timestep $t$", fontsize=12)
    ax.set_ylabel(r"$\chi$", fontsize=13)
    ax.set_title(r"$\chi$ tracking: global truth vs per-robot estimates", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_SI_curve(logger: SimLogger, output_path: str) -> None:
    """Plot the SI infection curve I(t).

    - Red line            : I(t) = fraction of informed robots.
    - Vertical dashed line: t* (first timestep where I > 0).
    - y-axis fixed to [0, 1].

    Parameters
    ----------
    logger      : populated SimLogger instance.
    output_path : file path for the saved figure.
    """
    summary = logger.to_summary()
    t   = summary["t"].values
    I_t = summary["I_t"].values

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(t, I_t, color="red", linewidth=2.0, label=r"$I(t)$")

    # Vertical line at t* — first timestep with I > 0
    seed_mask = I_t > 0
    if seed_mask.any():
        t_star = int(t[seed_mask][0])
        ax.axvline(t_star, color="red", linewidth=1.2, linestyle="--",
                   label=rf"$t^* = {t_star}$")

    ax.set_ylim(0, 1)
    ax.set_xlabel("Timestep $t$", fontsize=12)
    ax.set_ylabel(r"$I(t)$", fontsize=13)
    ax.set_title("SI diffusion curve", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
