import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def aggregate_experiment(exp_name: str) -> None:
    """Aggregate all run CSVs for an experiment and produce summary plots.

    Reads:   results/{exp_name}/run_*.csv
    Writes:
      results/{exp_name}/aggregate.csv
      results/{exp_name}/SI_curve_aggregated.png
      results/{exp_name}/chi_tracking_aggregated.png

    Parameters
    ----------
    exp_name : experiment label matching the folder name under results/.
    """
    out_dir = Path("results") / exp_name
    csv_files = sorted(out_dir.glob("run_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No run CSVs found in '{out_dir}'")

    print(f"[{exp_name}] Loading {len(csv_files)} runs …")
    frames = [pd.read_csv(f) for f in csv_files]
    # Stack into (n_runs, T) structure via a single tall DataFrame
    all_runs = pd.concat(frames, keys=range(len(frames)), names=["run", "row"])

    # ------------------------------------------------------------------
    # Aggregate by timestep
    # ------------------------------------------------------------------
    agg = (
        all_runs
        .groupby("t")
        .agg(
            I_mean         = ("I_t",          "mean"),
            I_std          = ("I_t",          "std"),
            chi_true_mean  = ("chi_true",      "mean"),
            chi_true_std   = ("chi_true",      "std"),
            chi_hat_mean   = ("chi_hat_mean",  "mean"),
            chi_hat_std    = ("chi_hat_mean",  "std"),
        )
        .reset_index()
    )

    agg_path = out_dir / "aggregate.csv"
    agg.to_csv(agg_path, index=False)
    print(f"  Aggregate CSV → {agg_path}")

    t             = agg["t"].values
    I_mean        = agg["I_mean"].values
    I_std         = agg["I_std"].values
    chi_true_mean = agg["chi_true_mean"].values
    chi_true_std  = agg["chi_true_std"].values
    chi_hat_mean  = agg["chi_hat_mean"].values
    chi_hat_std   = agg["chi_hat_std"].values

    # Detect t* from aggregate (first t where I_mean > 0)
    t_star_mask = I_mean > 0
    t_star = int(t[t_star_mask][0]) if t_star_mask.any() else None

    # ------------------------------------------------------------------
    # Plot 1: SI curve with ±std band
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4))

    ax.fill_between(t, np.clip(I_mean - I_std, 0, 1), np.clip(I_mean + I_std, 0, 1),
                    color="red", alpha=0.18, label=r"$\pm\sigma$")
    ax.plot(t, I_mean, color="red", linewidth=2.0, label=r"$\langle I(t) \rangle$")

    if t_star is not None:
        ax.axvline(t_star, color="red", linewidth=1.2, linestyle="--",
                   label=rf"$t^* = {t_star}$")

    ax.set_ylim(0, 1)
    ax.set_xlabel("Timestep $t$", fontsize=12)
    ax.set_ylabel(r"$I(t)$", fontsize=13)
    ax.set_title(
        rf"SI diffusion — {len(csv_files)} runs — $\langle I \rangle \pm \sigma$",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    si_path = out_dir / "SI_curve_aggregated.png"
    fig.savefig(si_path, dpi=150)
    plt.close(fig)
    print(f"  SI curve plot  → {si_path}")

    # ------------------------------------------------------------------
    # Plot 2: chi tracking — chi_true_mean and chi_hat_mean with ±std bands
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.fill_between(t, chi_true_mean - chi_true_std, chi_true_mean + chi_true_std,
                    color="black", alpha=0.12)
    ax.plot(t, chi_true_mean, color="black", linewidth=2.5,
            label=r"$\langle \chi_{\rm true} \rangle$")

    ax.fill_between(t, chi_hat_mean - chi_hat_std, chi_hat_mean + chi_hat_std,
                    color="steelblue", alpha=0.18)
    ax.plot(t, chi_hat_mean, color="steelblue", linewidth=2.0, linestyle="--",
            label=r"$\langle \hat{\chi} \rangle$")

    ax.set_xlabel("Timestep $t$", fontsize=12)
    ax.set_ylabel(r"$\chi$", fontsize=13)
    ax.set_title(
        rf"$\chi$ tracking — {len(csv_files)} runs — mean $\pm\,\sigma$",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    chi_path = out_dir / "chi_tracking_aggregated.png"
    fig.savefig(chi_path, dpi=150)
    plt.close(fig)
    print(f"  χ tracking plot → {chi_path}")

    print(f"[{exp_name}] Done.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python postprocess.py <exp_name>")
        sys.exit(1)
    aggregate_experiment(sys.argv[1])
