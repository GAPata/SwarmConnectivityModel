import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
from pathlib import Path
from PIL import Image


def plot_snapshot(
    positions: np.ndarray,
    k_i: np.ndarray,
    chi_hat: np.ndarray,
    chi_true: float,
    R: float,
    L: float,
    t: int,
    output_path: str,
    informed: np.ndarray | None = None,
    vmin: float = 0.0,
    vmax: float = 10.0,
) -> None:
    """Save a single simulation frame.

    Informed robots are drawn in red, non-informed in grey.
    Interaction circles follow the same colour scheme.

    Parameters
    ----------
    positions   : (N, 2) robot coordinates.
    k_i         : (N,)  per-robot degree.
    chi_hat     : (N,)  per-robot chi estimate (reserved for future encoding).
    chi_true    : global chi ground truth at this timestep.
    R           : interaction radius (circle radius).
    L           : arena side length.
    t           : current timestep (shown in title).
    output_path : file path for the saved figure.
    informed    : (N,) boolean array — True for informed robots.
                  If None, all robots are rendered grey.
    vmin, vmax  : reserved for future colour encoding.
    """
    N = len(positions)
    if informed is None:
        informed = np.zeros(N, dtype=bool)

    fig, ax = plt.subplots(figsize=(6, 6))

    for mask, color in [(~informed, "grey"), (informed, "red")]:
        if not mask.any():
            continue

        # Interaction circles
        circles = [Circle((x, y), R) for x, y in positions[mask]]
        ax.add_collection(
            PatchCollection(circles, facecolor=color, alpha=0.12,
                            edgecolor=color, linewidth=0.4)
        )

        # Robot dots
        ax.scatter(
            positions[mask, 0], positions[mask, 1],
            color=color, s=40, zorder=3,
            edgecolors="white", linewidths=0.4,
        )

    I_t = informed.sum() / N
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("y", fontsize=11)
    ax.set_title(
        rf"$t = {t}$ — $\chi_{{\rm true}} = {chi_true:.3f}$ — $I = {I_t:.2f}$",
        fontsize=12,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def make_gif(
    snapshot_dir: str,
    output_path: str,
    duration: int = 100,
) -> None:
    """Assemble all PNG frames in snapshot_dir into an animated GIF.

    Parameters
    ----------
    snapshot_dir : directory containing the PNG snapshots (sorted by name).
    output_path  : destination path for the GIF file.
    duration     : delay between frames in milliseconds.
    """
    frames = sorted(Path(snapshot_dir).glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"No PNG files found in '{snapshot_dir}'")

    images = [Image.open(f).convert("RGBA") for f in frames]

    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        loop=0,
        duration=duration,
    )
