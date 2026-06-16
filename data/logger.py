import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List


@dataclass
class _Record:
    t: int
    chi_true: float
    chi_hat: np.ndarray   # (N,)
    k_i: np.ndarray       # (N,) int
    I_t: float            # fraction of informed robots in [0, 1]


class SimLogger:
    """Accumulates per-timestep simulation data and exports to DataFrame/CSV.

    Usage
    -----
    logger = SimLogger()
    for t in range(T):
        ...
        logger.log(t, chi_true_val, chi_hat_arr, k_i_arr)

    df       = logger.to_dataframe()   # one row per (t, robot)
    df_agg   = logger.to_summary()     # one row per t  (means + stds)
    logger.save("run.csv")
    """

    def __init__(self) -> None:
        self._records: List[_Record] = []

    def log(
        self,
        t: int,
        chi_true: float,
        chi_hat: np.ndarray,
        k_i: np.ndarray,
        I_t: float = 0.0,
    ) -> None:
        """Append one timestep."""
        self._records.append(
            _Record(
                t=int(t),
                chi_true=float(chi_true),
                chi_hat=np.asarray(chi_hat, dtype=float).copy(),
                k_i=np.asarray(k_i, dtype=int).copy(),
                I_t=float(I_t),
            )
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Wide-form DataFrame: one row per timestep.

        Columns: t, chi_true, chi_hat_0, chi_hat_1, ..., chi_hat_{N-1}
        """
        rows = []
        for rec in self._records:
            row = {"t": rec.t, "chi_true": rec.chi_true, "I_t": rec.I_t}
            for i, v in enumerate(rec.chi_hat):
                row[f"chi_hat_{i}"] = v
            rows.append(row)
        return pd.DataFrame(rows)

    def to_summary(self) -> pd.DataFrame:
        """Wide-form DataFrame: one row per timestep with aggregated statistics.

        Columns: t, chi_true,
                 k_mean, k_std,
                 chi_hat_mean, chi_hat_std, chi_hat_median
        """
        rows = []
        for rec in self._records:
            finite = np.isfinite(rec.chi_hat)
            rows.append({
                "t":              rec.t,
                "chi_true":       rec.chi_true,
                "I_t":            rec.I_t,
                "k_mean":         float(rec.k_i.mean()),
                "k_std":          float(rec.k_i.std()),
                "chi_hat_mean":   float(rec.chi_hat[finite].mean()) if finite.any() else np.nan,
                "chi_hat_std":    float(rec.chi_hat[finite].std())  if finite.any() else np.nan,
                "chi_hat_median": float(np.median(rec.chi_hat[finite])) if finite.any() else np.nan,
            })
        return pd.DataFrame(rows)

    def save(self, path: str, summary: bool = False) -> None:
        """Write to CSV. Pass summary=True to save the aggregated form."""
        df = self.to_summary() if summary else self.to_dataframe()
        df.to_csv(path, index=False)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"SimLogger({len(self._records)} timesteps logged)"
