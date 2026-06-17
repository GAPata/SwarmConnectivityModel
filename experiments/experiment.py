from pathlib import Path

from simulations.runner import SimRunner


class Experiment:
    """Run multiple independent realisations of the swarm simulation.

    Each run uses a progressive seed (0, 1, 2, …) so results are
    reproducible. Snapshots and GIFs are disabled for all runs.

    Parameters
    ----------
    exp_name   : label used as the output sub-folder name.
    n_runs     : number of independent realisations.
    All remaining keyword arguments are forwarded verbatim to SimRunner.

    Output
    ------
    results/{mobility_mode}/{exp_name}/run_{i:03d}.csv   — summary CSV for each run.
    """

    def __init__(
        self,
        exp_name: str,
        n_runs: int,
        **runner_kwargs,
    ) -> None:
        self.exp_name   = exp_name
        self.n_runs     = n_runs
        self.runner_kwargs = dict(runner_kwargs, snapshot_every=0)  # always off during experiments

    def run(self) -> None:
        mobility_mode = self.runner_kwargs["mobility_mode"]
        out_dir = Path("results") / mobility_mode / self.exp_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{mobility_mode}/{self.exp_name}] Starting {self.n_runs} runs …")

        for i in range(self.n_runs):
            runner = SimRunner(seed=i, **self.runner_kwargs)
            logger = runner.run()
            csv_path = out_dir / f"run_{i:03d}.csv"
            logger.save(str(csv_path), summary=True)
            print(f"  run {i+1:>{len(str(self.n_runs))}}/{self.n_runs}  →  {csv_path}")

        print(f"[{mobility_mode}/{self.exp_name}] Done. Results in {out_dir}/")
