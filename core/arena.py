import numpy as np


class Arena:
    """Square 2D arena of side L with reflective boundaries."""

    def __init__(self, L: float, N: int, rng: np.random.Generator | None = None):
        self.L = float(L)
        self.N = int(N)
        self.rng = rng if rng is not None else np.random.default_rng()

    # ------------------------------------------------------------------
    # Boundary helpers
    # ------------------------------------------------------------------

    def _reflect(self, x: np.ndarray) -> np.ndarray:
        """Map coordinates into [0, L] with reflective boundaries.

        Works for any number of bounces via fold-and-mirror on the period 2L.
        """
        L = self.L
        x = x % (2 * L)           # fold into [0, 2L)
        return np.where(x > L, 2 * L - x, x)

    def clip(self, positions: np.ndarray) -> np.ndarray:
        """Return positions reflected inside the arena. Does not modify in-place."""
        return self._reflect(positions)

    def _wrap_torus(self, x: np.ndarray) -> np.ndarray:
        """Map coordinates into [0, L) with periodic (toroidal) boundaries."""
        return x % self.L

    def _toroidal_delta(self, delta: np.ndarray) -> np.ndarray:
        """Minimal-image displacement: wrap delta into [-L/2, L/2).

        Used to compute distances/directions on the torus, so that two
        robots near opposite edges of the arena are correctly seen as
        close neighbours.
        """
        L = self.L
        return (delta + L / 2.0) % L - L / 2.0

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def random_walk_step(
        self,
        positions: np.ndarray,
        step_size: float,
    ) -> np.ndarray:
        """One fixed-step random-walk move with reflective walls.

        Each robot draws a heading θ ~ U[0, 2π) independently and moves
        exactly step_size in that direction. The control parameter is
        step_size, not a variance.

        Parameters
        ----------
        positions : (N, 2) array of current robot positions.
        step_size : fixed displacement magnitude per step.

        Returns
        -------
        new_positions : (N, 2) array, reflected into [0, L]^2.
        """
        theta = self.rng.uniform(0.0, 2.0 * np.pi, size=len(positions))
        displacement = step_size * np.column_stack((np.cos(theta), np.sin(theta)))
        return self._reflect(positions + displacement)

    def aggregation_step(
        self,
        positions: np.ndarray,
        k_i: np.ndarray,
        G,
        step_size: float,
        noise: float = 0.05,
    ) -> np.ndarray:
        """One aggregation move: each robot steps toward its neighbours' centre of mass.

        Robots with no neighbours draw a random heading instead.
        A small isotropic Gaussian noise is added to prevent full collapse.
        Reflective boundaries are applied at the end.

        Parameters
        ----------
        positions : (N, 2) array of current robot positions.
        k_i       : (N,)  per-robot degree (used to detect isolated robots).
        G         : networkx.Graph — RGG at the current timestep.
        step_size : fixed displacement magnitude toward the local CoM.
        noise     : std-dev of the additive Gaussian perturbation.

        Returns
        -------
        new_positions : (N, 2) array, reflected into [0, L]^2.
        """
        N = len(positions)
        displacement = np.empty((N, 2))

        isolated = k_i == 0

        # --- Robots with neighbours: step toward local centre of mass ---
        for i in np.where(~isolated)[0]:
            neighbours = list(G.neighbors(int(i)))
            com = positions[neighbours].mean(axis=0)
            direction = com - positions[i]
            dist = np.linalg.norm(direction)
            if dist > 0:
                displacement[i] = step_size * direction / dist
            else:
                # Already at the CoM — random nudge
                theta = self.rng.uniform(0.0, 2.0 * np.pi)
                displacement[i] = step_size * np.array([np.cos(theta), np.sin(theta)])

        # --- Isolated robots: random direction ---
        if isolated.any():
            theta = self.rng.uniform(0.0, 2.0 * np.pi, size=int(isolated.sum()))
            displacement[isolated] = step_size * np.column_stack(
                (np.cos(theta), np.sin(theta))
            )

        # --- Gaussian noise to prevent full collapse ---
        displacement += self.rng.normal(scale=noise, size=(N, 2))

        return self._reflect(positions + displacement)

    def _reflect_vel(
        self, raw_pos: np.ndarray, velocities: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reflect positions into [0, L]^2 and flip the corresponding velocity components.

        Handles a single bounce per axis per step (correct for step_size << L).
        """
        new_pos = raw_pos.copy()
        new_vel = velocities.copy()
        L = self.L
        for d in range(2):
            x = raw_pos[:, d]
            below = x < 0.0
            above = x > L
            new_pos[below, d] = -x[below]
            new_vel[below, d] = -velocities[below, d]
            new_pos[above, d] = 2.0 * L - x[above]
            new_vel[above, d] = -velocities[above, d]
        return new_pos, new_vel

    def flocking_step(
        self,
        positions: np.ndarray,
        velocities: np.ndarray | None,
        step_size: float,
        R_sep: float = 1.0,
        R_ali: float = 3.0,
        R_coh: float = 5.0,
        W_sep: float = 2.0,
        W_ali: float = 1.0,
        W_coh: float = 1.0,
        noise: float = 0.3,
        dt: float = 0.1,
        max_force: float = 0.5,
        min_speed: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """One Reynolds flocking step with inertia, on a toroidal arena.

        Three interaction zones (all centred on robot i, self excluded),
        each yielding a force rather than an instantaneous heading:
          Separation  (R_sep): mean of unit repulsion vectors from
                                too-close neighbours, weighted by 1/dist.
          Alignment   (R_ali): mean neighbour velocity minus own velocity.
          Cohesion    (R_coh): mean toroidal offset to neighbours minus
                                own velocity (steer toward their centroid).

        All neighbour distances/directions use the minimal-image (toroidal)
        convention, so robots near opposite edges of the arena can see each
        other as close. Forces are combined, clamped to max_force, and
        integrated into velocity over dt. Noise is applied as a random
        rotation of the velocity vector (not an additive term), and the
        resulting speed is clamped to [min_speed, step_size * 1.5]. Position
        is then advanced by velocity * dt and wrapped on the torus.

        Parameters
        ----------
        positions  : (N, 2) current robot positions.
        velocities : (N, 2) current robot velocities, or None for random init.
        step_size  : sets the maximum speed (step_size * 1.5).
        R_sep      : separation radius.
        R_ali      : alignment radius.
        R_coh      : cohesion radius.
        W_sep      : separation weight.
        W_ali      : alignment weight.
        W_coh      : cohesion weight.
        noise      : max random rotation applied to velocity (radians).
        dt         : integration timestep.
        max_force  : clamp on the combined steering force norm.
        min_speed  : lower clamp on the resulting speed.

        Returns
        -------
        new_positions  : (N, 2) wrapped into [0, L)^2.
        new_velocities : (N, 2) updated velocity vectors.
        """
        N = len(positions)
        max_speed = step_size * 1.5

        # Initialise velocities with random unit headings scaled to step_size
        if velocities is None:
            theta0 = self.rng.uniform(0.0, 2.0 * np.pi, size=N)
            velocities = step_size * np.column_stack(
                (np.cos(theta0), np.sin(theta0))
            )

        # Pairwise toroidal displacement  delta[i, j] = minimal-image(pos_j - pos_i)
        delta = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]  # (N,N,2)
        delta = self._toroidal_delta(delta)
        dist  = np.linalg.norm(delta, axis=2)                              # (N,N)
        np.fill_diagonal(dist, np.inf)                                     # ignore self

        new_vel = np.empty((N, 2))

        for i in range(N):
            d = dist[i]

            # Separation — mean unit repulsion vector, weighted by 1/dist
            sep_force = np.zeros(2)
            sep_mask = d < R_sep
            if sep_mask.any():
                away_unit = -delta[i, sep_mask] / d[sep_mask, np.newaxis]
                weight = 1.0 / d[sep_mask]
                sep_force = (away_unit * weight[:, np.newaxis]).mean(axis=0)

            # Alignment — steer velocity toward the neighbours' mean velocity
            ali_force = np.zeros(2)
            ali_mask = d < R_ali
            if ali_mask.any():
                ali_force = velocities[ali_mask].mean(axis=0) - velocities[i]

            # Cohesion — steer toward the neighbours' toroidal centroid
            coh_force = np.zeros(2)
            coh_mask = d < R_coh
            if coh_mask.any():
                target_offset = delta[i, coh_mask].mean(axis=0)
                coh_force = target_offset - velocities[i]

            force = W_sep * sep_force + W_ali * ali_force + W_coh * coh_force

            force_norm = np.linalg.norm(force)
            if force_norm > max_force:
                force *= max_force / force_norm

            vel = velocities[i] + force * dt

            # Noise as a random rotation of the velocity vector
            angle = self.rng.uniform(-noise, noise)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            vel = np.array(
                [vel[0] * cos_a - vel[1] * sin_a, vel[0] * sin_a + vel[1] * cos_a]
            )

            # Clamp speed into [min_speed, max_speed]
            speed = np.linalg.norm(vel)
            if speed < 1e-10:
                theta = self.rng.uniform(0.0, 2.0 * np.pi)
                vel = min_speed * np.array([np.cos(theta), np.sin(theta)])
            elif speed < min_speed:
                vel *= min_speed / speed
            elif speed > max_speed:
                vel *= max_speed / speed

            new_vel[i] = vel

        new_pos = self._wrap_torus(positions + new_vel * dt)
        return new_pos, new_vel
