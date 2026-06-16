import numpy as np
import networkx as nx
from scipy.spatial import cKDTree


def build_rgg(
    positions: np.ndarray,
    R: float,
    L: float | None = None,
) -> tuple[nx.Graph, np.ndarray]:
    """Build the Random Geometric Graph (RGG) for a swarm.

    Two robots i, j are connected iff ||pos_i - pos_j|| <= R.

    Uses a cKDTree so the pair-search scales as O(N log N) rather than O(N^2).

    Parameters
    ----------
    positions : (N, 2) array of robot coordinates.
    R         : interaction radius.
    L         : arena side length. Pass it for periodic (toroidal) arenas —
                e.g. mobility_mode='flocking' — so that distances wrap around
                the box edges consistently with Arena._toroidal_delta. None
                (default) uses plain Euclidean distance, correct for the
                reflective-boundary modes (random_walk, aggregation).

    Returns
    -------
    G       : networkx.Graph with N nodes (labelled 0..N-1) and the RGG edges.
    degrees : (N,) int array where degrees[i] = degree of robot i.
    """
    N = len(positions)

    tree = cKDTree(positions, boxsize=L) if L is not None else cKDTree(positions)
    pairs = tree.query_pairs(R)          # set of (i, j) with i < j, dist <= R

    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(pairs)

    degrees = np.array([G.degree(i) for i in range(N)], dtype=int)
    return G, degrees
