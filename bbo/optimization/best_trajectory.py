import numpy as np
import numbers


def test_best_trajectory():
    distances = np.array([[0,1,1],
                         [1,0,1],
                         [1,1,0]])
    np.testing.assert_equal(best_trajectory(distances, transitionmatrix="triangle"), [0,1,2])



def best_trajectory(distances, transitionmatrix = None):
    """
    Computes the shortest path over multiple timepoints using a variation of Dijkstra's algorithm.
    This function finds an optimal trajectory based on a given distance matrix and a
    transition cost matrix penalizing switches between the different paths.

    Parameters:
    -----------
    distances : np.ndarray
        A 2D NumPy array of shape (num_timepoints, num_paths) where each entry represents
        the distance at a given timepoint for a particular path. NaN values indicate missing data.

    transitionmatrix : np.ndarray, str, or float, optional (default=None)
        Defines the cost of transitioning between different paths at each time step.
        - If "triangle", a transition matrix is created where transitions are only allowed
          between the current path and all previous paths (lower triangular matrix with zeroes).
        - If a numeric value, a full transition matrix is created with the given value as
          the transition cost, except for the diagonal, which remains zero (no self-transition cost).
        - If a 2D NumPy array, it is used directly as the transition matrix.
        - If None, no transition cost is applied.

    Returns:
    --------
    trajectory_indices : np.ndarray
        A 1D NumPy array of length `num_timepoints`, representing the indices of the optimal
        path at each timepoint.
    """
    num_timepoints = distances.shape[0]
    num_paths = distances.shape[1]
    dijkstra_dist = np.zeros(shape=num_paths, dtype=float)
    dijkstra_indices = np.empty(shape=distances.shape, dtype=int)
    if transitionmatrix=="triangle":
        transitionmatrix = np.full(shape=(num_paths, num_paths), dtype=float, fill_value=np.inf)
        #transitionmatrix[np.triu_indices(num_paths, 1)] = 0 #Todo check which one is correct
        #transitionmatrix[np.tril_indices(num_paths, 1)] = 0
        for i in range(num_paths):
            transitionmatrix[i:, i] = 0
    elif isinstance(transitionmatrix, numbers.Number):
        transitionmatrix = np.full(shape=(num_paths, num_paths), dtype=float, fill_value=transitionmatrix)
        transitionmatrix[np.diag_indices(num_paths)] = 0
    for i in range(num_timepoints):
        if np.all(np.isnan(distances[i])):
            dijkstra_indices[i] = np.arange(num_paths)
        else:
            nextdist = dijkstra_dist[np.newaxis, :] + distances[i, :, np.newaxis] + transitionmatrix
            nextdist[np.isnan(nextdist)] = np.inf
            dijkstra_indices[i] = np.nanargmin(nextdist, axis=1)
            dijkstra_dist = nextdist[(np.arange(num_paths), dijkstra_indices[i])]
        dijkstra_dist -= np.min(dijkstra_dist)
    # backtrack
    trajectory_indices = np.empty(shape=num_timepoints, dtype=int)
    trajectory_indices[-1] = np.nanargmin(dijkstra_dist)
    for i in range(num_timepoints - 2, -1, -1):
        trajectory_indices[i] = dijkstra_indices[i+1, trajectory_indices[i + 1]]
    return trajectory_indices

