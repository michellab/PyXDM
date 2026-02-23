"""Geometry-related utilities for XDM calculations."""
import numpy as np
from numpy.typing import NDArray


def compute_distance_matrix(coords: NDArray[np.float64],) -> NDArray[np.float64]:
    """
    Compute pairwise distance matrix for a set of coordinates.
    
    Parameters
    ----------
    coords : np.ndarray
        Array of shape (N, 3) containing the coordinates of N atoms.

    Returns
    -------
    np.ndarray
        Distance matrix of shape (N, N) where element (i, j) is the distance between atoms i and j.
    """
    coords = np.asarray(coords, dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diff, axis=-1)