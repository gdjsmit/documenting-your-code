import numpy as np
from scipy.stats import poisson, uniform
from typing import Tuple


def generate_poisson_points(bounds: Tuple[float, float, float, float],
                            rate: float) -> np.ndarray:
    """
    Generate random points within a given bounding box at a given rate
    (density) following a Poisson process.

    Parameters
    ----------
    bounds : Tuple[float, float, float, float]
        Bounding box coordinates (xmin, ymin, xmax, ymax) within
        which the points are generated.
    rate : float
        Poisson rate (average number of points per unit area).

    Returns
    -------
    np.array
       array of generated points.

    """
    dx = bounds[2] - bounds[0]
    dy = bounds[3] - bounds[1]

    # how many points to generate (cf. poisson distribution with given rate)
    N = poisson(rate * dx * dy).rvs()

    # generate the given number of points (x and y coordinates separetely)
    xs = uniform.rvs(0, dx, ((N, 1))) + bounds[0]
    ys = uniform.rvs(0, dy, ((N, 1))) + bounds[1]

    return np.hstack((xs, ys))  # rearrange as array of coordinates
