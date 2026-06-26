"""Synthetic datasets exhibiting Constrained Perfect Multivariate Conditional
Redundancy (CPMCR).

Both datasets use binary variables and a 3-way-XOR style target
``A = 1{X1 == X2 == X3}`` combining redundancy with synergy:

* ``4red_varbs`` (Four Redundant Variables): ``X4 = X5 = X6 = X1``.
* ``2red_trips`` (Two Redundant Triplets): ``(X4, X5, X6) = (X1, X2, X3)``.

Pairwise feature-selection methods cannot resolve these; TERC can.
"""

import random
from pathlib import Path

import numpy as np


def generate_data(name, n_points=10000, data_dir="."):
    """Generate a synthetic dataset and save ``obs_<name>.npy`` / ``acs_<name>.npy``.

    Parameters
    ----------
    name : {"4red_varbs", "2red_trips"}
    n_points : int
        Number of samples (the paper uses 10,000).
    data_dir : str or Path
        Output directory for the ``.npy`` files.

    Returns
    -------
    (obs, acs) : tuple of np.ndarray
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    obs, acs = [], []

    if name == "4red_varbs":
        for _ in range(n_points):
            base = [random.randint(0, 1) for _ in range(3)]
            obs.append(base + [base[0]] * 3)
            acs.append(int(base[0] == base[1] == base[2]))
    elif name == "2red_trips":
        for _ in range(n_points):
            base = [random.randint(0, 1) for _ in range(3)]
            obs.append(base + base)
            acs.append(int(base[0] == base[1] == base[2]))
    else:
        raise ValueError(f"Unknown synthetic dataset: {name!r}")

    obs = np.array(obs)
    acs = np.array(acs)
    np.save(data_dir / f"obs_{name}.npy", obs)
    np.save(data_dir / f"acs_{name}.npy", acs)
    return obs, acs
