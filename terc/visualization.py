"""Plotting helpers for TERC outputs.

``plot_terc_curves`` visualizes the per-variable MINE mutual-information curves
written by :class:`terc.estimator.TERC`, with the null model drawn in black.
``plot_learning_curves`` visualizes the reward curves written by
:class:`terc.pipelines.learning_curves.LearningCurveGenerator`.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def moving_average(a, window_size=100):
    return [np.mean(a[i:i + window_size]) for i in range(0, len(a) - window_size)]


def plot_terc_curves(name, results_dir=".", ci_scale=2 / 3.33, show=True, save_path=None):
    """Plot TERC MI curves for each variable plus the null model.

    Parameters
    ----------
    name : str
        Dataset name (loads ``<name>_means.npy`` / ``<name>_stdv.npy``).
    results_dir : str or Path
        Directory containing the saved curves.
    ci_scale : float
        Scaling applied to the std band for display.
    """
    results_dir = Path(results_dir)
    means_big = np.load(results_dir / f"{name}_means.npy")
    stdv_big = np.load(results_dir / f"{name}_stdv.npy")

    fig, ax = plt.subplots(1, figsize=(8, 6))
    for i in range(1, len(means_big)):
        means = means_big[i]
        stdv = stdv_big[i]
        x = np.arange(len(means))
        ax.plot(x, means, label=f"variable {i - 1}")
        ax.fill_between(x, means - stdv * ci_scale, means + stdv * ci_scale, alpha=0.1)
    nm_means = means_big[0]
    nm_stdv = stdv_big[0]
    x = np.arange(len(nm_means))
    ax.plot(x, nm_means, label="Null Model", color="black")
    ax.fill_between(x, nm_means - nm_stdv * ci_scale, nm_means + nm_stdv * ci_scale,
                    alpha=0.1, color="black")
    ax.set_xlabel("MINE iteration (moving average)")
    ax.set_ylabel(r"$\Phi_{X_i; \mathcal{X} \rightarrow A}$")
    ax.set_title(f"TERC: {name}")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def plot_learning_curves(name, data_dir=".", window_size=100, labels=("Full state", "TERC"),
                         show=True, save_path=None):
    """Plot reward learning curves for the full vs. TERC-selected state."""
    data_dir = Path(data_dir)
    lcs = np.load(data_dir / f"learning_curve_{name}.npy")
    fig, ax = plt.subplots(1, figsize=(8, 6))
    for i in range(len(lcs)):
        means = np.array(moving_average(lcs[i][0], window_size))
        stdv = np.array(moving_average(lcs[i][1], window_size))
        x = np.arange(len(means))
        label = labels[i] if i < len(labels) else f"subset {i}"
        ax.plot(x, means, label=label)
        ax.fill_between(x, means - stdv, means + stdv, alpha=0.1)
    ax.set_xlabel("Episode (moving average)")
    ax.set_ylabel("Cumulative reward")
    ax.set_title(f"Learning curves: {name}")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax
