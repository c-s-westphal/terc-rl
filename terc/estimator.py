"""The Transfer Entropy Redundancy Criterion (TERC).

TERC estimates, for each observable state variable ``X_i``, the transfer-entropy
based quantity

    Phi(X_i; X -> A) = H(A | X_without_i) - H(A | X) >= 0,

i.e. the reduction in the entropy of the actions ``A`` attributable to ``X_i``.
A variable is kept iff its estimated ``Phi`` is statistically distinguishable
from a null model (an injected uninformative random variable) at the 95%
confidence level. Variables are removed one at a time (Algorithm 1 in the
paper), so that perfectly redundant variables are correctly resolved.

The conditional entropies are estimated via mutual information using a MINE-style
neural estimator (Belghazi et al., 2018), implemented in :class:`MineNet`.

Inputs are read from ``obs_<name>.npy`` and ``acs_<name>.npy`` in ``data_dir``;
the per-variable MI curves are written to ``<name>_means.npy`` /
``<name>_stdv.npy`` in ``results_dir``.
"""

import math
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F


class MineNet(nn.Module):
    """Single-hidden-layer statistics network for the MINE MI lower bound."""

    def __init__(self, input_size, hidden_size=50):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        return self.fc2(F.elu(self.fc1(x)))


class TERC:
    """Transfer Entropy Redundancy Criterion estimator.

    Parameters
    ----------
    name : str
        Dataset name; trajectories are read from ``obs_<name>.npy`` /
        ``acs_<name>.npy``. The special names ``"TFMT"`` / ``"TitForManyTats"``
        trigger a sweep over opponents TF2T..TF10T.
    num_iterations : int
        Number of MINE training iterations (``N`` in Algorithm 2).
    full : bool
        If True, run the standard selection on the full trajectory set. If
        False, run the per-training-quartile analysis used for interpretability.
    batch_size : int
        Mini-batch size ``b`` for MINE.
    lr : float
        Learning rate for the MINE optimizer.
    n_experiments : int
        Number of repeated MINE runs averaged per variable (default 5).
    subsample : int or None
        If the trajectory set is larger than this, randomly subsample to this
        many rows (the original implementation caps at 10,000).
    data_dir, results_dir : str or Path
        Where trajectories are read from and MI curves are written to.
    device : str or None
        Torch device; defaults to CUDA when available.
    verbose : bool
        Whether to print progress.
    """

    def __init__(self, name, num_iterations, full=True, batch_size=100,
                 lr=0.0001, n_experiments=5, subsample=10000,
                 data_dir=".", results_dir=".", device=None, verbose=True):
        self.name = name
        self.lr = lr
        self.full = full
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.ma_window_size = max(1, int(num_iterations / 10))
        self.n_experiments = n_experiments
        self.subsample = subsample
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self._is_tfmt = name in ("TFMT", "TitForManyTats")
        if not self._is_tfmt:
            self._load(name)

    # ------------------------------------------------------------------ I/O
    def _log(self, *args):
        if self.verbose:
            print(*args)

    def _load(self, name):
        self.acs = np.load(self.data_dir / f"acs_{name}.npy").round(decimals=2)
        self.obs = np.load(self.data_dir / f"obs_{name}.npy").round(decimals=2)
        if self.obs.ndim == 1:
            self.obs = np.expand_dims(self.obs, 1)
        if self.acs.ndim == 1:
            self.acs = np.expand_dims(self.acs, 1)
        if self.subsample and self.obs.shape[0] > self.subsample:
            idx = np.random.choice(self.obs.shape[0], self.subsample, replace=False)
            self.obs = self.obs[idx]
            self.acs = self.acs[idx]
        self._log(self.obs.shape, "obs")
        self._log(self.acs.shape, "acs")

    # --------------------------------------------------------- MINE training
    def mutual_information(self, joint, marginal, mine_net):
        t = mine_net(joint)
        et = torch.exp(mine_net(marginal))
        mi_lb = torch.mean(t) - torch.log(torch.mean(et))
        return mi_lb, t, et

    def learn_mine(self, batch, mine_net, optimizer, ma_et, ma_rate=0.001):
        joint, marginal = batch
        joint = torch.FloatTensor(joint).to(self.device)
        marginal = torch.FloatTensor(marginal).to(self.device)
        mi_lb, t, et = self.mutual_information(joint, marginal, mine_net)
        ma_et = (1 - ma_rate) * ma_et + ma_rate * torch.mean(et)
        loss = -(torch.mean(t) - (1 / ma_et.mean()).detach() * torch.mean(et))
        optimizer.zero_grad()
        autograd.backward(loss)
        optimizer.step()
        return mi_lb, ma_et

    def _sample_batch(self, obs, sample_mode="joint"):
        index = np.random.choice(range(obs.shape[0]), size=self.batch_size, replace=False)
        if sample_mode == "marginal":
            marginal_index = np.random.choice(range(obs.shape[0]), size=self.batch_size, replace=False)
            return np.concatenate((obs[index, :], np.array(self.acs[marginal_index, :])), axis=1)
        return np.concatenate((obs[index, :], np.array(self.acs[index, :])), axis=1)

    def _train_mi(self, obs):
        mine_net = MineNet(obs.shape[1] + 1).to(self.device)
        optimizer = optim.Adam(mine_net.parameters(), lr=self.lr)
        results = []
        for _ in range(self.n_experiments):
            ma_et = 1.0
            run = []
            for _ in range(self.num_iterations):
                batch = (self._sample_batch(obs), self._sample_batch(obs, "marginal"))
                mi_lb, ma_et = self.learn_mine(batch, mine_net, optimizer, ma_et)
                run.append(mi_lb.detach().cpu().numpy())
            results.append(self._moving_average(run))
        return np.array(results)

    def _moving_average(self, a):
        w = self.ma_window_size
        return [np.mean(a[i:i + w]) for i in range(0, len(a) - w)]

    def _phi(self, obs, drop_index):
        """Estimate Phi = I(A; X) - I(A; X_without_drop_index)."""
        trips = self._train_mi(obs)
        dubs = self._train_mi(np.delete(obs, drop_index, axis=1))
        return trips - dubs

    # --------------------------------------------------------- selection loop
    def _threshold(self, means, stdv):
        return means[-1] + 2 * stdv[-1] / math.sqrt(self.n_experiments)

    def run_inner(self, obs, acs):
        time0 = time()
        self.acs = acs
        # Null model: inject a known-uninformative binary variable.
        nm_col = np.random.randint(2, size=(obs.shape[0], 1))
        obs = np.concatenate((obs, nm_col), axis=1)
        nm = self._phi(obs, obs.shape[-1] - 1)
        means_nm = nm.mean(axis=0)
        stdv_nm = nm.std(axis=0)
        nm_upper = self._threshold(means_nm, stdv_nm)
        self._log(nm_upper, "Null Model Threshold")

        big_means = [means_nm]
        big_stdv = [stdv_nm]
        obs = np.delete(obs, obs.shape[-1] - 1, axis=1)

        keepers = []
        for i in range(obs.shape[-1] - 1, -1, -1):
            if obs.shape[-1] > 1:
                results = self._phi(obs, i)
                means = results.mean(axis=0)
                stdv = results.std(axis=0)
                big_means.append(means)
                big_stdv.append(stdv)
                self._save_curves(big_means, big_stdv)
                var_lower = means[-1] - 2 * stdv[-1] / math.sqrt(self.n_experiments)
                if var_lower < nm_upper:
                    obs = np.delete(obs, i, axis=1)
                    self._log(f"Excluded: variable {i} transferred a statistically "
                              "insignificant amount of entropy to the actions.")
                else:
                    keepers.append(i)
                    self._log(f"Included: variable {i} transferred a statistically "
                              "significant amount of entropy to the actions.")
            else:
                # Single remaining variable: compare its own MI to the null model.
                results = self._train_mi(obs)
                means = results.mean(axis=0)
                stdv = results.std(axis=0)
                big_means.append(means)
                big_stdv.append(stdv)
                self._save_curves(big_means, big_stdv)
                var_lower = means[-1] - 2 * stdv[-1] / math.sqrt(self.n_experiments)
                if var_lower > nm_upper:
                    keepers.append(i)
                    self._log(f"Included: variable {i} transferred a statistically "
                              "significant amount of entropy to the actions.")
                else:
                    self._log(f"Excluded: variable {i} transferred a statistically "
                              "insignificant amount of entropy to the actions.")
        self._log(f"\nElapsed (minutes) = {(time() - time0) / 60:.2f}")
        return keepers

    def _save_curves(self, big_means, big_stdv):
        np.save(self.results_dir / f"{self.name}_means.npy", np.array(big_means))
        np.save(self.results_dir / f"{self.name}_stdv.npy", np.array(big_stdv))

    def run_all(self):
        if self.full:
            keepers = self.run_inner(self.obs, self.acs)
            self._log(f"The indices of the optimal subset of variables are {keepers}")
            return keepers

        self._log("WARNING: running the per-quartile TERC analysis; variables are "
                  "selected separately for each training quartile.")
        n = self.obs.shape[0]
        q = n // 4
        obs_q = [self.obs[:q], self.obs[q:2 * q], self.obs[2 * q:3 * q], self.obs[3 * q:]]
        acs_q = [self.acs[:q], self.acs[q:2 * q], self.acs[2 * q:3 * q], self.acs[3 * q:]]
        keepers = None
        for obs, acs in zip(obs_q, acs_q):
            keepers = self.run_inner(obs, acs)
        return keepers

    def run(self):
        """Run TERC, returning the indices of the selected variables.

        For the TFMT sweep, the final opponent's keepers are returned.
        """
        if self._is_tfmt:
            keepers = None
            for N in range(2, 11):
                self._load(f"{self.name}{N}")
                keepers = self.run_all()
            return keepers
        return self.run_all()
