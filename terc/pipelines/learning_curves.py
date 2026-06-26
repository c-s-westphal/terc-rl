"""Learning-curve generation: retrain agents on the full state vs. the
TERC-selected subset and record reward curves, demonstrating the improvement in
learning efficiency from the reduced state representation (stage 3 of the
pipeline).

Output is written to ``learning_curve_<name>.npy`` in ``data_dir`` as an array
of shape ``(n_subsets, 2, n_trajectories)`` holding the mean and std reward
curves (averaged over ``n_experiments`` seeds) for each subset.
"""

import random
from pathlib import Path

import numpy as np

from terc.agents import ActorCriticAgent, PPOAgent, QLearningAgent
from terc.envs import SecretKeyGameEnv, TitForNTatsEnv, make_gym_env, resolve_gym_name, GYM_ENVS


class LearningCurveGenerator:
    def __init__(self, name, keep_tene, lra=0.0001, lrc=0.001,
                 num_trajectories=3000, n_experiments=5, n_random=3,
                 data_dir=".", skg_state_length=50, tfmt_opponent=5, verbose=True):
        self.name = name
        self.orig_name = name
        self.lra = lra
        self.lrc = lrc
        self.num_trajectories = num_trajectories
        self.n_experiments = n_experiments
        self.n_random = n_random
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.skg_state_length = skg_state_length
        self.tfmt_opponent = tfmt_opponent
        self.verbose = verbose
        self.is_tfmt = name in ("TFMT", "TitForManyTats")

        canonical = resolve_gym_name(name)
        if canonical in GYM_ENVS:
            self.env = make_gym_env(canonical)
            self.keep_all = range(len(self.env.reset()[0]) + self.n_random)
        elif name in ("SKG", "SecretKeyGame"):
            self.env = SecretKeyGameEnv(state_length=self.skg_state_length)
            self.keep_all = range(len(self.env.reset()[0]))
        elif self.is_tfmt:
            self.env = TitForNTatsEnv(1)
            self.keep_all = range(len(self.env.reset()[0]))
        else:
            raise ValueError(f"Environment incorrectly specified: {name!r}")

        self.keep_tene = keep_tene
        self.keepers = [self.keep_all, self.keep_tene]

    def _log(self, *args):
        if self.verbose:
            print(*args)

    def _make_obs(self, obs, keep):
        state = list(obs) + [random.randint(0, 1) for _ in range(self.n_random)]
        return [e for i, e in enumerate(state) if i in keep]

    def _make_agent(self, keep):
        canonical = resolve_gym_name(self.name)
        if canonical == "Pendulum":
            return PPOAgent(n_actions=self.env.action_space.shape[0], batch_size=64,
                            alpha=self.lra, n_epochs=10, input_dims=[int(len(keep))])
        if self.is_tfmt:
            self.env = TitForNTatsEnv(self.N, max_mem_size=int(len(keep)))
            return QLearningAgent(2, 4 ** self.env.mem_size, 0.9, 0.99, 1,
                                  self.num_trajectories)
        return ActorCriticAgent(self.lra, self.lrc, input_dims=int(len(keep)),
                                fc1_dims=64, fc2_dims=64,
                                n_actions=self.env.action_space.n)

    def run_inner(self):
        lcs = []
        is_pendulum = resolve_gym_name(self.name) == "Pendulum"
        for keep in self.keepers:
            rewards_expt = []
            for _ in range(self.n_experiments):
                self.agent = self._make_agent(keep)
                rewards = []
                for traj in range(self.num_trajectories):
                    score = 0
                    done = done2 = False
                    observation = self.env.reset()[0]
                    up_obs = self._make_obs(observation, keep)
                    while not done and not done2:
                        action = self.agent.choose_action(up_obs)
                        observation_, reward, done, done2, _ = self.env.step(action)
                        up_obs_ = self._make_obs(observation_, keep)
                        self.agent.learn(up_obs, reward, up_obs_, action, done)
                        up_obs = up_obs_
                        score += reward
                    rewards.append(score)
                rewards_expt.append(rewards)
            means = np.array(rewards_expt).mean(axis=0)
            std = np.array(rewards_expt).std(axis=0)
            lcs.append([means, std])
        np.save(self.data_dir / f"learning_curve_{self.name}.npy", np.array(lcs))

    def run(self):
        self._log(f"Subsets evaluated (full, TERC): {list(self.keep_all)}, {list(self.keep_tene)}")
        if self.is_tfmt:
            self.N = self.tfmt_opponent
            self._log(f"Now collecting TF{self.N}T learning curves")
            self.name = f"{self.orig_name}{self.N}"
            self.run_inner()
            self.name = self.orig_name
        else:
            self.run_inner()
        self._log("Learning curves generated and saved.")
