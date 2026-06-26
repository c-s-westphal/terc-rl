"""Trajectory generation: train an agent to convergence on the full observable
state and save the resulting state-action trajectories for TERC.

This implements stages 1-2 of the TERC pipeline (initial training and
trajectory sampling). Output is written to ``obs_<name>.npy`` /
``acs_<name>.npy`` in ``data_dir``.

For the Gym physics environments the state is "doped" with extra uniform random
variables (Grooten et al., 2023); TERC should later discard them.
"""

import random
from pathlib import Path

import numpy as np

from terc.agents import ActorCriticAgent, PPOAgent, QLearningAgent
from terc.envs import SecretKeyGameEnv, TitForNTatsEnv, make_gym_env, resolve_gym_name, GYM_ENVS
from terc.pipelines.synthetic import generate_data


class TrajectoryGenerator:
    def __init__(self, name, lra=0.0001, lrc=0.001, num_trajectories=10000,
                 data_dir=".", skg_state_length=50, verbose=True):
        self.name = name
        self.orig_name = name
        self.lra = lra
        self.lrc = lrc
        self.num_trajectories = num_trajectories
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.skg_state_length = skg_state_length
        self.verbose = verbose
        self.adding_varbs = False
        self.n_random = 0
        self.env = None
        self.agent = None

        canonical = resolve_gym_name(name)
        if canonical in GYM_ENVS:
            self._setup_gym(canonical)
        elif name in ("SKG", "SecretKeyGame"):
            self._setup_skg()
        elif name in ("TFMT", "TitForManyTats"):
            pass  # configured per-opponent in run()
        elif name in ("2red_trips", "4red_varbs"):
            pass  # handled directly in run()
        else:
            raise ValueError(f"Environment incorrectly specified: {name!r}")

    def _log(self, *args):
        if self.verbose:
            print(*args)

    def _setup_gym(self, canonical):
        self.env = make_gym_env(canonical)
        self.adding_varbs = True
        self.n_random = GYM_ENVS[canonical]["n_random"]
        base_dims = len(self.env.reset()[0])
        input_dims = base_dims + self.n_random
        if canonical == "Pendulum":
            self.agent = PPOAgent(
                n_actions=self.env.action_space.shape[0], batch_size=64,
                alpha=self.lra, n_epochs=10, input_dims=[int(input_dims)])
            self._log("PPO agent selected (hyperparameters per Schulman et al., 2017)")
        else:
            self.agent = ActorCriticAgent(
                self.lra, self.lrc, input_dims=input_dims, fc1_dims=64,
                fc2_dims=64, n_actions=self.env.action_space.n)

    def _setup_skg(self):
        self.env = SecretKeyGameEnv(state_length=self.skg_state_length)
        self.adding_varbs = False
        self.agent = ActorCriticAgent(
            self.lra, self.lrc, input_dims=len(self.env.reset()[0]),
            fc1_dims=64, fc2_dims=64, n_actions=self.env.action_space.n)

    def _dope(self, obs):
        if not self.adding_varbs:
            return obs
        add = np.array([random.randint(0, 1) for _ in range(self.n_random)])
        return np.concatenate((obs, add))

    def run_inner(self):
        acs, obs, scores = [], [], []
        self._log(self.name)
        is_pendulum = resolve_gym_name(self.name) == "Pendulum"
        for traj in range(self.num_trajectories):
            score = 0
            done = done2 = False
            observation = self.env.reset()[0]
            up_obs = self._dope(observation)
            while not done and not done2:
                action = self.agent.choose_action(up_obs)
                observation_, reward, done, done2, _ = self.env.step(action)
                up_obs_ = self._dope(observation_)
                acs.append(action[0] if is_pendulum else action)
                self.agent.learn(up_obs, reward, up_obs_, action, done)
                obs.append(list(up_obs))
                up_obs = up_obs_
                score += reward
            scores.append(score)
            if traj % 100 == 0:
                self._log(f"Mean score over last 100 trajectories: "
                          f"{np.array(scores[-100:]).mean():.3f}")
        np.save(self.data_dir / f"obs_{self.name}.npy", np.array(obs))
        np.save(self.data_dir / f"acs_{self.name}.npy", np.array(acs))

    def run(self):
        if self.name in ("2red_trips", "4red_varbs"):
            generate_data(self.name, data_dir=self.data_dir)
        elif self.name in ("TFMT", "TitForManyTats"):
            for N in range(2, 11):
                self._log(f"Now collecting TF{N}T data")
                self.env = TitForNTatsEnv(N)
                self.agent = QLearningAgent(
                    2, 4 ** self.env.mem_size, 0.9, 0.99, 1, self.num_trajectories)
                self.name = f"{self.orig_name}{N}"
                self.run_inner()
            self.name = self.orig_name
        else:
            self.run_inner()
        self._log(f"Trajectories generated for {self.orig_name}")
