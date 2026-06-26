"""Tabular Q-learning agent for the Iterated Prisoner's Dilemma (TFNT).

The state is a one-hot-style encoding of the last ``mem_size`` action pairs in
base 4 (see :mod:`terc.envs.tit_for_n_tats`). Epsilon decays linearly to zero
over the course of training, matching the paper's appendix.
"""

import numpy as np
import random
from random import randint


class QLearningAgent:
    def __init__(self, n_actions, n_states, lr, gamma, eps, q_trajectories):
        self.n_a = n_actions
        self.n_s = n_states
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.q_table = np.zeros((self.n_s, self.n_a))
        self.q_trajectories = q_trajectories

    def choose_action(self, state):
        ru = random.uniform(0, 1)
        state = self.state_conv(state)
        if ru > self.eps and self.q_table[state][0] != self.q_table[state][1]:
            return np.argmax(self.q_table[state])
        return randint(0, 1)

    def reset(self):
        self.q_table = np.zeros((self.n_s, self.n_a))

    def state_conv(self, obs):
        """Convert a base-4 action-pair history into a flat table index."""
        if len(obs) > 1:
            return int(sum(obs[i] * (4 ** i) for i in range(len(obs))))
        return int(obs[0])

    def learn(self, state, reward, state_, action, done):
        state = self.state_conv(state)
        state_ = self.state_conv(state_)
        self.q_table[state][action] += self.lr * (
            reward + np.max(self.q_table[state_]) * self.gamma
            - self.q_table[state][action]
        )
        if self.eps > 0:
            self.eps = self.eps - (1 / (self.q_trajectories * 20 - 500))
