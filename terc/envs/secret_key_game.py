"""The Secret Key Game environment.

Inspired by Shamir's secret-sharing protocol. A numerical secret ``y_0`` is the
y-intercept of a second-order polynomial ``f(x) = y_0 - a x + b x^2``. Three of
the ``state_length`` keys in the observable state are the y-values of points on
this curve at ``x in {1, 2, 3}``; the remaining keys are uninformative decoys.

The agent observes the full key vector and must output the secret. The reward is
the negative absolute error ``r = -|a - y_0|``. TERC is expected to identify the
three secret-forming keys and discard the decoys.

Notes
-----
Following the paper, the three secret keys occupy the first three indices of the
state vector. The Gym/Gymnasium-style API is preserved: ``reset`` returns
``[state]`` and ``step`` returns a 5-tuple ``(state, reward, terminated,
truncated, info)``.
"""

import random
import numpy as np
from numpy.polynomial import polynomial as P


class SecretKeyGameEnv:
    class _ActionSpace:
        def __init__(self, n):
            self.n = n

    def __init__(self, state_length=50, n_keys=3, key_max=10, action_range=40,
                 episode_length=10):
        self.state_length = state_length
        self.n_keys = n_keys
        self.key_max = key_max
        self.action_range = action_range
        self.episode_length = episode_length
        self.counter = 0
        self.state = self._draw_state()
        # Discrete action space of size 2*action_range (the paper uses 80 for
        # a secret range of [-40, 40]).
        self.action_space = self._ActionSpace(2 * action_range)

    def _draw_state(self):
        return np.array([random.randint(0, self.key_max)
                         for _ in range(self.state_length)])

    def reset(self):
        self.state = self._draw_state()
        return [self.state]

    def step(self, action):
        action = action - self.action_range
        x = np.arange(1.0, self.n_keys + 1.0)
        y = np.array([self.state[i] for i in range(self.n_keys)])
        c = P.polyfit(x, y, self.n_keys - 1)
        secret = c[0]
        reward = -abs(int(round(secret - action)))
        self.counter += 1
        done = self.counter % self.episode_length == 0
        self.state = self.reset()[0]
        return self.state, reward, done, done, {}
