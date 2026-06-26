"""Tit-For-N-Tats (TFNT) opponent in the Iterated Prisoner's Dilemma.

The opponent defects only after ``N`` consecutive defections by the learning
agent and otherwise cooperates. The optimal counter-strategy is cyclic with
period ``N`` and requires a history length of at least ``N - 1`` to learn,
making this a setting where TERC discovers the minimal sufficient state-history
length.

The observable state is the last ``mem_size`` action pairs, each encoded as a
single base-4 integer ``a_you + 2 * a_opponent`` (0..3).
"""

import numpy as np


class TitForNTatsEnv:
    def __init__(self, number_of_tats, n_states=4, n_actions=2,
                 max_mem_size=10, episode_length=20):
        self.x = np.random.randint(0, n_states)
        self.N = number_of_tats
        self.n_states = n_states
        self.n_actions = n_actions
        self.episode_length = episode_length
        # Payoff matrix (player gains): see Appendix "TFNT in the IPD".
        self.payout_mat = np.array([[2.0, 3.0], [0.0, 1.0]])
        self.mem_cntr_a = 0
        self.done_counter = 0
        self.mem_size = max_mem_size
        self.mem_size_a = max_mem_size * int(n_actions)
        self.last_state = 0
        self.penultimate_state = 0
        self.index_a = 0
        zeros = [0] * (self.mem_size * int(n_actions))
        self.action_memory_yours = np.array(zeros, dtype=np.int32)
        self.action_memory_op = np.array(zeros, dtype=np.int32)

    def reset(self):
        return [self.state_conv()]

    def rewards(self, ac0, ac1):
        return self.payout_mat[ac1][ac0]

    def state_update(self, action_yours, action_op):
        self.index_a = self.mem_cntr_a % self.mem_size_a
        self.action_memory_yours[self.index_a] = action_yours
        self.action_memory_op[self.index_a] = action_op
        self.mem_cntr_a += 1

    def state_conv(self):
        """Convert the action memory into the base-4 encoded state vector."""
        state = [self.action_memory_yours[(i + self.index_a + 1) % self.mem_size]
                 + self.action_memory_op[(i + self.index_a + 1) % self.mem_size] * 2
                 for i in range(self.mem_size)]
        return state[::-1]

    def step(self, action_yours):
        if sum(self.action_memory_yours[(i + self.index_a + 1) % self.mem_size]
               for i in range(self.N)) == self.N:
            action_op = 1
        else:
            action_op = 0
        reward = self.rewards(action_yours, action_op)
        self.state_update(action_yours, action_op)
        self.done_counter += 1
        state = self.state_conv()
        done = self.done_counter % self.episode_length == 0
        return state, reward, done, done, {}
