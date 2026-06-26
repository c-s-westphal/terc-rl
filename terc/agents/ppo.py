"""Proximal Policy Optimization agent for continuous control (Pendulum).

The actor parameterizes a Beta distribution over the (rescaled) action space
and is updated with the clipped surrogate objective of Schulman et al. (2017).
Hyperparameters follow the paper's "OpenAI's Gym: Pendulum" appendix.
"""

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta


class ContinuousActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=128, fc2_dims=128):
        super().__init__()
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.alpha = nn.Linear(fc2_dims, n_actions)
        self.beta = nn.Linear(fc2_dims, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = T.tanh(self.fc1(state))
        x = T.tanh(self.fc2(x))
        alpha = F.relu(self.alpha(x)) + 1.0
        beta = F.relu(self.beta(x)) + 1.0
        return Beta(alpha, beta)


class ContinuousCriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=128, fc2_dims=128):
        super().__init__()
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.v = nn.Linear(fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = T.tanh(self.fc1(state))
        x = T.tanh(self.fc2(x))
        return self.v(x)


class PPOMemory:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.clear_memory()

    def recall(self):
        return (np.array(self.states), np.array(self.new_states),
                np.array(self.actions), np.array(self.probs),
                np.array(self.rewards), np.array(self.dones))

    def generate_batches(self):
        n_states = len(self.states)
        n_batches = int(n_states // self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        return [indices[i * self.batch_size:(i + 1) * self.batch_size]
                for i in range(n_batches)]

    def store_memory(self, state, state_, action, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.new_states.append(state_)

    def store_probs(self, prob):
        self.probs.append(prob)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.new_states = []


class PPOAgent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=3e-3,
                 gae_lambda=0.95, policy_clip=0.2, batch_size=64, n_epochs=10,
                 horizon=2048, entropy_coefficient=1e-3):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coefficient = entropy_coefficient
        self.actor = ContinuousActorNetwork(n_actions, input_dims, alpha)
        self.critic = ContinuousCriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)
        self.N = horizon
        self.counter = 0

    def learn(self, state, reward, state_, action, done):
        self.memory.store_memory(state, state_, action, reward, done)
        self.counter += 1
        if (self.counter + 1) % self.N == 0:
            self.learn_remembered()

    def choose_action(self, observation):
        with T.no_grad():
            state = T.tensor([observation], dtype=T.float).to(self.actor.device)
            dist = self.actor(state)
            action = dist.sample()
            probs = dist.log_prob(action)
            self.memory.store_probs(probs.cpu().numpy().flatten())
        return action.cpu().numpy().flatten()

    def calc_adv_and_returns(self, memories):
        states, new_states, r, dones = memories
        with T.no_grad():
            values = self.critic(states)
            values_ = self.critic(new_states)
            deltas = r + self.gamma * values_ - values
            deltas = deltas.cpu().flatten().numpy()
            adv = [0]
            for dlt, mask in zip(deltas[::-1], dones[::-1]):
                advantage = dlt + self.gamma * self.gae_lambda * adv[-1] * (1 - mask)
                adv.append(advantage)
            adv.reverse()
            adv = adv[:-1]
            adv = T.tensor(adv).float().unsqueeze(1).to(self.critic.device)
            returns = adv + values
            adv = (adv - adv.mean()) / (adv.std() + 1e-4)
        return adv, returns

    def learn_remembered(self):
        state_arr, new_state_arr, action_arr, old_prob_arr, reward_arr, dones_arr = \
            self.memory.recall()
        state_arr = T.tensor(state_arr, dtype=T.float).to(self.critic.device)
        action_arr = T.tensor(action_arr, dtype=T.float).to(self.critic.device)
        old_prob_arr = T.tensor(old_prob_arr, dtype=T.float).to(self.critic.device)
        new_state_arr = T.tensor(new_state_arr, dtype=T.float).to(self.critic.device)
        r = T.tensor(reward_arr, dtype=T.float).unsqueeze(1).to(self.critic.device)
        adv, returns = self.calc_adv_and_returns((state_arr, new_state_arr, r, dones_arr))
        for _ in range(self.n_epochs):
            for batch in self.memory.generate_batches():
                states = state_arr[batch]
                old_probs = old_prob_arr[batch]
                actions = action_arr[batch]
                dist = self.actor(states)
                new_probs = dist.log_prob(actions)
                prob_ratio = T.exp(new_probs.sum(1, keepdim=True) -
                                   old_probs.sum(1, keepdim=True))
                weighted_probs = adv[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(
                    prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * adv[batch]
                entropy = dist.entropy().sum(1, keepdims=True)
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs)
                actor_loss -= self.entropy_coefficient * entropy
                self.actor.optimizer.zero_grad()
                actor_loss.mean().backward()
                T.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                self.actor.optimizer.step()

                critic_value = self.critic(states)
                critic_loss = (critic_value - returns[batch]).pow(2).mean()
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()
        self.memory.clear_memory()
