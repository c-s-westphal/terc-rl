"""One-step temporal-difference Actor-Critic agent.

Used for the discrete-action environments in the paper (Secret Key Game,
Cart Pole, Lunar Lander). Both the actor and critic are single-hidden-layer
fully connected networks with ReLU activations, following the hyperparameters
reported in Appendix "Implementation Details".
"""

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ActorNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, fc1_dims=256, fc2_dims=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state), inplace=False)
        x = F.relu(self.fc2(x), inplace=False)
        return self.pi(x)


class CriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, fc1_dims=256, fc2_dims=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.v = nn.Linear(fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.v(x)


class ActorCriticAgent:
    """One-step TD Actor-Critic agent.

    Parameters
    ----------
    lra, lrc : float
        Learning rates for the actor and critic respectively.
    input_dims : int
        Dimensionality of the (possibly doped) state vector.
    fc1_dims, fc2_dims : int
        Hidden layer widths. The paper uses a single fully connected layer of
        size 64; both are exposed for flexibility.
    n_actions : int
        Size of the discrete action space.
    gamma : float
        Discount factor.
    """

    def __init__(self, lra, lrc, input_dims, fc1_dims, fc2_dims, n_actions,
                 max_mem_size=1, gamma=0.99):
        self.gamma = gamma
        self.lra = lra
        self.lrc = lrc
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.actor = ActorNetwork(lra, input_dims, n_actions, fc1_dims, fc2_dims)
        self.critic = CriticNetwork(lrc, input_dims, n_actions, fc1_dims, fc2_dims)
        self.log_prob = None
        self.prob = None
        self.mem_cntr = 0
        self.mem_size = max_mem_size
        self.state_memory = np.zeros((self.mem_size, int(input_dims)), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, int(input_dims)), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.int32)

    def store_transition(self, state, action, reward, state_):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.mem_cntr += 1

    def choose_action(self, observation):
        state = T.tensor(np.asarray([observation]), dtype=T.float).to(self.actor.device)
        probabilities = F.softmax(self.actor.forward(state), dim=1)
        self.prob = probabilities
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        self.log_prob = action_probs.log_prob(action)
        return action.item()

    def learn(self, state, reward, state_, action, done):
        state = T.tensor(np.asarray([state]), dtype=T.float).to(self.critic.device)
        state_ = T.tensor(np.asarray([state_]), dtype=T.float).to(self.critic.device)
        reward = T.tensor(reward, dtype=T.float).to(self.critic.device).detach()

        critic_value = self.critic.forward(state)
        critic_value_ = self.critic.forward(state_)
        mask = 0 if done else 1
        delta = reward + self.gamma * critic_value_ * mask - critic_value
        actor_loss = -(self.log_prob * delta.detach())
        critic_loss = delta ** 2

        self.critic.optimizer.zero_grad()
        self.actor.optimizer.zero_grad()
        critic_loss.backward()
        actor_loss.backward()
        self.critic.optimizer.step()
        self.actor.optimizer.step()
