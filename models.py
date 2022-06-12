import torch
from torch import nn
from torch.optim import Adam
import copy
from collections import deque
import random
import numpy as np
from torch.nn import functional as F

GAMMA = 0.999
TAU = 0.002
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3
DEVICE = 'cuda'
BATCH_SIZE = 64


def soft_update(target, source):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1 - TAU) * tp.data + TAU * sp.data)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.model(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        return self.model(torch.cat([state, action], dim=-1)).view(-1)


class DDPG:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(DEVICE)
        self.critic = Critic(state_dim, action_dim).to(DEVICE)

        self.actor_optim = Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optim = Adam(self.critic.parameters(), lr=CRITIC_LR)

        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

        self.replay_buffer = deque(maxlen=100000)

        self.state_mean = None
        self.state_sq_mean = None

    def update(self, transition):
        self._append_transition(transition)
        if len(self.replay_buffer) > 16 * BATCH_SIZE:
            batch = [self.replay_buffer[random.randint(0, len(self.replay_buffer) - 1)] for _ in range(BATCH_SIZE)]
            state, action, next_state, reward = zip(*batch)
            state_std = self.state_sq_mean - self.state_mean ** 2
            state_std[state_std < 0] = 0
            state_std = state_std ** 0.5
            state = (np.array(state) - self.state_mean) / (state_std + 1e-16)
            state = torch.tensor(state, device=DEVICE, dtype=torch.float)
            action = torch.tensor(np.array(action), device=DEVICE, dtype=torch.float)
            next_state = (np.array(next_state) - self.state_mean) / (state_std + 1e-16)
            next_state = torch.tensor(next_state, device=DEVICE, dtype=torch.float)
            reward = torch.tensor(np.array(reward), device=DEVICE, dtype=torch.float)

            output = self.critic(state, action)
            with torch.no_grad():
                action_predicted = self.target_actor(next_state)
                target = reward + GAMMA * self.target_critic(next_state, action_predicted)
            loss = F.mse_loss(output, target)
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()

            action_predicted = self.actor(state)
            loss = -torch.sum(self.critic(state, action_predicted))
            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()

            soft_update(self.target_actor, self.actor)
            soft_update(self.target_critic, self.critic)

    def _append_transition(self, transition):
        size = len(self.replay_buffer)
        state = transition[0]
        self.state_mean = state if size == 0 else (self.state_mean * size + state) / (size + 1)
        self.state_sq_mean = state ** 2 if size == 0 else (self.state_sq_mean * size + state ** 2) / (size + 1)
        self.replay_buffer.append(transition)

    def act(self, state):
        with torch.no_grad():
            if self.state_mean is not None:
                state_std = self.state_sq_mean - self.state_mean ** 2
                state_std[state_std < 0] = 0
                state_std = state_std ** 0.5
                state = (state - self.state_mean) / (state_std + 1e-16)
            state = torch.tensor(np.array([state]), device=DEVICE, dtype=torch.float)
            return self.actor(state).cpu().numpy()[0]

    def save(self):
        torch.save(self.actor, 'agent.pkl')

