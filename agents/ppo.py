import os
import random

import numpy as np
import torch
import torch as T
import torch.nn as nn
import torch.optim as optim
# import matplotlib.pyplot as plt
from torch.distributions.categorical import Categorical
from torch.nn import Identity
from tqdm import tqdm



class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def _concat_memory(self):
        self.states = T.cat(self.states)
        self.probs = T.cat(self.probs)
        self.vals = T.cat(self.vals)
        self.actions = T.cat(self.actions)
        self.rewards = T.cat(self.rewards)
        self.dones = T.cat(self.dones)

    def generate_batches(self):
        # if the type of the memory is list, convert it to tensor
        if isinstance(self.states, list):
            self._concat_memory()
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return self.states, self.actions, self.probs, self.vals, self.rewards, self.dones, \
            batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
                 fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()
        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')

        self.actor = nn.Sequential(
            nn.LazyLinear(fc1_dims),
            nn.ReLU(),
            nn.LazyLinear(fc2_dims),
            nn.ReLU(),
            nn.LazyLinear(n_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)

        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
                 chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
            nn.LazyLinear(fc1_dims),
            nn.ReLU(),
            nn.LazyLinear(fc2_dims),
            nn.ReLU(),
            nn.LazyLinear(1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent:
    def __init__(self, env, hidden_dim=128, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, n_epochs=10, encoder=Identity()):
        self.epsilon = 1.0
        self.env = env
        self.state_dim = torch.prod(torch.tensor(env.observation_space.shape)).item()
        self.action_dim = env.action_space.n
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.encoder = encoder.to(device)
        self.actor = ActorNetwork(self.action_dim, self.state_dim, alpha).to(device)
        self.critic = CriticNetwork(self.state_dim, alpha).to(device)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, state):
        state = self.encoder(state)
        dist = self.actor(state)
        values = self.critic(state)
        actions = dist.sample().detach()

        probs = dist.log_prob(actions).detach()
        values = T.squeeze(values).detach()

        return actions, probs, values

    def chose_actions(self, states):
        dist = self.actor(states)
        values = self.critic(states)
        actions = dist.sample()

        probs = dist.log_prob(actions)
        values = T.squeeze(values)

        return actions, probs, values

    def act(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return torch.randint(0, self.action_dim, (state.shape[0],)).to(device)
        else:
            with torch.no_grad():
                state = state.unsqueeze(0) if len(state.shape) == 3 else state
                state = self.encoder(state)
                return self.actor(state).probs.argmax(axis=1)

    def update(self, total_timesteps):
        for _ in tqdm(range(total_timesteps), desc='Train PPO'):
            state_arr, action_arr, old_prob_arr, vals_arr, \
                reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            advantage = torch.zeros(reward_arr.shape)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = advantage.to(self.actor.device)

            values = values.to(self.actor.device)
            for batch in batches:
                states = state_arr[batch].to(self.actor.device)
                old_probs = old_prob_arr[batch].to(self.actor.device)
                actions = action_arr[batch].to(self.actor.device)

                states = self.encoder(states)
                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                # prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip,
                                                 1 + self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()

    def learn(self, total_timesteps=500, N=50):
        self.epsilon = 1.0
        learn_iters = 0
        avg_score = 0
        n_steps = 0
        score_history = []
        best_score = -np.inf
        for i in range(total_timesteps):
            state = self.env.reset()
            done = False
            score = 0

            action, prob, val = self.choose_action(state)
            observation_, reward, done, info = self.env.step(action)
            n_steps += 1
            self.remember(state, action, prob, val, reward, done)
            if n_steps % N == 0:
                self.update(1)
                learn_iters += 1
            observation = observation_
            score_history.append(reward.mean().item())
            avg_score = np.mean(score_history[-100:])

            if avg_score > best_score:
                best_score = avg_score
                self.save_models()

            print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                  'time_steps', n_steps, 'learning_steps', learn_iters)
        self.epsilon = 0.01


# def plot_learning_curve(x, scores, figure_file):
#     running_avg = np.zeros(len(scores))
#     for i in range(len(running_avg)):
#         running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])
#     plt.plot(x, running_avg)
#     plt.title('Running average of previous 100 scores')
#     plt.savefig(figure_file)
