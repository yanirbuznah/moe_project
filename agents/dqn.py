import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import models.utils as ut
from agents.custom_env import CustomEnv
from models.MOE import MixtureOfExperts


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, backbone):
        super(DQN, self).__init__()
        self.backbone = backbone
        self.fc1 = nn.LazyLinear(hidden_dim)
        self.fc2 = nn.LazyLinear(action_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


class ReplayBuffer:
    def __init__(self, capacity):
        self.state = torch.FloatTensor()
        self.action = torch.LongTensor()
        self.reward = torch.FloatTensor()
        self.next_state = torch.FloatTensor()
        self.capacity = capacity

    def append(self, state, action, reward):
        self.state = torch.cat([self.state, state.cpu()])
        self.action = torch.cat([self.action, action.cpu()])
        self.reward = torch.cat([self.reward, reward.cpu()])
        # self.next_state = torch.cat([self.next_state, torch.FloatTensor(next_state)])
        if len(self.state) > self.capacity:
            self.state = self.state[-self.capacity:]
            self.action = self.action[-self.capacity:]
            self.reward = self.reward[-self.capacity:]
            # self.next_state = self.next_state[-self.capacity:]

    def sample(self, batch_size, device='cpu'):
        idxs = np.random.randint(0, len(self.state), size=batch_size)
        return self.state[idxs].to(device), self.action[idxs].to(device), self.reward[idxs].to(device)

    def sample_with_exponentially_smoothing(self, batch_size, alpha=0.0005, device='cpu'):
        # create a list of probabilities for each sample
        probs = np.array([alpha * (1 - alpha) ** i for i in range(len(self.state), 0, -1)])
        # normalize the probabilities
        probs = probs / probs.sum()
        # sample from the list of probabilities
        idxs = np.random.choice(len(self.state), size=batch_size, p=probs)
        return self.state[idxs].to(device), self.action[idxs].to(device), self.reward[idxs].to(device)

    def __len__(self):
        return len(self.state)


class Agent:

    def __init__(self, model: MixtureOfExperts):
        self.model = model
        self.config = model.router_config['model_config']
        backbone = self._get_backbone()
        self.env = CustomEnv(model)
        self.state_dim = torch.prod(torch.tensor(self.env.observation_space.shape)).item()
        self.action_dim = self.env.action_space.n
        self.hidden_dim = self.config.get('hidden_dim', 128)
        self.buffer_capacity = self.config.get('buffer_capacity', 10000)
        self.batch_size = self.config.get('batch_size', 64)
        self.gamma = self.config.get('gamma', 0.99)
        self.epsilon = self.config.get('epsilon', 1.0)
        self.lr = self.config.get('lr', 0.001)
        self.num_of_episodes = self.config.get('num_of_episodes', 500)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.999)
        self.epsilon_min = self.config.get('epsilon_min', 0.01)
        self.q_net = DQN(self.state_dim, self.action_dim, self.hidden_dim, backbone).to(model.device)
        self.target_net = DQN(self.state_dim, self.action_dim, self.hidden_dim, backbone).to(model.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.memory = ReplayBuffer(self.buffer_capacity)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)

    def __call__(self, *args, **kwargs):
        return nn.Softmax(dim=1)(self.q_net(*args, **kwargs))
        # return self.q_net(*args, **kwargs)
    def _get_backbone(self):
        backbone_config = self.config.get('backbone', None)
        backbone_output_shape = self.config.get('backbone_output_shape', None)
        backbone = ut.get_model(backbone_config, output_shape=backbone_output_shape)
        return backbone

    def to(self, device):
        self.q_net.to(device)
        self.target_net.to(device)
        return self

    def act(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return torch.randint(0, self.action_dim, (state.shape[0],)).to(self.model.device)
        else:
            with torch.no_grad():
                state = state.unsqueeze(0) if len(state.shape) == 3 else state
                return self.q_net(state).argmax(axis=1)

    def predict(self, state):
        with torch.no_grad():
            return self.q_net(state).argmax(axis=1)

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        state_batch, action_batch, reward_batch = self.memory.sample_with_exponentially_smoothing(self.batch_size,5e-4,
                                                                                                  self.model.device)
        self.update_by_batch(state_batch, action_batch, reward_batch)

    def update_by_batch(self, state_batch, action_batch, reward_batch):
        q_values = self.q_net(state_batch)
        q_values = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # next_q_values = self.target_net(next_state_batch)
            # next_q_values = next_q_values.max(1)[0]
            expected_q_values = reward_batch
        # loss = nn.HuberLoss()(q_values, expected_q_values)
        loss = nn.MSELoss()(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def learn(self):
        rewards = 0
        self.epsilon = self.config.get('epsilon', 1.0)
        episode = 0
        for episode in tqdm(range(self.num_of_episodes), desc=f"RL Training"):
            state = self.env.reset()
            # if state is tensor convert to device
            if isinstance(state, torch.Tensor):
                state = state.to(self.model.device)
            else:
                state = torch.FloatTensor(state).to(self.model.device)

            # while not done:
            # one step in the environment
            action = self.act(state)
            _, reward, _, _ = self.env.step(action)
            self.memory.append(state, action, reward)
            self.update()
            if episode % 10 == 0:
                self.update_target_net()
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            rewards += (reward.mean().item())
            tqdm.write(f"\rEpisode: {episode}, Epsilon: {round(self.epsilon, 3)},  mean Reward: {rewards / (episode + 1)}", end="")
            # print("\rEpisode: {}\{}, Epsilon: {},  mean Reward: {}".format(episode, self.num_of_episodes,
            #                                                                round(self.epsilon, 3),
            #                                                                rewards / (episode + 1)), end="")
        # print("\n")
        # return rewards

# def run_example():
#     import gym
#     env = gym.make('CartPole-v0')
#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.n
#     hidden_dim = 128
#     buffer_capacity = 10000
#     batch_size = 64
#     gamma = 0.99
#     epsilon = 1.0
#     lr = 0.001
#     max_episodes = 1000
#     agent = Agent(env)
#     rewards = agent.learn(total_timesteps=max_episodes)
#     print('Average reward: {}'.format(np.mean(rewards[-100:])))
#

# run_example()
