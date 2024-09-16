import gym
import numpy as np
import torch
from torch import nn

from agents.reward_strategy import RewardStrategy
from logger import Logger
from models.MOE import MixtureOfExperts

logger = Logger().logger(__name__)
# from utils.general_utils import device


class CustomEnv(gym.Env):
    def __init__(self, model: MixtureOfExperts, config):
        self.action_space = gym.spaces.Discrete(model.num_experts)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=model.input_shape, dtype=np.float32)
        self.num_of_experts = model.num_experts
        self.train_set = model.train_set
        self.means = np.random.normal(0, 1, model.num_experts)
        self.model = model
        self.sample = (0, 0, 0)
        self.last_actions = torch.zeros((100, self.num_of_experts))
        self.cf_entropy = nn.CrossEntropyLoss(reduction='none')
        self.reward_function = RewardStrategy(config['reward_function'], model.num_experts,
                                              model.output_shape).get_reward_function()
        logger.info(self.reward_function)
        self.batch_size = config['batch_size']

    def step(self, action):
        reward = self.reward_function(self.sample, action, self.model)
        return self.sample[0], reward, torch.ones(action.shape, dtype=torch.bool), {}

    def reset(self, x=0):
        self.model.eval()
        self.sample = self.train_set.get_random_mini_batch_after_transform(self.batch_size)
        with torch.no_grad():
            obs = self.model.encoder(self.sample[0].to(self.model.device))
        return obs

    def get_reward_for_given_sample(self, sample, action, out=None):
        return self.reward_function(sample[0], action, self.model, out, sample[1])

    def render(self, mode='human'):
        pass

# def run(moe: MoE):
#     experiment = experiments.Cifar10Experiment()
#
#     env = CustomEnv(experiment, moe)
#     # create a vectorized environment
#     vec_env = DummyVecEnv([lambda: env])
#     # define the model
#
#     policy_kwargs = dict(
#         features_extractor_class=CustomPPOModel,
#         features_extractor_kwargs=dict(features_dim=moe.num_experts),
#     )
#
#     model = PPO("CNNPolicy", vec_env, policy_kwargs=policy_kwargs, verbose=1)
#
#     vec_env.reset()
#     moe.eval()
#     obs = moe.encoder(vec_env.sample[0])
#     action, _ = model.predict(obs)
#     vec_env.step(action)
#
#     # train the model
#     model.learn(total_timesteps=1000000)
#     # save the trained model
#     model.save("ppo2_custom_multiarm_bandit_env")
#     print("model saved")
