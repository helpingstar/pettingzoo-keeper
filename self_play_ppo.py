# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from pettingzoo.utils import AECEnv, ParallelEnv

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ParallelPPO:
    def __init__(self, env: ParallelEnv, agent_0: nn.Module, agent_1: nn.Module) -> None:
        self.envs = env
        self.agent_0 = agent_0
        self.agent_1 = agent_1
        # self.exp_name: str = os.path.basename(file)[: -len(".py")]
        # """the name of this experiment"""
        self.seed: int = 1
        """seed of the experiment"""
        self.torch_deterministic: bool = True
        """if toggled, torch.backends.cudnn.deterministic=False"""
        self.cuda: bool = True
        """if toggled, cuda will be enabled by default"""
        # self.track: bool = False
        # """if toggled, this experiment will be tracked with Weights and Biases"""
        # self.wandb_project_name: str = "cleanRL"
        # """the wandb's project name"""
        # self.wandb_entity: str = None
        # """the entity (team) of wandb's project"""
        # self.capture_video: bool = False
        # """whether to capture videos of the agent performances (check out videos folder)"""

        # Algorithm specific arguments
        self.env_id: str = "CartPole-v1"
        """the id of the environment"""
        self.total_timesteps: int = 500000
        """total timesteps of the experiments"""
        self.learning_rate: float = 2.5e-4
        """the learning rate of the optimizer"""
        self.num_envs: int = 4
        """the number of parallel game environments"""
        self.num_steps: int = 128
        """the number of steps to run in each environment per policy rollout"""
        self.anneal_lr: bool = True
        """Toggle learning rate annealing for policy and value networks"""
        self.gamma: float = 0.99
        """the discount factor gamma"""
        self.gae_lambda: float = 0.95
        """the lambda for the general advantage estimation"""
        self.num_minibatches: int = 4
        """the number of mini-batches"""
        self.update_epochs: int = 4
        """the K epochs to update the policy"""
        self.norm_adv: bool = True
        """Toggles advantages normalization"""
        self.clip_coef: float = 0.2
        """the surrogate clipping coefficient"""
        self.clip_vloss: bool = True
        """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
        self.ent_coef: float = 0.01
        """coefficient of the entropy"""
        self.vf_coef: float = 0.5
        """coefficient of the value function"""
        self.max_grad_norm: float = 0.5
        """the maximum norm for the gradient clipping"""
        self.target_kl: float = None
        """the target KL divergence threshold"""

        # to be filled in runtime
        self.batch_size: int = 0
        """the batch size (computed in runtime)"""
        self.minibatch_size: int = 0
        """the mini-batch size (computed in runtime)"""
        self.num_iterations: int = 0
        """the number of iterations (computed in runtime)"""

    def train(self):
        self.batch_size = int(self.num_envs * self.num_steps)
        self.minibatch_size = int(self.batch_size // self.num_minibatches)
        self.num_iterations = self.total_timesteps // self.batch_size

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = self.torch_deterministic

        device = torch.device("cuda" if torch.cuda.is_available() and self.cuda else "cpu")

        optimizer_0 = optim.Adam(self.agent_0.parameters(), lr=self.learning_rate, eps=1e-5)
        optimizer_1 = optim.Adam(self.agent_1.parameters(), lr=self.learning_rate, eps=1e-5)

        obs = torch.zeros((self.num_steps, self.num_envs) + self.envs.single_observation_space.shape).to(device)
        actions_0 = torch.zeros((self.num_steps, self.num_envs) + self.envs.single_action_space.shape).to(device)
        actions_1 = torch.zeros((self.num_steps, self.num_envs) + self.envs.single_action_space.shape).to(device)
        logprobs_0 = torch.zeros((self.num_steps, self.num_envs)).to(device)
        logprobs_1 = torch.zeros((self.num_steps, self.num_envs)).to(device)
        rewards_0 = torch.zeros((self.num_steps, self.num_envs)).to(device)
        rewards_1 = torch.zeros((self.num_steps, self.num_envs)).to(device)
        dones = torch.zeros((self.num_steps, self.num_envs)).to(device)
        values_0 = torch.zeros((self.num_steps, self.num_envs)).to(device)
        values_1 = torch.zeros((self.num_steps, self.num_envs)).to(device)
        
        global_step = 0
        start_time = time.time()
        next_obs, _ = self.envs.reset(seed=self.seed)
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(self.num_envs).to(device)

        for iteration in range(1, self.num_iterations + 1):
            # Annealing the rate if instructed to do so.
            if self.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.num_iterations
                lrnow = frac * self.learning_rate
                optimizer_0.param_groups[0]["lr"] = lrnow
                optimizer_1.param_groups[0]["lr"] = lrnow
            for step in range(0, self.num_steps):
                global_step += self.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # AlGO LOGIC: action logic
                with torch.no_grad():
                    action_0, logprob_0, _, value_0 = self.agent_0.get_action_and_value(next_obs)
                    action_1, logprob_1, _, value_1 = self.agent_1.get_action_and_value(next_obs)
                    values_0[step] = value_0.flatten()
                    values_1[step] = value_1.flatten()
                actions_0[step] = action_0
                actions_1[step] = action_1
                logprobs_0[step] = logprob_0
                logprobs_1[step] = logprob_1

                # TRY NOT TO MODIFY: execute the game and log data.
                
                
                
                    
                    