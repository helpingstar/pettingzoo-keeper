# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass
from gymnasium.experimental.wrappers import RecordVideoV0
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from network import Agent
from pettingzoo.utils import ParallelEnv
from typing import Dict, Any
import yaml
from network import Agent
from pikazoo import pikazoo_v0
import supersuit as ss
from tqdm import tqdm


class OnePlayerPPO:
    def __init__(self, args) -> None:

        #
        args["batch_size"] = int(args["num_envs"] * args["num_steps"])
        args["minibatch_size"] = int(args["batch_size"] // args["num_minibatches"])
        args["num_iterations"] = args["total_timesteps"] // args["batch_size"]

        for key, value in args.items():
            setattr(self, key, value)

    def train(
        self,
        env: ParallelEnv,
        agent_train: Agent,
        agent_no_train: Agent,
        is_right: int,
        run_name: str,
    ):
        assert is_right in (0, 1)
        if self.track:
            import wandb

            wandb.init(
                project=self.wandb_project_name,
                entity=self.wandb_entity,
                config=vars(self),
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )
        # TRY NOT TO MODIFY: seeding
        self.env = env
        self.is_right = is_right
        player_train = env.possible_agents[self.is_right]
        player_no_train = env.possible_agents[self.is_right ^ 1]
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = self.torch_deterministic

        device = torch.device(
            "cuda" if torch.cuda.is_available() and self.cuda else "cpu"
        )

        agent_train = agent_train.to(device)
        agent_no_train = agent_no_train.to(device)

        optimizer = optim.Adam(
            agent_train.parameters(), lr=self.learning_rate, eps=1e-5
        )

        # ALGO Logic: Storage setup
        obs = torch.zeros(
            (self.num_steps, self.num_envs) + env.observation_space("player_1").shape
        ).to(device)
        actions = torch.zeros(
            (self.num_steps, self.num_envs) + env.action_space("player_1").shape
        ).to(device)
        logprobs = torch.zeros((self.num_steps, self.num_envs)).to(device)
        rewards = torch.zeros((self.num_steps, self.num_envs)).to(device)
        dones = torch.zeros((self.num_steps, self.num_envs)).to(device)
        values = torch.zeros((self.num_steps, self.num_envs)).to(device)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_obs, _ = env.reset(seed=self.seed)
        next_obs_train, next_obs_no_train = next_obs.values()
        next_obs = torch.Tensor(next_obs_train).to(device)
        next_obs_no_train = torch.Tensor(next_obs_no_train).to(device)
        next_done = torch.zeros(self.num_envs).to(device)

        for iteration in tqdm(range(1, self.num_iterations + 1)):
            # Annealing the rate if instructed to do so.
            if self.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.num_iterations
                lrnow = frac * self.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.num_steps):
                global_step += self.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent_train.get_action_and_value(
                        next_obs
                    )
                    values[step] = value.flatten()
                    action_no_train, _, _, _ = agent_no_train.get_action_and_value(
                        next_obs_no_train
                    )
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = env.step(
                    {
                        player_train: action.cpu().numpy(),
                        player_no_train: action_no_train.cpu().numpy(),
                    }
                )
                if terminations[player_train] or truncations[player_train]:
                    wandb.log(
                        {
                            "charts/score": infos[player_train]["score"] - infos[player_no_train]["score"]
                        }
                    )
                    next_obs, infos = env.reset()
                next_obs = next_obs[player_train]
                reward = reward[player_train]
                terminations = terminations[player_train]
                truncations = truncations[player_train]
                infos = infos[player_train]

                next_done = np.array(any([terminations, truncations]))
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                    next_done
                ).to(device)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent_train.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = (
                        rewards[t]
                        + self.gamma * nextvalues * nextnonterminal
                        - values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + env.observation_space("player_1").shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + env.action_space("player_1").shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(self.batch_size)
            clipfracs = []
            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent_train.get_action_and_value(
                        b_obs[mb_inds], b_actions.long()[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                        ]

                    mb_advantages = b_advantages[mb_inds]
                    if self.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - self.clip_coef, 1 + self.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        agent_train.parameters(), self.max_grad_norm
                    )
                    optimizer.step()

                if self.target_kl is not None and approx_kl > self.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            wandb.log(
                {"charts/learning_rate": optimizer.param_groups[0]["lr"]},
                step=global_step,
            )
            wandb.log({"losses/value_loss": v_loss.item()}, step=global_step)
            wandb.log({"losses/policy_loss": pg_loss.item()}, step=global_step)
            wandb.log({"losses/entropy": entropy_loss.item()}, step=global_step)
            wandb.log({"losses/old_approx_kl": old_approx_kl.item()}, step=global_step)
            wandb.log({"losses/approx_kl": approx_kl.item()}, step=global_step)
            wandb.log({"losses/clipfrac": np.mean(clipfracs)}, step=global_step)
            wandb.log({"losses/explained_variance": explained_var}, step=global_step)
            wandb.log(
                {"charts/SPS": int(global_step / (time.time() - start_time))},
                step=global_step,
            )

        env.close()

if __name__ == "__main__":
    env = pikazoo_v0.env(render_mode=None)
    env = ss.dtype_v0(env, np.float32)
    env = ss.normalize_obs_v0(env)
    # env = RecordVideoV0(env, video_folder='.', episode_trigger=lambda x: (x % 50 == 0), disable_logger=True)
    agent_train = Agent()
    agent_no_train = Agent()
    with open("ppo.yaml") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    agent_train = Agent()
    agent_no_train = Agent()

    learner = OnePlayerPPO(args)

    learner.train(env, agent_train, agent_no_train, is_right=0, run_name="pikazoo_test_2")