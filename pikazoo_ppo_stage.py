from pikazoo import pikazoo_v0
from pikazoo.wrappers import (
    RewardInNormalState,
    RewardByBallPosition,
    RecordEpisodeStatistics,
)
import os
import numpy as np
import yaml
import torch
import random
import torch.optim as optim
import time
import torch.nn as nn
from pettingzoo.utils import ParallelEnv
import wandb
import config
import supersuit as ss


class PikaZooPPOStage:
    def __init__(self, agent_type, reward=None) -> None:
        assert (agent_type == "nml" and reward is None) or (
            agent_type != "nml" and reward is not None
        )
        self.agent_type = agent_type
        self.reward = reward
        self.get_env_list()

        with open("ppo.yaml") as f:
            args = yaml.load(f, Loader=yaml.FullLoader)
        # `* 2`: left (player_1) and right (player_2)
        args["total_timesteps"] = args["total_timesteps"] // 2

        args["batch_size"] = int(args["num_envs"] * args["num_steps"])
        args["minibatch_size"] = int(args["batch_size"] // args["num_minibatches"])
        args["num_iterations"] = args["total_timesteps"] // args["batch_size"]
        for key, value in args.items():
            setattr(self, key, value)

    def get_env_list(self):
        assert self.agent_type in config.agent["agent_type"]

        self.env_list = []

        if self.agent_type == "nml":
            env1 = pikazoo_v0.env(**config.env)
        elif self.agent_type == "nal":
            env1 = pikazoo_v0.env(**config.env)
            env1 = RewardInNormalState(env1, self.reward)
        elif self.agent_type == "phf":
            env1 = pikazoo_v0.env(**config.env)
            env1 = RewardByBallPosition(env1, {1, 4}, self.reward, False)
            env2 = pikazoo_v0.env(**config.env)
            env2 = RewardByBallPosition(env2, {1, 4}, self.reward, True)
        elif self.agent_type == "pqd":
            env1 = pikazoo_v0.env(**config.env)
            env1 = RewardByBallPosition(env1, {4}, self.reward, False)
            env2 = pikazoo_v0.env(**config.env)
            env2 = RewardByBallPosition(env2, {4}, self.reward, True)
        else:  # pqu, "pos_on_quarter_up"
            env1 = pikazoo_v0.env(**config.env)
            env1 = RewardByBallPosition(env1, {1}, self.reward, False)
            env2 = pikazoo_v0.env(**config.env)
            env2 = RewardByBallPosition(env2, {1}, self.reward, True)

        env1 = RecordEpisodeStatistics(env1)
        env1 = ss.dtype_v0(env1, np.float32)
        env1 = ss.normalize_obs_v0(env1)
        self.env_list.append(env1)

        if self.agent_type not in ["nml", "nal"]:
            env2 = RecordEpisodeStatistics(env2)
            env2 = ss.dtype_v0(env2, np.float32)
            env2 = ss.normalize_obs_v0(env2)
            self.env_list.append(env2)

        self.n_env = len(self.env_list)

    def train(self, agent_train, agent_no_train, run_name):
        for is_right, side in enumerate(["l", "r"]):  # is_right: 0=left, 1=right
            if self.n_env == 1:
                env = self.env_list[0]
            else:
                env = self.env_list[is_right]
            self.train_one_side(
                env, agent_train, agent_no_train, is_right, f"{run_name}_{side}"
            )

    def train_one_side(
        self, env: ParallelEnv, agent_train, agent_no_train, is_right, run_name
    ):
        assert is_right in (0, 1)
        # TRY NOT TO MODIFY: seeding
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

        # agent_train = agent_train.to(device)
        # agent_no_train = agent_no_train.to(device)

        optimizer = optim.Adam(
            agent_train.parameters(), lr=self.learning_rate, eps=1e-5
        )

        # ALGO Logic: Storage setup
        obs = torch.zeros(
            (self.num_steps, self.num_envs) + env.observation_space(player_train).shape
        ).to(device)
        actions = torch.zeros(
            (self.num_steps, self.num_envs) + env.action_space(player_train).shape
        ).to(device)
        logprobs = torch.zeros((self.num_steps, self.num_envs)).to(device)
        rewards = torch.zeros((self.num_steps, self.num_envs)).to(device)
        dones = torch.zeros((self.num_steps, self.num_envs)).to(device)
        values = torch.zeros((self.num_steps, self.num_envs)).to(device)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_obs, _ = env.reset(seed=self.seed)
        # TODO : check
        next_obs_train, next_obs_no_train = (
            next_obs[player_train],
            next_obs[player_no_train],
        )
        next_obs_train = torch.Tensor(next_obs_train).to(device)
        next_obs_no_train = torch.Tensor(next_obs_no_train).to(device)
        next_done = torch.zeros(self.num_envs).to(device)

        for iteration in range(1, self.num_iterations + 1):
            # Annealing the rate if instructed to do so.
            if self.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.num_iterations
                lrnow = frac * self.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.num_steps):
                global_step += self.num_envs
                obs[step] = next_obs_train
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent_train.get_action_and_value(
                        next_obs_train
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
                    w_log = {
                        f"charts/score/{run_name}": infos[player_train]["score"][
                            self.is_right
                        ]
                        - infos[player_no_train]["score"][self.is_right ^ 1],
                        f"charts/cumulative_reward/{run_name}": infos[player_train][
                            "episode"
                        ]["r"],
                        f"charts/episode_length/{run_name}": infos[player_train][
                            "episode"
                        ]["l"],
                        f"charts/frame_per_round/{run_name}": infos[player_train][
                            "episode"
                        ]["l"]
                        // sum(infos[player_train]["score"]),
                        "global_step": global_step,
                    }
                    wandb.log(w_log)
                    next_obs, infos = env.reset()
                next_obs_train = next_obs[player_train]
                reward = reward[player_train]
                terminations = terminations[player_train]
                truncations = truncations[player_train]
                infos = infos[player_train]

                next_done = np.array(any([terminations, truncations]))
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs_train = torch.Tensor(next_obs_train).to(device)
                next_done = torch.Tensor(next_done).to(device)
            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent_train.get_value(next_obs_train).reshape(1, -1)
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
            b_obs = obs.reshape((-1,) + env.observation_space(player_train).shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + env.action_space(player_train).shape)
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
            log = {
                f"charts/learning_rate/{run_name}": optimizer.param_groups[0]["lr"],
                f"losses/value_loss/{run_name}": v_loss.item(),
                f"losses/policy_loss/{run_name}": pg_loss.item(),
                f"losses/entropy/{run_name}": entropy_loss.item(),
                f"losses/old_approx_kl/{run_name}": old_approx_kl.item(),
                f"losses/approx_kl/{run_name}": approx_kl.item(),
                f"losses/clipfrac/{run_name}": np.mean(clipfracs),
                f"losses/explained_variance/{run_name}": explained_var,
                f"charts/SPS/{run_name}": int(global_step / (time.time() - start_time)),
                f"global_step": global_step,
            }
            wandb.log(log)

        env.close()
