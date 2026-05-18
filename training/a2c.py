import copy
import json
import os
import signal
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import product
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from misc.utils import get_grids_by_number
from training.models import DQN

interrupted = False
train_obj_global = None
run_name_global = None


def siguser_handler(signum, frame):
    global train_obj_global
    if train_obj_global.env.cur_phase < 4:
        train_obj_global.env.cur_phase += 1
        print(
            "RECEIVED SIGUSER1 - UPDATED CUR PHASE TO ", train_obj_global.env.cur_phase
        )
    else:
        print("RECIEVED SIGUSER1, DID NOT UPDATE BECAUSE CUR PHASE IS 4")


@dataclass
class EpisodeData:
    states_seq: List[torch.Tensor]
    actions_seq: List[torch.Tensor]
    rewards_seq: List[torch.Tensor]
    policy_logits_seq: List[torch.Tensor]
    values_tensor: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    mask: torch.Tensor
    episode_lengths: torch.Tensor
    best_lengths: torch.Tensor
    total_rewards: torch.Tensor


class ActorCriticTrainer:
    """
    Trainer class implementing advantage actor-critic algorithm for batched brick environment.
    Handles variable-length episodes with proper masking.
    """

    def __init__(
        self,
        model,
        env,
        val_env,
        test_env=None,
        conv_deets=None,
        feed_state=False,
        lr=7e-4,
        lr_init=0,
        gamma=0.9,
        beta_entropy=0.005,
        beta_critic=0.05,
        epsilon=0.2,
        wd=0.0,
        rl_method="a2c",
        logger=None,
        device=None,
        p_mode="asis",
    ):
        """
        Args:
            model: RNN model that outputs both policy logits and value estimates
            env: BatchedBrickEnvironment instance
            lr: Learning rate for actor (policy) parameters
            lr_init: Learning rate for initial states
            gamma: Discount factor
            beta_entropy: Entropy regularization coefficient
            beta_critic: Critic loss coefficient
            logger: Optional logger
            device: Device to run on
        """

        self.model = model
        self.env = env
        self.val_env = val_env
        self.test_env = test_env
        self.feed_state = feed_state
        self.gamma = gamma
        self.epsilon = epsilon
        self.beta_entropy = beta_entropy
        self.beta_critic = beta_critic

        params_to_optimize = []

        if not isinstance(model, DQN) and model.learn_init:
            params_to_optimize.extend(
                [
                    {
                        "params": [
                            p
                            for name, p in model.named_parameters()
                            if "initial_states" not in name
                        ],
                        "lr": lr,
                    },
                    {"params": model.initial_states.parameters(), "lr": lr_init},
                ]
            )

        else:
            params_to_optimize.append({"params": model.parameters(), "lr": lr})

        if conv_deets is not None:
            conv_ins, conv_targs = conv_deets
            params_to_optimize.extend(
                [
                    {"params": conv_ins.parameters(), "lr": lr},
                    {"params": conv_targs.parameters(), "lr": lr},
                ]
            )

        self.optimizer = (
            optim.Adam(params_to_optimize, eps=1e-7)
            if wd == 0
            else optim.AdamW(params_to_optimize, eps=1e-7, weight_decay=wd)
        )
        self.logger = logger
        self.p_mode = p_mode

        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        match rl_method:
            case "ppo":
                self.loss_fn = self.ppo_actor_loss
                self.ads_fn = self.compute_advantages_masked
            case "a2c":
                self.loss_fn = self.a2c_actor_loss
                self.ads_fn = self.compute_advantages_masked

        self.model.to(self.device)

    def save(self, path):
        os.makedirs(path[: path.rfind("/")], exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def get_model_outputs_batched(self, input_tensor, hidden_state, old_model=False):
        """
        Get policy logits and value estimate from model for batched input.
        Assumes model outputs (batch, seq, features) where features == num_actions + 1.
        """
        if type(self.model) is DQN:
            outputs = (
                self.model(input_tensor)
                if not old_model
                else self.old_model(input_tensor)
            )
            new_hidden = None
        else:
            outputs, new_hidden = self.model(input_tensor, hidden_state)

        # Assuming single step output: (batch, 1, features)
        policy_logits = outputs[:, 0, :-1]  # (batch, num_actions)
        value_estimate = outputs[:, 0, -1]  # (batch,)

        return policy_logits, value_estimate, new_hidden

    def select_action_batched(self, policy_logits, inference=False):
        """
        Select actions for entire batch based on policy logits.

        Args:
            policy_logits: (batch, num_actions) tensor of logits
            inference: If True, use greedy selection; if False, sample
        """
        if not inference:
            action_probs = F.softmax(policy_logits, dim=-1)
            return torch.multinomial(action_probs, 1).squeeze(-1)  # (batch,)
        else:
            return policy_logits.argmax(-1)  # (batch,)

    def compute_advantages_masked(self, rewards, values, dones_mask):
        """
        Compute discounted returns for variable-length episodes with masking.

        Args:
            rewards: (batch, max_seq_len) tensor of rewards
            values: (batch, max_seq_len) tensor of value estimates
            dones_mask: (batch, max_seq_len) tensor indicating valid steps
        """
        batch_size, seq_len = rewards.shape
        ads = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        # For each batch item, work backwards from its actual end
        for b in range(batch_size):
            # Find the last valid step for this batch item
            valid_steps = dones_mask[b].sum().item()
            if valid_steps == 0:
                continue

            R = 0.0  # No bootstrap for terminal states

            # Work backwards through valid steps only
            for t in reversed(range(valid_steps)):
                R = rewards[b, t] + self.gamma * R
                returns[b, t] = R
                ads[b, t] = R - values[b, t]

        return ads, returns

    def a2c_actor_loss(self, log_probs, old_log_probs, advantages):
        return -(log_probs * advantages)

    def compute_advantage_loss_batched(self, episode_data: EpisodeData):
        """
        Compute batched actor-critic loss components with masking.

        Args:
        EpisodeData object containing:
            policy_logits_seq: List of (batch, num_actions) tensors, one per timestep
            actions_seq: List of (batch,) tensors, one per timestep
            returns: (batch, seq_len) tensor of returns
            values: (batch, seq_len) tensor of value estimates
            advantages: (batch, seq_len) tensor of advantages
            mask: (batch, seq_len) tensor indicating valid steps
        """
        batch_size, seq_len, _ = episode_data.policy_logits_seq.shape

        # Collect log probs and entropies for all valid steps
        log_probs = torch.zeros(batch_size, seq_len, device=self.device)
        old_log_probs = torch.zeros(batch_size, seq_len, device=self.device)
        entropies = torch.zeros(batch_size, seq_len, device=self.device)

        for t in range(seq_len):
            logits = episode_data.policy_logits_seq[:, t]  # (batch, num_actions)
            actions = episode_data.actions_seq[:, t]  # (batch,)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs[:, t] = dist.log_prob(actions)
            entropies[:, t] = dist.entropy()

        # Apply mask to only include valid steps
        masked_probs = log_probs * episode_data.mask
        masked_old_probs = old_log_probs * episode_data.mask
        masked_entropies = entropies * episode_data.mask
        masked_advantages = episode_data.advantages * episode_data.mask

        valid_steps = episode_data.mask.sum()

        if valid_steps > 0:
            actor_loss = (
                self.loss_fn(
                    masked_probs, masked_old_probs.detach(), masked_advantages.detach()
                ).sum()
                / valid_steps
            )
            entropy_bonus = masked_entropies.sum() / valid_steps
            actor_loss = actor_loss - self.beta_entropy * entropy_bonus

            # Critic loss: only on valid steps
            masked_returns = episode_data.returns * episode_data.mask
            masked_values = episode_data.values * episode_data.mask
            critic_loss = (masked_returns - masked_values).pow(2).sum() / valid_steps
        else:
            actor_loss = torch.tensor(0.0, device=self.device)
            critic_loss = torch.tensor(0.0, device=self.device)

        return (
            actor_loss + self.beta_critic * critic_loss,
            {
                "actor_loss": actor_loss.item(),
                "critic_loss": critic_loss.item(),
            },
            masked_advantages.detach(),
        )

    def run_episode_batch(self, inference=False, test=False):
        """
        Run a full batch of episodes to completion.

        Returns:
            episode_data: Dictionary containing all episode information
        """
        env = self.env if not inference else self.val_env
        env = env if not test else self.test_env
        batch_size = env.batch_size

        # Initialize episode storage
        states_seq = []  # List of (batch, state_dim) tensors
        actions_seq = []  # List of (batch,) tensors
        rewards_seq = []  # List of (batch,) tensors
        policy_logits_seq = []  # List of (batch, num_actions) tensors
        values_seq = []  # List of (batch,) tensors

        hidden_state = None
        states = env.get_current_states().to(self.device)  # (batch, state_dim)

        # Track which episodes are still running
        max_possible_steps = max(env.max_lens)
        best_lens = np.array(env.max_lens) // 4
        active_episodes = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        mask = torch.ones(
            batch_size, max_possible_steps, dtype=torch.bool, device=self.device
        )

        step_count = 0
        while active_episodes.any() and step_count < max_possible_steps:
            if not states_seq and self.feed_state:
                model_input = torch.concat(
                    [states, torch.zeros(batch_size, 2).to(states.device)], dim=-1
                )
            elif self.feed_state:
                model_input = torch.concat(
                    [
                        states,
                        rewards_seq[-1].unsqueeze(-1),
                        actions_seq[-1].unsqueeze(-1),
                    ],
                    dim=-1,
                )
            else:
                model_input = states

            model_input = model_input.unsqueeze(1)  # (batch, 1, state_dim+2) for RNN
            policy_logits, values, new_hidden = self.get_model_outputs_batched(
                model_input, hidden_state
            )

            actions = self.select_action_batched(policy_logits, inference=inference)
            new_states, rewards, dones = env.step(actions)

            dones = dones.to(self.device)
            states_seq.append(states.to(self.device))
            actions_seq.append(actions.to(self.device))
            rewards_seq.append(rewards.to(self.device))
            policy_logits_seq.append(policy_logits)
            values_seq.append(values.to(self.device))

            states = new_states.to(self.device)
            hidden_state = new_hidden

            # Update active episodes
            mask[:, step_count] = active_episodes.clone()
            active_episodes = torch.logical_and(
                active_episodes, torch.logical_not(dones)
            )  # TODO: bitwise invert?
            step_count += 1

        seq_len = len(states_seq)
        mask = mask[:, :seq_len]
        # Stack sequences
        rewards_tensor = torch.stack(rewards_seq, dim=1)  # (batch, seq_len)
        values_tensor = torch.stack(values_seq, dim=1)  # (batch, seq_len)
        actions_seq = torch.stack(actions_seq, dim=1)
        policy_logits_seq = torch.stack(policy_logits_seq, dim=1)

        # Compute returns
        advantages, returns = self.ads_fn(rewards_tensor, values_tensor, mask)

        return EpisodeData(
            states_seq=states_seq,
            actions_seq=actions_seq,
            rewards_seq=rewards_seq,
            policy_logits_seq=policy_logits_seq,
            values_tensor=values_tensor,
            advantages=advantages,
            returns=returns,
            mask=mask,
            episode_lengths=mask.sum(dim=1)
            / torch.from_numpy(best_lens).to(mask.device),
            best_lengths=best_lens,
            total_rewards=(rewards_tensor * mask).sum(dim=1),
        )

    def train_step(self):
        """
        Run one training step: episode batch to completion + backpropagation.
        """
        self.model.train()
        episode_data = self.run_episode_batch(inference=False)

        total_loss, losses, advantages = self.compute_advantage_loss_batched(
            episode_data
        )
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Reinitialize environments for next batch
        self.env._initialize_all_environments()

        return {
            "total_loss": total_loss.item(),
            "train_mean_episode_length": episode_data.episode_lengths.float()
            .mean()
            .item(),
            "train_mean_episode_reward": episode_data.total_rewards.float()
            .mean()
            .item(),
            "advantage_mean": advantages.mean().item(),
            "advantage_std": advantages.std().item() ** losses,
        }

    def train(self, run_name):
        """
        Train for specified number of steps.
        """
        best_train_loss = float("inf")
        best_test_acc = -float("inf")
        best_val_acc = -float("inf")
        best_val_grid_correctness = -float("inf")
        window_unsat = 10  # atleast 10 in a sequence great performacnce
        step = 0
        pbar = tqdm(total=500)
        while best_val_acc < 0.99 or best_val_grid_correctness < 0.99 or window_unsat:
            stats = self.train_step()
            if stats["actor_loss"] + stats["critic_loss"] < best_train_loss:
                best_train_loss = stats["actor_loss"] + stats["critic_loss"]
                self.save(f"data/run_data/{run_name}/train/best_train_goal_net.pt")
                if self.p_mode == "conv":
                    self.save(
                        self.env.conv_layer_ins.state_dict(),
                        f"data/run_data/{run_name}/best_train_goal_net_conv_in.pt",
                    )
                    self.save(
                        self.env.conv_layer_targs.state_dict(),
                        f"data/run_data/{run_name}/best_train_goal_net_conv_targs.pt",
                    )

            if step % 500 == 0:
                self.env.last_few_performances.append(
                    stats["train_mean_episode_reward"]
                )
                eval_stats = self.evaluate()
                eval_stats_test = self.evaluate(48, test=True)

                acc = eval_stats["accuracy"]
                correctness = eval_stats["grid_correctness"]
                # if correctness > best_test_grid_correctness:
                if acc > best_val_acc:
                    self.save(f"data/run_data/{run_name}/val/best_val_goal_net.pt")
                    if self.p_mode == "conv":
                        self.save(
                            self.val_env.conv_layer_ins.state_dict(),
                            f"data/run_data/{run_name}/best_val_goal_net_conv_in.pt",
                        )
                        self.save(
                            self.val_env.conv_layer_targs.state_dict(),
                            f"data/run_data/{run_name}/best_val_goal_net_conv_targs.pt",
                        )
                    best_val_acc = acc
                    best_val_grid_correctness = correctness
                    if best_val_acc >= 0.99 and best_val_grid_correctness >= 0.99:
                        window_unsat -= 1
                else:
                    window_unsat = 10

                if eval_stats_test["accuracy_test"] > best_test_acc:
                    self.save(f"data/run_data/{run_name}/test/best_test_goal_net.pt")
                    if self.p_mode == "conv":
                        self.save(
                            self.val_env.conv_layer_ins.state_dict(),
                            f"data/run_data/{run_name}/best_test_goal_net_conv_in.pt",
                        )
                        self.save(
                            self.val_env.conv_layer_targs.state_dict(),
                            f"data/run_data/{run_name}/best_test_goal_net_conv_targs.pt",
                        )
                    best_test_acc = eval_stats_test["accuracy_test"]

                stats = eval_stats | stats | eval_stats_test
                self.logger.log(stats)
                pbar.reset()
            step += 1
            pbar.update(1)

    def evaluate(self, num_episodes=128, test=False):
        """
        Evaluate model performance over multiple episodes.
        """
        self.model.eval()

        assert not test or (test and self.test_env)
        env = self.val_env if not test else self.test_env

        total_rewards = []
        episode_lengths = []
        correctness = 0
        correctness_denom = 0
        grid_correctness = 0

        aux_loss = 0
        aux_denom = 0

        with torch.no_grad():
            for _ in range(max(num_episodes // env.batch_size, 1)):
                episode_data = self.run_episode_batch(inference=True, test=test)
                total_rewards.extend(episode_data.total_rewards.cpu().numpy())
                episode_lengths.extend(episode_data.episode_lengths.cpu().numpy())
                if isinstance(episode_data.rewards_seq, list):
                    episode_data.rewards_seq = torch.stack(
                        episode_data.rewards_seq, dim=-1
                    )
                rews = episode_data.rewards_seq
                mask = episode_data.mask
                correctness += torch.sum(torch.logical_and((rews != -1), (rews != 0)))
                correctness_denom += torch.sum(mask)
                if t := torch.count_nonzero(rews == 1):
                    grid_correctness += t

                if hasattr(episode_data, "aux_loss"):
                    aux_loss += episode_data.aux_loss.item()
                    aux_denom += 1

                # Reinitialize for next evaluation batch
                env._initialize_all_environments()

        return {
            f"mean_reward{'' if not test else '_test'}": np.mean(total_rewards),
            f"mean_length{'' if not test else '_test'}": np.mean(episode_lengths),
            f"std_length{'' if not test else '_test'}": np.std(episode_lengths),
            f"accuracy{'' if not test else '_test'}": correctness / correctness_denom,
            f"grid_correctness{'' if not test else '_test'}": grid_correctness
            / max(num_episodes, env.batch_size),
        } | (
            {f"{'test_' if test else 'val_'}aux_loss": aux_loss / aux_denom}
            if aux_denom > 0
            else {}
        )
        # | ({"incorrectness_tally": incorrectness_tally} if test else {})


def load_test_data(
    test_data_dir="/w/150/lambda_squad/misc/clarion_replay/data/processed/regular/test_data/test_stims",
):
    files = os.listdir(test_data_dir)
    return [files]


def load_from_dirname(data_dir):
    g_n = get_grids_by_number(os.listdir(data_dir), data_dir, start_from=1, end_at=5)
    # unroll generator:
    data = []
    for grid_names in g_n:
        if grid_names:
            data.append(grid_names)
    return data
