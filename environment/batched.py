import os
import random
from collections import Counter, deque
from copy import deepcopy
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
from torch.nn import Module

from data_prep.batched import (
    ACTION_KEYS,
    action_is_right,
    make_grid_after_action,
    make_mlp_dict,
)
from misc.utils import get_pairwise_relations, get_start_relations


class ExhaustiveSampler:
    def __init__(self, grids):
        self.grids = deepcopy(grids)
        self.no_grids = len(grids)
        self.current_grids = list(range(len(grids)))
        self.generator = np.random.Generator(np.random.PCG64())
        self.generator.shuffle(self.current_grids)

    def sample(self, sample_size):
        if sample_size > len(self.current_grids):
            ret_indices = self.current_grids
            self.current_grids = list(range(self.no_grids))
            self.generator.shuffle(self.current_grids)
            return [self.grids[i] for i in ret_indices] + self.sample(
                sample_size - len(ret_indices)
            )
        else:
            ret_indices = [self.current_grids.pop() for _ in range(sample_size)]
            return [self.grids[i] for i in ret_indices]

    def __len__(self):
        return self.no_grids


class BatchedBrickEnvironment:
    def __init__(
        self,
        file_paths: List[List[str]],
        batch_size: int,
        base_dir: str,
        inf=False,
        device=torch.device("cpu" if not torch.cuda.is_available() else "cuda:0"),
        conv: Optional[Tuple[Module, Module]] = None,
        p_mode: Literal["conv", "asis", "clarion"] = "asis",
        curriculum=False,
        close_rewards=False,
        test=False,  # track some additional metrics
        manual_init=False,
    ):
        self.file_paths = file_paths
        self.batch_size = batch_size

        self.goal_grids = []
        for file_type in file_paths:
            self.goal_grids.append([])
            for file_path in file_type:
                goal_grid = np.load(os.path.join(base_dir, file_path))
                self.goal_grids[-1].append(goal_grid)
        self.goal_grids = [ExhaustiveSampler(grids) for grids in self.goal_grids]
        # Initialize batch environments
        self.grid_possibles = []  # list of possible actions for each sampled grid
        self.current_grids = []  # List of current grid states (numpy arrays)
        self.goal_grids_batch = []  # List of target grids for current batch
        self.grid_tensors = []  # List of current states in tensor format
        self.step_counts = []  # Track steps for each environment
        self.dones = []
        self.max_lens = []  # Max length for each environment (4 * num_shapes)
        self.inf = inf
        self.device = device
        self.p_mode = p_mode
        self.close_rewards = close_rewards
        if conv is not None:
            self.conv_layer_ins, self.conv_layer_targs = conv

        self.last_few_performances = deque(maxlen=100)
        self.cur_phase = 1 if curriculum else 4
        self.curriculum = curriculum

        self.test = test
        self.manual_init = manual_init
        # Initialize all environments
        if not self.manual_init:
            self._initialize_all_environments()

    def _reset(self):
        self.current_grids = []
        self.goal_grids_batch = []
        self.grid_tensors = []
        self.step_counts = []
        self.dones = []
        self.max_lens = []

    def get_brick_weights(self):
        if (
            np.mean(self.last_few_performances) > 0.78
            and self.cur_phase < 4
            and len(self.last_few_performances) == 100
        ):
            self.last_few_performances.clear()
            self.cur_phase += 1
            print(f"PHASE UPDATE TO: {self.cur_phase}")

        match self.cur_phase:
            case 1:
                weights = [0.7, 0.2, 0.1, 0.0]
            case 2:
                weights = [0.25, 0.3, 0.4, 0.05]
            case 3:
                weights = [0.2, 0.2, 0.3, 0.3]
            case 4:
                weights = [0.1, 0.14, 0.38, 0.38]
            case _:
                raise ValueError(f"Incorrect model phase value {self.phase}")
        return weights

    def make_input_tensor(self, current_grid, goal_grid):
        if self.p_mode == "conv":
            current_grid = (
                torch.from_numpy(current_grid.copy())
                .to(self.device)
                .to(dtype=torch.float32)
                .unsqueeze(0)
            )
            goal_grid = (
                torch.from_numpy(goal_grid.copy())
                .to(self.device)
                .to(dtype=torch.float32)
                .unsqueeze(0)
            )

            input_tensor = self.conv_layer_ins(current_grid).flatten()
            target_tensor = self.conv_layer_targs(goal_grid).flatten()
            return torch.concat([target_tensor, input_tensor])
        elif self.p_mode == "clarion":
            target_tensor = make_mlp_dict(goal_grid, target=False)
            input_tensor = make_mlp_dict(current_grid, target=True)
            return (target_tensor + input_tensor >= 1).to(dtype=torch.float32)
        elif self.p_mode == "asis":
            return torch.from_numpy(
                np.concatenate((goal_grid.flatten(), current_grid.flatten()))
            ).to(dtype=torch.float32)

    def _initialize_all_environments(self):
        """Initialize all batch environments with random targets"""
        self._reset()
        if not hasattr(self, "manual_init") or not self.manual_init:
            goal_types = Counter(
                random.choices(
                    range(len(self.goal_grids)),
                    weights=(
                        self.get_brick_weights()
                        if (len(self.goal_grids) == 4)
                        else [1 / 4] * len(self.goal_grids)
                    ),
                    k=self.batch_size,
                )
            )
            goal_grids = []
            for i in range(len(self.goal_grids)):
                goal_grids.extend(self.goal_grids[i].sample(goal_types.get(i, 0)))

            for i in range(self.batch_size):
                self._initialize_single_environment(i, goal_grids[i])

    def initialize_custom_grid(self, grid):
        """Initialize environment with a specific grid"""
        self._reset()
        self._initialize_single_environment(0, grid)

    def _initialize_single_environment(self, idx: int, goal_grid: np.array):
        """Initialize a single environment at given index"""
        start_grid = np.zeros_like(goal_grid, dtype=np.uint8)
        num_shapes = len(np.unique(goal_grid)[np.unique(goal_grid) != 0])
        max_len = 4 * num_shapes

        grid_tensor = self.make_input_tensor(
            np.zeros_like(goal_grid), goal_grid
        )  # current state - contains goal state

        if idx >= len(self.current_grids):
            # Append new environment
            self.current_grids.append(start_grid)
            self.goal_grids_batch.append(goal_grid)
            self.grid_tensors.append(grid_tensor)
            self.step_counts.append(0)
            self.dones.append(0)
            self.max_lens.append(max_len)
            if self.test:
                self.grid_possibles.append(
                    get_start_relations(goal_grid) + get_pairwise_relations(goal_grid)
                )
        else:
            # Replace existing environment
            self.current_grids[idx] = start_grid
            self.goal_grids_batch[idx] = goal_grid
            self.grid_tensors[idx] = grid_tensor
            self.step_counts[idx] = 0
            self.dones[idx] = 0
            self.max_lens[idx] = max_len
            if self.test:
                self.grid_possibles[idx] = get_start_relations(
                    goal_grid
                ) + get_pairwise_relations(goal_grid)

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Take a step in all environments

        Args:
            actions: [batch_size] tensor of action indices

        Returns:
            states: [batch_size, state_dim] tensor of new states
            rewards: [batch_size] tensor of rewards
            dones: [batch_size] tensor of done flags
            infos: List of info dicts for each environment
        """
        states = []
        rewards = []
        dones = []

        for i in range(self.batch_size):
            if self.dones[i]:
                states.append(torch.zeros_like(self.grid_tensors[i]))
                rewards.append(0)
                dones.append(1)
                # Update grid tensor for next step
                self.grid_tensors[i] = torch.zeros_like(self.grid_tensors[i])
                continue

            action = actions[i].item()
            current_grid = self.current_grids[i]
            goal_grid = self.goal_grids_batch[i]

            # Check if action is right
            reward = -1
            done = False
            action_str = ACTION_KEYS[action] if action != -1 else None
            # print(current_grid, goal_grid, action_str)
            if action_str is not None and action_is_right(
                goal_grid, current_grid, action_str
            ):
                # Calculate number of steps needed
                current_grid_shapes = (t := np.unique(current_grid))[t != 0].tolist()
                goal_grid_shapes = (t := np.unique(goal_grid))[t != 0].tolist()
                num_steps = len(goal_grid_shapes) - len(current_grid_shapes)

                if num_steps == 1:
                    reward = 1
                    done = True
                    self.dones[i] = 1
                elif not self.close_rewards:
                    reward = -0.1
                else:
                    reward = 1 - (num_steps - 1) / len(goal_grid_shapes)

                # Apply action and get new grid
                new_grid = make_grid_after_action(goal_grid, current_grid, action_str)
                new_state = self.make_input_tensor(new_grid, goal_grid)

                # Update environment state
                self.current_grids[i] = new_grid
            else:
                # Wrong action, state doesn't change
                new_state = self.grid_tensors[i].clone()

            # Check if max length reached
            self.step_counts[i] += 1
            if self.step_counts[i] >= self.max_lens[i]:
                done = True
                self.dones[i] = 1

            # Store results
            states.append(new_state)
            rewards.append(reward)
            dones.append(done)
            # Update grid tensor for next step
            self.grid_tensors[i] = new_state

        return (
            torch.stack(states),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.bool),
        )

    def replace_done_episodes(self, dones: torch.Tensor):
        """
        Replace environments that are done with new random samples

        Args:
            dones: [batch_size] tensor of done flags
        """
        for i in range(self.batch_size):
            if dones[i]:
                goal_grid = self.goal_grid[
                    random.choices(range(len(self.goal_grids)))
                ].sample(1)
                self._initialize_single_environment(i, goal_grid)

    def get_current_states(self) -> torch.Tensor:
        """Get current states for all environments"""
        return torch.stack(self.grid_tensors)

    def get_batch_info(self, i) -> dict:
        """Get info for all environments in batch"""
        return {
            "step_count": self.step_counts[i],
            "max_len": self.max_lens[i],
            "num_shapes": len(
                np.unique(self.goal_grids_batch[i])[
                    np.unique(self.goal_grids_batch[i]) != 0
                ]
            ),
            "current_grid_shape": self.current_grids[i].shape,
            "goal_grid_shape": self.goal_grids_batch[i].shape,
            "allowed_actions": self.grid_possibles[i],
        }

    def __len__(self):
        return sum(map(lambda x: len(x), self.goal_grids))
