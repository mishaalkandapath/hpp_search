import random
import re

import numpy as np
import torch

from data_prep.batched import STATE_KEYS, action_is_right, make_grid_after_action
from environment.batched import BatchedBrickEnvironment
from misc.config import ACTION_KEYS, SHAPE_DICT
from data_prep.actions import (
    encode_one_hot_pixel,
    get_action_from_factorized,
    get_factorized_action_indices,
)
from misc.evaluation import brick_connectedness
from misc.utils import get_pairwise_relations, get_start_relations


class NewPipelineEnv(BatchedBrickEnvironment):
    def __init__(
        self,
        *args,
        goal_repr="pixel",
        state_repr="asis",
        action_repr="standard",
        **kwargs,
    ):
        """
        Extended environment.
        goal_repr: 'pixel', 'asis', 'clarion'
        state_repr: 'asis', 'clarion'
        action_repr: 'standard', 'factored'
        extra args passed to BatchedBrickEnvironment.
        """
        self.goal_repr_mode = goal_repr
        self.state_repr_mode = state_repr
        self.action_repr_mode = action_repr

        # we maintain goals separately from states here
        self.goal_tensors = []
        super().__init__(*args, **kwargs)

    def _reset(self):
        super()._reset()
        self.goal_tensors = []

    def make_input_tensor(self, current_grid, goal_grid):
        """
        Overriding or separate method to creating inputs for Hypothesizer/Planner.
        Returns separate tensors for Current(S) and Goal(G).
        """
        # Goal Encoding
        if self.goal_repr_mode == "pixel":
            g_tensor = encode_one_hot_pixel(goal_grid).to(self.device)  # (36,)
        elif self.goal_repr_mode == "asis":
            g_tensor = (
                torch.from_numpy(goal_grid.flatten())
                .float()
                .to(self.device)
                .to(torch.float32)
            )
        elif self.goal_repr_mode == "clarion":
            # Reuse batched_data_prep logic
            # g_tensor = make_mlp_dict(goal_grid, target=False, split=True).to(self.device).to(torch.float32)
            g_tensor = (
                make_mlp_dict_optimized(goal_grid).to(self.device).to(torch.float32)
            )
        # State Encoding
        if self.state_repr_mode == "asis":
            s_tensor = (
                torch.from_numpy(current_grid.flatten())
                .float()
                .to(self.device)
                .to(torch.float32)
            )
        elif self.state_repr_mode == "clarion":
            if np.any(current_grid):
                # s_tensor = make_mlp_dict(current_grid, target=True, split=True).to(self.device).to(torch.float32)
                s_tensor = (
                    make_mlp_dict_optimized(current_grid)
                    .to(self.device)
                    .to(torch.float32)
                )
            else:
                s_tensor = torch.zeros((144,), dtype=torch.float32).to(self.device)

        return s_tensor, g_tensor

    def _initialize_single_environment(self, idx: int, goal_grid: np.array):
        """Initialize a single environment at given index"""
        start_grid = np.zeros_like(goal_grid, dtype=np.uint8)
        num_shapes = len(np.unique(goal_grid)[np.unique(goal_grid) != 0])
        max_len = 4 * num_shapes

        state_tensor, goal_tensor = self.make_input_tensor(
            np.zeros_like(goal_grid), goal_grid
        )

        if idx >= len(self.current_grids):
            # Append new environment
            self.current_grids.append(start_grid)
            self.goal_grids_batch.append(goal_grid)
            self.grid_tensors.append(state_tensor)
            self.goal_tensors.append(goal_tensor)
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
            self.grid_tensors[idx] = state_tensor
            self.goal_tensors[idx] = goal_tensor
            self.step_counts[idx] = 0
            self.dones[idx] = 0
            self.max_lens[idx] = max_len
            if self.test:
                self.grid_possibles[idx] = get_start_relations(
                    goal_grid
                ) + get_pairwise_relations(goal_grid)

    def get_current_states(self):
        self.state_tensors = torch.stack(self.grid_tensors)
        self.goal_tensors = torch.stack(self.goal_tensors)

        return self.state_tensors, self.goal_tensors

    def generate_aux_questions(self, indices=None):
        """
        Generates 4 auxiliary questions for each active environment.
        return:
            questions: Tensor (B, 4, InputDim)
            labels: Tensor (B, 4)
        """
        if indices is None:
            indices = range(self.batch_size)

        questions_batch = []
        labels_batch = []

        skip_indices = torch.ones(len(indices), dtype=torch.bool)

        for i in indices:
            goal_grid = self.goal_grids_batch[i]

            shapes_in_grid = [s for s in np.unique(goal_grid) if s != 0]
            if len(shapes_in_grid) < 2:
                dim = 52 if self.action_repr_mode == "standard" else 21
                questions_batch.append(torch.zeros(4, dim, device=self.device))
                labels_batch.append(torch.zeros(4, device=self.device))
                skip_indices[i] = False
                continue

            qs_for_env = []
            ls_for_env = []

            # We want 4 questions.
            all_good = True
            for _ in range(4):
                # Decide if we want Yes (25%) or No (75%)
                want_yes = random.random() < 0.25

                if not want_yes and len(shapes_in_grid) < 4 and random.random() < 0.33:
                    # Let's get an absent question
                    not_in_grid = set(range(1, 5)) - set(shapes_in_grid)
                    A_id = random.choice(list(not_in_grid))
                    B_id = random.choice(shapes_in_grid)
                    A_name = [k for k, v in SHAPE_DICT.items() if v == A_id][0]
                    B_name = [k for k, v in SHAPE_DICT.items() if v == B_id][0]
                    chosen_action_str = f"{B_name}_{A_name}_absent"
                    label = 0.0
                else:
                    # Pick A
                    A_id = random.choice(shapes_in_grid)
                    A_name = [k for k, v in SHAPE_DICT.items() if v == A_id][0]

                    # Others
                    others = [s for s in shapes_in_grid if s != A_id]
                    random.shuffle(others)

                    # Find a B
                    chosen_action_str = None
                    label = 0.0 if not want_yes else 1.0

                    for B_cand in others:
                        mask = (goal_grid == A_id) | (goal_grid == B_cand)
                        filtered_grid = goal_grid * mask
                        _, rels = brick_connectedness(filtered_grid)
                        rels = list(map(int, rels))

                        connected = any(rels)
                        if want_yes and connected:
                            rel = ["left", "above", "right", "below"][rels.index(A_id)]
                            B_name = [k for k, v in SHAPE_DICT.items() if v == B_cand][
                                0
                            ]
                            chosen_action_str = f"{B_name}_{A_name}_{rel}"
                            break
                        elif not want_yes:
                            B_name = [k for k, v in SHAPE_DICT.items() if v == B_cand][
                                0
                            ]
                            if not connected:
                                rel = random.choice(["left", "above", "right", "below"])
                                chosen_action_str = f"{B_name}_{A_name}_{rel}"
                                break
                            else:
                                rel = random.choice(
                                    ["left", "above", "right", "below"][
                                        : rels.index(A_id)
                                    ]
                                    + ["left", "above", "right", "below"][
                                        rels.index(A_id) + 1 :
                                    ]
                                )
                                chosen_action_str = f"{B_name}_{A_name}_{rel}"
                                break
                    if chosen_action_str is None:
                        skip_indices[i] = False
                        questions_batch.append(torch.zeros(4, dim, device=self.device))
                        labels_batch.append(torch.zeros(4, device=self.device))
                        all_good = False
                        break

                # Encode
                # Input: Exactly the action vector.
                # Standard: OneHot(52)
                # Factored: OneHot(4) + OneHot(17) -> 21

                # Check action repr
                if self.action_repr_mode == "standard":
                    q_vec = torch.zeros(52, device=self.device)
                    if chosen_action_str in ACTION_KEYS:
                        a_idx = ACTION_KEYS.index(chosen_action_str)
                        q_vec[a_idx] = 1.0
                else:
                    # Factored
                    q_vec = torch.zeros(4 + 17, device=self.device)
                    if chosen_action_str in ACTION_KEYS:
                        ent_idx, rel_idx = get_factorized_action_indices(
                            chosen_action_str
                        )
                        # ent_idx: 0-3
                        # rel_idx: 0-16
                        q_vec[ent_idx] = 1.0
                        q_vec[4 + rel_idx] = 1.0

                qs_for_env.append(q_vec)
                ls_for_env.append(label)

            if all_good:
                questions_batch.append(torch.stack(qs_for_env))  # (4, 52 or 21)
                labels_batch.append(
                    torch.tensor(ls_for_env, device=self.device)
                )  # (4,)

        return (
            torch.stack(questions_batch),
            torch.stack(labels_batch),
            skip_indices.unsqueeze(-1).expand(-1, 4).to(questions_batch[0].device),
        )

    def step(self, action_indices: list[torch.Tensor] | torch.Tensor):
        """
        action_indices:
            - If standard: Tensor (B,) of indices.
            - If factored: List of [Tensor(B, ent), Tensor(B, rel)].
        Returns:
            next_states (Tensor S), next_goals (Tensor G), rewards (Tensor), dones (Tensor)
        """

        # 1. Map Actions to Strings/Indices
        real_action_indices = []
        bs = self.batch_size

        if self.action_repr_mode == "factored":
            # action_indices is list of tensors
            ent_idxs = action_indices[0].cpu().numpy()
            rel_idxs = action_indices[1].cpu().numpy()

            assert ent_idxs.shape == (self.batch_size,)
            assert rel_idxs.shape == (self.batch_size,)

            #  # Debug Diversity (moved from train.py)
            #  dist_E = Counter(ent_idxs.flatten().tolist())
            #  dist_R = Counter(rel_idxs.flatten().tolist())
            #  total = len(ent_idxs)
            #  dist_E = {k: round(v/total, 2) for k, v in sorted(dist_E.items())}
            #  dist_R = {k: round(v/total, 2) for k, v in sorted(dist_R.items())}
            #  print(f"DEBUG: Diversity E: {dist_E}")
            #  print(f"DEBUG: Diversity R: {dist_R}")

            for i in range(bs):
                act_str = get_action_from_factorized(ent_idxs[i], rel_idxs[i])
                if act_str is None:
                    real_action_indices.append(-1)  # Invalid
                else:
                    real_action_indices.append(ACTION_KEYS.index(act_str))
        else:
            # Standard
            real_action_indices = action_indices[0].cpu().numpy().tolist()

        states = []
        rewards = []
        dones = []

        action_strs_for_debug = []
        for i in range(bs):
            if self.dones[i]:
                states.append(torch.zeros_like(self.grid_tensors[i]))
                rewards.append(0)
                dones.append(1)
                # Update grid tensor for next step
                self.grid_tensors[i] = torch.zeros_like(self.grid_tensors[i])
                continue

            action = real_action_indices[i]
            current_grid = self.current_grids[i]
            goal_grid = self.goal_grids_batch[i]

            # Check if action is right
            reward = -1.0 if action != -1 else -2.5
            done = False
            action_str = ACTION_KEYS[action] if action != -1 else None
            action_strs_for_debug.append(str(action_str))

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
                new_state, _ = self.make_input_tensor(new_grid, goal_grid)

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

        # # # Debug Diversity
        # if action_strs_for_debug:
        #     total = len(action_strs_for_debug)
        #     dist = Counter(action_strs_for_debug)
        #     dist = {k: round(v/total, 2) for k, v in dist.items()}
        #     print(f"DEBUG: Diversity Action Strings: {dist}")

        return (
            torch.stack(states),
            torch.tensor(rewards, device=self.device, dtype=torch.float32),
            torch.tensor(dones, device=self.device, dtype=torch.bool),
        )


# Pre-compute lookup table for decode_clarion optimization
# STATE_KEYS is list of (key_string, value_string)
# We map each index 0..287 to (shape_id, point_idx, axis, value)
CLARION_LOOKUP = np.zeros((288, 4), dtype=np.int8)
for idx, (key, val_str) in enumerate(STATE_KEYS):
    # key like 'input_half_T_row1'
    match = re.match(
        r"^(input|target)_(mirror_L|half_T|vertical|horizontal)_(row|col)(\d+)", key
    )
    shape_str = match.group(2)
    axis_str = match.group(3)  # row or col
    point_idx = int(match.group(4)) - 1  # 0-based

    val = int(val_str)
    shape_id = SHAPE_DICT[shape_str]
    axis = 0 if axis_str == "row" else 1

    CLARION_LOOKUP[idx] = [shape_id, point_idx, axis, val]

# Pre-compute Inverse Lookup for Encoding
# Map: [shape_id-1, point_idx, axis, val-1] -> index (0-143)
# We only map the first 144 (Input) entries. Target entries match these but +144.
# Dimensions: Shapes(4) x Points(3) x Axis(2) x Values(6)
INVERSE_CLARION_LOOKUP = np.full((4, 3, 2, 6), -1, dtype=np.int16)

# Iterate only first 144 (Input keys)
# CLARION_LOOKUP structure: [shape_id, point_idx, axis, val]
# derived from STATE_KEYS 0..143 (which are input_...)
for idx in range(144):
    props = CLARION_LOOKUP[idx]
    s_id = props[0]
    p_idx = props[1]
    ax = props[2]
    val = props[3]

    # Validation
    if s_id > 0:  # Valid entry
        INVERSE_CLARION_LOOKUP[s_id - 1, p_idx, ax, val - 1] = idx


def make_mlp_dict_optimized(grid):
    """
    Vectorized version of make_mlp_dict.
    Returns 144-dim cpu tensor (float).
    Equivalent to make_mlp_dict(grid, target=True/False, split=True)
    because we only return the 144-dim structure (0s and 1s).
    The caller (algorithmic_step) expects a 144-dim tensor.
    """
    tensor = torch.zeros(144, dtype=torch.float32)

    # Iterate over 4 shapes
    # This loop is small constant time (4 iters)
    for s_id in range(1, 5):
        # Find pixels
        rows, cols = np.where(grid == s_id)
        if len(rows) == 0:
            continue

        # Expect 3 pixels for valid shapes
        # If partial shapes exist, we handle them (up to 3)
        # Original code iterated ranges(3).
        limit = min(len(rows), 3)

        for k in range(limit):
            r_val = rows[k] + 1
            c_val = cols[k] + 1

            # Row attr
            idx_r = INVERSE_CLARION_LOOKUP[s_id - 1, k, 0, r_val - 1]
            if idx_r != -1:
                tensor[idx_r] = 1.0

            # Col attr
            idx_c = INVERSE_CLARION_LOOKUP[s_id - 1, k, 1, c_val - 1]
            if idx_c != -1:
                tensor[idx_c] = 1.0

    return tensor


def decode_clarion_optimized(tensor, target=True):
    """
    Optimized version of decode_clarion using pre-computed lookup table and numpy operations.
    Accepts tensor or numpy array.
    """
    if isinstance(tensor, torch.Tensor):
        arr = tensor.cpu().numpy()
    else:
        arr = tensor

    idxs = np.nonzero(arr)[0]

    # Offset for target (matches how keys are ordered in STATE_KEYS vs tensor halves)
    # The tensor passed is usually 144-dim.
    # If target=True, it corresponds to the second half of attributes in the global STATE_KEYS list.
    if target:
        idxs += 144

    entries = CLARION_LOOKUP[idxs]  # (N, 4)

    # Structure: coords[shape_id-1, point_idx, axis]
    # We use a temp array to gather coordinates.
    # Shapes are 1-4, so index 0-3.
    # Initialize with 0.
    coords = np.zeros((4, 3, 2), dtype=np.int8)

    # Vectorized assignment
    # entries columns: 0:shape, 1:point, 2:axis, 3:val
    # We trust that the encoding is valid (no conflicting values for same slot)
    shape_idxs = entries[:, 0] - 1
    point_idxs = entries[:, 1]
    axes = entries[:, 2]
    vals = entries[:, 3]

    coords[shape_idxs, point_idxs, axes] = vals

    clean_grid = np.zeros((6, 6), dtype=np.uint8)

    # Reconstruct grid
    # We only care about shapes that appeared in the entries
    present_shape_idxs = np.unique(shape_idxs)

    for s_idx in present_shape_idxs:
        s_id = s_idx + 1
        for p in range(3):
            r = coords[s_idx, p, 0]
            c = coords[s_idx, p, 1]
            # Valid coordinate check (both row and col must be present > 0)
            if r > 0 and c > 0:
                clean_grid[r - 1, c - 1] = s_id

    return clean_grid


def decode(tensor, mode, idx=None, current=True):
    if mode == "asis":
        if isinstance(tensor, torch.Tensor):
            arr = tensor.cpu().numpy()
        else:
            arr = tensor
        return arr.reshape(6, 6).astype(np.uint8)
    elif mode == "clarion":
        # Use optimized version
        return decode_clarion_optimized(tensor, current)
    else:
        raise NotImplementedError(f"Decode mode {mode} not supported")


def decode_clarion(tensor, target=True):
    nonzero_idxs = np.nonzero(tensor.cpu().numpy())[0]
    prefix = "target" if target else "input"
    clean_grid = np.zeros((6, 6), dtype=np.uint8)

    if target:
        nonzero_idxs += 144

    shapes = {}
    key_pattern = re.compile(
        r"^(input|target)_(mirror_L|half_T|vertical|horizontal)_(row|col)(\d+)"
    )
    for idx in nonzero_idxs:
        key = STATE_KEYS[idx]
        pos_idx = int(key[1])

        match = re.match(key_pattern, key[0])
        assert match
        is_target = match.group(1)
        assert is_target == prefix

        shape = match.group(2)
        if shape not in shapes:
            shapes[shape] = {}

        row_or_col_idx = match.group(3) + match.group(4)
        shapes[shape][row_or_col_idx] = pos_idx

    for shape in shapes:
        for i in range(1, 4):
            row, col = shapes[shape][f"row{i}"], shapes[shape][f"col{i}"]
            assert clean_grid[row - 1][col - 1] == 0, (
                f"Row {row}, col {col} is not empty: current map: {clean_grid}{shapes}"
            )
            clean_grid[row - 1][col - 1] = SHAPE_DICT[shape]

    return clean_grid


def algorithmic_step(
    state_tensors,
    action_inputs,
    state_decode_mode="clarion",
    close_rewards=False,
    original_goal_grids=None,
):
    """
    Functional world model step. Batched.
    state_tensors: (N, S)
    action_inputs: Tensor(N,) OR List[Tensor(N,)] for factored actions
    original_goal_grids: List[np.array] of length N. Required if goal_decode_mode="pixel".

    Returns: next_states (Tensor), rewards (Tensor), dones (Tensor)
    """
    device = state_tensors.device
    N = state_tensors.shape[0]

    next_states_list = []
    rewards_list = []
    dones_list = []

    # Optimization: Batch conversion to numpy
    state_np = state_tensors.cpu().numpy()

    # Process actions to indices
    real_action_indices = []
    # Check if factorized (list of 2 tensors) or standard (tensor or list of 1 tensor)
    if isinstance(action_inputs, list) and len(action_inputs) > 1:
        # Factorized
        ent_idxs = action_inputs[0].cpu().numpy()
        rel_idxs = action_inputs[1].cpu().numpy()
        # Map to strings can also be optimized but might be fast enough
        # For now keeping loop as string mapping is complex
        for i in range(N):
            act_str = get_action_from_factorized(ent_idxs[i], rel_idxs[i])
            if act_str is None:
                real_action_indices.append(-1)
            else:
                try:
                    real_action_indices.append(ACTION_KEYS.index(act_str))
                except ValueError:
                    real_action_indices.append(-1)
    else:
        # Standard
        if isinstance(action_inputs, list):
            action_inputs = action_inputs[0]
        real_action_indices = action_inputs.cpu().numpy().tolist()

    # Iterate over batch
    for i in range(N):
        # Pass numpy array to decode
        current_grid = decode(state_np[i], state_decode_mode, i)

        goal_grid = original_goal_grids[i]

        action_idx = real_action_indices[i]

        action_str = ACTION_KEYS[action_idx] if action_idx != -1 else None

        reward = -1.0 if action_str is not None else -2.5
        done = False
        new_grid = current_grid.copy()
        if action_str is not None and action_is_right(
            goal_grid, current_grid, action_str
        ):
            current_grid_shapes = (t := np.unique(current_grid))[t != 0].tolist()
            goal_grid_shapes = (t := np.unique(goal_grid))[t != 0].tolist()
            num_steps = len(goal_grid_shapes) - len(current_grid_shapes)

            if num_steps == 1:
                reward = 1.0
                done = True
            elif not close_rewards:
                reward = -0.1
            else:
                reward = 1.0 - (num_steps - 1) / len(goal_grid_shapes)

            new_grid = make_grid_after_action(goal_grid, current_grid, action_str)

        # Re-encode state
        if state_decode_mode == "asis":
            new_state = torch.from_numpy(new_grid.flatten()).float()  # Stay on CPU
        elif state_decode_mode == "clarion":
            if np.any(new_grid):
                # ns = make_mlp_dict(new_grid, target=True, split=True)
                # new_state = ns.float() # Stay on CPU (make_mlp_dict output is CPU)
                # Use vectorized version
                new_state = make_mlp_dict_optimized(new_grid)
            else:
                new_state = torch.zeros((144,), dtype=torch.float32)  # Default CPU

        next_states_list.append(new_state)
        rewards_list.append(reward)
        dones_list.append(done)

    # Batch transfer to device
    next_states = torch.stack(next_states_list).to(device)
    rewards = torch.tensor(rewards_list, device=device, dtype=torch.float32)
    dones = torch.tensor(dones_list, device=device, dtype=torch.bool)

    return next_states, rewards, dones
