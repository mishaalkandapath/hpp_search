import re
from typing import List, Tuple

import numpy as np
import torch

from misc.config import ACTION_KEYS, SHAPE_DICT, SHAPES, STATE_KEYS
from misc.evaluation import mk_besideness, mk_ontopness

SHAPE_SHAPE_REL = r"(half_T|mirror_L|vertical|horizontal)_(half_T|mirror_L|vertical|horizontal)_(left|right|above|below)"
SHAPE_START = r"(half_T|mirror_L|vertical|horizontal)_start"


def pyc_to_torch(d: dict, s_indices: List[str], a_indices: List[str] = None):
    a_indices = a_indices or []
    in_len = len(s_indices) + int("reward" in d) * 2
    data_array = torch.zeros(in_len)
    for k, value in d.items():
        if k in ("reward", "action"):
            continue
        data_array[s_indices.index(k)] = value
    if a_indices and "reward" in d:
        data_array[-2] = d["reward"]
        data_array[-1] = a_indices.index(d["action"])
    return data_array


def make_mlp_dict(grid: np.array, target=False, split=False) -> torch.Tensor:
    mlp_input = {}
    grid_shapes = (t := np.unique(grid))[t != 0].tolist()
    name = "target" if target else "input"
    for shape in grid_shapes:
        shape_name = SHAPES[int(shape) - 1]
        rows, cols = np.where(grid == int(shape))
        for i in range(3):
            mlp_input[f"{name}_{shape_name}_row{i + 1}_{int(rows[i]) + 1}"] = 1
            mlp_input[f"{name}_{shape_name}_col{i + 1}_{int(cols[i]) + 1}"] = 1
    tensor = pyc_to_torch(mlp_input, s_indices=STATE_KEYS)
    if not split:
        return tensor
    nonzero = tensor.nonzero().squeeze(-1)
    if nonzero.numel() and nonzero[0] >= 144:
        return tensor[144:]
    return tensor[:144]


def convert_tensor_to_dict(grid_tensor: torch.Tensor):
    return {STATE_KEYS[idx]: 1 for idx in grid_tensor.nonzero().squeeze(-1)}


def make_mlp_input(gridname: str) -> Tuple[np.ndarray, torch.Tensor]:
    grid = np.load(gridname)
    return grid, make_mlp_dict(grid)


def make_grid_after_action(
    goal_grid: np.array, current_grid: np.array, action: str
) -> np.array:
    if "start" in action:
        shape = re.match(SHAPE_START, action).group(1)
    else:
        shape = re.match(SHAPE_SHAPE_REL, action).group(2)
    new_grid = np.zeros_like(goal_grid)
    new_grid += current_grid
    new_grid += goal_grid * (goal_grid == SHAPE_DICT[shape])
    assert np.max(new_grid) <= 4
    return new_grid


def action_is_right(goal_grid: np.array, current_grid: np.array, action: str) -> bool:
    if (match := re.match(SHAPE_START, action)) and not np.count_nonzero(current_grid):
        return np.count_nonzero(goal_grid == SHAPE_DICT[match.group(1)]) != 0
    if re.match(SHAPE_START, action):
        return False

    match = re.match(SHAPE_SHAPE_REL, action)
    shape1, shape2, rel = match.group(1), match.group(2), match.group(3)
    check_grid = goal_grid * (goal_grid == SHAPE_DICT[shape1])
    check_grid += goal_grid * (goal_grid == SHAPE_DICT[shape2])

    if not (
        np.count_nonzero(goal_grid == SHAPE_DICT[shape1])
        and np.count_nonzero(goal_grid == SHAPE_DICT[shape2])
        and np.count_nonzero(current_grid == SHAPE_DICT[shape1])
        and not np.count_nonzero(current_grid == SHAPE_DICT[shape2])
    ):
        return False

    if rel in ("above", "below"):
        ontopness, _, ontop, below = mk_ontopness(check_grid)
        if not ontopness:
            return False
        return (
            SHAPES[ontop.item() - 1] == shape1 and SHAPES[below.item() - 1] == shape2
            if rel == "above"
            else SHAPES[ontop.item() - 1] == shape2
            and SHAPES[below.item() - 1] == shape1
        )

    besideness, _, left, right = mk_besideness(check_grid)
    if not besideness:
        return False
    return (
        SHAPES[left.item() - 1] == shape1 and SHAPES[right.item() - 1] == shape2
        if rel == "left"
        else SHAPES[left.item() - 1] == shape2 and SHAPES[right.item() - 1] == shape1
    )


def make_transitions_for_grid(
    grid: np.array,
    target_grid: np.array,
    grid_tensor: torch.Tensor,
) -> List[torch.Tensor]:
    dataset = []
    grid_shapes = (t := np.unique(grid))[t != 0].tolist()
    target_grid_shapes = (t := np.unique(target_grid))[t != 0].tolist()
    num_steps = len(grid_shapes) - len(target_grid_shapes)
    for action_index, action in enumerate(ACTION_KEYS):
        reward = -1
        new_state = torch.clone(grid_tensor)
        if action_is_right(grid, target_grid, action):
            reward = 1 if num_steps == 1 else -0.1
            new_grid = make_grid_after_action(grid, target_grid, action)
            new_state += make_mlp_dict(new_grid, target=True)
            new_state[new_state > 1] = 1
            if reward != 1:
                dataset.extend(make_transitions_for_grid(grid, new_grid, new_state))
        dataset.append([grid_tensor, action_index, reward, new_state])
    return dataset
