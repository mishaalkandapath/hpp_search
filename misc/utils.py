import os
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F

from misc.config import SHAPES
from misc.evaluation import mk_besideness, mk_ontopness


def categorical_entropy_from_logits(logits, dim=-1):
    p = F.softmax(logits, dim=dim)
    logp = F.log_softmax(logits, dim=dim)
    return -(p * logp).sum(dim=dim)


def bin_centers(num_bins: int, device):
    edges = torch.linspace(0.0, 1.0, num_bins + 1, device=device)
    return 0.5 * (edges[:-1] + edges[1:])  # (K,)


def continuous_to_bin(x01, num_bins: int):
    # x01: (B,) in [0,1]
    idx = torch.clamp((x01 * num_bins).long(), 0, num_bins - 1)
    return idx


def split_policy_logits(policy_logits, action_dims):
    # policy_logits: (B, M, sum(action_dims))
    return torch.split(policy_logits, action_dims, dim=-1)


def entropy_sum_over_factors(policy_logits, action_dims):
    parts = split_policy_logits(policy_logits, action_dims)
    ent = 0.0
    for lg in parts:
        ent = ent + categorical_entropy_from_logits(lg, dim=-1)  # (B,M)
    return ent  # (B,M)


def gaussian_density(z, z_hist, sigma):
    # z: (B,M,D), z_hist: (B,H,D) or None
    if z_hist is None:
        return torch.zeros(z.shape[0], z.shape[1], device=z.device)
    diff = z.unsqueeze(2) - z_hist.unsqueeze(1)  # (B,M,H,D)
    dist2 = (diff * diff).sum(dim=-1)  # (B,M,H)
    return torch.exp(-dist2 / (2 * sigma * sigma)).sum(dim=-1)  # (B,M)


def update_history(z_hist, z_new, max_hist):
    # store detached to prevent backprop through history
    z_new = z_new.detach()
    if z_hist is None:
        z_hist = z_new
    else:
        z_hist = torch.cat([z_hist, z_new], dim=1)
        if z_hist.size(1) > max_hist:
            z_hist = z_hist[:, -max_hist:, :]
    return z_hist


def gather_hidden(hidden, idx):
    """
    hidden: list of tensors or list of None, one per RNN layer.
    idx: (B_active,) LongTensor
    """
    if hidden is None:
        return None
    new_hidden = []
    for h in hidden:
        if h is None:
            new_hidden.append(None)
        else:
            assert h.dim() == 2, (
                f"Expected dims to be 2 but got shape {h.shape}"
            )  # (B,H)
            new_hidden.append(h.index_select(0, idx))
    return new_hidden


def shrink_batch(idx, **tensors):
    """
    idx: (B_active,) LongTensor selecting rows along dim=0.
    tensors: any number of batch-first tensors
    returns: dict of shrunk tensors
    """
    out = {}
    for k, v in tensors.items():
        if v is None:
            out[k] = None
        else:
            out[k] = v.index_select(0, idx)
    return out


def get_start_relations(grid):
    all_grids = np.unique(grid[grid != 0])
    return [f"{SHAPES[grid_no - 1]}_start" for grid_no in all_grids]


def get_pairwise_relations(grid):
    all_grids = np.unique(grid[grid != 0])
    all_rels = []

    def get_shape(x):
        return SHAPES[x - 1]

    for grid1_no, grid2_no in combinations(all_grids, 2):
        form_grid = grid * ((grid == grid1_no) | (grid == grid2_no))
        ontopness, count_ontop, ontop, below = mk_ontopness(form_grid)
        besideness, count_beside, left, right = mk_besideness(form_grid)

        assert (
            (not ontopness and not besideness)
            or (ontopness and not besideness)
            or (besideness and not ontopness)
        )
        assert count_ontop <= 1
        assert count_beside <= 1

        if ontopness:
            all_rels.append(f"{get_shape(ontop)}_{get_shape(below)}_above")
            all_rels.append(f"{get_shape(below)}_{get_shape(ontop)}_below")
        elif besideness:
            all_rels.append(f"{get_shape(left)}_{get_shape(right)}_left")
            all_rels.append(f"{get_shape(right)}_{get_shape(left)}_right")

    return all_rels


def get_grids_by_number(grid_names, base_dir, start_from=1, end_at=5):
    grids_by_size = {1: [], 2: [], 3: [], 4: []}
    for grid_name in grid_names:
        if grid_name[:2] == "._":
            continue
        stim_grid = np.load(os.path.join(base_dir, grid_name))
        grids_by_size[len(np.unique(stim_grid)) - 1].append(grid_name)
    for size in range(start_from, end_at):
        yield grids_by_size[size]


def load_from_dirname(data_dir):
    g_n = get_grids_by_number(os.listdir(data_dir), data_dir, start_from=1, end_at=5)
    # unroll generator:
    data = []
    for grid_names in g_n:
        if grid_names:
            data.append(grid_names)
    return data
