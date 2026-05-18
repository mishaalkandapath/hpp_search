import torch
import torch.nn.functional as F

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
    diff = z.unsqueeze(2) - z_hist.unsqueeze(1)          # (B,M,H,D)
    dist2 = (diff * diff).sum(dim=-1)                    # (B,M,H)
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
            assert h.dim() == 2, f"Expected dims to be 2 but got shape {h.shape}"    # (B,H)
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