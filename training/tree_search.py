import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from tqdm import tqdm

from environment.pipeline import NewPipelineEnv, algorithmic_step, decode
from misc.utils import (
    bin_centers,
    categorical_entropy_from_logits,
    continuous_to_bin,
    entropy_sum_over_factors,
    gather_hidden,
    gaussian_density,
    shrink_batch,
    update_history,
)
from training.a2c import load_from_dirname
from training.models import TreeSearcher, TreeSearcherConfig, WorldModel, WorldModelConfig

train_obj_global = None
run_name_global = None

def signal_handler(signum, frame):
    global train_obj_global, run_name_global
    print("\n\nReceived interrupt signal (Ctrl+C). Saving model and exiting gracefully...")
    
    if train_obj_global is not None and run_name_global is not None:
        try:
            save_dir = Path(f"data/run_data/{run_name_global}/interrupted")
            train_obj_global.save(save_dir, prefix="interrupted")
            print(f"Model saved to: {save_dir.parent}")
        except Exception as e:
            print(f"Error saving model: {e}")
            import traceback
            traceback.print_exc()
    
    print("Exiting...")
    sys.exit(0)

def sigusr2_handler(signum, frame):
    global train_obj_global, run_name_global
    print("\n\nReceived SIGUSR2. Saving model to 'user_asked' directory...")
    
    if train_obj_global is not None and run_name_global is not None:
        try:
            target_dir = Path(f"data/run_data/{run_name_global}/user_asked")
            train_obj_global.save(target_dir, prefix="user_asked")
            print(f"Model saved to: {target_dir.parent}")
        except Exception as e:
            print(f"Error saving model: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Trainer object or run name not available. Cannot save.")

@dataclass
class LossWeights:
    ce: float = 1.0
    ent: float = 0.01
    continue_pen: float = 0.01
    ig_mono: float = 0.1
    value: float = 1.0
    revisit: float = 0.1
    ub_elim: float = 0.5
    cov: float = 0.1
    parent_div: float = 0.1

class IGStopper:
    def __init__(self, eps_ig=1e-3, eps_entropy=0.05, patience=2):
        self.eps_ig = eps_ig
        self.eps_entropy = eps_entropy
        self.patience = patience
        self.prev_probs = None     # (B,K)
        self.counter = None        # (B,)

    def step(self, belief_logits):
        probs = F.softmax(belief_logits, dim=-1)  # (B,K)
        H = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1)  # (B,)

        if self.prev_probs is None:
            IG = torch.zeros_like(H)
            self.counter = torch.zeros_like(H, dtype=torch.long)
            return torch.zeros_like(H, dtype=torch.bool), {"entropy": H.detach(), "ig": IG.detach()}

        prev_H = -(self.prev_probs * self.prev_probs.clamp_min(1e-9).log()).sum(dim=-1)
        IG = prev_H - H
        small = IG < self.eps_ig
        self.counter = torch.where(small, self.counter + 1, torch.zeros_like(self.counter))

        self.prev_probs = probs.detach()
        stop = (H < self.eps_entropy) | (self.counter >= self.patience)
        return stop, {"entropy": H.detach(), "ig": IG.detach()}

    def shrink(self, idx):
        # idx: (B_keep,) local indices
        new = IGStopper(self.eps_ig, self.eps_entropy, self.patience)
        if self.prev_probs is not None:
            new.prev_probs = self.prev_probs.index_select(0, idx)
        if self.counter is not None:
            new.counter = self.counter.index_select(0, idx)
        return new

class SearchLossState:
    def __init__(self, B, M, device):
        self.prev_probs = None       # (B,K)
        self.z_hist = None           # (B,H,D)
        self.depths = torch.zeros(B, M, device=device)  # per-slot depth
        self.best_ub = torch.ones(B, device=device)     # best upper bound in [0,1]

    def shrink(self, idx):
        new = SearchLossState(idx.numel(), self.depths.size(1), device=self.depths.device)
        if self.prev_probs is not None:
            new.prev_probs = self.prev_probs.index_select(0, idx)
        if self.z_hist is not None:
            new.z_hist = self.z_hist.index_select(0, idx)
        new.depths = self.depths.index_select(0, idx)
        new.best_ub = self.best_ub.index_select(0, idx)
        return new
    

class TreeSearchTrainer:
    def __init__(self, 
                 model: TreeSearcher,
                 env,
                 val_env,
                 test_env,
                 config, # dict or dataclass with training params (lr, etc)
                 wm_config: WorldModelConfig,
                 logger = None,
                 weights: LossWeights = LossWeights(),
                 gamma=0.99,
                 eps_entropy=0.05,
                 eps_ig=1e-3,
                 ig_patience=2,
                 revisit_sigma=1.0,
                 hist_max=512
                 ):
        self.model = model
        self.env = env
        self.val_env = val_env
        self.test_env = test_env
        self.config = config
        self.logger = logger
        self.weights = weights
        self.gamma = gamma
        self.eps_entropy = eps_entropy
        self.eps_ig = eps_ig
        self.ig_patience = ig_patience
        self.revisit_sigma = revisit_sigma
        self.hist_max = hist_max
        self.device = next(model.parameters()).device
        
        self.internal_step = 0

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=config.lr, 
            weight_decay=config.wd
        )

        # World Model
        self.world_model = WorldModel(
            state_dim=wm_config.state_dim,
            action_dims=wm_config.action_dims,
            action_embedding_dim=wm_config.action_embedding_dim,
            goal_dim=wm_config.goal_dim,
            hidden_size=wm_config.hidden_size,
            norm=wm_config.norm,
            dropout=wm_config.dropout
        ).to(self.device)
        
        self.wm_optimizer = optim.Adam(
            self.world_model.parameters(), 
            lr=wm_config.wm_lr, 
            weight_decay=wm_config.wm_weight_decay
        )
        
        self.wm_buffer = []

    def save(self, path, prefix="best"):
        if isinstance(path, str):
            path = Path(path)
        base = path.parent
        os.makedirs(base, exist_ok=True)
        
        torch.save(self.model.state_dict(), base / f"{prefix}_model.pt")
        torch.save(self.optimizer.state_dict(), base / f"{prefix}_opt.pt")
        torch.save(self.world_model.state_dict(), base / f"{prefix}_wm.pt")
        torch.save(self.wm_optimizer.state_dict(), base / f"{prefix}_wm_opt.pt")

    def collect_wm_buffer(self, out, goals):
        """
        Collect transitions for World Model training from inference output.
        out: dict from inference_step
        goals: (B, G)
        """
        B = goals.size(0)
        M = self.model.M
        
        # Flatten everything 
        wm_s = out["states_selected"].reshape(-1, self.model.state_dim)
        wm_a = out["policy_one_hot"].reshape(-1, self.model.total_action_dim)
        wm_ns = out["states"].reshape(-1, self.model.state_dim)
        wm_r = out["rewards"].reshape(-1)
        
        # Goal needs to be repeated M times then flattened
        wm_g = goals.unsqueeze(1).expand(B, M, -1).reshape(-1, self.model.goal_dim)
        
        self.wm_buffer.append((wm_s.detach(), wm_a.detach(), wm_g.detach(), wm_ns.detach(), wm_r.detach()))
        
    def train_wm_offline(self, batch_size=64):
        if not self.wm_buffer:
            return 0.0
        
        def cat_buffer(idx):
             return torch.cat([t[idx] for t in self.wm_buffer], dim=0)

        all_states = cat_buffer(0)
        all_actions = cat_buffer(1)
        all_goals = cat_buffer(2)
        all_next_states = cat_buffer(3)
        all_rewards = cat_buffer(4)
        
        total_samples = all_states.size(0)
        indices = torch.randperm(total_samples)
        
        total_loss = 0
        num_batches = 0
        
        # Simple training loop over buffer
        self.world_model.train()
        for start_idx in range(0, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)
            batch_indices = indices[start_idx:end_idx]
            
            b_states = all_states[batch_indices]
            b_actions = all_actions[batch_indices]
            b_goals = all_goals[batch_indices]
            b_next = all_next_states[batch_indices]
            b_rewards = all_rewards[batch_indices]

            pred_next, pred_rewards = self.world_model(b_states, b_actions, b_goals)
            loss = F.mse_loss(pred_next, b_next) + F.mse_loss(pred_rewards, b_rewards)
            
            self.wm_optimizer.zero_grad()
            loss.backward()
            self.wm_optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            
        # Clear buffer
        self.wm_buffer = []
        return total_loss / max(num_batches, 1)

    def evaluate(self, env, num_steps, T=12):
        self.model.eval()
        total_val_loss = 0.0
        total_norm_stop_steps = 0.0
        total_success_count = 0.0
        count = 0
        
        with torch.no_grad():
            for _ in range(num_steps):
                data = env.get_current_states()
                start_states, goals = data
                start_states = start_states.to(self.device)
                goals = goals.to(self.device)
                
                original_goal_grids = list(env.goal_grids_batch)
                
                # Calculate goal lengths
                # Handle possible float/int differences in unique
                # np.unique returns sorted unique elements
                goal_lengths = torch.tensor([
                    len(np.unique(g)[np.unique(g) != 0]) for g in original_goal_grids
                ], device=self.device, dtype=torch.float32)

                B = goals.size(0)
                M = self.model.M

                # init frontier
                states = start_states.unsqueeze(1).expand(B, M, self.model.state_dim).contiguous()

                prev_actions = torch.zeros(B, M, self.model.total_action_dim, device=self.device)
                prev_rewards = torch.zeros(B, M, device=self.device)
                prev_values  = torch.zeros(B, M, device=self.device)
                hidden = None

                loss_state = SearchLossState(B, M, device=self.device)
                stopper = IGStopper(eps_ig=self.eps_ig, eps_entropy=self.eps_entropy, patience=self.ig_patience)
                
                step_loss = 0.0
                active_idx = torch.arange(B, device=self.device)
                
                stop_steps = torch.zeros(B, device=self.device)
                is_stopped = torch.zeros(B, dtype=torch.bool, device=self.device)

                for t in range(T):
                    if not goals.size(0):
                        break

                    out = self.inference_step(
                        states, goals, prev_actions, prev_rewards, prev_values, hidden, original_goal_grids
                    )
                    
                    # Track success
                    # next_states_flat is implicitly in out["states"]
                    next_states = out["states"]
                    B_curr = next_states.size(0)
                    next_states_flat = next_states.reshape(B_curr * M, self.model.state_dim)
                    expanded_goal_grids = [g for g in original_goal_grids for _ in range(M)]
                    dists = self.compute_env_lens(next_states_flat, expanded_goal_grids).view(B_curr, M)
                    reached = (dists == 0.0).any(dim=1)
                    total_success_count += reached.sum().item()
                    
                    L_t, _ = self.compute_losses(
                        out, goals, start_states, original_goal_grids, prev_values, prev_rewards, loss_state, first=(t==0)
                    )
                    step_loss += L_t
                    
                    states = out["states"]
                    prev_actions = out["policy_one_hot"]
                    prev_rewards = out["rewards"]
                    prev_values = out["values"]
                    hidden = out["hidden"]
                    
                    stop_mask, _ = stopper.step(out["beliefs"])
                    
                    # Track stopping
                    newly_stopped_mask = stop_mask & (~is_stopped[active_idx])
                    stop_steps[active_idx[newly_stopped_mask]] = t + 1
                    is_stopped[active_idx[stop_mask]] = True
                    
                    if stop_mask.all():
                        break
                    
                    keep_idx = torch.nonzero(~stop_mask, as_tuple=False).squeeze(1)
                    active_idx = active_idx.index_select(0, keep_idx)
                    
                    shrunk = shrink_batch(keep_idx, start_states=start_states, goals=goals, states=states, prev_actions=prev_actions, prev_rewards=prev_rewards, prev_values=prev_values)
                    start_states, goals, states, prev_actions, prev_rewards, prev_values = (shrunk[k] for k in ["start_states", "goals", "states", "prev_actions", "prev_rewards", "prev_values"])
                    
                    hidden = gather_hidden(hidden, keep_idx)
                    loss_state = loss_state.shrink(keep_idx)
                    stopper = stopper.shrink(keep_idx)
                    original_goal_grids = [original_goal_grids[i] for i in keep_idx.cpu().numpy()]

                total_val_loss += step_loss.item()
                
                # Metrics
                stop_steps[~is_stopped] = T
                norm_stop_steps = stop_steps / goal_lengths
                total_norm_stop_steps += norm_stop_steps.mean().item()
                
                count += 1
                env._initialize_all_environments()

        return {
            "loss": total_val_loss / max(count, 1),
            "norm_stop_steps": total_norm_stop_steps / max(count, 1),
            "success_count": total_success_count / max(count, 1) # avg success per batch
        }

    def compute_env_lens(self, states_flat, goal_grids_list):
        # states_flat: (N, S)
        # goal_grids_list: List of np.array, length N
        
        # We need to decode states to grids first
        # Assuming states are clarion or asis, use decode function
        # This is essentially calculating distance to goal
        
        dists = []
        states_np = states_flat.cpu().numpy()
        
        for i in range(len(goal_grids_list)):
            current_grid = decode(states_np[i], self.env.state_repr_mode, i)
            goal_grid = goal_grids_list[i]
            
            current_grid_shapes = np.unique(current_grid)
            current_grid_shapes = current_grid_shapes[current_grid_shapes != 0]
            
            goal_grid_shapes = np.unique(goal_grid)
            goal_grid_shapes = goal_grid_shapes[goal_grid_shapes != 0]
            
            num_steps = len(goal_grid_shapes) - len(current_grid_shapes)
            # Normalized distance (max len is 4)
            d = float(num_steps) / 4.0
            dists.append(d)
            
        return torch.tensor(dists, device=self.device, dtype=torch.float32).clamp(0.0, 1.0)

    def inference_step(
        self,
        states,          # (B,M,S)
        goals,           # (B,G)
        prev_actions,    # (B,M,A)
        prev_rewards,    # (B,M)
        prev_values,     # (B,M)
        hidden,          # list or None
        original_goal_grids # list of np.arrays (B items)
    ):
        rnn_out, new_hiddens, beliefs = self.model(
            prev_actions,
            prev_rewards,
            prev_values,
            states,
            goals,
            hidden
        )  # beliefs: (B, K) logits

        # Sample frontier + action
        (
            policy_one_hot,   # (B,M,A) one-hot over concatenated action space
            policy_indices,   # list of (B,M) int tensors, one per factor
            policy_logits,    # (B,M,A) logits over concatenated action space
            values,           # (B,M)
            parent_one_hot,   # (B,M,M)
            frontier_logits,  # (B,M,M)
        ) = self.model.sample_action(rnn_out, states, goals)

        # Select parent states: (B,M,M) x (B,M,S) -> (B,M,S)
        states_selected = torch.bmm(parent_one_hot, states)

        B, M, S = states_selected.shape
        states_flat = states_selected.reshape(B * M, S)
        
        # policy_indices is a list of tensors (factorized)
        # flatten them
        flat_indices = [x.reshape(B * M) for x in policy_indices]
        
        # Expand original_goal_grids for M hypotheses
        # original_goal_grids is list of length B
        expanded_goal_grids = [g for g in original_goal_grids for _ in range(M)]

        if len(flat_indices) == 1:
            alg_input = flat_indices[0]
        else:
            alg_input = flat_indices

        next_states_flat, rewards_flat, dones_flat = algorithmic_step(
            states_flat.detach(),
            alg_input,
            state_decode_mode=self.env.state_repr_mode,
            original_goal_grids=expanded_goal_grids,
            close_rewards=self.env.close_rewards
        )

        next_states = next_states_flat.view(B, M, S)
        rewards = rewards_flat.view(B, M)

        return {
            "states_prev": states,
            "states_selected": states_selected,
            "states": next_states,
            "rewards": rewards,
            "values": values,
            "beliefs": beliefs,               # logits (B,K)
            "policy_logits": policy_logits,   # (B,M,sumA)
            "frontier_logits": frontier_logits,  # (B,M,M)
            "parent_one_hot": parent_one_hot, # (B,M,M)
            "policy_one_hot": policy_one_hot, # (B,M,sumA)
            "policy_indices": policy_indices,
            "hidden": new_hiddens,
        }

    def compute_losses(
        self,
        out,                 # dict from inference_step
        goals,               # (B,G)
        start_states,        # (B,S) empty grids (for env.lens CE anchor)
        original_goal_grids, # list of np arrays
        prev_values_undetached, 
        prev_rewards,
        loss_state: SearchLossState,
        first=False
    ):
        device = goals.device
        B = goals.size(0)
        M = self.model.M
        K = self.model.config.num_belief_bins

        belief_logits = out["beliefs"]          # (B,K)
        belief_probs = F.softmax(belief_logits, dim=-1)
        H = categorical_entropy_from_logits(belief_logits, dim=-1)  # (B,)

        # --- (1) CE anchor using env.lens ---
        # env.lens(start_states, goals) -> (B,) normalized [0,1]
        # We need env.lens equivalent.
        d_star = self.compute_env_lens(start_states, original_goal_grids).detach()
        y = continuous_to_bin(d_star, K)
        L_ce = F.cross_entropy(belief_logits, y)

        # --- (2) IG monotonicity penalty ---
        if loss_state.prev_probs is None:
            L_ig_mono = torch.tensor(0.0, device=device)
            prev_H = None
        else:
            prev_H = -(loss_state.prev_probs * loss_state.prev_probs.clamp_min(1e-9).log()).sum(dim=-1)
            L_ig_mono = F.relu(H - prev_H).mean()

        loss_state.prev_probs = belief_probs.detach()

        # --- (3) Continue-search penalty ---
        L_continue = H.mean()

        # --- (4) Entropy regularization: frontier + per-factor action entropy ---
        frontier_logits = out["frontier_logits"]    # (B,M,M)
        policy_logits = out["policy_logits"]        # (B,M,sumA)
        
        ent_frontier = categorical_entropy_from_logits(frontier_logits, dim=-1)  # (B,M)
        ent_policy = entropy_sum_over_factors(policy_logits, self.model.config.action_dims)  # (B,M)
        # Encourage exploration early
        L_ent = -(ent_frontier.mean() + ent_policy.mean())

        # --- (5) Value TD loss ---
        values = out["values"]          # (B,M)
        if not first:
            target = prev_rewards + self.gamma * values.detach()
            L_value = F.mse_loss(prev_values_undetached, target)
        else:
            L_value = torch.tensor(0.0, device=device)

        # --- (6) Revisit-over-time density loss ---
        # Use embeddings of the *new* frontier states (out["states"]) to penalize revisits
        z_next = self.model.state_encoder(out["states"])  # (B,M,D)
        dens = gaussian_density(z_next, loss_state.z_hist, sigma=self.revisit_sigma)  # (B,M)
        L_revisit = dens.mean()
        loss_state.z_hist = update_history(loss_state.z_hist, z_next, max_hist=self.hist_max)

        # --- (7) within frontier same state repulsion
        z = F.normalize(z_next, dim=-1)                 # (B,M,D)
        sim = torch.matmul(z, z.transpose(1,2))         # (B,M,M)
        mask = 1 - torch.eye(M, device=z.device).unsqueeze(0)
        L_cov = (sim * mask).sum(dim=(1,2)) / (M*(M-1))
        L_cov = L_cov.mean()

        # --- (8) Choose varying parents
        P = F.softmax(frontier_logits, dim=-1)    # (B,M,M)
        # encourage P rows to be orthogonal
        G_mat = torch.matmul(P, P.transpose(1,2))     # (B,M,M)
        L_parent_div = (G_mat * mask).sum(dim=(1,2)) / (M*(M-1))
        L_parent_div = L_parent_div.mean()


        # --- (9) Evidence consistency: upper-bound elimination using env.lens on frontier ---
        # Compute remaining distance from each next state:
        next_states = out["states"]  # (B,M,S)
        next_states_flat = next_states.reshape(B * M, self.model.state_dim)
        
        expanded_goal_grids = [g for g in original_goal_grids for _ in range(M)]
        
        # true distance left to goal
        d_rem = self.compute_env_lens(next_states_flat, expanded_goal_grids).view(B, M).detach() # (B,M) in [0,1]

        # Update depths: new depth for each new slot = parent depth + 1
        # We need parent selection to propagate depth
        parent_one_hot = out["parent_one_hot"]  # (B,M,M)
        parent_depth = torch.bmm(parent_one_hot, loss_state.depths.unsqueeze(-1)).squeeze(-1)# (B,M)
        # Check if state changed significantly (not strict equality, maybe based on reward or something)
        # Using exact match on states here:
        changed = 1.0 - torch.isclose(out["states"], out["states_selected"], atol=1e-5).all(dim=-1).float()
        loss_state.depths = parent_depth + changed

        # normalize depth to [0,1] using env.max_lens (which is 4*shapes)
        # BUT here we want normalized by MAX POSSIBLE STEPS in task which is 4
        depth_norm = loss_state.depths / 4.0

        ub_candidates = (depth_norm + d_rem).clamp(0.0, 1.0)  # (B,M)
        ub_t = ub_candidates.min(dim=1).values               # (B,)
        loss_state.best_ub = torch.minimum(loss_state.best_ub, ub_t)

        bins = bin_centers(K, device=device)  # (K,)
        # belief probs should NOT put mass above best_ub
        mask_above = (bins.unsqueeze(0) > loss_state.best_ub.unsqueeze(1)).float()  # (B,K)
        L_ub_elim = (belief_probs * mask_above).sum(dim=1).mean()
        
        L = (
            self.weights.ce * L_ce
            + self.weights.ig_mono * L_ig_mono
            + self.weights.continue_pen * L_continue
            + self.weights.ent * L_ent
            + self.weights.value * L_value
            + self.weights.revisit * L_revisit
            + self.weights.ub_elim * L_ub_elim
            + self.weights.cov * L_cov
            + self.weights.parent_div * L_parent_div
        )

        logs = {
            "L_ce": float(L_ce.detach().cpu()),
            "L_ig_mono": float(L_ig_mono.detach().cpu()),
            "L_continue": float(L_continue.detach().cpu()),
            "L_ent": float(L_ent.detach().cpu()),
            "L_value": float(L_value.detach().cpu()),
            "L_revisit": float(L_revisit.detach().cpu()),
            "L_ub_elim": float(L_ub_elim.detach().cpu()),
            "L_cov": float(L_cov.detach().cpu()),
            "L_parent_div": float(L_parent_div.detach().cpu()),
            "entropy_mean": float(H.mean().detach().cpu()),
            "best_ub_mean": float(loss_state.best_ub.mean().detach().cpu()),
            "total_loss": float(L.detach().cpu())
        }
        return L, logs

    def train_step(self, T):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        
        data = self.env.get_current_states() # (states, goals)
        # states: (B, S), goals: (B, G)
        start_states, goals = data
        start_states = start_states.to(self.device)
        goals = goals.to(self.device)
        
        original_goal_grids = list(self.env.goal_grids_batch) # List of np arrays
        
        # Calculate goal lengths (number of blocks) for normalization
        goal_lengths = torch.tensor([
            len(np.unique(g)[np.unique(g) != 0]) for g in original_goal_grids
        ], device=self.device, dtype=torch.float32)

        B = goals.size(0)
        M = self.model.M

        # init frontier (repeat empty start state)
        states = start_states.unsqueeze(1).expand(B, M, self.model.state_dim).contiguous()

        # Capture for WM
        # self.collect_wm_buffer(out=None, goals=goals) # wait, we need out... 

        prev_actions = torch.zeros(B, M, self.model.total_action_dim, device=self.device)
        prev_rewards = torch.zeros(B, M, device=self.device)
        prev_values  = torch.zeros(B, M, device=self.device)
        hidden = None

        loss_state = SearchLossState(B, M, device=self.device)
        stopper = IGStopper(eps_ig=self.eps_ig, eps_entropy=self.eps_entropy, patience=self.ig_patience)

        total_loss = 0.0
        active_idx = torch.arange(B, device=self.device)
        
        # Metrics
        stop_steps = torch.zeros(B, device=self.device)
        is_stopped = torch.zeros(B, dtype=torch.bool, device=self.device)
        success_count = 0
        
        accumulated_logs = {}

        for t in range(T):
            if not goals.size(0):
                break

            out = self.inference_step(
                states, goals,
                prev_actions, prev_rewards, prev_values.detach(),
                hidden,
                original_goal_grids
            )
            
            # Check for success (distance to goal == 0)
            next_states = out["states"] # (B', M, S)
            B_curr = next_states.size(0)
            next_states_flat = next_states.reshape(B_curr * M, self.model.state_dim)
            expanded_goal_grids = [g for g in original_goal_grids for _ in range(M)]
            
            dists = self.compute_env_lens(next_states_flat, expanded_goal_grids).view(B_curr, M) # (B', M)
            reached = (dists == 0.0).any(dim=1) # (B',)
            success_count += reached.sum().item()

            L_t, logs = self.compute_losses(
                out=out,
                goals=goals,
                start_states=start_states,
                original_goal_grids=original_goal_grids,
                prev_values_undetached=prev_values,
                prev_rewards=prev_rewards,
                loss_state=loss_state,
                first=(t == 0)
            )

            total_loss = total_loss + L_t

            # Aggregate logs
            for k, v in logs.items():
                accumulated_logs[k] = accumulated_logs.get(k, 0.0) + v
            
            # WM Buffer Collection
            self.collect_wm_buffer(out, goals)

            # update recurrent inputs
            states = out["states"]
            prev_actions = out["policy_one_hot"].detach()
            prev_rewards = out["rewards"].detach()
            prev_values  = out["values"]
            hidden = out["hidden"]

            # stopping
            stop_mask, stop_info = stopper.step(out["beliefs"])
             
            # Record stop steps for those stopping NOW
            # active_idx maps current batch index to original index
            # stop_mask corresponds to current batch
            newly_stopped_mask = stop_mask & (~is_stopped[active_idx])
            stop_steps[active_idx[newly_stopped_mask]] = t + 1
            is_stopped[active_idx[stop_mask]] = True
             
            if not stop_mask.any():
                continue
            elif stop_mask.all():
                break
            
            keep_idx = torch.nonzero(~stop_mask, as_tuple=False).squeeze(1)
            active_idx = active_idx.index_select(0, keep_idx)
            
            shrunk = shrink_batch(
                keep_idx,
                start_states=start_states,
                goals=goals,
                states=states,
                prev_actions=prev_actions,
                prev_rewards=prev_rewards,
                prev_values=prev_values,
            )
            start_states = shrunk["start_states"]
            goals        = shrunk["goals"]
            states       = shrunk["states"]
            prev_actions = shrunk["prev_actions"]
            prev_rewards = shrunk["prev_rewards"]
            prev_values  = shrunk["prev_values"]

            hidden = gather_hidden(hidden, keep_idx)
            loss_state = loss_state.shrink(keep_idx)
            stopper = stopper.shrink(keep_idx)
            
            # shrink original_goal_grids list
            original_goal_grids = [original_goal_grids[i] for i in keep_idx.cpu().numpy()]
            
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Train WM
        wm_loss = self.train_wm_offline()
        accumulated_logs["wm_loss"] = wm_loss
        
        self.internal_step += 1
        
        # Normalize logs
        for k in accumulated_logs:
            if k == "wm_loss": continue # Already averaged
            accumulated_logs[k] /= T
            
        # Add new metrics
        stop_steps[~is_stopped] = T
        norm_stop_steps = stop_steps / goal_lengths
        
        accumulated_logs["norm_stop_steps"] = norm_stop_steps.mean().item()
        accumulated_logs["goal_success_count"] = float(success_count)/(B*T)
            
        if self.logger:
             self.logger.log(accumulated_logs, step=self.internal_step)
        
        # Reset environment for next batch
        self.env._initialize_all_environments()
        
        return accumulated_logs

    def train(self, num_steps, T=12, run_name="default"):
        best_val_loss = float('inf')
        
        while True:
            # Progress bar for num_steps
            pbar = tqdm(range(num_steps), desc="Training")
            for i in pbar:
                logs = self.train_step(T)
                
                # Update progress bar
                pbar.set_postfix({
                    "Loss": f"{logs.get('total_loss', 0.0):.4f}", 
                    "WM": f"{logs.get('wm_loss', 0.0):.4f}"
                })
                
                # We'll save/eval every num_steps
                
            # End of num_steps chunk
            # Save epoch
            self.save(f"data/run_data/{run_name}/epochs/step_{self.internal_step}", prefix=f"step_{self.internal_step}")
            
            # Evaluate Validation
            val_metrics = self.evaluate(self.val_env, num_steps=10, T=T)
            print(f"Validation Metrics: {val_metrics}")
            
            # Evaluate Test
            test_metrics = self.evaluate(self.test_env, num_steps=10, T=T)
            print(f"Test Metrics: {test_metrics}")
            
            if self.logger:
                self.logger.log({
                    "val_loss": val_metrics["loss"],
                    "val_norm_stop_steps": val_metrics["norm_stop_steps"],
                    "val_success_count": val_metrics["success_count"],
                    "test_loss": test_metrics["loss"],
                    "test_norm_stop_steps": test_metrics["norm_stop_steps"],
                    "test_success_count": test_metrics["success_count"],
                    "epoch": self.internal_step // num_steps
                })
            
            val_loss = val_metrics["loss"]
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save(f"data/run_data/{run_name}/best", prefix="best")
                print("New best model saved!")

def initialize_from_saved(model, wm, ctd_from, trainer=None):
    if not ctd_from.is_dir():
        return

    #list all ckpt in ctd_from dir
    ckpt_files = [f for f in ctd_from.iterdir() if f.is_file() and f.name.endswith(".pt")]
    
    wm_files = [f for f in ckpt_files if "wm" in f.name and "opt" not in f.name]
    model_files = [f for f in ckpt_files if "model" in f.name]
    opt_files = [f for f in ckpt_files if "opt" in f.name and "wm" not in f.name]
    wm_opt_files = [f for f in ckpt_files if "wm_opt" in f.name]

    
    if wm_files:
        wm.load_state_dict(torch.load(wm_files[0]))
        print(f"Loaded WM from {wm_files[0]}")
        
    if model_files:
        model.load_state_dict(torch.load(model_files[0]))
        print(f"Loaded Model from {model_files[0]}")

    if trainer is not None:
        if opt_files:
            trainer.optimizer.load_state_dict(torch.load(opt_files[0]))
            print(f"Loaded Agent Optimizer from {opt_files[0]}")
        if wm_opt_files:
            trainer.wm_optimizer.load_state_dict(torch.load(wm_opt_files[0]))
            print(f"Loaded WM Optimizer from {wm_opt_files[0]}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--ctd_from", type=Path, default=None)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")

    # Model Config
    parser.add_argument("--d_hidden", type=int, default=128, help="Hidden size")
    parser.add_argument("--n_layers", type=int, default=2, help="Num RNN layers")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--layer_norm", action="store_true")
    parser.add_argument("--include_state_ins_for_actions", action="store_true")
    parser.add_argument("--num_hypo", type=int, default=4)
    parser.add_argument("--state_embed_dim", type=int, default=64)
    parser.add_argument("--action_embed_dim", type=int, default=32)

    # Optimization
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_wm", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)

    # Training Environment
    parser.add_argument("--data_method", choices=["regular", "all_shuffled", "no_test_in_train", "gen_flat_train"], default="gen_flat_train")
    parser.add_argument("--state_repr", choices=["asis", "clarion"], default="clarion")
    parser.add_argument("--goal_repr", choices=["asis", "clarion", "pixel"], default="pixel")
    parser.add_argument("--action_repr", choices=["standard", "factored"], default="standard")
    parser.add_argument("--close_rewards", action="store_true")

    # Loss Weights
    parser.add_argument("--w_ce", type=float, default=1.0)
    parser.add_argument("--w_ent", type=float, default=0.01)
    parser.add_argument("--w_continue", type=float, default=0.01)
    parser.add_argument("--w_ig_mono", type=float, default=0.1)
    parser.add_argument("--w_value", type=float, default=1.0)
    parser.add_argument("--w_revisit", type=float, default=0.1)
    parser.add_argument("--w_ub_elim", type=float, default=0.5)
    parser.add_argument("--w_cov", type=float, default=0.1)
    parser.add_argument("--w_parent_div", type=float, default=0.1)

    # Constants / Epsilons
    parser.add_argument("--eps_entropy", type=float, default=0.05)
    parser.add_argument("--eps_ig", type=float, default=1e-3)
    parser.add_argument("--ig_patience", type=int, default=2)
    parser.add_argument("--revisit_sigma", type=float, default=1.0)
    parser.add_argument("--hist_max", type=int, default=512)


    args = parser.parse_args()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGUSR1, sigusr2_handler)
    signal.signal(signal.SIGUSR2, sigusr2_handler)
    
    # WandB
    run = wandb.init(project="brickworld-tree-search", name=args.run_name, config=vars(args))
    
    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    
    # Data loading (simplified matching train.py logic or simplified)
    match args.data_method:
        case "regular":
            train_base, val_base, test_base = "data/processed/regular/train_data/train_stims/", "data/processed/regular/test_data_training/test_stims/", "data/processed/regular/test_data/test_stims"
        case "gen_flat_train":
            train_base, val_base, test_base = "data/processed/gen_grids/flattened_removed_train/", "data/processed/gen_grids/flattened_removed_val/", "data/processed/regular/test_data/test_stims"
        case _:
            raise ValueError("No such data method implemented yet")
        
    env = NewPipelineEnv(load_from_dirname(train_base), args.batch_size, train_base, device=device, close_rewards=args.close_rewards, goal_repr=args.goal_repr, state_repr=args.state_repr, action_repr=args.action_repr)
    val_env = NewPipelineEnv(load_from_dirname(val_base), args.batch_size, val_base, device=device, close_rewards=args.close_rewards, goal_repr=args.goal_repr, state_repr=args.state_repr, action_repr=args.action_repr)
    test_env = NewPipelineEnv(load_from_dirname(test_base), 48, test_base, device=device, close_rewards=args.close_rewards, goal_repr=args.goal_repr, state_repr=args.state_repr, action_repr=args.action_repr)

    # Configs
    S=36 if not args.state_repr == "clarion" else 144
    G=36 if not args.goal_repr == "clarion" else 144
    
    # Action dims calculation
    if args.action_repr == "standard":
        a_dims = (52,)
        a_embed_dims = (args.action_embed_dim,)
    else:
        a_dims = (4, 17)
        a_embed_dims = (args.action_embed_dim, args.action_embed_dim)

    tree_config = TreeSearcherConfig(
        M=args.num_hypo,
        state_dim=S,
        state_embedding_dim=args.state_embed_dim,
        goal_dim=G,
        goal_embedding_dim=args.state_embed_dim, # Sharing state embed dim
        action_dims=a_dims,
        action_embedding_dim=args.action_embed_dim
    )
    
    # World Model Config
    wm_config = WorldModelConfig(
        hidden_size=args.d_hidden,
        wm_lr=args.lr_wm,
        dropout=args.dropout,
        norm=args.layer_norm,
        wm_weight_decay=args.wd,
    )
    
    # We populate it from args + tree_config properties
    wm_config.state_dim = S
    wm_config.action_dims = a_dims
    wm_config.action_embedding_dim = a_embed_dims
    wm_config.goal_dim = G
    
    model = TreeSearcher(
        tree_config, 
        hidden_size=args.d_hidden,
        num_layers=args.n_layers,
        norm=args.layer_norm,
        include_state_ins_for_actions=args.include_state_ins_for_actions,
        dropout=args.dropout,
    ).to(device)
    
    weights = LossWeights(
        ce=args.w_ce,
        ent=args.w_ent,
        continue_pen=args.w_continue,
        ig_mono=args.w_ig_mono,
        value=args.w_value,
        revisit=args.w_revisit,
        ub_elim=args.w_ub_elim,
        cov=args.w_cov,
        parent_div=args.w_parent_div
    )
    
    trainer = TreeSearchTrainer(
        model, env, val_env, test_env, args, 
        wm_config=wm_config, 
        logger=run,
        weights=weights,
        gamma=args.gamma,
        eps_entropy=args.eps_entropy,
        eps_ig=args.eps_ig,
        ig_patience=args.ig_patience,
        revisit_sigma=args.revisit_sigma,
        hist_max=args.hist_max
    )
    
    train_obj_global = trainer
    run_name_global = args.run_name
    
    if args.ctd_from:
        initialize_from_saved(model, trainer.world_model, args.ctd_from, trainer=trainer)
        
    trainer.train(num_steps=500, run_name=args.run_name)
