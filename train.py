
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import datetime
import json
import wandb
import sys
import signal
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import logging
import subprocess
import copy
import shutil

from .models import Orchestrator, HypoConfig, PlannerConfig, WorldModelConfig
from .config import ACTION_KEYS, STATE_KEYS
from .data_prep import get_action_from_factorized
from .env import NewPipelineEnv
from batched_rl_env import BatchedBrickEnvironment
from models import RNN
from .profiler import SimpleProfiler, NoOpProfiler

# Import data loading utils from batched_a2c_train
# Assuming we can import from the parent directory scope or provided files
from batched_a2c_train import load_data, load_test_data, load_from_dirname, setup_conv, ActorCriticTrainer, siguser_handler, signal_handler, EpisodeData

def signal_handler(signum, frame):
    global interrupted, train_obj_global, run_name_global
    print("\n\nReceived interrupt signal (Ctrl+C). Saving model and exiting gracefully...")
    interrupted = True
    
    if train_obj_global is not None and run_name_global is not None:
        try:
            # os.makedirs is handled by trainer.save but we preserve the path structure logic
            # Passing path as "data/run_data/{run_name_global}/interrupted" so prefix becomes "interrupted"
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
class ModelTrainConfig:
    lr: float
    beta_entropy: float
    beta_critic: float
    beta_diversity: float
    wd: float
    gamma: float
    run_name: str
    device: str
    profile: bool = False
    save_inference_data: bool = False
    cmd_args: object = None

@dataclass
class ExtendedEpisodeData(EpisodeData):
    wm_loss: torch.Tensor
    div_loss: torch.Tensor
    aux_loss: torch.Tensor = None

class NewPipelineTrainer(ActorCriticTrainer):
    """
    Trainer for the New Pipeline (Orchestrator-based).
    Shadows ActorCriticTrainer from batched_a2c_train.py.
    """
    def __init__(self, 
                 orchestrator: Orchestrator, 
                 env: NewPipelineEnv, 
                 val_env: NewPipelineEnv, 
                 config: ModelTrainConfig,
                 test_env: NewPipelineEnv = None,
                 logger = None
                 ):
        self.model = orchestrator
        self.env = env
        self.val_env = val_env
        self.test_env = test_env

        #uninitialized
        # self.feed_state -> that responsibility is now in the model side
        # self.epsilon -> unused
        # self.beta_entropy --> moved to config.beta_entropy
        # self.beta_critic --> moved to config.beta_critic
        # self.loss_fn 
        # self.ads_fn both of these are manual now, since we only have a2c. 
        
        self.p_mode = "sdfsdf" # j doesnt need to be conv - we dont have taht anymore
        self.gamma = config.gamma
        
        self.config = config
        self.logger = logger
        self.device = torch.device(config.device)
        
        self.optimizer = optim.AdamW(self.model.policy_parameters(), lr=config.lr, weight_decay=config.wd)
        
        # Replay buffer for WM training
        # List of tuples: (state, action_one_hot, goal, next_state, reward)
        self.wm_buffer = []

        self.profiler = SimpleProfiler() if config.profile else NoOpProfiler()
        self.internal_step_count = 0 
        self.aux_loss_fn = nn.BCEWithLogitsLoss(reduction='none') 

    def save(self, path, prefix="best"):
        #cut off the filename in the path
        if isinstance(path, str):
            path = Path(path)
        base = path.parent
        os.makedirs(base, exist_ok=True)

        if self.model.hypothesizer is None:
            try:
                torch.save(self.model.melded_model.state_dict(), base / f"{prefix}_melded.pt")
            except AttributeError:
                torch.save(self.model.state_dict(), base / f"{prefix}_melded.pt")
        else:
            torch.save(self.model.melded_model.state_dict(), base / f"{prefix}_melded.pt")
            torch.save(self.model.hypothesizer.state_dict(), base / f"{prefix}_hypo.pt")
        try:
            torch.save(self.model.world_model.state_dict(), base / f"{prefix}_wm.pt")
        except AttributeError:
            pass
        
        # Save Optimizers
        try:
            torch.save(self.optimizer.state_dict(), base / f"{prefix}_agent_opt.pt")
            if hasattr(self.model, "wm_optimizer"):
                torch.save(self.model.wm_optimizer.state_dict(), base / f"{prefix}_wm_opt.pt")
        except Exception as e:
            print(f"Error saving optimizers: {e}")
    
    def compute_advantage_loss_batched(
            self, 
            episode_data: ExtendedEpisodeData
        ):
        """
        Compute A2C loss for factorized actions.
        Episode data containing:
        policy_logits_seq: (B, T, TotalActionDim)
        indices_list_seq, # indices of factorized? actions 
        returns, 
        values, 
        advantages,
        mask
        """
        
        B, T, _ = episode_data.policy_logits_seq.shape
        log_probs = torch.zeros(B, T, device=self.device)
        entropies = torch.zeros(B, T, device=self.device)

        assert episode_data.policy_logits_seq.size(-1) == sum(self.model.action_dims), f"Policy logits shape: {episode_data.policy_logits_seq.shape}, Action dims: {self.model.action_dims}"
        
        for t in range(T):
            step_logits = episode_data.policy_logits_seq[:, t] 
            step_indices = episode_data.actions_seq[:, t]
            
            # Compute log prob for this step summing over factors
            step_log_prob = 0
            step_entropy = 0
            current_idx = 0
            
            for i, dim in enumerate(self.model.action_dims):
                end_idx = current_idx + dim
                factor_logits = step_logits[:, current_idx:end_idx]
                factor_idx = step_indices[:, i]
                
                dist_factor = torch.distributions.Categorical(logits=factor_logits)
                step_log_prob = step_log_prob + dist_factor.log_prob(factor_idx)
                step_entropy = step_entropy + dist_factor.entropy()
                
                current_idx = end_idx
                
            log_probs[:, t] = step_log_prob
            entropies[:, t] = step_entropy
        
        valid_steps = episode_data.mask.sum()
        if valid_steps > 0:
            masked_log_probs = log_probs * episode_data.mask
            masked_advantages = episode_data.advantages.detach() * episode_data.mask
            masked_entropies = entropies * episode_data.mask #* (- masked_advantages.detach())
            
            actor_loss = -(masked_log_probs * masked_advantages).sum() / valid_steps
            entropy_loss = (masked_entropies.sum() / valid_steps)
            
            masked_values = episode_data.values_tensor * episode_data.mask
            masked_returns = episode_data.rewards_seq * episode_data.mask
            critic_loss = (masked_returns - masked_values).pow(2).sum() / valid_steps
        else:
            actor_loss = torch.tensor(0.0, device=self.device)
            critic_loss = torch.tensor(0.0, device=self.device)
            entropy_loss = torch.tensor(0.0, device=self.device)

        # Use stack to preserve gradients!
        # wm_loss = torch.stack(episode_data.wm_loss).sum() / valid_steps if episode_data.wm_loss else torch.tensor(0.0, device=self.device)
        # div_loss = torch.stack(episode_data.div_loss).sum() / valid_steps if episode_data.div_loss else torch.tensor(0.0, device=self.device)
        
        #but we are going to detach for now: TODO: not detaching causes divergence in div loss
        wm_loss = torch.tensor(episode_data.wm_loss).sum() / valid_steps
        div_loss = torch.tensor(episode_data.div_loss).sum() / valid_steps
        aux_loss = episode_data.aux_loss
        
        return (
            actor_loss + self.config.beta_critic * critic_loss + self.config.beta_diversity * div_loss - self.config.beta_entropy * entropy_loss + aux_loss,
            {
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "entropy_loss": entropy_loss,
                "wm_loss": wm_loss,
                "div_loss": div_loss,
                "aux_loss": aux_loss
            },
            masked_advantages.detach()
        )

    def run_episode_batch(self, inference=False, test=False):
        env = self.env if not inference else self.val_env
        env = env if not test else self.test_env

        #reset env
        self.model.env = env
        
        batch_size = env.batch_size
        
        # Initialize episode storage
        states_seq = []      # List of (batch, state_dim) tensors
        actions_seq = []     # List of (batch,) tensors  
        one_hot_actions_seq = [] # List of (batch, total_action_dim) tensors
        rewards_seq = []     # List of (batch,) tensors
        policy_logits_seq = [] # List of (batch, num_actions) tensors
        values_seq = []      # List of (batch,) tensors
        wm_losses = []
        div_losses = []

        # Generate Aux questions
        if self.model.aux_task:
             aux_qs, aux_labels, aux_mask = env.generate_aux_questions()
             # aux_qs: (B, 4, Inp)
             # aux_labels: (B, 4)
             # aux_mask: (B, 4)
        
        hidden_state = None
        hypo_hidden_state = None
        final_hidden_states = None

        manual_split = False
        try:
            states, goals = env.get_current_states()  # (batch, state_dim)
        except Exception:
            states = env.get_current_states()
            goals, states = states[..., :144], states[..., 144:]
            manual_split = True

        states = states.to(self.device)
        goals = goals.to(self.device)
        
        # Track which episodes are still running
        max_possible_steps = max(env.max_lens)
        best_lens = np.array(env.max_lens)//4
        active_episodes = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        mask = torch.ones(batch_size, max_possible_steps, 
                          dtype=torch.bool, device=self.device)
        
        step_count = 0
        
        while active_episodes.any() and step_count < max_possible_steps:
            self.profiler.start("model_forward")

            (
                rnn_out, values, 
                new_hidden, new_hypo_hidden, 
                div_loss, wm_loss
            ) = self.model(
                    states, goals,
                    rewards_seq[-1].unsqueeze(-1) if states_seq else None,
                    one_hot_actions_seq[-1] if states_seq else None,
                    hidden_state, hypo_hidden_state,
                    profiler=self.profiler
                )
            
            one_hot_actions, action_indices_list, policy_logits, _ = self.model.sample_action(
                rnn_out,
                states,
                goals
            )# action_indices_list is [Tensor(B), Tensor(B)]
            
            self.profiler.stop("model_forward")
            
            self.profiler.start("env_step")
            new_states, rewards, dones = env.step(
                action_indices_list if not manual_split else action_indices_list[0]
            )
            self.profiler.stop("env_step")

            if manual_split:
                new_states = new_states[..., 144:]
                new_states = new_states.to(self.device)
                rewards = rewards.to(self.device)
                dones = dones.to(self.device)
            
            if not inference:
                self.profiler.start("data_storage")
                # Items: state, action_one_hot, goal, next_state, reward
                active_mask = active_episodes
                if active_mask.any():
                    # Mask and detach
                    b_states = states[active_mask].detach()
                    b_actions = one_hot_actions[active_mask].detach()
                    b_goals = goals[active_mask].detach()
                    b_next_states = new_states[active_mask].detach()
                    b_rewards = rewards[active_mask].detach()
                    
                    self.wm_buffer.append((b_states, b_actions, b_goals, b_next_states, b_rewards))
                self.profiler.stop("data_storage")
        
            dones = dones.to(self.device)
            states_seq.append(states.to(self.device))      
            actions_seq.append(torch.stack(action_indices_list, dim=1))
            one_hot_actions_seq.append(one_hot_actions)

            rewards_seq.append(rewards.to(self.device))
            policy_logits_seq.append(policy_logits)
            values_seq.append(values.to(self.device))
            wm_losses.append(wm_loss)
            div_losses.append(div_loss)
            
            states = new_states.to(self.device)
            hidden_state = new_hidden
            hypo_hidden_state = new_hypo_hidden

            if self.model.aux_task:
                 if not self.model.disable_hypo:
                     current_h = new_hypo_hidden[-1]
                 else:
                     current_h = new_hidden[-1]
                     
                 if final_hidden_states is None:
                     final_hidden_states = current_h
                 else:
                     # Update only active episodes, keep old values for finished ones
                     # active_episodes is (B,)
                     # current_h is (B, H)
                     # final_hidden_states is (B, H)
                     final_hidden_states = torch.where(active_episodes.unsqueeze(-1), current_h, final_hidden_states)
            
            # Update active episodes
            mask[:, step_count] = active_episodes.clone()
            active_episodes = torch.logical_and(active_episodes, torch.logical_not(dones)) 
            step_count += 1
        
        seq_len = len(states_seq)
        mask = mask[:, :seq_len]
        # Stack sequences
        rewards_tensor = torch.stack(rewards_seq, dim=1)  # (batch, seq_len)
        values_tensor = torch.stack(values_seq, dim=1)    # (batch, seq_len)
        actions_seq = torch.stack(actions_seq, dim=1)     # (batch, seq_len, total_action_dim)
        policy_logits_seq = torch.stack(policy_logits_seq, dim=1)

        # Compute Aux Loss
        final_aux_loss = torch.tensor(0.0, device=self.device)
        if self.model.aux_task and final_hidden_states is not None:
            aux_logits = self.model.compute_aux_logits(final_hidden_states, aux_qs)
            aux_loss = self.aux_loss_fn(aux_logits, aux_labels)
            final_aux_loss = (aux_loss * aux_mask).mean()
            

        
        # Compute returns
        advantages, returns = self.compute_advantages_masked(
            rewards_tensor, values_tensor, mask
        )
        
        return ExtendedEpisodeData(
            states_seq=states_seq,
            actions_seq=actions_seq,
            rewards_seq=rewards_tensor, 
            policy_logits_seq=policy_logits_seq,
            values_tensor=values_tensor,
            advantages=advantages,
            returns=returns,
            mask=mask,
            episode_lengths=mask.sum(dim=1)/torch.from_numpy(best_lens).to(mask.device),
            best_lengths=best_lens,
            total_rewards=(rewards_tensor * mask).sum(dim=1),
            wm_loss=wm_losses,
            div_loss=div_losses,
            aux_loss=final_aux_loss
        )


    def evaluate(self, num_episodes=128, test=False):
        """
        Evaluate model performance over multiple episodes. Overriden to spawn inference.
        """
        eval_stats = super().evaluate(num_episodes, test)
        
        # Condition to trigger saving inference data: 
        # Evaluate is usually called every 500 steps. 
        # We only want to trigger on the validation run (test=False) 
        if not test and getattr(self.config, 'save_inference_data', False):
            # Save temporary checkpoint
            temp_dir = Path(f"data/run_data/{self.config.run_name}/temp_inference_ckpt")
            temp_dir.mkdir(parents=True, exist_ok=True)
            self.save(temp_dir / "yada", prefix="temp")
            
            # Just pass the filename, inference.py will prepend the save path.
            output_file = "inference_data_log.pt"
            
            # Build the absolute path to inference.py
            inference_script = Path(__file__).resolve().parent / "inference.py"
            
            cmd = [
                sys.executable,
                "-m", "new_pipeline.inference",
                "--run_name", self.config.run_name,
                "--ctd_from", str(temp_dir),
                "--stats_file", str(output_file)
            ]
            
            if self.config.cmd_args is not None:
                c_args = self.config.cmd_args
                cmd.extend([
                    "--d_hidden", str(c_args.d_hidden),
                    "--n_layers", str(c_args.n_layers),
                    "--batch_size", "48", 
                    "--dropout", str(c_args.dropout), 
                    "--state_repr", str(c_args.state_repr),
                    "--goal_repr", str(c_args.goal_repr),
                    "--action_repr", str(c_args.action_repr),
                    "--num_hypo", str(c_args.num_hypo),
                ])
                for flag in ["feed_state", "disable_hypo", "melded", "layer_norm", 
                             "close_rewards", "include_state_ins_for_actions", 
                             "permute_hypo", "decouple_wm"]:
                    if getattr(c_args, flag, False):
                        cmd.append(f"--{flag}")
            else:
                cmd.extend([
                    "--d_hidden", str(getattr(self.config, 'n_hidden', 32)),
                    "--n_layers", str(getattr(self.config, 'n_layers', 1))
                ])

            # Check if there is at least 2GB of disk space available
            total, used, free = shutil.disk_usage(".")
            free_gb = free / (1024 ** 3)
            if free_gb < 2:
                print(f"CRITICAL WARNING: Only {free_gb:.2f} GB of disk space remaining! Less than 2GB available.")
                print("Suspending process to prevent disk full errors. Run 'kill -CONT' to resume when space is cleared.")
                os.kill(os.getpid(), signal.SIGSTOP)
                
            # Spawn asynchronously and route stdout and stderr to DEVNULL to avoid clutter
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
        return eval_stats

    def train_wm_offline(self, batch_size=64):
        """
        Train WM on collected buffer.
        """
        self.profiler.start("wm_offline_train")
        if not self.wm_buffer:
            self.profiler.stop("wm_offline_train")
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
        
        for start_idx in range(0, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)
            batch_indices = indices[start_idx:end_idx]
            
            loss = self.model.train_world_model_batch(
                all_states[batch_indices],
                all_actions[batch_indices],
                all_goals[batch_indices],
                all_next_states[batch_indices],
                all_rewards[batch_indices]
            )
            total_loss += loss
            num_batches += 1
            
        # Clear buffer
        self.wm_buffer_len = total_samples
        self.wm_buffer = []
        self.profiler.stop("wm_offline_train")
        
        return total_loss / max(num_batches, 1)

    def train_step(self):
        """
        Run one training step: episode batch to completion + backpropagation.
        """
        self.profiler.start("total_train_step")
        self.model.train()
        
        self.profiler.start("episode_rollout")
        episode_data = self.run_episode_batch(inference=False)
        self.profiler.stop("episode_rollout")
        
        self.profiler.start("loss_compute")
        total_loss, losses, advantages = self.compute_advantage_loss_batched(
            episode_data
        )
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.policy_parameters(), max_norm=1.0)
        self.optimizer.step()
        self.profiler.stop("loss_compute")

        # Offline WM Training
        wm_offline_loss = self.train_wm_offline(batch_size=self.env.batch_size)
        losses["wm_offline_loss"] = wm_offline_loss

        # Reinitialize environments for next batch
        self.env._initialize_all_environments()
        
        self.profiler.stop("total_train_step")
        
        if self.config.profile:
            self.internal_step_count += 1
            if self.internal_step_count % 10 == 0:
                print(self.profiler.report())
                self.profiler.reset()
        
        # # Log to file for visibility
        # with open("debug_log.txt", "a") as f:
        #     import datetime
        #     ts = datetime.datetime.now().isoformat()
        #     pid = os.getpid()
        #     f.write(f"[{ts}] [PID:{pid}] Step: {self.internal_step_count}, Loss: {total_loss.item():.4f}, "
        #             f"MeanLen: {episode_data.episode_lengths.float().mean().item():.2f}, "
        #             f"WM_Loss: {wm_offline_loss:.4f}, "
        #             f"WM_Samples: {self.wm_buffer_len if hasattr(self, 'wm_buffer_len') else 'N/A'}\n")
            
            # if self.config.profile and self.internal_step_count % 10 == 0:
            #      f.write(self.profiler.report() + "\n")
            #      f.flush()
            # if self.config.profile and self.internal_step_count % 10 == 0:
            #      f.write(self.profiler.report() + "\n")
        
        return {
            'total_loss': total_loss.item(),
            'train_mean_episode_length': episode_data.episode_lengths.float().mean().item(),
            'train_mean_episode_reward': episode_data.total_rewards.float().mean().item(),
            'advantage_mean': advantages.mean().item(),
            'advantage_std': advantages.std().item(),
            **losses
        }

def initialize_from_saved(orchestrator, ctd_from, melded):
    if not ctd_from.is_dir():
        orchestrator.load_state_dict(torch.load(ctd_from))
        return

    #list all ckpt in ctd_from dir
    ckpt_files = [f for f in ctd_from.iterdir() if f.is_file() and f.name.endswith(".pt")]
    
    #check for file with wm in it: if there are many check for best_wm
    wm_files = [f for f in ckpt_files if "wm" in f.name and "opt" not in f.name]
    if len(wm_files) == 0:
        raise ValueError("No WM checkpoint found in directory")
    if len(wm_files) > 1:
        best_wm_file = [f for f in wm_files if "best" in f.name][0]
    else:
        best_wm_file = wm_files[0]

    orchestrator.world_model.load_state_dict(torch.load(best_wm_file))
    print("===== Loaded WM model =====")

    if melded:
        melded_files = [f for f in ckpt_files if "melded" in f.name]
        if len(melded_files) == 0:
            raise ValueError("No Melded checkpoint found in directory")
        if len(melded_files) > 1:
            best_melded_file = [f for f in melded_files if "best" in f.name][0]
        else:
            best_melded_file = melded_files[0]

        orchestrator.melded_model.load_state_dict(torch.load(best_melded_file))
        print("===== Loaded melded model =====")
    else:
        planner_files = [f for f in ckpt_files if "planner" in f.name]
        if len(planner_files) == 0:
            raise ValueError("No Planner checkpoint found in directory")
        if len(planner_files) > 1:
            best_planner_file = [f for f in planner_files if "best" in f.name][0]
        else:
            best_planner_file = planner_file[0]
        hypothesizer_file = [f for f in ckpt_files if "hypothesizer" in f.name]
        if len(hypothesizer_file) == 0:
            raise ValueError("No Hypothesizer checkpoint found in directory")
        if len(hypothesizer_file) > 1:
            best_hypothesizer_file = [f for f in hypothesizer_file if "best" in f.name][0]
        else:
            best_hypothesizer_file = hypothesizer_file[0]

        orchestrator.planner.load_state_dict(torch.load(best_planner_file))
        orchestrator.hypothesizer.load_state_dict(torch.load(best_hypothesizer_file))

    # Load WM Optimizer if exists
    wm_opt_files = [f for f in ckpt_files if "wm_opt" in f.name]
    if wm_opt_files:
        if len(wm_opt_files) > 1:
            best_wm_opt = [f for f in wm_opt_files if "best" in f.name][0]
        else:
            best_wm_opt = wm_opt_files[0]
        
        try:
            orchestrator.wm_optimizer.load_state_dict(torch.load(best_wm_opt))
            print(f"Loaded WM optimizer from {best_wm_opt}")
        except Exception as e:
            print(f"Failed to load WM optimizer: {e}")

def load_agent_optimizer(trainer, ctd_from):
    if not ctd_from.is_dir():
        return
    ckpt_files = [f for f in ctd_from.iterdir() if f.is_file() and f.name.endswith(".pt")]
    agent_opt_files = [f for f in ckpt_files if "agent_opt" in f.name]
    
    if agent_opt_files:
        if len(agent_opt_files) > 1:
            best_agent_opt = [f for f in agent_opt_files if "best" in f.name][0]
        else:
            best_agent_opt = agent_opt_files[0]
            
        try:
            trainer.optimizer.load_state_dict(torch.load(best_agent_opt))
            print(f"Loaded Agent optimizer from {best_agent_opt}")
        except Exception as e:
            print(f"Failed to load Agent optimizer: {e}")

# Main execution block
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)

    parser.add_argument("--d_hidden", type=int, required=True, help="Hidden layer size")
    parser.add_argument("--n_layers", type=int, required=True, help="Number of RNN layers")
    parser.add_argument("--ctd_from", type=Path, default=None)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_wm", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--beta_entropy", type=float, default=0.05)
    parser.add_argument("--beta_diversity", type=float, default=0.1)
    parser.add_argument("--beta_critic", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--layer_norm", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--data_method", choices=["regular", "all_shuffled", "no_test_in_train", "gen_flat_train"], default="gen_flat_train")
    parser.add_argument("--state_repr", choices=["asis", "clarion"], default="clarion")
    parser.add_argument("--goal_repr", choices=["asis", "clarion", "pixel"], default="pixel")
    parser.add_argument("--action_repr", choices=["standard", "factored"], default="standard")
    parser.add_argument("--feed_state", action="store_true")
    parser.add_argument("--close_rewards", action="store_true")
    
    parser.add_argument("--permute_hypo", action="store_true")
    parser.add_argument("--disable_hypo", action="store_true")
    parser.add_argument("--num_hypo", type=int, default=4)
    parser.add_argument("--decouple_wm", action="store_true")
    parser.add_argument("--melded", action="store_true")
    parser.add_argument("--include_state_ins_for_actions", action="store_true")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    parser.add_argument("--aux_task", action="store_true", help="Enable auxiliary task")
    parser.add_argument("--save_inference_data", action="store_true", help="Save inference data every evaluate call")
    
    args = parser.parse_args()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGVTALRM, siguser_handler) # SIGUSR1 is sometimes taken by other processes/libraries, maybe user meant just a signal. 
    # But code says SIGUSR1. Let's stick to what was there and add SIGUSR2.
    # Actually wait, the existing code has signal.SIGUSR1.
    signal.signal(signal.SIGUSR1, sigusr2_handler)
    signal.signal(signal.SIGUSR2, sigusr2_handler)
    
    # WandB
    run = wandb.init(project="brickworld-new-pipeline", name=args.run_name, config={
                "learning_rate": args.lr,
                "learning_rate_wm": args.lr_wm,
                "batch_size": args.batch_size,
                "gamma": args.gamma,
                "num_layers": args.n_layers,
                "beta_entropy": args.beta_entropy,
                "beta_critic": args.beta_critic,
                "beta_diversity": args.beta_diversity,
                "wd": args.wd,
                "layer_norm": args.layer_norm,
                "dropout": args.dropout,
                "data_method": args.data_method,
                "state_repr": args.state_repr,
                "goal_repr": args.goal_repr,
                "feed_state": args.feed_state,
                "close_rewards": args.close_rewards,
                "decouple_wm": args.decouple_wm,
                "melded": args.melded,
                "n_hidden": args.d_hidden,
                "disable_hypo": args.disable_hypo,
                "num_hypo": args.num_hypo,
                "action_repr": args.action_repr,
                "include_state_ins_for_actions": args.include_state_ins_for_actions,
                "aux_task": args.aux_task
            })
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    match args.data_method:
        case "regular":
            train_data = load_data("data/processed/regular/train_data/train_stims/")
            val_data = load_data("data/processed/regular/test_data_training/test_stims/")
            test_data = load_test_data()
            train_base, val_base, test_base = "data/processed/regular/train_data/train_stims/", "data/processed/regular/test_data_training/test_stims/", "/w/150/lambda_squad/misc/clarion_replay/data/processed/regular/test_data/test_stims"
        case "no_test_in_train":
            # although this suggests val data is in regular, it is not - there is acheck for that. its j that there are some duplication of file sin some directories, so this  method controls for that
            train_data = load_from_dirname("data/processed/regular/train_data/train_stims_wo_test/")
            val_data = load_from_dirname("data/processed/regular/test_data_training/test_stims/")
            test_data = load_test_data()

            train_base, val_base, test_base = "data/processed/regular/train_data/train_stims_wo_test/", "data/processed/regular/test_data_training/test_stims/", "/w/150/lambda_squad/misc/clarion_replay/data/processed/regular/test_data/test_stims"
        case "gen_flat_train":
            train_data = load_from_dirname("data/processed/gen_grids/flattened_removed_train/")
            val_data = load_from_dirname("data/processed/gen_grids/flattened_removed_val/")
            test_data = load_test_data()

            train_base, val_base, test_base = "data/processed/gen_grids/flattened_removed_train/", "data/processed/gen_grids/flattened_removed_val/", "/w/150/lambda_squad/misc/clarion_replay/data/processed/regular/test_data/test_stims"
        case "all_shuffled":
            # mixes in human test data into the overall data, and forms a set from that. 
            train_data = load_from_dirname("/w/150/lambda_squad/misc/clarion_replay/all_shuffled/train_new")
            val_data = load_from_dirname("/w/150/lambda_squad/misc/clarion_replay/all_shuffled/val_new")
            test_data = load_from_dirname("/w/150/lambda_squad/misc/clarion_replay/all_shuffled/test_new")

            train_base, val_base, test_base = "/w/150/lambda_squad/misc/clarion_replay/all_shuffled/train_new", "/w/150/lambda_squad/misc/clarion_replay/all_shuffled/val_new", "/w/150/lambda_squad/misc/clarion_replay/all_shuffled/test_new"
        case _:
            raise ValueError("No such data method")
        
    # Environments
    env = NewPipelineEnv(train_data, args.batch_size, train_base, device=device, close_rewards=args.close_rewards, goal_repr=args.goal_repr, state_repr=args.state_repr, action_repr=args.action_repr)
    val_env = NewPipelineEnv(val_data, args.batch_size, val_base, device=device, close_rewards=args.close_rewards, goal_repr=args.goal_repr, state_repr=args.state_repr, action_repr=args.action_repr)
    test_env = NewPipelineEnv(test_data, 48, test_base, device=device, close_rewards=args.close_rewards, goal_repr=args.goal_repr, state_repr=args.state_repr, action_repr=args.action_repr)
    
    # Configs
    hypo_conf = HypoConfig(
        beta_diversity=args.beta_diversity,
        hidden_size=args.d_hidden,
        num_layers=args.n_layers,
        norm=args.layer_norm,
        num_hypotheses=args.num_hypo,
        include_state_ins_for_actions=args.include_state_ins_for_actions,
        permute_hypotheses=args.permute_hypo
    )
    plan_conf = PlannerConfig(
        hidden_size=args.d_hidden,
        num_layers=args.n_layers,
        norm=args.layer_norm,
        feed_state=args.feed_state,
        include_state_ins_for_actions=args.include_state_ins_for_actions    
    )
    wm_conf = WorldModelConfig(
        hidden_size=args.d_hidden,
        wm_lr=args.lr_wm,
        decouple_wm=args.decouple_wm,
        dropout=args.dropout,
        norm=args.layer_norm,
        wm_weight_decay=args.wd
    )
    
    S=36 if not args.state_repr == "clarion" else 144
    G=36 if not args.goal_repr == "clarion" else 144
    
    # Orchestrator
    orchestrator = Orchestrator(
        env=env,
        state_dim=S,
        goal_dim=G,
        action_dims=(52,) if args.action_repr == "standard" else (4, 17),
        action_embedding_dim=(32,) if args.action_repr == "standard" else (8, 24),
        hypo_config=hypo_conf,
        planner_config=plan_conf,
        world_model_config=wm_conf,
        use_melded_mode=args.melded,
        disable_hypo=args.disable_hypo,
        aux_task=args.aux_task,
        device=device
    )

    if args.ctd_from is not None:
        print(f"Loading checkpoint from {args.ctd_from}")
        initialize_from_saved(orchestrator, args.ctd_from, args.melded)
    
    train_config = ModelTrainConfig(
        lr=args.lr,
        beta_entropy=args.beta_entropy,
        beta_critic=args.beta_critic,
        beta_diversity=args.beta_diversity,
        wd=args.wd,
        gamma=args.gamma,
        run_name=args.run_name,
        device=device,
        profile=args.profile,
        save_inference_data=args.save_inference_data,
        cmd_args=args
    )

    os.makedirs(f"data/run_data/{args.run_name}/figures", exist_ok=True)
    os.makedirs(f"data/run_data/{args.run_name}/train", exist_ok=True)
    os.makedirs(f"data/run_data/{args.run_name}/val", exist_ok=True)
    os.makedirs(f"data/run_data/{args.run_name}/test", exist_ok=True)
    
    trainer = NewPipelineTrainer(orchestrator, env, val_env, train_config, test_env=test_env, logger=run)

    if args.ctd_from is not None:
         load_agent_optimizer(trainer, args.ctd_from)

    train_obj_global = trainer
    run_name_global = args.run_name 

    # repro command
    with open(f"data/run_data/{args.run_name}/run_command.sh", "w") as f:
        f.write("python train.py " + " ".join(sys.argv[1:]))
    
    trainer.train(run_name=args.run_name)
        