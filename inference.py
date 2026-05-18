import argparse
import re
from pathlib import Path

import numpy as np
import torch

from plot_stats import plot_sequences
from train import initialize_from_saved
from misc.config import ACTION_KEYS
from data_prep.actions import get_action_from_factorized
from environment.pipeline import NewPipelineEnv
from misc.evaluation import (
    REVERSE_SHAPE_MAP,
    SHAPE_MAP,
    brick_connectedness,
)
from training.models import Orchestrator


class InstrumentedOrchestrator(Orchestrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Buffer to store (B, M, A) logits per step
        self.step_hypothesis_logits = []
        self.step_hypothesis_states = []
        self.hypo_phase_actions = []
        self.step_hypothesis_masks = []

    def diversity_loss(self, logits, in_states_actions, hypo_mask=None):
        # Capture the mask used during hypothesizing
        # hypo_mask is (B, 1) or None
        if hypo_mask is not None:
            self.step_hypothesis_masks.append(hypo_mask.detach().cpu())

        return super().diversity_loss(logits, in_states_actions, hypo_mask)

    def sample_subroutine(self, logits, planner=False, gumbel=False):
        # Capture logits before they are processed/collapsed
        # logits shape: (B, M, A)

        ret = super().sample_subroutine(logits, planner, gumbel)

        # ret is (action_one_hot, action_indices, items)

        if planner:
            self.step_hypothesis_logits.append(logits.detach().cpu())
        else:
            self.hypo_phase_actions.append(ret[1])

        return ret

    def forward(
        self,
        curr_states,
        curr_goals,
        rewards=None,
        actions=None,
        planner_hidden=None,
        hypothesizer_state=None,
        profiler=None,
    ):
        # Capture current states input
        # current_states: (B, M, S)
        if curr_states is not None:
            self.step_hypothesis_states.append(curr_states.detach().cpu())
        return super().forward(
            curr_states,
            curr_goals,
            rewards,
            actions,
            planner_hidden,
            hypothesizer_state,
            profiler,
        )

    # We need to clear buffers between episodes or batches
    def clear_buffers(self):
        self.step_hypothesis_logits = []
        self.step_hypothesis_states = []
        self.hypo_phase_actions = []
        self.step_hypothesis_masks = []


def simple_goal_sequencessness_multi_head(goals, grids, stable_blocks, M=1):
    """
    Multi-head version of simple_goal_sequencessness_elaborate.
    goals: list (batch) of list (steps) of list (M actions).
    grids: list (batch) of grids.
    stable_blocks: list (batch) of stable block names (strings).
    M: int, number of hypotheses for normalization.
    """
    if not goals:
        return {}

    max_len = max([len(g) for g in goals]) - 1
    if max_len < 1:
        return {}

    stable_to_present = np.zeros((len(goals), max_len))
    stable_to_distant_present = np.zeros((len(goals), max_len))
    stable_to_absent = np.zeros((len(goals), max_len))
    present_to_stable = np.zeros((len(goals), max_len))
    present_to_distant_present = np.zeros((len(goals), max_len))
    present_to_present = np.zeros((len(goals), max_len))
    present_to_absent = np.zeros((len(goals), max_len))
    absent_to_present = np.zeros((len(goals), max_len))
    absent_to_stable = np.zeros((len(goals), max_len))
    absent_to_distant_present = np.zeros((len(goals), max_len))
    distant_present_to_stable = np.zeros((len(goals), max_len))
    distant_present_to_present = np.zeros((len(goals), max_len))
    distant_present_to_absent = np.zeros((len(goals), max_len))
    stable_to_distant_present = np.zeros((len(goals), max_len))
    rest = np.zeros((len(goals), max_len))

    sequences = {
        "Stable to present": stable_to_present,
        "Present to stable": present_to_stable,
        "Present to distant present": present_to_distant_present,
        "Present to present": present_to_present,
        "Stable to absent": stable_to_absent,
        "Present to absent": present_to_absent,
        "Absent to present": absent_to_present,
        "Absent to stable": absent_to_stable,
        "Absent to distant present": absent_to_distant_present,
        "Distant present to stable": distant_present_to_stable,
        "Distant present to present": distant_present_to_present,
        "Distant present to absent": distant_present_to_absent,
        "Stable to distant present": stable_to_distant_present,
        "Rest": rest,
    }

    valids = 0

    for i, step_actions in enumerate(goals):
        if not step_actions:
            continue

        stable_block = stable_blocks[i]
        if not stable_block:
            continue

        # Grid Analysis
        _, brick_rel = brick_connectedness(grids[i])
        try:
            t = brick_rel.index(SHAPE_MAP[stable_block])
        except ValueError:
            continue

        present = brick_rel[t - 2 if t >= 2 else t + 2]

        grid_unique = np.unique(grids[i])
        mask = (
            (grid_unique != present)
            & (grid_unique != SHAPE_MAP[stable_block])
            & (grid_unique != 0)
        )
        filtered = grid_unique[mask]
        if len(filtered) == 0:
            continue

        present2 = filtered.item()
        present2_block = REVERSE_SHAPE_MAP[present2]
        present_block = REVERSE_SHAPE_MAP[present]
        block2_isdistant = brick_rel.count(SHAPE_MAP[stable_block]) != 2

        valids += 1

        # Iterate steps (starting from 1 to max_len)
        for j, m_actions_at_step in enumerate(step_actions[1:]):
            assert len(m_actions_at_step) == M
            for action in m_actions_at_step:
                if not action or len(action) != 2:
                    rest[i, j] += 1
                    continue
                block1, block2 = action

                if block1 == stable_block and block2 == present_block:
                    stable_to_present[i, j] += 1
                elif (
                    block1 == stable_block
                    and block2 == present2_block
                    and not block2_isdistant
                ):
                    stable_to_present[i, j] += 1
                elif block1 == stable_block and block2 == present2_block:
                    stable_to_distant_present[i, j] += 1
                elif block1 == present_block and block2 == stable_block:
                    present_to_stable[i, j] += 1
                elif (
                    block1 == present2_block
                    and block2 == stable_block
                    and not block2_isdistant
                ):
                    present_to_stable[i, j] += 1
                elif block1 == present2_block and block2 == stable_block:
                    distant_present_to_stable[i, j] += 1
                elif (
                    block1 == present_block
                    and block2 == present2_block
                    and block2_isdistant
                ):
                    present_to_distant_present[i, j] += 1
                elif block1 == present_block and block2 == present2_block:
                    present_to_present[i, j] += 1
                elif (
                    block1 == present2_block
                    and block2 == present_block
                    and not block2_isdistant
                ):
                    present_to_present[i, j] += 1
                elif block1 == present2_block and block2 == present_block:
                    distant_present_to_present[i, j] += 1
                elif block1 == stable_block:
                    stable_to_absent[i, j] += 1
                elif block2 == stable_block:
                    absent_to_stable[i, j] += 1
                elif block1 == present_block:
                    present_to_absent[i, j] += 1
                elif block2 == present_block:
                    absent_to_present[i, j] += 1
                elif block1 == present2_block and not block2_isdistant:
                    present_to_absent[i, j] += 1
                elif block1 == present2_block:
                    distant_present_to_absent[i, j] += 1
                elif block2 == present2_block and not block2_isdistant:
                    absent_to_present[i, j] += 1
                elif block2 == present2_block:
                    absent_to_distant_present[i, j] += 1
                else:
                    print("YOU SHOULD HAVE COVERED ALL CASES")

    # Normalize by M (number of hypotheses)
    if M > 0:
        for k in sequences:
            sequences[k] /= M

    # note - i norm by len(goals) in the plot function
    return sequences


def format_action_for_eval(action_str, config_action_repr="factored"):
    """
    Format action string into evaluation format.
    goals list structure:
    - If start action: ["Shape"]
    - If relational: [Shape1_idx, Shape2_idx] (ints) or Strings?
      User instruction:
      "one string if ... start action"
      "list of two ints if its not a start action"
      "first action is: shape of entity logit, and then shape in factored relation action logit"
    """
    if "start" in action_str:
        # e.g. "half_T_start"
        shape = action_str.split("_start")[0]
        return [shape]
    else:
        # e.g. "horizontal_half_T_above" (Ref_Ent_Rel)
        # data_prep logic:
        # Ref = Shape1, Ent = Shape2.
        # User wants: [Ent, Ref] (Shape of Entity Logit, Shape of Relation (Ref) Logit)

        # Let's re-parse using regex from data_prep to be sure
        match = re.match(
            r"(mirror_L|vertical|horizontal|half_T)_(mirror_L|vertical|horizontal|half_T)_(left|right|above|below)",
            action_str,
        )
        if match:
            ref_shape = match.group(1)  # Shape1
            ent_shape = match.group(2)  # Shape2

            # User says "list of two ints" or "strings"?
            # Evaluation.py usually expects strings for comparison.
            # But User explicitly said "list of two ints" for non-start actions.
            # AND "you can get the names from the indices".
            # This implies I should provide names (strings) derived from indices?
            # Or pass indices?
            # Given simple_goal_sequencessness_elaborate compares with stable_block (string),
            # I will return [EntString, RefString].

            return [ent_shape, ref_shape]
        else:
            return None


def run_inference(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1. Load Data/Env (Validation/Test)
    # We use validation or test data based on args
    from train import load_test_data

    print("Loading data...")
    # Simplified data loading for inference - reusing train.py logic
    # Assuming 'gen_flat_train' style for now or whatever user used
    # But for inference we usually want a specific set.
    # Let's use test data.
    test_data = load_test_data()
    test_base = "/w/150/lambda_squad/misc/clarion_replay/data/processed/regular/test_data/test_stims"

    env = NewPipelineEnv(
        test_data,
        args.batch_size,
        test_base,
        device=device,
        close_rewards=args.close_rewards,
        goal_repr=args.goal_repr,
        state_repr=args.state_repr,
        action_repr=args.action_repr,
    )

    # 2. Initialize Model
    from train import HypoConfig, PlannerConfig, WorldModelConfig

    hypo_conf = HypoConfig(
        beta_diversity=args.beta_diversity,
        hidden_size=args.d_hidden,
        num_layers=args.n_layers,
        norm=args.layer_norm,
        num_hypotheses=args.num_hypo,
        include_state_ins_for_actions=args.include_state_ins_for_actions,
        permute_hypotheses=args.permute_hypo,
    )
    plan_conf = PlannerConfig(
        hidden_size=args.d_hidden,
        num_layers=args.n_layers,
        norm=args.layer_norm,
        feed_state=args.feed_state,
        include_state_ins_for_actions=args.include_state_ins_for_actions,
    )
    wm_conf = WorldModelConfig(
        hidden_size=args.d_hidden,
        wm_lr=args.lr_wm,
        decouple_wm=args.decouple_wm,
        dropout=args.dropout,
        norm=args.layer_norm,
        wm_weight_decay=args.wd,
    )

    S = 144 if args.state_repr == "clarion" else 36
    G = 144 if args.goal_repr == "clarion" else 36

    orchestrator = InstrumentedOrchestrator(
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
        device=device,
    )

    if args.ctd_from:
        initialize_from_saved(orchestrator, Path(args.ctd_from), args.melded)

    orchestrator.eval()

    # 3. Create Trainer wrapper (just for convenience of run_episode_batch if we wanted to use it)
    # But we want to instrument the loop to collect our specific data.
    # So we will write a custom inference loop.

    print("Starting Inference Rollout...")

    batch_size = env.batch_size
    orchestrator.clear_buffers()

    # Lists to store collected data per episode
    # Buffers for current episodes in batch
    batch_hypo_traces = [
        [] for _ in range(batch_size)
    ]  # List of list of M actions (the hypothesis rollout actions)
    batch_actions_seq = [
        [] for _ in range(batch_size)
    ]  # List of planner actions (strings)
    batch_goal_grids = [None for _ in range(batch_size)]

    # Results
    collected_hypo_sequences = []  # Formatted M-head sequences for evaluation
    collected_planner_sequences = []  # Formatted planner sequences
    collected_stable_blocks = []  # Stable block name for each episode
    collected_grids = []  # goal grids

    # Tracking for correct grids
    num_correct = 0
    num_total = 0

    # Get initial state
    states, goals = env.get_current_states()
    states = states.to(device)
    goals = goals.to(device)

    # Store initial goal grids
    for i in range(batch_size):
        batch_goal_grids[i] = env.goal_grids_batch[i].copy()

    hidden_state = None
    hypo_hidden_state = None

    rewards_seq = []
    one_hot_actions_seq = []

    max_steps = max(env.max_lens)
    active_episodes = torch.ones(batch_size, dtype=torch.bool, device=device)

    for step in range(max_steps):
        if not active_episodes.any():
            break

        orchestrator.clear_buffers()  # clear step buffers

        # Forward pass
        rnn_out, values, new_hidden, new_hypo_hidden, div_loss, wm_loss = orchestrator(
            states,
            goals,
            rewards_seq[-1].unsqueeze(-1) if rewards_seq else None,
            one_hot_actions_seq[-1] if one_hot_actions_seq else None,
            hidden_state,
            hypo_hidden_state,
        )

        # Capture Hypothesis Trace (only available at step 0 usually)
        if orchestrator.hypo_phase_actions:
            # orchestrator.hypo_phase_actions is List of Tensors.
            # If factored, it's [Ent_Tensor, Rel_Tensor, Ent_Tensor, Rel_Tensor, ...].
            # If not, it's [Act_Tensor, Act_Tensor, ...].

            num_hypo_steps = len(orchestrator.hypo_phase_actions)
            for b in range(batch_size):
                if active_episodes[b]:
                    # Extract sequence for batch b
                    b_trace = []

                    # Check if factored
                    is_factored = args.action_repr == "factored"

                    if is_factored:
                        if args.permute_hypo:
                            step_size = 3
                            ent_offset = 1
                            rel_offset = 2
                        else:
                            step_size = 2
                            ent_offset = 0
                            rel_offset = 1
                    else:
                        step_size = 1

                    for t in range(0, num_hypo_steps, step_size):
                        if is_factored:
                            ent_tensor = orchestrator.hypo_phase_actions[t + ent_offset]
                            rel_tensor = orchestrator.hypo_phase_actions[t + rel_offset]

                            ent_idxs = ent_tensor[b].cpu().numpy()  # (M,)
                            rel_idxs = rel_tensor[b].cpu().numpy()  # (M,)

                            m_actions = []
                            for m in range(len(ent_idxs)):
                                a_str = get_action_from_factorized(
                                    ent_idxs[m], rel_idxs[m]
                                )
                                if a_str:
                                    m_actions.append(format_action_for_eval(a_str))
                                else:
                                    m_actions.append(None)
                        else:
                            # Standard
                            act_tensor = orchestrator.hypo_phase_actions[t]
                            idxs = act_tensor[b].cpu().numpy()  # (M,)

                            m_actions = []
                            for m in range(len(idxs)):
                                if idxs[m] >= 0 and idxs[m] < len(ACTION_KEYS):
                                    a_str = ACTION_KEYS[idxs[m]]
                                    m_actions.append(format_action_for_eval(a_str))
                                else:
                                    m_actions.append(None)

                        # Apply Mask
                        # orchestrator.step_hypothesis_masks is List of Tensors (B, 1)
                        # corresponding to steps.
                        # Since we might have step_size=2 for factored (Ent, Rel),
                        # the mask should correspond to the step.
                        # diversity_loss is called once per algorithmic step.
                        # So step_hypothesis_masks[t] should be valid?
                        # Wait, t increments by step_size.
                        # If factored: Ent (idx t), Rel (idx t+1). One diversity loss call?
                        # Let's check models.py: loop range(10) -> one div_loss call.
                        # So we have M masks for M steps of the loop.
                        # orchestrator.hypo_phase_actions length is M (or 2M).
                        # We need to map t to the correct mask index.

                        mask_idx = t // step_size
                        should_keep = True
                        if mask_idx < len(orchestrator.step_hypothesis_masks):
                            mask_val = orchestrator.step_hypothesis_masks[mask_idx][
                                b
                            ].item()
                            if mask_val < 0.5:  # 0.0 means switched to planning (done)
                                should_keep = False

                        if should_keep:
                            b_trace.append(m_actions)

                    if not batch_hypo_traces[b]:  # Only store if empty (first time)
                        batch_hypo_traces[b] = b_trace

        # Sample (Planner) - Normal execution continues

        # Sample
        one_hot_actions, action_indices_list, policy_logits, _ = (
            orchestrator.sample_action(rnn_out, states, goals)
        )
        # orchestrator.step_hypothesis_logits contains the raw logits now
        # orchestrator.step_hypothesis_states contains the states used in forward

        # Env step
        new_states, rewards, dones = env.step(action_indices_list)

        # Parse actions to strings for curr step
        if args.action_repr == "factored":
            ent_idxs = action_indices_list[0].cpu().numpy()
            rel_idxs = action_indices_list[1].cpu().numpy()
            step_actions = []
            for i in range(batch_size):
                a_str = get_action_from_factorized(ent_idxs[i], rel_idxs[i])
                step_actions.append(a_str)
        else:
            # Standard
            act_idxs = action_indices_list[0].cpu().numpy()
            step_actions = []
            for i in range(batch_size):
                if act_idxs[i] >= 0 and act_idxs[i] < len(ACTION_KEYS):
                    step_actions.append(ACTION_KEYS[act_idxs[i]])
                else:
                    step_actions.append(None)

        for i in range(batch_size):
            if active_episodes[i]:
                # Append data
                if step_actions[i] is not None:
                    batch_actions_seq[i].append(step_actions[i])

                if dones[i]:
                    # Episode finished, save to results
                    # Check if solved correctly (final reward is 1.0)
                    num_total += 1
                    if rewards[i].item() > 0.5:
                        num_correct += 1

                    # Save Hypo Trace
                    collected_hypo_sequences.append(batch_hypo_traces[i])

                    # Save Planner Trace
                    # batch_actions_seq[i] contains the strings
                    collected_planner_sequences.append(batch_actions_seq[i])

                    collected_grids.append(batch_goal_grids[i])

                    # Determine Stable Block from Planner's first action
                    # Expected format: "Shape_start"
                    first_act = (
                        batch_actions_seq[i][0] if batch_actions_seq[i] else None
                    )
                    stable_b = None
                    if first_act and "_start" in first_act:
                        stable_b = first_act.split("_start")[0]
                    collected_stable_blocks.append(stable_b)

                    # Mark inactive
                    active_episodes[i] = False

        # Prepare for next step
        states = new_states.to(device)
        # Goals are static for this batch implementation
        # goals = goals.to(device)

        hidden_state = new_hidden
        hypo_hidden_state = new_hypo_hidden

        rewards_seq.append(rewards.to(device))
        one_hot_actions_seq.append(one_hot_actions)

    # Handle Unfinished Episodes
    for i in range(batch_size):
        if active_episodes[i]:
            # Unfinished episodes count as incorrect
            num_total += 1
            # (num_correct stays the same - unfinished = incorrect)

            # Save them anyway as per user request
            collected_hypo_sequences.append(batch_hypo_traces[i])
            collected_planner_sequences.append(batch_actions_seq[i])
            collected_grids.append(batch_goal_grids[i])

            first_act = batch_actions_seq[i][0] if batch_actions_seq[i] else None
            stable_b = None
            if first_act and "_start" in first_act:
                stable_b = first_act.split("_start")[0]
            collected_stable_blocks.append(stable_b)

    print(f"Collected {len(collected_hypo_sequences)} episodes.")
    print(
        f"Correct grids: {num_correct}/{num_total} ({100 * num_correct / num_total:.2f}%)"
        if num_total > 0
        else "No episodes collected."
    )

    # 4. Run Evaluation
    sequences = None
    if len(collected_hypo_sequences) > 0:
        print("Running simple_goal_sequencessness_multi_head...")

        sequences = simple_goal_sequencessness_multi_head(
            collected_hypo_sequences,
            collected_grids,
            collected_stable_blocks,
            M=args.num_hypo,
        )

        print("Sequences calculated.")

    planner_sequences = None
    if len(collected_planner_sequences) > 0:
        print("Running simple_goal_sequencessness_elaborate on Planner traces...")
        formatted_planner_goals = []
        for seq in collected_planner_sequences:
            ep_fmt = []
            for act_str in seq:
                if act_str:
                    fmt = format_action_for_eval(act_str)
                    if fmt:
                        ep_fmt.append([fmt])  # Wrap in list to simulate M=1?
            formatted_planner_goals.append(ep_fmt)

        planner_sequences = simple_goal_sequencessness_multi_head(
            formatted_planner_goals, collected_grids, collected_stable_blocks, M=1
        )
        print("Planner sequences calculated.")

    # Basic plotting logic
    save_path = Path(f"data/inference/{args.run_name}/inference_results")
    save_path.mkdir(parents=True, exist_ok=True)
    if len(collected_hypo_sequences) == 0 and len(collected_planner_sequences) == 0:
        print("No episodes completed.")
    elif hasattr(args, "stats_file") and args.stats_file:
        import pickle

        stats_dict = {}
        if sequences is not None:
            stats_dict["sequences"] = sequences
        if planner_sequences is not None:
            stats_dict["planner_sequences"] = planner_sequences

        with open(save_path / args.stats_file, "ab") as f:
            pickle.dump(stats_dict, f)
        print(f"Stats appended to {args.stats_file}")
    else:
        if sequences is not None:
            plot_sequences(sequences, path=save_path / "sequenceness.png")

        # Also save the raw collected data as user requested
        torch.save(
            {
                "hypo_sequences": collected_hypo_sequences,
                "planner_sequences": collected_planner_sequences,
                "grids": collected_grids,
            },
            save_path / "inference_data.pt",
        )
        print(f"Results saved to {save_path}")

        if planner_sequences is not None:
            plot_sequences(
                planner_sequences, path=save_path / "sequenceness_planner.png"
            )
            print("Planner sequences plotted.")


if __name__ == "__main__":
    # Argument parsing (simplified subset of train.py)
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--d_hidden", type=int, required=True)
    parser.add_argument("--n_layers", type=int, required=True)
    parser.add_argument("--ctd_from", type=str, required=True)

    # Default args
    parser.add_argument("--lr", type=float, default=1e-4)  # needed for config
    parser.add_argument("--lr_wm", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--beta_entropy", type=float, default=0.05)
    parser.add_argument("--beta_diversity", type=float, default=0.1)
    parser.add_argument("--beta_critic", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--layer_norm", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--data_method", default="gen_flat_train")
    parser.add_argument("--state_repr", default="clarion")
    parser.add_argument("--goal_repr", default="pixel")
    parser.add_argument("--action_repr", default="factored")
    parser.add_argument("--feed_state", action="store_true")
    parser.add_argument("--close_rewards", action="store_true")
    parser.add_argument("--permute_hypo", action="store_true")
    parser.add_argument("--disable_hypo", action="store_true")
    parser.add_argument("--num_hypo", type=int, default=4)
    parser.add_argument("--decouple_wm", action="store_true")
    parser.add_argument("--melded", action="store_true")
    parser.add_argument("--include_state_ins_for_actions", action="store_true")
    parser.add_argument(
        "--stats_file",
        type=str,
        default=None,
        help="File to append stats to instead of plotting",
    )
    parser.add_argument("--permute_hypo", action="store_true")
    parser.add_argument("--disable_hypo", action="store_true")
    parser.add_argument("--num_hypo", type=int, default=4)
    parser.add_argument("--decouple_wm", action="store_true")
    parser.add_argument("--melded", action="store_true")
    parser.add_argument("--include_state_ins_for_actions", action="store_true")
    parser.add_argument(
        "--stats_file",
        type=str,
        default=None,
        help="File to append stats to instead of plotting",
    )

    # Just to prevent error if extra args
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    run_inference(args)
