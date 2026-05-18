import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_sequences(
    sequences: dict,
    path="data/figures/sequences_simple_goal.png",
    max_legend_items=15,
    legend_outside=True,
    use_extended_colors=True,
):
    plt.figure(figsize=(10, 4) if legend_outside else (8, 4))

    colors = (
        plt.cm.tab20.colors + plt.cm.Set3.colors
        if use_extended_colors
        else [f"C{i}" for i in range(10)]
    )

    sequence_names = list(sequences.keys())
    for i, sequence in enumerate(sequence_names):
        sequences[sequence] = sequences[sequence].mean(axis=0)
        plt.plot(sequences[sequence], label=sequence, color=colors[i % len(colors)])

    plt.xlabel("Time steps")
    plt.ylabel("Sequence occurrence average")
    plt.title("Sequences across steps")

    if len(sequence_names) <= max_legend_items:
        if legend_outside:
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
        else:
            plt.legend(loc="best")
    else:
        print(
            f"Warning: {len(sequence_names)} sequences detected. Legend skipped to avoid crowding."
        )

    plt.savefig(path, bbox_inches="tight" if legend_outside else None, dpi=300)
    plt.close()


def aggregate_sequence_dicts(dict_list):
    if not dict_list:
        return None
    keys = dict_list[0].keys()
    agg = {}
    for k in keys:
        arrays = [d[k] for d in dict_list if k in d and d[k] is not None]
        if not arrays:
            continue
        max_len = max([a.shape[1] for a in arrays])
        padded_arrays = []
        for a in arrays:
            if a.shape[1] < max_len:
                pad_width = ((0, 0), (0, max_len - a.shape[1]))
                a_padded = np.pad(a, pad_width, mode="constant", constant_values=0)
                padded_arrays.append(a_padded)
            else:
                padded_arrays.append(a)
        agg[k] = np.concatenate(padded_arrays, axis=0)
    return agg


def main():
    parser = argparse.ArgumentParser(description="Plot appended inference stats.")
    parser.add_argument(
        "--stats_file",
        type=str,
        required=True,
        help="Path to the pickle file containing stats.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Run name, used for output directory.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="If provided, plot only the stat at this index (0-based). Otherwise, aggregate all.",
    )
    parser.add_argument(
        "--bounds",
        type=str,
        default=None,
        help="JSON string representing a list of lists of ints. Each sublist should be [start_index, end_index] (inclusive), e.g. '[[0, 5], [10, 15]]'.",
    )
    args = parser.parse_args()

    stats_list = []
    try:
        with open(args.stats_file, "rb") as f:
            while True:
                try:
                    stats_list.append(pickle.load(f))
                except EOFError:
                    break
    except FileNotFoundError:
        print(f"Error: file {args.stats_file} not found.")
        sys.exit(1)

    if not stats_list:
        print("No stats found in the file.")
        sys.exit(0)

    print(f"Loaded {len(stats_list)} stat objects from {args.stats_file}.")

    if args.bounds is not None:
        bounds = json.loads(args.bounds)
        new_stats_list = []
        for i, stat in enumerate(stats_list):
            # Check if this index falls within any of the provided bounds
            for start, end in bounds:
                if start <= i <= end:
                    new_stats_list.append(stat)
                    break
        stats_list = new_stats_list
        print(f"Filtered to {len(stats_list)} stat objects based on bounds.")

    if not stats_list:
        print("No stats left to plot after applying bounds.")
        sys.exit(0)

    if args.index is not None:
        if args.index < 0 or args.index >= len(stats_list):
            print(
                f"Error: index {args.index} out of bounds (0 to {len(stats_list) - 1})."
            )
            sys.exit(1)
        print(f"Using stats at index {args.index}.")
        selected_stats = stats_list[args.index]
        sequences = selected_stats.get("sequences")
        planner_sequences = selected_stats.get("planner_sequences")
    else:
        print("Aggregating all stats.")
        seq_list = [
            s.get("sequences") for s in stats_list if s.get("sequences") is not None
        ]
        sequences = aggregate_sequence_dicts(seq_list) if seq_list else None

        plan_seq_list = [
            s.get("planner_sequences")
            for s in stats_list
            if s.get("planner_sequences") is not None
        ]
        planner_sequences = (
            aggregate_sequence_dicts(plan_seq_list) if plan_seq_list else None
        )

    # Plot
    save_path = Path(f"data/inference/{args.run_name}/inference_results")
    if args.index is not None:
        save_path = save_path / f"index_{args.index}"
    elif args.bounds is not None:
        save_path = save_path / "bounded"
    else:
        save_path = save_path / "aggregated"

    save_path.mkdir(parents=True, exist_ok=True)

    if sequences is not None:
        out_file = save_path / "sequenceness.png"
        print(f"Plotting sequences to {out_file}")
        plot_sequences(sequences, path=out_file)

    if planner_sequences is not None:
        out_file = save_path / "sequenceness_planner.png"
        print(f"Plotting planner sequences to {out_file}")
        plot_sequences(planner_sequences, path=out_file)

    print("Done.")

    print("Done.")


if __name__ == "__main__":
    main()
