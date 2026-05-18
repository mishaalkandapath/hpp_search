import numpy as np

SHAPE_MAP = {"half_T": 1, "mirror_L": 2, "vertical": 3, "horizontal": 4}
REVERSE_SHAPE_MAP = {v: k for k, v in SHAPE_MAP.items()}


def mk_ontopness(required_form):
    ontopness = False
    count_ontop = 0
    ontop = 0
    below = 0

    if required_form.size > 0:
        for i in range(1, required_form.shape[0]):
            row_current = required_form.shape[0] - i
            row_above = required_form.shape[0] - (i + 1)
            diff_idx = np.where(
                required_form[row_current, :] - required_form[row_above, :] != 0
            )[0]

            if diff_idx.size and np.any(
                required_form[row_current, diff_idx]
                * required_form[row_above, diff_idx]
                != 0
            ):
                ontopness = True
                count_ontop += 1
                elements_form = np.unique(required_form)
                elements_form = elements_form[elements_form != 0]

                row1 = np.where(required_form == elements_form[0])[0]
                row2 = np.where(required_form == elements_form[1])[0]

                if np.min(row1) < np.min(row2):
                    ontop = elements_form[0]
                    below = elements_form[1]
                elif np.min(row1) > np.min(row2):
                    ontop = elements_form[1]
                    below = elements_form[0]

    return ontopness, count_ontop, ontop, below


def mk_besideness(required_form):
    besideness = False
    count_beside = 0
    left = 0
    right = 0

    if required_form.size > 0:
        required_form = required_form.T

        for i in range(1, required_form.shape[0]):
            row_current = required_form.shape[0] - i
            row_above = required_form.shape[0] - (i + 1)
            diff_idx = np.where(
                required_form[row_current, :] - required_form[row_above, :] != 0
            )[0]

            if diff_idx.size and np.any(
                required_form[row_current, diff_idx]
                * required_form[row_above, diff_idx]
                != 0
            ):
                besideness = True
                count_beside += 1
                elements_form = np.unique(required_form)
                elements_form = elements_form[elements_form != 0]

                row1 = np.where(required_form == elements_form[0])[0]
                row2 = np.where(required_form == elements_form[1])[0]

                if np.min(row1) < np.min(row2):
                    left = elements_form[0]
                    right = elements_form[1]
                elif np.min(row1) > np.min(row2):
                    left = elements_form[1]
                    right = elements_form[0]

    return besideness, count_beside, left, right


def brick_connectedness(stim_grid):
    bricks_conn_trial = [0, 0, 0]
    bricks_rel_trial = [0, 0, 0, 0]

    bricks = np.unique(stim_grid)[1:]
    if len(bricks) == 2:
        bricks = np.array([bricks[0], bricks[1], 5])

    part1 = np.copy(stim_grid)
    part1[part1 == bricks[0]] = 0
    part2 = np.copy(stim_grid)
    part2[part1 == bricks[1]] = 0
    part3 = np.copy(stim_grid)
    part3[part1 == bricks[2]] = 0

    bricks_order = np.array(
        [
            [
                mk_ontopness(part3)[0] + mk_ontopness(part2)[0],
                mk_ontopness(part1)[0] + mk_ontopness(part3)[0],
                mk_ontopness(part1)[0] + mk_ontopness(part2)[0],
            ],
            [
                mk_besideness(part3)[0] + mk_besideness(part2)[0],
                mk_besideness(part1)[0] + mk_besideness(part3)[0],
                mk_besideness(part1)[0] + mk_besideness(part2)[0],
            ],
        ]
    )
    bricks_order = [
        np.where(~bricks_order[0, :] & bricks_order[1, :])[0],
        np.where(bricks_order[0, :] & bricks_order[1, :])[0],
        np.where(bricks_order[0, :] & ~bricks_order[1, :])[0],
    ]
    try:
        bricks_conn_trial = bricks[bricks_order].T
    except Exception:
        pass

    if mk_ontopness(part1)[0]:
        _, _, bricks_rel_trial[1], bricks_rel_trial[3] = mk_ontopness(part1)
    elif mk_ontopness(part2)[0]:
        _, _, bricks_rel_trial[1], bricks_rel_trial[3] = mk_ontopness(part2)
    elif mk_ontopness(part3)[0]:
        _, _, bricks_rel_trial[1], bricks_rel_trial[3] = mk_ontopness(part3)

    if mk_besideness(part1)[0]:
        _, _, bricks_rel_trial[0], bricks_rel_trial[2] = mk_besideness(part1)
    elif mk_besideness(part2)[0]:
        _, _, bricks_rel_trial[0], bricks_rel_trial[2] = mk_besideness(part2)
    elif mk_besideness(part3)[0]:
        _, _, bricks_rel_trial[0], bricks_rel_trial[2] = mk_besideness(part3)

    return (
        bricks_conn_trial.flatten() if 5 not in bricks else bricks_conn_trial,
        bricks_rel_trial,
    )


def simple_goal_sequencessness_elaborate(goals, grids):
    good_indices = [i for i in range(len(goals)) if goals[i]]
    goals = [goals[i] for i in good_indices]
    grids = [grids[i] for i in good_indices]

    max_len = max([len(g) for g in goals]) - 1
    sequences = {
        "Stable to present": np.zeros((len(goals), max_len)),
        "Present to stable": np.zeros((len(goals), max_len)),
        "Present to distant present": np.zeros((len(goals), max_len)),
        "Present to present": np.zeros((len(goals), max_len)),
        "Stable to absent": np.zeros((len(goals), max_len)),
        "Present to absent": np.zeros((len(goals), max_len)),
        "Absent to present": np.zeros((len(goals), max_len)),
        "Absent to stable": np.zeros((len(goals), max_len)),
        "Absent to distant present": np.zeros((len(goals), max_len)),
        "Distant present to stable": np.zeros((len(goals), max_len)),
        "Distant present to present": np.zeros((len(goals), max_len)),
        "Distant present to absent": np.zeros((len(goals), max_len)),
        "Stable to distant present": np.zeros((len(goals), max_len)),
    }

    for i, choices_in_trial in enumerate(goals):
        stable_block = choices_in_trial[0][0]
        _, brick_rel = brick_connectedness(grids[i])
        t = brick_rel.index(SHAPE_MAP[stable_block])
        present = brick_rel[t - 2 if t >= 2 else t + 2]
        present2 = np.unique(grids[i])[
            (np.unique(grids[i]) != present)
            & (np.unique(grids[i]) != SHAPE_MAP[stable_block])
            & (np.unique(grids[i]) != 0)
        ].item()
        present_block = REVERSE_SHAPE_MAP[present]
        present2_block = REVERSE_SHAPE_MAP[present2]
        block2_isdistant = brick_rel.count(SHAPE_MAP[stable_block]) != 2

        for j, other_blocks in enumerate(choices_in_trial[1:]):
            if len(other_blocks) != 2:
                continue

            block1, block2 = other_blocks
            if block1 == stable_block and block2 == present_block:
                sequences["Stable to present"][i, j] = 1
            elif (
                block1 == stable_block
                and block2 == present2_block
                and not block2_isdistant
            ):
                sequences["Stable to present"][i, j] = 1
            elif block1 == stable_block and block2 == present2_block:
                sequences["Stable to distant present"][i, j] = 1
            elif block1 == present_block and block2 == stable_block:
                sequences["Present to stable"][i, j] = 1
            elif (
                block1 == present2_block
                and block2 == stable_block
                and not block2_isdistant
            ):
                sequences["Present to stable"][i, j] = 1
            elif block1 == present2_block and block2 == stable_block:
                sequences["Distant present to stable"][i, j] = 1
            elif (
                block1 == present_block
                and block2 == present2_block
                and block2_isdistant
            ):
                sequences["Present to distant present"][i, j] = 1
            elif block1 == present_block and block2 == present2_block:
                sequences["Present to present"][i, j] = 1
            elif (
                block1 == present2_block
                and block2 == present_block
                and not block2_isdistant
            ):
                sequences["Present to present"][i, j] = 1
            elif block1 == present2_block and block2 == present_block:
                sequences["Distant present to present"][i, j] = 1
            elif block1 == stable_block:
                sequences["Stable to absent"][i, j] = 1
            elif block2 == stable_block:
                sequences["Absent to stable"][i, j] = 1
            elif block1 == present_block:
                sequences["Present to absent"][i, j] = 1
            elif block2 == present_block:
                sequences["Absent to present"][i, j] = 1
            elif block1 == present2_block and not block2_isdistant:
                sequences["Present to absent"][i, j] = 1
            elif block1 == present2_block:
                sequences["Distant present to absent"][i, j] = 1
            elif block2 == present2_block and not block2_isdistant:
                sequences["Absent to present"][i, j] = 1
            elif block2 == present2_block:
                sequences["Absent to distant present"][i, j] = 1

    return sequences
