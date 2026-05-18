
from itertools import product
import numpy as np

SHAPES = ["half_T", "mirror_L", "vertical", "horizontal"]
RELS = ["left", "right", "above", "below"]

# Standard Action Keys (copied from batched_data_prep.py)
ACTION_KEYS = ['half_T_start', 'mirror_L_start', 'vertical_start',
 'horizontal_start', 'half_T_horizontal_left', 'half_T_horizontal_right', 'half_T_horizontal_above', 'half_T_horizontal_below', 'horizontal_half_T_left', 'horizontal_half_T_right', 'horizontal_half_T_above', 'horizontal_half_T_below', 'half_T_vertical_left', 'half_T_vertical_right', 'half_T_vertical_above', 'half_T_vertical_below', 'vertical_half_T_left', 'vertical_half_T_right', 'vertical_half_T_above', 'vertical_half_T_below', 'half_T_mirror_L_left', 'half_T_mirror_L_right', 'half_T_mirror_L_above', 'half_T_mirror_L_below', 'mirror_L_half_T_left', 'mirror_L_half_T_right', 'mirror_L_half_T_above', 'mirror_L_half_T_below', 'mirror_L_horizontal_left', 'mirror_L_horizontal_right', 'mirror_L_horizontal_above', 'mirror_L_horizontal_below', 'horizontal_mirror_L_left', 'horizontal_mirror_L_right', 'horizontal_mirror_L_above', 'horizontal_mirror_L_below', 'mirror_L_vertical_left', 'mirror_L_vertical_right', 'mirror_L_vertical_above', 'mirror_L_vertical_below', 'vertical_mirror_L_left', 'vertical_mirror_L_right', 'vertical_mirror_L_above', 'vertical_mirror_L_below', 'vertical_horizontal_left', 'vertical_horizontal_right', 'vertical_horizontal_above', 'vertical_horizontal_below', 'horizontal_vertical_left', 'horizontal_vertical_right', 'horizontal_vertical_above', 'horizontal_vertical_below']

SHAPE_DICT = {"half_T": 1, "mirror_L": 2, "vertical": 3, "horizontal": 4}

STATE_KEYS = ['input_half_T_row1', 'input_half_T_row2', 'input_half_T_row3',
 'input_half_T_col1', 'input_half_T_col2', 'input_half_T_col3', 'input_mirror_L_row1', 'input_mirror_L_row2', 'input_mirror_L_row3', 'input_mirror_L_col1', 'input_mirror_L_col2', 'input_mirror_L_col3', 'input_vertical_row1', 'input_vertical_row2', 'input_vertical_row3', 'input_vertical_col1', 'input_vertical_col2', 'input_vertical_col3', 'input_horizontal_row1', 'input_horizontal_row2', 'input_horizontal_row3', 'input_horizontal_col1', 'input_horizontal_col2', 'input_horizontal_col3', 'target_half_T_row1', 'target_half_T_row2', 'target_half_T_row3', 'target_half_T_col1', 'target_half_T_col2', 'target_half_T_col3', 'target_mirror_L_row1', 'target_mirror_L_row2', 'target_mirror_L_row3', 'target_mirror_L_col1', 'target_mirror_L_col2', 'target_mirror_L_col3', 'target_vertical_row1', 'target_vertical_row2', 'target_vertical_row3', 'target_vertical_col1', 'target_vertical_col2', 'target_vertical_col3', 'target_horizontal_row1', 'target_horizontal_row2', 'target_horizontal_row3', 'target_horizontal_col1', 'target_horizontal_col2', 'target_horizontal_col3']

BASE_KEYS = STATE_KEYS[:]
STATE_KEYS = [f"{k}_{v}" for k in BASE_KEYS for v in ["1", "2", "3", "4", "5", "6"]]
