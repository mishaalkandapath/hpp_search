
import numpy as np
import torch
import re
from .config import ACTION_KEYS, SHAPES, RELS, SHAPE_DICT

# Mappings
# Entities: 0: half_T, 1: mirror_L, 2: vertical, 3: horizontal
ENTITIES = SHAPES
# Relations: 0: start. 
# Then 1..16: (Shape x Rel). 
# Order: For each shape in SHAPES, for each rel in RELS.
# r_index = 1 + shape_idx * 4 + rel_idx
BRICK_RELATIONS = ["start"]
for shape in SHAPES:
    for rel in RELS:
        BRICK_RELATIONS.append(f"{shape}_{rel}")

def encode_one_hot_pixel(grid: np.array) -> torch.Tensor:
    """
    Encodes the grid as a flattened one-hot vector indicating occupancy.
    Output size: H*W (so 6*6 = 36).
    1 if occupied (non-zero), 0 otherwise.
    """
    return torch.from_numpy((grid != 0).astype(np.float32).flatten())

def get_factorized_action_indices(action_str: str):
    """
    Returns (entity_idx, relation_idx) for a given action string.
    """
    if "start" in action_str:
        # Format: Shape_start
        match = re.match(r"(.*)_start", action_str)
        shape_name = match.group(1)
        ent_idx = SHAPES.index(shape_name)
        rel_idx = 0 # start
        return ent_idx, rel_idx
    else:
        # Format: Shape1_Shape2_Rel
        # Shape1 is the entity being placed. Shape2 is the reference.
        # Relation is Shape2_Rel.
        # Example: half_T_horizontal_above -> half_T (Ent) is above horizontal (Ref).
        # Relation: horizontal_above.
        match = re.match(r"(mirror_L|vertical|horizontal|half_T)_(mirror_L|vertical|horizontal|half_T)_(left|right|above|below)", action_str)
        shape1 = match.group(1)
        shape2 = match.group(2)
        rel = match.group(3)
        
        ent_idx = SHAPES.index(shape2)
        ref_shape_idx = SHAPES.index(shape1)
        
        rel_sub_idx = RELS.index(rel)
        rel_idx = 1 + ref_shape_idx * 4 + rel_sub_idx
        return ent_idx, rel_idx

def get_action_from_factorized(ent_idx: int, rel_idx: int) -> str:
    """
    Returns the action string from indices. Returns None if invalid.
    Invalid cases: 
    - factorization yields an action string not in ACTION_KEYS 
      (e.g., placing a shape relative to itself, which implies Shape1==Shape2).
    """
    entity_name = SHAPES[ent_idx]
    
    if rel_idx == 0:
        action_str = f"{entity_name}_start"
    else:
        # Decode relation
        adj_rel_idx = rel_idx - 1
        ref_shape_idx = adj_rel_idx // 4
        rel_sub_idx = adj_rel_idx % 4
        
        ref_shape_name = SHAPES[ref_shape_idx]
        rel_name = RELS[rel_sub_idx]
        
        action_str = f"{ref_shape_name}_{entity_name}_{rel_name}"

    return action_str if action_str in ACTION_KEYS else None

def make_factorized_mapping():
    """
    Creates a mapping tensor/lookup for converting standard action indices to factorized indices.
    Returns:
       action_to_factorized: Tensor of shape (NumActions, 2) -> [ent_idx, rel_idx]
    """
    mapping = []
    for action in ACTION_KEYS:
        ent, rel = get_factorized_action_indices(action)
        mapping.append([ent, rel])
    return torch.tensor(mapping, dtype=torch.long)
