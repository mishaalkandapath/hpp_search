import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from dataclasses import dataclass
from collections import OrderedDict
from .config import ACTION_KEYS
from .data_prep import get_action_from_factorized
from .env import algorithmic_step
import torch.optim as optim
from typing import List

class myModule(nn.Module):
    def device(self):
        return next(self.parameters()).device

@dataclass
class HypoConfig:
    num_hypotheses: int = 4
    hidden_size: int = 128
    num_layers: int = 2
    norm: bool = False
    beta_diversity: float = 0.1
    include_state_ins_for_actions: bool = True
    permute_hypotheses: bool = False

@dataclass
class PlannerConfig:
    hidden_size: int = 128
    num_layers: int = 2
    norm: bool = False
    feed_state: bool = False
    include_state_ins_for_actions: bool = True

@dataclass
class WorldModelConfig:
    hidden_size: int = 64
    wm_lr: float = 1e-3
    decouple_wm: bool = False
    dropout: float = 0.0
    norm: bool = False
    wm_weight_decay: float = 0.0

class AuxiliaryNetwork(myModule):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim* 2),
            nn.ReLU(),
            nn.Linear(hidden_dim* 2, hidden_dim* 2),
            nn.ReLU(),
            nn.Linear(hidden_dim* 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Logit output
        )
    
    def forward(self, x):
        return self.net(x)

class GRULayer(myModule):
    """Custom GRU layer copied from original models.py to ensure independence."""
    def __init__(self, input_size, hidden_size, bias=True, norm=False):
        super(GRULayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
    
        self.input_to_reset = nn.Sequential(nn.Linear(input_size, hidden_size, bias=False), nn.LayerNorm(hidden_size)) if norm else nn.Linear(input_size, hidden_size, bias=bias)
        self.hidden_to_reset = nn.Sequential(nn.Linear(hidden_size, hidden_size, bias=bias), nn.LayerNorm(hidden_size)) if norm else nn.Linear(hidden_size, hidden_size, bias=bias)
        
        self.input_to_update = nn.Sequential(nn.Linear(input_size, hidden_size, bias=False), nn.LayerNorm(hidden_size)) if norm else nn.Linear(input_size, hidden_size, bias=bias)
        self.hidden_to_update = nn.Sequential(nn.Linear(hidden_size, hidden_size, bias=bias), nn.LayerNorm(hidden_size)) if norm else nn.Linear(hidden_size, hidden_size, bias=bias)
        
        self.input_to_new = nn.Sequential(nn.Linear(input_size, hidden_size, bias=False), nn.LayerNorm(hidden_size)) if norm else nn.Linear(input_size, hidden_size, bias=bias)
        self.hidden_to_new = nn.Sequential(nn.Linear(hidden_size, hidden_size, bias=bias), nn.LayerNorm(hidden_size)) if norm else nn.Linear(hidden_size, hidden_size, bias=bias)
        
    def forward(self, x, h_0=None):
        batch_size, seq_len, _ = x.size()
        if h_0 is None:
            h_0 = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        
        outputs = []
        h_t = h_0
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            r_t = torch.sigmoid(self.input_to_reset(x_t) + self.hidden_to_reset(h_t))
            z_t = torch.sigmoid(self.input_to_update(x_t) + self.hidden_to_update(h_t))
            n_t = torch.tanh(self.input_to_new(x_t) + self.hidden_to_new(r_t * h_t))
            h_t = (1 - z_t) * h_t + z_t * n_t
            outputs.append(h_t.unsqueeze(1)) # add time dim back

        outputs = torch.cat(outputs, dim=1) # (batch_size, seq_len, hidden_size)
        return outputs, h_t

class Hypothesizer(myModule):
    def __init__(self, 
                 M, state_dim, goal_dim, 
                 action_dims, action_embedding_dim, hidden_size=128, num_layers=2, norm=False
                ):
        """
        M: Number of hypotheses
        state_dim: Dimension of ONE state vector (S)
        goal_dim: Dimension of goal vector (G)
        action_dims: Tuple of dimensions for each action factor
        action_embedding_dim: Dimension of embedding for each action factor
        """
        super().__init__()
        self.M = M
        self.action_dims = action_dims
        self.action_embedding_dim = action_embedding_dim
        self.total_action_dim = sum(action_dims)
        
        # Embeddings for each factor
        self.action_embeddings = nn.ModuleList([
            nn.Linear(dim, self.action_embedding_dim[i], bias=False) 
            for i, dim in enumerate(action_dims)
        ])
        
        # Input: M * (1 scalar reward + (num_factors * embedding_dim)) + G + M*S
        total_embed_size = sum(self.action_embedding_dim)
        self.input_size = M * (1 + total_embed_size) + goal_dim + M * state_dim
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.ModuleList()
        # First layer
        self.rnn.append(GRULayer(self.input_size, hidden_size, norm=norm))
        # Subsequent layers
        curr_size = hidden_size
        for _ in range(num_layers - 1):
            self.rnn.append(GRULayer(curr_size, hidden_size, norm=norm))
            
        # Output head: M * (total_action_dim)
        self.output_dim = self.total_action_dim
        self.head = nn.Linear(hidden_size, M * self.output_dim)

    def forward(self, 
                prev_actions_one_hot, prev_rewards, 
                current_states, goal, 
                hidden=None
            ):
        """
        prev_actions_one_hot: (B, M, SumOfActionDims) - Concatenated one-hot vectors (differentiable)
        prev_rewards: (B, M) scalars
        current_states: (B, M, S)
        goal: (B, G)
        """
        B = goal.size(0)
        
        act_embeds_list = []
        start_idx = 0
        for i, dim in enumerate(self.action_dims):
            end_idx = start_idx + dim
            # factor_one_hot: (B, M, dim)
            factor_one_hot = prev_actions_one_hot[..., start_idx:end_idx]
            embed = self.action_embeddings[i](factor_one_hot)
            act_embeds_list.append(embed)
            start_idx = end_idx
            
        # (B, M, num_factors*emb_dim)
        act_embeds = torch.cat(act_embeds_list, dim=-1)
            
        act_flat = act_embeds.reshape(B, -1) # (B, M*TotalEmbed)
        
        rew_flat = prev_rewards.reshape(B, -1) # (B, M)
        
        states_flat = current_states.reshape(B, -1) # (B, M*S)
        
        # Concatenate: (B, M*TotalEmbed + M + G + M*S)
        rnn_input = torch.cat([act_flat, rew_flat, goal, states_flat], dim=1)
        
        rnn_input = rnn_input.unsqueeze(1) # (B, 1, InputSize)
        
        new_hiddens = []
        x = rnn_input
        
        if hidden is None:
            hidden = [None] * len(self.rnn)
            
        for i, layer in enumerate(self.rnn):
            out, h = layer(x, hidden[i])
            x = out
            new_hiddens.append(h)
            
        logits_flat = self.head(x.squeeze(1)) # (B, M * output_dim)
        
        # Reshape to (B, M, output_dim)
        logits = logits_flat.view(B, self.M, self.output_dim)
        
        return logits, new_hiddens

class WorldModel(myModule):
    def __init__(self, 
                state_dim, action_dims, action_embedding_dim, goal_dim, hidden_size=64, norm=False, dropout=0.2,
                ):
        """
        Input: (S + Embed(A) + G)
        Output: (S + 1) -> Next State + Reward
        """
        super().__init__()
        self.action_dims = action_dims
        self.action_embedding_dim = action_embedding_dim
        self.total_action_dim = sum(action_dims)
        
        self.action_embeddings = nn.ModuleList([
            nn.Linear(dim, self.action_embedding_dim[i], bias=False) 
            for i, dim in enumerate(action_dims)
        ])
        
        total_embed_size = sum(self.action_embedding_dim)
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + total_embed_size + goal_dim,
                      hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size) if norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size) if norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, state_dim + 1) # Next State + Reward
        )

        self.state_dim = state_dim
        self.goal_dim = goal_dim
        
    def forward(self, state, action_one_hot, goal):
        """
        state: (B*M, S)
        action_one_hot: (B*M, SumOfActionDims)
        goal: (B*M, G)
        """

        assert action_one_hot.size(1) == self.total_action_dim, f"action_one_hot should be (B*M, {self.total_action_dim}) but is {action_one_hot.shape}"
        assert state.size(1) == self.state_dim, f"state should be (B*M, {self.state_dim}) but is {state.shape}"
        assert goal.size(1) == self.goal_dim, f"goal should be (B*M, {self.goal_dim}) but is {goal.shape}"
        
        # Embed actions
        act_embeds_list = []
        start_idx = 0
        for i, dim in enumerate(self.action_dims):
            end_idx = start_idx + dim
            # factor_one_hot: (B*M, dim)
            factor_one_hot = action_one_hot[..., start_idx:end_idx]
            embed = self.action_embeddings[i](factor_one_hot)
            act_embeds_list.append(embed)
            start_idx = end_idx
            
        act_embeds = torch.cat(act_embeds_list, dim=-1)
        assert act_embeds.size(-1) == sum(self.action_embedding_dim), f"act_embeds should be (B*M, {sum(self.action_embedding_dim)}) but is {act_embeds.shape}"
        
        inp = torch.cat([state, act_embeds, goal], dim=1)
        assert inp.size(-1) == self.state_dim + sum(self.action_embedding_dim) + self.goal_dim, f"inp should be (B*M, {self.state_dim + sum(self.action_embedding_dim) + self.goal_dim}) but is {inp.shape}"
        out = self.net(inp)
        assert out.size(-1) == self.state_dim + 1, f"out should be (B*M, {self.state_dim + 1}) but is {out.shape}"
        next_state = out[:, :-1]
        reward = out[:, -1]
        return next_state, reward

class Planner(myModule):
    def __init__(self, 
                state_dim, goal_dim, hypo_hidden_dim, output_dim, 
                hidden_size=128, num_layers=2, norm=False, feed_state=False):
        """
        Planner input: Current State (S) + Goal (G) + Hypo Hidden (V)
        """
        super().__init__()
        self.input_size = state_dim + goal_dim + hypo_hidden_dim
        
        self.rnn = nn.ModuleList()
        self.rnn.append(GRULayer(self.input_size, hidden_size, norm=norm))
        for _ in range(num_layers - 1):
            self.rnn.append(GRULayer(hidden_size, hidden_size, norm=norm))
            
        self.head = nn.Linear(hidden_size, output_dim)

        self.feed_state = feed_state
        
    def forward(self, model_input, hypo_hidden, hidden=None):
        """
        state: (B, S)
        goal: (B, G)
        hypo_hidden: (B, V) -> This comes from Hypothesizer's last hidden state
        """
        
        B = model_input.size(0)
        inp = torch.cat([model_input, hypo_hidden], dim=1)
        inp = inp.unsqueeze(1) # (B, 1, Inp)
        
        if hidden is None:
            hidden = [None] * len(self.rnn)
            
        new_hiddens = []
        x = inp
        for i, layer in enumerate(self.rnn):
            out, h = layer(x, hidden[i])
            x = out
            new_hiddens.append(h)
            
        logits = self.head(x.squeeze(1))
        return logits, new_hiddens


class MeldedHypothesizerPlanner(myModule):
    def __init__(
                    self, 
                    M, state_dim, goal_dim, 
                    action_dims, action_embedding_dim, 
                    hidden_size=128, num_layers=2, norm=False,
                    include_state_ins_for_actions=False,
                    permute_hypotheses=False
                ):
        """
        Melded model that does both Hypothesizing and Planning.
        Inputs: M * (Reward + Embed(Action) + Mode) + G + M*S
        Mode: Scalar (-1 for Hypothesizing, 1 for Planning)
        
        Outputs: 
            Logits: M * TotalActionDim
            Value: M * 1 (Planner Value)
        """
        super().__init__()
        self.M = M
        self.action_dims = action_dims
        self.action_embedding_dim = action_embedding_dim
        self.total_action_dim = sum(action_dims)
        
        # Embeddings for each factor
        self.action_embeddings = nn.ModuleList([
            nn.Linear(dim, self.action_embedding_dim[i], bias=False) 
            for i, dim in enumerate(action_dims)
        ])
        
        # Input: M * (1 scalar value + (num_factors * embedding_dim)) + G + M*S + 1 (Mode)
        total_embed_size = sum(self.action_embedding_dim)
        self.input_size = M * (1 + total_embed_size) + goal_dim + M * state_dim + 1

        self.state_dim = state_dim
        self.goal_dim = goal_dim
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.ModuleList()
        # First layer
        self.rnn.append(GRULayer(self.input_size, hidden_size, norm=norm))
        # Subsequent layers
        curr_size = hidden_size
        for _ in range(num_layers - 1):
            self.rnn.append(GRULayer(curr_size, hidden_size, norm=norm))
            
        # Heads
        # Action Logits: M * (total_action_dim)
        # self.action_head = nn.Linear(hidden_size, M * self.total_action_dim)
        self.is_factored = len(action_dims) > 1
        self.include_state_ins_for_actions = include_state_ins_for_actions or permute_hypotheses
        self.permute_hypotheses = permute_hypotheses

        
        if self.permute_hypotheses:
            frontier_inp_size = hidden_size + self.M * self.state_dim + self.goal_dim
            self.frontier_head = nn.Sequential(
                nn.Linear(frontier_inp_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size * 2),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size * 2),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, M * M)
            ) # to choose which among the last M nodes you want to take actions against
        if not self.is_factored:
            # Standard Mode (Single Action Head)
            if self.permute_hypotheses:
                inp_size = hidden_size + self.M * self.state_dim + self.goal_dim
                self.action_head = nn.Sequential(
                    nn.Linear(inp_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size * 2),
                    nn.ReLU(),
                    nn.Linear(hidden_size * 2, hidden_size * 2),
                    nn.ReLU(),
                    nn.Linear(hidden_size * 2, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, M * self.total_action_dim)
                )
            else:
                inp_size = hidden_size
                self.action_head = nn.Linear(inp_size, M * self.total_action_dim)
        else:
            # Conditional
            # 1. Entity Head: Hidden -> M * 4 (entities)
            # 2. Relation MLP: (Hidden + M * 4) -> Hidden -> M * 17 (relations)
            if self.permute_hypotheses:
                inp_size = hidden_size + self.M * self.state_dim + self.goal_dim
                self.entity_head = nn.Sequential(
                    nn.Linear(inp_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size * 2),
                    nn.ReLU(),
                    nn.Linear(hidden_size * 2, hidden_size * 2),
                    nn.ReLU(),
                    nn.Linear(hidden_size * 2, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, M * self.action_dims[0])
                )
            else:
                inp_size = hidden_size
                self.entity_head = nn.Linear(hidden_size, self.M * self.action_dims[0])
            
            # Relation MLP input: Hidden + M * 4 (OneHot Entities)
            inp_dim = hidden_size + self.M * self.action_dims[0]
            if self.include_state_ins_for_actions:
                inp_dim += self.M * self.state_dim + self.goal_dim
                
            self.relation_mlp = nn.Sequential(
                nn.Linear(inp_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size * 2),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size * 2),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.M * self.action_dims[1])
            )

        # Value Head: M * 1
        self.value_head = nn.Linear(hidden_size, M)


    def forward(
            self, 
            prev_actions_one_hot, prev_rewards, mode_tensor,
            current_states, goals, 
            hidden=None
        ):
        """
        prev_actions_one_hot: (B, M, SumOfActionDims)
        prev_rewards: (B, M)
        mode_tensor: (B, 1) -> 0 or 1
        current_states: (B, M, S)
        goals: (B, G)
        ---------------------------------------------
        Roughly this function:
        1. embeds previous actions 
        2. squeezes tensors into 2D ones
        3. passes them through the rnn
        4. last hidden state is passed through the critic
        
        Returns:
        1. rnn last layer output
        2. value head output
        3. rnn all hidden layers output
        """
        B = goals.size(0)

        #shape checking
        assert prev_actions_one_hot.shape == (B, self.M, sum(self.action_dims)), f"prev_actions_one_hot should be (B, M, SumOfActionDims) but is {prev_actions_one_hot.shape}"  
        assert prev_rewards.shape == (B, self.M), f"prev_rewards should be (B, M) but is {prev_rewards.shape}"
        assert mode_tensor.shape == (B, 1), f"mode_tensor should be (B, 1) but is {mode_tensor.shape}"
        assert current_states.shape == (B, self.M, self.state_dim), f"current_states should be (B, M, S) but is {current_states.shape}"
        assert goals.shape == (B, self.goal_dim), f"goals should be (B, G) but is {goals.shape}"

        assert torch.unique(mode_tensor).numel() <= 2, f"mode_tensor should be -1 or 1 but is {torch.unique(mode_tensor)}"
        assert torch.unique(mode_tensor).min() in [0, 1] and torch.unique(mode_tensor).max() in [0, 1], f"mode_tensor should be 0 or 1 but is {torch.unique(mode_tensor)}"


        # Embed previous actions
        act_embeds_list = []
        start_idx = 0
        for i, dim in enumerate(self.action_dims):
            end_idx = start_idx + dim
            # factor_one_hot: (B, M, dim)
            factor_one_hot = prev_actions_one_hot[..., start_idx:end_idx]
            #check that it is indeed one hot
            assert torch.all(factor_one_hot.sum(dim=-1) == 1) or torch.all(factor_one_hot.sum(dim=-1) == 0), f"factor_one_hot should be one hot but is {factor_one_hot.sum(dim=-1)}"
            embed = self.action_embeddings[i](factor_one_hot)
            act_embeds_list.append(embed)
            start_idx = end_idx
            
        # (B, M, num_factors*emb_dim)
        act_embeds = torch.cat(act_embeds_list, dim=-1)
        
        assert act_embeds.shape == (B, self.M, sum(self.action_embedding_dim)), f"act_embeds should be (B, M, num_factors*emb_dim) but is {act_embeds.shape}"
            
        act_flat = act_embeds.reshape(B, -1) # (B, M*TotalEmbed)
        rew_flat = prev_rewards.reshape(B, -1) # (B, M)
        
        states_flat = current_states.reshape(B, -1) # (B, M*S)
        
        # Concatenate: (B, M*TotalEmbed + M + 1 + G + M*S)
        rnn_input = torch.cat([act_flat, rew_flat, mode_tensor, goals, states_flat], dim=1)
        assert rnn_input.shape == (B, self.input_size), f"rnn_input should be (B, InputSize) but is {rnn_input.shape}"
        
        rnn_input = rnn_input.unsqueeze(1) # (B, 1, InputSize)
        
        new_hiddens = []
        x = rnn_input
        
        if hidden is None:
            hidden = [None] * len(self.rnn)
            
        for i, layer in enumerate(self.rnn):
            out, h = layer(x, hidden[i])
            x = out
            new_hiddens.append(h)
            
        rnn_out = x.squeeze(1) # (B, hidden)
        
        values = self.value_head(rnn_out) # (B, M)
        assert values.shape == (B, self.M), f"values should be (B, M) but is {values.shape}"
        
        return rnn_out, values, new_hiddens

    def get_standard_logits(self, rnn_out, states, goals):
        """
        Standard mode: rnn_out -> logits matched to single action dim.
        """
        assert not self.is_factored, "Called get_standard_logits but model is in Factored mode"
        B = rnn_out.shape[0]
        
        if len(states.shape) == 2:
            states = states.unsqueeze(1).expand(-1, self.M, -1)
        
        states_flat = states.reshape(B, -1)
        
        if self.permute_hypotheses:
            inp = torch.cat([rnn_out, states_flat, goals], dim=1)
        else: 
            inp = rnn_out
        logits_flat = self.action_head(inp)
        return logits_flat.view(B, self.M, self.total_action_dim)

    def get_entity_logits(self, rnn_out, states, goals):
        """
        rnn_out: (B, Hidden)
        states: (B, M, S)
        goals: (B, G)
        returns:
        (B, M, 4) tensor of probabilities over entities (4 here)
        """
        assert self.is_factored, "Called get_entity_logits but model is in Standard mode"
        B = rnn_out.shape[0]
        if len(states.shape) == 2:
            states = states.unsqueeze(1).expand(-1, self.M, -1)
        
        states_flat = states.reshape(B, -1)
        
        if self.permute_hypotheses:
            inp = torch.cat([rnn_out, states_flat, goals], dim=1)
        else:
            inp = rnn_out
        
        logits_flat = self.entity_head(inp) # (B, M*4)
        return logits_flat.view(B, self.M, self.action_dims[0])

    def get_relation_logits(self, rnn_out, states, goals, chosen_entities):
        """
        rnn_out: (B, Hidden)
        states: (B, M, S)
        goals: (B, G)
        chosen_entities: (B, M, 4) One-Hot tensors
        Returns: (B, M, 17)
        """
        assert self.is_factored, "Called get_relation_logits but model is in Standard mode"
        B = rnn_out.shape[0]
        if len(chosen_entities.shape) == 2:
            chosen_entities = chosen_entities.unsqueeze(1).expand(-1, self.M, -1)
        
        if len(states.shape) == 2:
            states = states.unsqueeze(1).expand(-1, self.M, -1)
            
        entities_flat = chosen_entities.reshape(B, -1) # (B, M*4)
        states_flat = states.reshape(B, -1)
        
        inp = torch.cat([rnn_out, goals, states_flat, entities_flat], dim=1)
        logits_flat = self.relation_mlp(inp)
        return logits_flat.view(B, self.M, self.action_dims[1])

    def get_frontier_logits(self, rnn_out, states, goals):
        """
        rnn_out: (B, Hidden)
        states: (B, M, S)
        goals: (B, G)
        Returns: (B, M, M) - M distributions over M previous hypotheses
        """
        
        if self.permute_hypotheses:
            B = rnn_out.shape[0]

            if len(states.shape) == 2:
                states = states.unsqueeze(1).expand(-1, self.M, -1)

            states_flat = states.reshape(B, -1)
            inp = torch.cat([rnn_out, states_flat, goals], dim=1)
            logits_flat = self.frontier_head(inp) # (B, M*M)
            return logits_flat.view(B, self.M, self.M)
        else:
            #ret diag
            return torch.eye(self.M, device=rnn_out.device).expand(B, self.M, self.M)



class Orchestrator(myModule):

    def __init__(
                self, 
                env,
                state_dim, goal_dim, 
                action_dims, action_embedding_dim, 
                hypo_config: HypoConfig,
                planner_config: PlannerConfig,
                world_model_config: WorldModelConfig,
                hypothesize_always: bool = False,
                use_melded_mode: bool = False,
                disable_hypo: bool = False,
                aux_task: bool = False,
                device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            ):
        super().__init__() 
        self.env = env
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dims = action_dims
        self.action_embedding_dim = action_embedding_dim
        self.total_action_dim = sum(action_dims)
        self.use_melded_mode = use_melded_mode
        self.hypo_config = hypo_config
        self.disable_hypo = disable_hypo
        self.aux_task = aux_task
        
        # Models
        if use_melded_mode:
            self.melded_model = MeldedHypothesizerPlanner(
                M=hypo_config.num_hypotheses,
                state_dim=state_dim,
                goal_dim=goal_dim,
                action_dims=action_dims,
                action_embedding_dim=action_embedding_dim,
                hidden_size=hypo_config.hidden_size,
                num_layers=hypo_config.num_layers,
                norm=hypo_config.norm,
                include_state_ins_for_actions=planner_config.include_state_ins_for_actions,
                permute_hypotheses=hypo_config.permute_hypotheses
            )
            self.hypothesizer = None
            self.planner = None
            self.melded_model.to(device)
        else:
            self.hypothesizer = Hypothesizer(
                M=hypo_config.num_hypotheses,
                state_dim=state_dim,
                goal_dim=goal_dim,
                action_dims=action_dims,
                action_embedding_dim=action_embedding_dim,
                hidden_size=hypo_config.hidden_size,
                num_layers=hypo_config.num_layers,
                norm=hypo_config.norm,
                include_state_ins_for_actions=planner_config.include_state_ins_for_actions
            )
            # Planner outputs logits (TotalActionDim) + Value (1)
            self.planner = Planner(
                state_dim=state_dim,
                goal_dim=goal_dim,
                hypo_hidden_dim=hypo_config.hidden_size,
                output_dim=self.total_action_dim + 1,
                hidden_size=planner_config.hidden_size,
                num_layers=planner_config.num_layers,
                norm=planner_config.norm,
                feed_state=planner_config.feed_state,
                include_state_ins_for_actions=planner_config.include_state_ins_for_actions
            )
            self.hypothesizer.to(device)
            self.planner.to(device)
            
        if self.aux_task: # are we training with the question task?
            # Question vector dim: matches action dim
            q_dim = sum(action_dims)
            
            if not disable_hypo:
                input_dim = hypo_config.hidden_size + q_dim
            else:
                input_dim = planner_config.hidden_size + q_dim
                
            self.aux_model = AuxiliaryNetwork(input_dim)
            self.aux_model.to(device)
        else:
            self.aux_model = None

        self.world_model = WorldModel(
            state_dim=state_dim,
            action_dims=action_dims,
            action_embedding_dim=action_embedding_dim,
            goal_dim=goal_dim,
            hidden_size=world_model_config.hidden_size,
            dropout=world_model_config.dropout,
            norm=world_model_config.norm
        )
        self.world_model.to(device)
        self.wm_optimizer = optim.Adam(self.world_model.parameters(), lr=world_model_config.wm_lr, weight_decay=world_model_config.wm_weight_decay)
        self.decouple_wm = world_model_config.decouple_wm
        self.M = hypo_config.num_hypotheses
        self.beta_diversity = hypo_config.beta_diversity
        self.is_training = False
        self.hypothesize_always = hypothesize_always
        self.permute_hypotheses = hypo_config.permute_hypotheses
        
    def policy_parameters(self):
        """
        Returns parameters for the policy optimization (excluding World Model).
        """
        if self.use_melded_mode:
            params = list(self.melded_model.parameters())
            if self.aux_model is not None:
                params += list(self.aux_model.parameters())
            return params
        else:
            # Union of planner and hypothesizer parameters
            return list(self.planner.parameters()) + (list(self.hypothesizer.parameters()) if not self.disable_hypo else []) + (list(self.aux_model.parameters()) if self.aux_model is not None else [])

    def planner_sample_norm(self, logits):
        """
        Take the mean of the probabilities of action over distributions. 
        Return log softmax of this mean dist. 
        """
        # logits: (B, M, A)
        assert len(logits.shape) == 3, f"logits should be (B, M, A) but is {logits.shape}"
        return torch.log(F.softmax(logits, dim=-1).mean(dim=1) + 1e-10)
    
    def sample_subroutine(self, logits, planner=False, gumbel=False):
        # logits: (B, M, A)
        assert len(logits.shape) == 3, f"logits should be (B, M, A) but is {logits.shape}"
        if planner:
            action_logits = self.planner_sample_norm(logits)
            if not gumbel:
                action_indices = torch.multinomial(F.softmax(action_logits, dim=-1), num_samples=1).squeeze(-1)
                action_one_hot = F.one_hot(action_indices, num_classes=action_logits.shape[-1]).to(dtype=action_logits.dtype)
            else:
                action_one_hot = F.gumbel_softmax(action_logits, tau=1, hard=True)
                action_indices = torch.argmax(action_one_hot, dim=-1)
        else:
            action_one_hot = F.gumbel_softmax(logits, tau=1, hard=True)
            action_indices = torch.argmax(action_one_hot, dim=-1)

        return action_one_hot, action_indices, action_logits if planner else logits

    
    def sample_action(self, rnn_out, states, goals, planner=True, gumbel=True):
        """
        Handles sampling for Arbitrary Factorized actions.
        rnn_out: (..., hidden_size)
        Returns:
            one_hot_concatenated: (..., total_action_dim)
            indices_list: List of (...,) tensors, one for each factor.
            parent_indices: (B, M) if not planner, else None
        """
        parent_indices = None

        if len(states.shape) == 2:
            states = states.unsqueeze(1).expand(-1, self.M, -1)
        
        # If not planner, we first sample parents
        if not planner and self.permute_hypotheses:
            # 1. Frontier Sampling
            frontier_logits = self.melded_model.get_frontier_logits(rnn_out, states, goals) # (B, M, M)
            parent_one_hot, parent_indices, _ = self.sample_subroutine(frontier_logits, planner=False, gumbel=True)
            # parent_one_hot: (B, M, M) -> for each of M new slots, one-hot vector over M old slots.
            
            # 2. Permute inputs for Action Sampling Differentiably
            # states: (B, M, S)

            # (B, M, M) x (B, M, S) -> (B, M, S)
            states_permuted = torch.bmm(parent_one_hot, states)
            
        else:
            states_permuted = states
            parent_one_hot = None
            
        
        if len(self.action_dims) > 1:
            assert len(self.action_dims) == 2
            entity_logits = self.melded_model.get_entity_logits(
                rnn_out,
                states_permuted,
                goals
            )
            entity_one_hot, entity_indices, new_entity_logits = self.sample_subroutine(entity_logits, planner=planner, gumbel=True)
            relation_logits = self.melded_model.get_relation_logits(rnn_out, states_permuted, goals, entity_one_hot)
            relation_one_hot, relation_indices, new_relation_logits = self.sample_subroutine(relation_logits, planner=planner, gumbel=True)

            one_hot_concatenated = torch.cat([entity_one_hot, relation_one_hot], dim=-1)
            indices_list = [entity_indices, relation_indices]

            if planner:
                new_logits_concatenated = torch.cat([new_entity_logits, new_relation_logits], dim=-1)
            else:
                new_logits_concatenated = torch.cat([entity_logits, relation_logits], dim=-1) 

            return one_hot_concatenated, indices_list, new_logits_concatenated, parent_one_hot
        else:
            policy_logits = self.melded_model.get_standard_logits(
                rnn_out,
                states_permuted,
                goals
            )
            policy_one_hot, policy_indices, new_policy_logits = self.sample_subroutine(policy_logits, planner=planner, gumbel=True)

            return policy_one_hot, [policy_indices], (new_policy_logits if planner else policy_logits), parent_one_hot

    def diversity_loss(self, logits, in_states_actions, hypo_mask=None):
        """
        logits: (B, M, A)
        mask: (B,1)
        Computes average KL divergence across all factors.
        """

        assert logits.shape[-1] == self.total_action_dim, f"logits should have last dim {self.total_action_dim} but is {logits.shape[-1]}"
        assert len(logits.shape) == 3, f"logits should be (B, M, {self.total_action_dim}) but is {logits.shape}"
        assert hypo_mask is None or hypo_mask.shape == (logits.shape[0], 1), f"hypo_mask should be (B,) but is {hypo_mask.shape}"
        assert len(in_states_actions.shape) == 3, f"in_states_actions should be (B, M, {144+self.total_action_dim}) but is {in_states_actions.shape}"

        B = logits.shape[0]
        M = logits.shape[1]

        if M == 1:
            return 0
        
        total_kl = 0
        current_idx = 0

        #first calculate, for each batch the difference between the in_state_actions in across hypotheses
        inp_diff = (in_states_actions.unsqueeze(1) - in_states_actions.unsqueeze(2)).pow(2).sum(-1) # (B, M, M)
        
        for dim in self.action_dims:
            end_idx = current_idx + dim
            factor_logits = logits[..., current_idx:end_idx] # (B, M, dim)

            log_probs = F.log_softmax(factor_logits, dim=-1)
            # Compute pairwise KL divergence
            log_p1 = log_probs.unsqueeze(2)  # (B, M, 1, dim)
            log_p2 = log_probs.unsqueeze(1)  # (B, 1, M, dim)

            kl_matrix = F.kl_div(log_p2, log_p1, reduction='none', log_target=True).sum(-1) #* inp_diff  # (B, M, M)
            
            if hypo_mask is not None:
                # Mask out invalid pairs
                mask_matrix = hypo_mask.unsqueeze(2) * hypo_mask.unsqueeze(1)  # (B, 1, 1)
                diagonal_mask = ~torch.eye(M, device=kl_matrix.device, dtype=torch.bool).unsqueeze(0)  # (1, M, M)
                mask_matrix = mask_matrix * diagonal_mask  # (B, M, M)
                kl_matrix = kl_matrix * mask_matrix
                disagreement = kl_matrix.sum() / mask_matrix.sum()
            else:
                # Average over all pairs (excluding diagonal)
                mask = ~torch.eye(M, device=kl_matrix.device, dtype=torch.bool)
                disagreement = kl_matrix[:, mask].mean()

            assert disagreement.shape == (), f"disagreement should be () but is {disagreement.shape}"
            total_kl += disagreement
            current_idx = end_idx   
        # its a diversity bonus so neg.    
        return -total_kl / len(self.action_dims)

    def compute_aux_logits(self, hidden_state, questions):
        """
        hidden_state: 
            - If melded: (B, Hidden)
            - If Standard+Hypo: (B, M, Hidden) -> will flatten
            - If Standard+NoHypo: (B, Hidden)
        questions: (B, 4, 56)
        """
        if self.aux_model is None:
            return None
            
        B = questions.size(0)
        flat_hidden = hidden_state
             
        # Expand hidden to match 4 questions per batch
        # flat_hidden: (B, H_flat) -> (B, 4, H_flat)
        flat_hidden_expanded = flat_hidden.unsqueeze(1).expand(-1, 4, -1)
        
        # Concat: (B, 4, H_flat + qdim)
        inp = torch.cat([flat_hidden_expanded, questions], dim=-1)
        
        # Forward pass
        # (B, 4, 1)
        logits = self.aux_model(inp)
        
        return logits.squeeze(-1) # (B, 4)

    def run_hypothesizer_loop(
            self, 
            init_states, 
            goal_tensor, 
            h_t=None
        ):
        """
        Runs the hypothesizer loop using the world model.
        Returns:
            h_final: (B, Hidden)
            total_div_loss: scalar tensor
            wm_loss: scalar tensor
        """
        raise NotImplementedError
        B = init_states.size(0)
        M = self.hypo_config.num_hypotheses
        device = self.device()
        
        curr_states = init_states.unsqueeze(1).expand(B, M, -1)
        prev_actions_one_hot = torch.zeros(B, M, self.total_action_dim, device=device)
        prev_rewards = torch.zeros(B, M, device=device)
        
        total_div_loss = 0
        total_wm_loss = 0
        
        expanded_goal_grids = []
        for grid in self.env.goal_grids_batch:
            for _ in range(M):
                expanded_goal_grids.append(grid)
        original_goal_grids = expanded_goal_grids

        state_mode = self.env.state_repr_mode
        goal_mode = self.env.goal_repr_mode

        for step in range(12):
            policy_logits, h_new = self.hypothesizer(
                prev_actions_one_hot, 
                prev_rewards, 
                curr_states, 
                goal_tensor, 
                h_t
            )
            h_t = h_new
            logits = policy_logits
            
            div_loss = self.diversity_loss(logits, torch.cat(
                [curr_states, prev_actions_one_hot], dim=-1
            ))
            total_div_loss += div_loss
            
            # Sampling
            logits_flat = logits.reshape(B*M, -1)
            one_hot_act, indices_list = self.sample_action(logits_flat)
            
            # World Model
            states_flat = curr_states.reshape(B*M, -1)
            goal_flat = goal_tensor.unsqueeze(1).expand(B, M, -1).reshape(B*M, -1)
            
            # Get Ground Truth using Algorithmic Step
            with torch.no_grad():
                # We need to flatten the indices list for algorithmic step -> from (B, M) to (B*M)
                flat_indices_list = [idx.view(-1) for idx in indices_list]
                
                gt_next_struct, gt_rewards, gt_dones = algorithmic_step(    
                   states_flat, 
                   flat_indices_list,
                   state_decode_mode=state_mode,
                   original_goal_grids=original_goal_grids,
                   close_rewards=True,#self.env.close_rewards
                )
            
            # Prediction
            pred_next_struct, pred_rewards = self.world_model(
                                                    states_flat, 
                                                    one_hot_act.detach() if self.decouple_wm else one_hot_act, 
                                                    goal_flat
                                                )
            
            # Compute Loss on the side
            wm_loss_val = F.mse_loss(pred_next_struct, gt_next_struct) + F.mse_loss(pred_rewards, gt_rewards)
            
            if self.is_training:
                self.wm_optimizer.zero_grad()
                wm_loss_val.backward()
                self.wm_optimizer.step()
            
            total_wm_loss += wm_loss_val.detach()
            
            curr_states = pred_next_struct.view(B, M, -1)
            prev_rewards = pred_rewards.view(B, M)
            prev_actions_one_hot = one_hot_act.view(B, M, -1)
            
            if self.decouple_wm:
                curr_states = curr_states.detach()
                prev_rewards = prev_rewards.detach()     
        #TODO: it seems i drive the hypothesizer for max steps - no early stopping conditions
        return h_t[-1], total_div_loss, total_wm_loss

    def compute_wm_loss(
            self, 
            curr_states, 
            one_hot_act, 
            goal_tensor, 
            indices_list, 
            incl_mask,
            profiler=None
        ):

        # shape of incl_mask is (B, 1)
        # World Model Step
        B = curr_states.size(0)
        M = curr_states.size(1)

        states_flat = curr_states.reshape(B*M, -1)
        one_hot_flat = one_hot_act.view(B*M, -1) # Use broadcasted actions
        goal_flat = goal_tensor.unsqueeze(1).expand(B, M, -1).reshape(B*M, -1)

        expanded_goal_grids = []
        for grid in self.env.goal_grids_batch:
            expanded_goal_grids.extend([grid] * M)
        original_goal_grids = expanded_goal_grids

        if profiler: profiler.start("hypo_alg_step")
        with torch.no_grad():
            flat_indices_list = [idx.view(-1) for idx in indices_list]
            gt_next_struct, gt_rewards, gt_dones = algorithmic_step(    
                states_flat, 
                flat_indices_list,
                state_decode_mode=self.env.state_repr_mode,
                original_goal_grids=original_goal_grids,
                close_rewards=self.env.close_rewards
            )
        if profiler: profiler.stop("hypo_alg_step")
        
        if profiler: profiler.start("hypo_wm_fwd")
        pred_next_struct, pred_rewards = self.world_model(
                                                states_flat, 
                                                one_hot_flat.detach() if self.decouple_wm else one_hot_flat, 
                                                goal_flat
                                            )
        if profiler: profiler.stop("hypo_wm_fwd")

        if profiler: profiler.start("hypo_wm_loss_calc")
        state_loss_val = F.mse_loss(pred_next_struct, gt_next_struct, reduction='none')
        reward_loss_val = F.mse_loss(pred_rewards, gt_rewards, reduction='none')
        
        state_loss_val = state_loss_val.view(B, M, -1).mean(dim=2).mean(dim=1, keepdim=True)
        reward_loss_val = reward_loss_val.view(B, M).mean(dim=1, keepdim=True)

        wm_loss_val = state_loss_val + reward_loss_val
        if profiler: profiler.stop("hypo_wm_loss_calc")
        wm_loss_val = (wm_loss_val * incl_mask).sum() / incl_mask.sum()

        # it turns out this may not be very worth - because of model collapse in the hypothesizer predicting bad actions. Temporarily disabling it for now. 
        # if self.is_training:
        #     self.wm_optimizer.zero_grad()
        #     wm_loss_val.backward()
        #     self.wm_optimizer.step()
        return wm_loss_val, gt_next_struct, gt_rewards
    
    def run_melded_loop(
            self, 
            in_states, 
            goal_tensor, 
            rewards=None,
            actions=None,
            h_t=None,
            profiler=None
        ):
        """
        Runs the Melded Hypothesizer-Planner loop.
        """
        B = in_states.size(0)
        M = self.hypo_config.num_hypotheses
        device = self.device()

        assert in_states.shape == (B, self.state_dim), f"in_states should be (B, {self.state_dim}) but is {in_states.shape}"
        assert goal_tensor.shape == (B, self.goal_dim), f"goal_tensor should be (B, {self.goal_dim}) but is {goal_tensor.shape}"
        
        if h_t is None:
            assert rewards is None and actions is None, "rewards and actions must be None if h_t is None"

            curr_states = in_states.unsqueeze(1).expand(B, M, -1)
            prev_actions_one_hot = torch.zeros(B, M, self.total_action_dim, device=device)
            prev_rewards = torch.zeros(B, M, device=device)
        
            # Start in hypothesize mode (1)
            mode_tensor = torch.ones((B, 1), device=device)
        else:
            assert rewards.shape == (B, 1), f"rewards should be (B, 1) but is {rewards.shape}"
            assert actions.shape == (B, self.total_action_dim), f"actions should be ({B}, {self.total_action_dim}) but is {actions.shape}"

            curr_states = in_states.unsqueeze(1).expand(B, M, -1)
            prev_actions_one_hot = actions.unsqueeze(1).expand(B, M, -1)
            prev_rewards = rewards.expand(B, M)
            mode_tensor = torch.zeros((B, 1), device=device)

        if h_t is None and not self.disable_hypo:
            if profiler: profiler.start("hypothesizing")
            h_t, total_div_loss, total_wm_loss = self.melded_hypothesize(
                prev_actions_one_hot,
                prev_rewards,
                mode_tensor,
                curr_states,
                goal_tensor,
                h_t,
                profiler=profiler
            )
            if profiler: profiler.stop("hypothesizing")
        else:
            total_div_loss, total_wm_loss = 0, 0

        
        # melded mode always feeds state.
        if profiler: profiler.start("planning_phase")
        rnn_out, values, new_h_t = self.melded_model(
            prev_actions_one_hot,
            prev_rewards,
            mode_tensor,
            curr_states,
            goal_tensor,
            h_t
        )
        if profiler: profiler.stop("planning_phase")

        # Return rnn_out instead of logits, so training loop can do split sampling
        return rnn_out, values, new_h_t, total_div_loss, total_wm_loss

    def melded_hypothesize(
            self, prev_actions_one_hot, 
            prev_rewards, mode_tensor, 
            curr_states, goal_tensor, 
            h_t = None,
            profiler=None
        ):

        B = prev_actions_one_hot.size(0)
        device = self.device()
        M = self.hypo_config.num_hypotheses
        
        total_div_loss = 0
        total_wm_loss = 0
        
        final_hiddens = [torch.zeros(
            (B, self.hypo_config.hidden_size), device=device
        ) for _ in range(len(self.melded_model.rnn))]

        for step in range(10):
            
            if profiler: profiler.start("hypo_nn")
            rnn_out, values, h_new = self.melded_model(
                prev_actions_one_hot,
                prev_rewards,
                mode_tensor,
                curr_states,
                goal_tensor,
                h_t
            )
            if profiler: profiler.stop("hypo_nn")

            # Sampling / Action Selection
            # Hyp Mode: Sample independently
            # Plan Mode: Average and sample
            if profiler: profiler.start("hypo_sample")
            # Note: sample_action now returns parent_one_hot for Permutation
            one_hot_actions, action_indices_list, policy_logits, parent_one_hot = self.sample_action(
                rnn_out,
                curr_states,
                goal_tensor,
                planner=False
            )
            if profiler: profiler.stop("hypo_sample")
            
            # Use parent_one_hot to permute everything diffrentiably
            if parent_one_hot is not None:
                # Permute curr_states: (B, M, M) x (B, M, S) -> (B, M, S)
                curr_states_permuted = torch.bmm(parent_one_hot, curr_states)
                
                # Permute prev_actions_one_hot: (B, M, M) x (B, M, A) -> (B, M, A)
                prev_actions_permuted = torch.bmm(parent_one_hot, prev_actions_one_hot)
                
            else:
                curr_states_permuted = curr_states
                prev_actions_permuted = prev_actions_one_hot

            h_t = h_new # h_t is (B, Hidden), no permutation needed.
            
            # Sample Gumbel for all
            assert policy_logits.shape == (B, M, self.total_action_dim), f"policy_logits should be (B, M, {self.total_action_dim}) but is {policy_logits.shape}"
            assert one_hot_actions.shape == (B, M, self.total_action_dim), f"one_hot_act should be (B, M, {self.total_action_dim}) but is {one_hot_actions.shape}"
            
            # Div Loss
            if profiler: profiler.start("hypo_div")
            div_loss = self.diversity_loss(policy_logits, torch.cat(
                [
                    curr_states_permuted,
                    prev_actions_permuted,
                ], dim=-1
            ), mode_tensor)
            total_div_loss += div_loss
            if profiler: profiler.stop("hypo_div")
            
            if profiler: profiler.start("hypo_wm_step")
            # wm_loss uses curr_states_permuted
            wm_loss_val, gt_next_struct, gt_rewards = self.compute_wm_loss(curr_states_permuted, one_hot_actions, goal_tensor, action_indices_list, mode_tensor, profiler=profiler)
            total_wm_loss += wm_loss_val.detach() 
            if profiler: profiler.stop("hypo_wm_step")
            
            curr_states = gt_next_struct.view(B, M, -1)
            prev_rewards = gt_rewards.view(B, M)
            prev_actions_one_hot = one_hot_actions
            
            if self.decouple_wm:
                # TODO:
                # right now this doesnt do anything, since we are using the ground truths
                # these dont have gradients anyway
                # but we should make it so that wm predictions are what is provided to next step in hypothesizing
                curr_states = curr_states.detach()
                prev_rewards = prev_rewards.detach()
                
            # Update Mode for NEXT step with hard switch logic
            # Strategy: If ANY hypothesis gets reward > 0.99 (==1), switch to planning (0).
            
            any_goal_reached = (prev_rewards > 0.99).any(dim=1, keepdim=True).float() # (B, 1)
            should_switch = (any_goal_reached > 0.5) & (mode_tensor > 0.5) # (B, 1) boolean
            
            switching_indices = should_switch.squeeze(1).nonzero(as_tuple=True)[0]
            
            if len(switching_indices) > 0:
                 for i in range(len(self.melded_model.rnn)):
                     final_hiddens[i][switching_indices] = h_new[i][switching_indices]
                     
                 # Update mode to 0
                 mode_tensor[switching_indices] = 0.0
                 
            # If everyone has switched to planning (mode 0), we can break early.
            if (mode_tensor < 0.5).all():
                break

        # For any that didn't switch by end of loop (reached 12 steps), their final hidden is the last one.
        still_hypothesizing = (mode_tensor > 0.5).squeeze(1)
        if still_hypothesizing.any():
            still_hypothesizing = still_hypothesizing.nonzero(as_tuple=True)[0]
            # They ran out of time, force switch to planning for the return
            # Capture their last hidden state
            for i in range(len(self.melded_model.rnn)):
                final_hiddens[i][still_hypothesizing] = h_new[i][still_hypothesizing]
            
            mode_tensor[still_hypothesizing] = 0.0
            
        return final_hiddens, total_div_loss, total_wm_loss


    def set_train(self):
        self.is_training = True

    def set_eval(self):
        self.is_training = False

    def melded_forward(
        self, 
        curr_states,
        curr_goals, 
        rewards=None,
        actions=None,
        hidden=None,
        profiler=None
    ):
        B = curr_goals.size(0)
        # Melded Logic
        (
            rnn_out, values, h_final, 
            div_loss, wm_loss
        ) = self.run_melded_loop(
            curr_states,
            curr_goals, 
            rewards,
            actions,
            hidden,
            profiler=profiler
        )

        mean_vals = values.mean(dim=1) # (B,)
        
        return (
            rnn_out, 
            mean_vals,
            h_final, # Planner hidden equivalent
            h_final, # Hypothesizer hidden equivalent
            div_loss,
            wm_loss
        )

    
    def forward(
            self, 
            curr_states,
            curr_goals, 
            rewards=None,
            actions=None,
            planner_hidden=None, 
            hypothesizer_state=None,
            profiler=None
        ):
        
        if self.use_melded_mode:
            return self.melded_forward(
                curr_states,
                curr_goals, 
                rewards,
                actions,
                hypothesizer_state,
                profiler=profiler
            )
        
        raise NotImplementedError
        if hypothesizer_state is None or self.hypothesize_always:# Run Hypothesizer Loop
            h_final, div_loss, wm_loss = self.run_hypothesizer_loop(curr_states, curr_goals)
        else:
            h_final = hypothesizer_state
            div_loss = 0
            wm_loss = 0

        # whether to feed state or not:
        if self.planner.feed_state:
            model_input = torch.cat(
                [
                    curr_states, 
                    curr_goals,
                    rewards,
                    actions
                ],
                dim=-1
            )
        else:
            model_input = torch.cat(
                [
                    curr_states, 
                    curr_goals,
                ],
                dim=-1
            )
        
        plan_out, planner_hidden = self.planner(model_input, curr_goals, h_final)
        logits = plan_out[:, :-1]
        val = plan_out[:, -1]

        return (
            logits, 
            val,
            planner_hidden,
            h_final, 
            div_loss,
            wm_loss
        ) 
    
    def train_world_model_batch(self, 
                                states, 
                                actions_one_hot, 
                                goals, 
                                next_states_target, 
                                rewards_target,
                                ):
        """
        Train World Model on a batch of transition data.
        states: (B, S)
        actions_one_hot: (B, TotalActionDim)
        goals: (B, G)
        next_states_target: (B, S)
        rewards_target: (B,) scalar
        """
        self.world_model.train()  

        B = states.size(0)
        
        assert rewards_target.shape == (B,), f"rewards_target should be ({B},) but is {rewards_target.shape}"
        assert next_states_target.shape == (B, self.state_dim), f"next_states_target should be ({B}, {self.state_dim}) but is {next_states_target.shape}"
        assert actions_one_hot.shape == (B, self.total_action_dim), f"actions_one_hot should be ({B}, {self.total_action_dim}) but is {actions_one_hot.shape}"
        assert goals.shape == (B, self.goal_dim), f"goals should be ({B}, {self.goal_dim}) but is {goals.shape}"

        pred_next_struct, pred_rewards = self.world_model(
                                                states, 
                                                actions_one_hot, 
                                                goals
                                            )
        loss = F.mse_loss(pred_next_struct, next_states_target) + F.mse_loss(pred_rewards, rewards_target)
        
        self.wm_optimizer.zero_grad()
        loss.backward()
        self.wm_optimizer.step()
        
        return loss.item()

    def train(self):
        self.world_model.train()
        if self.use_melded_mode:
            self.melded_model.train()
        else:
            self.planner.train()
            self.hypothesizer.train()
    
    def eval(self):
        self.world_model.eval()
        if self.use_melded_mode:
            self.melded_model.eval()
        else:
            self.planner.eval()
            self.hypothesizer.eval()


### Tree Search section
@dataclass
class TreeSearcherConfig:
    M: int
    state_dim: int
    state_embedding_dim: int
    goal_dim: int
    goal_embedding_dim: int
    action_dims: List[int]
    action_embedding_dim: int
    num_belief_bins: int = 8 # problems have anywhere from 1-6 steps to complete them - the brick task has atmost 4

class TreeSearcher(myModule):
    def __init__(
                    self, 
                    config: TreeSearcherConfig,
                    hidden_size=128, num_layers=2, norm=False,
                    dropout=0.0,
                    include_state_ins_for_actions=False
                ):
        """
        Melded model that does both Hypothesizing and Planning.
        Inputs: M * (Reward + Embed(Action) + Mode) + G + M*S
        Mode: Scalar (-1 for Hypothesizing, 1 for Planning)
        
        Outputs: 
            Logits: M * TotalActionDim
            Value: M * 1 (Planner Value)
            Switch: M * 1 (Logit for switching to Planning)
        """
        super().__init__()
        self.config = config
        self.include_state_ins_for_actions = include_state_ins_for_actions
        
        # embedding heads:
        self._setup_input_projections(hidden_size, norm=norm)

        # setup rnn backbone:
        self._setup_rnn(num_layers, hidden_size, norm)
        
        # Output Heads
        self._setup_policy_heads(hidden_size, dropout=dropout, norm=norm)
        self._setup_other_output_heads(hidden_size, dropout=dropout, norm=norm)

    ## Module setup functions
    def _setup_input_projections(self, hidden_state, norm=False):
        self.state_encoder = nn.Sequential(
            nn.Linear(self.config.state_dim, hidden_state),
            nn.LayerNorm(hidden_state) if norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_state, self.config.state_embedding_dim),
            nn.LayerNorm(self.config.state_embedding_dim) if norm else nn.Identity(),
            nn.ReLU(),
        )

        self.goal_encoder = nn.Sequential(
            nn.Linear(self.config.goal_dim, hidden_state),
            nn.LayerNorm(hidden_state) if norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_state, self.config.goal_embedding_dim),
            nn.LayerNorm(self.config.goal_embedding_dim) if norm else nn.Identity(),
            nn.ReLU(),
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(sum(self.config.action_dims), hidden_state),
            nn.LayerNorm(hidden_state) if norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_state, self.config.action_embedding_dim),
            nn.LayerNorm(self.config.action_embedding_dim) if norm else nn.Identity(),
            nn.ReLU(),
        )

    def _setup_rnn(self, num_layers, hidden_size, norm=False):
        # Input: M * (1 scalar value + 1 scalar return + (num_factors * embedding_dim)) + G + M*S
        input_size = self.config.M * (2 + self.config.action_embedding_dim) + self.config.goal_embedding_dim + self.config.M * self.config.state_embedding_dim

        rnn = nn.ModuleList()
        # First layer
        rnn.append(GRULayer(input_size, hidden_size, norm=norm))
        # Subsequent layers
        curr_size = hidden_size
        for _ in range(num_layers - 1):
            rnn.append(GRULayer(curr_size, hidden_size, norm=norm))

        self.rnn = rnn
        return rnn

    def _setup_policy_heads(self, hidden_size, dropout=0.0, norm=False):
        self._setup_frontier_head(hidden_size, dropout=dropout, norm=norm)
        self._setup_action_heads(hidden_size, dropout=dropout, norm=norm)

    def _setup_frontier_head(self, hidden_size, dropout=0.0, norm=False):
        frontier_inp_size = hidden_size + self.config.M * self.config.state_embedding_dim + self.config.goal_embedding_dim
        self.frontier_head = nn.Sequential(
            nn.Linear(frontier_inp_size, hidden_size),
            nn.LayerNorm(hidden_size) if norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2) if norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2) if norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size) if norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, self.config.M * self.config.M)
        ) # to choose which among the last M nodes you want to take actions against

    def _setup_action_heads(self, hidden_size, dropout=0.0, norm=False):
        if len(self.config.action_dims) == 1:
            inp_size = hidden_size + self.config.M * self.config.state_embedding_dim + self.config.goal_embedding_dim
            self.action_head = nn.Sequential(
                nn.Linear(inp_size, hidden_size),
                nn.LayerNorm(hidden_size) if norm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size * 2),
                nn.LayerNorm(hidden_size * 2) if norm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size * 2),
                nn.LayerNorm(hidden_size * 2) if norm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.LayerNorm(hidden_size) if norm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(hidden_size, self.config.M * self.config.action_dims[0])
            )
        else:
            # Conditional
            # 1. Entity Head: Hidden -> M * 4 (entities)
            # 2. Relation MLP: (Hidden + M * 4) -> Hidden -> M * 17 (relations)
            inp_size = hidden_size + self.config.M * self.config.state_embedding_dim + self.config.goal_embedding_dim
            self.entity_head = nn.Sequential(
                nn.Linear(inp_size, hidden_size),
                nn.LayerNorm(hidden_size) if norm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size * 2),
                nn.LayerNorm(hidden_size * 2) if norm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size * 2),
                nn.LayerNorm(hidden_size * 2) if norm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.LayerNorm(hidden_size) if norm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(hidden_size, self.config.M * self.config.action_dims[0])
            )

            inp_dim = hidden_size + self.config.M * self.config.action_dims[0] + self.config.M * self.config.state_embedding_dim + self.config.goal_embedding_dim
            self.relation_mlp = nn.Sequential(
                nn.Linear(inp_dim, hidden_size),
                nn.LayerNorm(hidden_size) if norm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size * 2),
                nn.LayerNorm(hidden_size * 2) if norm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size * 2),
                nn.LayerNorm(hidden_size * 2) if norm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.LayerNorm(hidden_size) if norm else nn.Identity(),
                nn.ReLU(),
                nn.Linear(hidden_size, self.config.M * self.config.action_dims[1])
            )

    def _setup_other_output_heads(self, hidden_size, dropout=0.0, norm=False):
        value_inp_dim = (
            hidden_size
            + self.config.M * self.config.state_embedding_dim
            + self.config.M * self.config.action_embedding_dim
            + self.config.goal_embedding_dim
        )

        self.value_head = nn.Sequential(
            nn.Linear(value_inp_dim, hidden_size),
            nn.LayerNorm(hidden_size) if norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size) if norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_size, self.config.M)
        )

        self.belief_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size) if norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size) if norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_size, self.config.num_belief_bins)
        )

    ### derived properties
    @property
    def input_size(self):
        return self.rnn[0].input_size

    @property
    def hidden_size(self):
        return self.rnn[0].hidden_size
    
    @property
    def total_action_dim(self):
        return sum(self.config.action_dims)

    @property
    def total_action_embedding_dim(self):
        if isinstance(self.config.action_embedding_dim, int):
            return self.config.action_embedding_dim
        return sum(self.config.action_embedding_dim)

    @property
    def is_factored(self):
        return len(self.config.action_dims) > 1

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            try:
                return getattr(self.config, name)
            except AttributeError:
                raise AttributeError(f"{self} has no attribute {name} nor does the config object")

    def forward(self, 
                prev_actions_one_hot, prev_rewards, prev_values,
                current_states, goals, 
                hidden=None
            ):
        """
        prev_actions_one_hot: (B, M, SumOfActionDims)
        prev_rewards: (B, M)
        prev_values: (B, M)
        current_states: (B, M, S)
        goals: (B, G)
        """
        B = goals.size(0)

        #shape checking
        assert prev_actions_one_hot.shape == (B, self.M, sum(self.action_dims)), f"prev_actions_one_hot should be (B, M, SumOfActionDims) but is {prev_actions_one_hot.shape}"  
        assert prev_rewards.shape == (B, self.M), f"prev_rewards should be (B, M) but is {prev_rewards.shape}"
        assert prev_values.shape == (B, self.M), f"prev_values should be (B, M) but is {prev_values.shape}"
        assert current_states.shape == (B, self.M, self.state_dim), f"current_states should be (B, M, S) but is {current_states.shape}"
        assert goals.shape == (B, self.goal_dim), f"goals should be (B, G) but is {goals.shape}"

        act_embeds = self.action_encoder(prev_actions_one_hot)
        assert act_embeds.shape == (B, self.M, self.total_action_embedding_dim), f"act_embeds should be (B, M, total_action_emb_dim) but is {act_embeds.shape}"
            
        act_flat = act_embeds.reshape(B, -1) # (B, M*TotalEmbed)
        rew_flat = prev_rewards.reshape(B, -1) # (B, M)
        val_flat = prev_values.reshape(B, -1) # (B, M)
        
        states_flat = self.state_encoder(current_states).reshape(B, -1) # (B, M*S)
        goals = self.goal_encoder(goals) # (B, G)
        
        # Concatenate: (B, M*TotalEmbed + M + 1 + G + M*S)
        rnn_input = torch.cat(
            [act_flat, rew_flat, val_flat, goals, states_flat], 
            dim=1
        )
        assert rnn_input.shape == (B, self.input_size), f"rnn_input should be (B, InputSize) but is {rnn_input.shape}"
        
        rnn_input = rnn_input.unsqueeze(1) # (B, 1, InputSize)
        
        new_hiddens = []
        x = rnn_input
        
        if hidden is None:
            hidden = [None] * len(self.rnn)
            
        for i, layer in enumerate(self.rnn):
            out, h = layer(x, hidden[i])
            x = out
            new_hiddens.append(h)
            
        rnn_out = x.squeeze(1) # (B, hidden)

        beliefs = self.belief_head(rnn_out) # (B, num_belief_bins)  
        
        return rnn_out, new_hiddens, beliefs

    def get_standard_logits(self, rnn_out, states, goals):
        """
        Standard mode: rnn_out -> logits matched to single action dim.
        """
        assert not self.is_factored, "Called get_standard_logits but model is in Factored mode"
        B = rnn_out.shape[0]
        
        states_flat = self.state_encoder(states).reshape(B, -1)
        goals = self.goal_encoder(goals)
        inp = torch.cat([rnn_out, states_flat, goals], dim=1)
        logits_flat = self.action_head(inp)
        return logits_flat.view(B, self.M, self.total_action_dim)

    def get_entity_logits(self, rnn_out, states, goals):
        """
        rnn_out: (B, Hidden)
        states: (B, M, S)
        goals: (B, G)
        Returns: (B, M, 4)
        """
        assert self.is_factored, "Called get_entity_logits but model is in Standard mode"
        B = rnn_out.shape[0]
        
        states_flat = self.state_encoder(states).reshape(B, -1)
        goals = self.goal_encoder(goals)
        inp = torch.cat([rnn_out, states_flat, goals], dim=1)
        
        logits_flat = self.entity_head(inp) # (B, M*4)
        return logits_flat.view(B, self.M, 4)

    def get_relation_logits(self, rnn_out, states, goals, chosen_entities):
        """
        rnn_out: (B, Hidden)
        states: (B, M, S)
        goals: (B, G)
        chosen_entities: (B, M, 4) One-Hot tensors
        Returns: (B, M, 17)
        """
        assert self.is_factored, "Called get_relation_logits but model is in Standard mode"
        B = rnn_out.shape[0]
            
        entities_flat = chosen_entities.reshape(B, -1) # (B, M*4)
        states_flat = self.state_encoder(states).reshape(B, -1)
        goals = self.goal_encoder(goals)
        
        if self.include_state_ins_for_actions:
            inp = torch.cat([rnn_out, goals, states_flat, entities_flat], dim=1)
        else:
            inp = torch.cat([rnn_out, entities_flat], dim=1)
        logits_flat = self.relation_mlp(inp)
        return logits_flat.view(B, self.M, 17)

    def get_frontier_logits(self, rnn_out, states, goals):
        """
        rnn_out: (B, Hidden)
        states: (B, M, S)
        goals: (B, G)
        Returns: (B, M, M) - M distributions over M previous hypotheses
        """
        B = rnn_out.shape[0]

        states_flat = self.state_encoder(states).reshape(B, -1)
        goals = self.goal_encoder(goals)
        inp = torch.cat([rnn_out, states_flat, goals], dim=1)
        logits_flat = self.frontier_head(inp) # (B, M*M)
        return logits_flat.view(B, self.M, self.M)

    def get_value(self, rnn_out, states, goals, curr_actions):
        """
        rnn_out: (B, Hidden)
        states: (B, M, S)
        goals: (B, G)
        curr_actions: (B, M, A)
        Returns: (B, M)
        """
        B = rnn_out.shape[0]
        
        states_flat = self.state_encoder(states).reshape(B, -1)
        goals = self.goal_encoder(goals)
        curr_actions_flat = self.action_encoder(curr_actions).reshape(B, -1)
        
        inp = torch.cat([rnn_out, states_flat, curr_actions_flat, goals], dim=1)
        return self.value_head(inp).view(B, self.M)

    def sample_subroutine(self, logits, gumbel=False):
        # logits: (B, M, A)
        assert len(logits.shape) == 3, f"logits should be (B, M, A) but is {logits.shape}"
        action_one_hot = F.gumbel_softmax(logits, tau=1, hard=True)
        action_indices = torch.argmax(action_one_hot, dim=-1)

        return action_one_hot, action_indices, logits
    
    def sample_action(self, rnn_out, states, goals, gumbel=True):
        """
        Handles sampling for Arbitrary Factorized actions.
        rnn_out: (..., hidden_size)
        states: (B, M, S)
        goals: (B, G)
        Returns:
            one_hot_concatenated: (..., total_action_dim)
            indices_list: List of (...,) tensors, one for each factor.
            parent_indices: (B, M) if not planner, else None
        """
        parent_indices = None
        # 1. Frontier Sampling
        frontier_logits = self.get_frontier_logits(rnn_out, states, goals) # (B, M, M)
        parent_one_hot, parent_indices, _ = self.sample_subroutine(frontier_logits, gumbel=True)
        # parent_one_hot: (B, M, M) -> for each of M new slots, one-hot vector over M old slots.
            
        # 2. Permute inputs for Action Sampling Differentiably
        # states: (B, M, S)

        # (B, M, M) x (B, M, S) -> (B, M, S)
        states_permuted = torch.bmm(parent_one_hot, states)
            
        if len(self.action_dims) > 1:
            assert len(self.action_dims) == 2
            entity_logits = self.get_entity_logits(
                rnn_out,
                states_permuted,
                goals
            )
            entity_one_hot, entity_indices, new_entity_logits = self.sample_subroutine(entity_logits, gumbel=True)
            relation_logits = self.get_relation_logits(rnn_out, states_permuted, goals, entity_one_hot)
            relation_one_hot, relation_indices, new_relation_logits = self.sample_subroutine(relation_logits, gumbel=True)

            policy_one_hot = torch.cat([entity_one_hot, relation_one_hot], dim=-1)
            policy_indices = [entity_indices, relation_indices]

            
            new_policy_logits = torch.cat([entity_logits, relation_logits], dim=-1) 
        else:
            policy_logits = self.get_standard_logits(
                rnn_out,
                states_permuted,
                goals
            )
            policy_one_hot, policy_indices, new_policy_logits = self.sample_subroutine(policy_logits, gumbel=True)
            policy_indices = [policy_indices]

        value = self.get_value(
            rnn_out,
            states_permuted,
            goals,
            policy_one_hot
        )
        return policy_one_hot, policy_indices, new_policy_logits, value, parent_one_hot, frontier_logits