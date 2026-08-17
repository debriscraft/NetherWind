"""
gre.py
======
Graph Relational Encoder (GRE) for BCA.

Implements the GRE component from the BCA paper (§4.2):
  - Dynamic graph construction with typed edges (TEAMMATE / OPPONENT / TARGET)
  - Geometry-based edge reweighting (distance + aspect angle)
  - 2-layer GraphSAGE with typed message passing (Mean → Max)

Input:
  obs: (batch, n_red, obs_dim)  where obs_dim = 19 * n_all
  The 19-dim per-agent features are:
    [rel_pos(3), rel_vel(3), pos(3), vel(3), attitude(3),
     missiles_left_norm, cooldown_norm, hp_norm, bullets_left_norm]

Output:
  embeddings: (batch, n_red, 2 * d_model)  — relationally-aware node embeddings
  attention_weights: dict with per-relation attention stats (for logging)

No external dependencies beyond PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


# ---------------------------------------------------------------------------
# Helper: reshape flattened obs to (batch, n_all, per_agent_dim)
# ---------------------------------------------------------------------------

def unpack_obs(obs: torch.Tensor, n_red: int, n_blue: int, per_agent_dim: int = 19):
    """
    Reshape flattened observation to per-agent slots.

    Args:
        obs: (batch, n_red, obs_dim)  flattened
        n_red, n_blue: team sizes
        per_agent_dim: 19

    Returns:
        x: (batch, n_all, per_agent_dim)  per-agent feature matrix
    """
    n_all = n_red + n_blue
    batch = obs.shape[0]
    # obs is (batch, n_red, n_all * per_agent_dim) — each red agent sees all
    # We need to build a single graph with n_all nodes.
    # For each red agent i, the obs[i] contains slots for all n_all agents.
    # We'll use the first red agent's observation to get all agent features.
    # In practice during training: batch=1 (PPO rollout), n_red agents share obs.
    #
    # Better: the environment returns (n_red, obs_dim) where each row is
    # the observation for that red agent. To build the graph we need all agents'
    # features. We'll assume the first 19* n_all dims contain all agents' features
    # in a consistent order: [self_slot, agent1_slot, ..., agentN_slot] * n_red?
    #
    # Actually from env.py: obs[i] = concat of all agent slots (n_all slots × 19 dims)
    # So from any red agent's observation we can extract all agents' features.
    # During centralized training with parameter sharing, we use the same features.

    # (batch, n_red, n_all * per_agent_dim) → (batch, n_red, n_all, per_agent_dim)
    x = obs.view(batch, n_red, n_all, per_agent_dim)

    # For graph construction we need (batch, n_all, per_agent_dim).
    # Use the first red agent's view (all agents' features from red[0]'s perspective).
    # In parameter-sharing setup, all red agents see the same world state.
    x_all = x[:, 0, :, :]  # (batch, n_all, per_agent_dim)

    return x_all


# ---------------------------------------------------------------------------
# Edge construction
# ---------------------------------------------------------------------------

def build_typed_edges(positions: torch.Tensor, n_red: int, n_blue: int,
                      fire_targets: Optional[torch.Tensor] = None):
    """
    Build typed edge lists and geometry-based weights.

    Vectorized implementation over all (i, j) pairs within each batch.

    Args:
        positions: (batch, n_all, 3)  aircraft positions (NED)
        n_red, n_blue: team sizes
        fire_targets: (batch, n_red)  current fire target per red agent
                      -1 = no target; otherwise index into all_aircraft (local index)
                      If None, no TARGET edges are added.

    Returns:
        edge_index: (2, n_edges)  source→target pairs (directed graph)
        edge_type:  (n_edges,)    0=TEAMMATE, 1=OPPONENT, 2=TARGET
        edge_weight: (n_edges,)    geometry-based weight in (0, 1]
    """
    batch, n_all, _ = positions.shape
    device = positions.device
    tau = 1000.0

    all_edges = []
    all_types = []
    all_weights = []

    for b in range(batch):
        # Vectorized distance computation
        pos_b = positions[b]  # (n_all, 3)
        diff = pos_b.unsqueeze(0) - pos_b.unsqueeze(1)  # (n_all, n_all, 3)
        dist = torch.norm(diff, dim=-1) + 1e-6  # (n_all, n_all)
        dist_weight = torch.sigmoid(-dist / tau)  # (n_all, n_all)

        for i in range(n_all):
            for j in range(n_all):
                if i == j:
                    continue

                i_red = (i < n_red)
                j_red = (j < n_red)

                # Skip blue-blue edges (not needed for red's graph)
                if (not i_red) and (not j_red):
                    continue

                # Relation type
                if i_red and j_red:
                    rel = 0  # TEAMMATE
                else:
                    rel = 1  # OPPONENT

                w = dist_weight[i, j].item()

                all_edges.append([b * n_all + i, b * n_all + j])
                all_types.append(rel)
                all_weights.append(w)

                # TARGET edge
                if fire_targets is not None and i < n_red:
                    if fire_targets[b, i] == j:
                        all_edges.append([b * n_all + i, b * n_all + j])
                        all_types.append(2)
                        all_weights.append(min(w * 1.5, 1.0))

    if len(all_edges) == 0:
        return (
            torch.empty((2, 0), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
            torch.empty((0,), device=device),
        )

    edge_index = torch.tensor(all_edges, dtype=torch.long, device=device).t()
    edge_type = torch.tensor(all_types, dtype=torch.long, device=device)
    edge_weight = torch.tensor(all_weights, dtype=torch.float32, device=device)

    return edge_index, edge_type, edge_weight


# ---------------------------------------------------------------------------
# GraphSAGE Layer with typed edges
# ---------------------------------------------------------------------------

class TypedGraphSAGE(nn.Module):
    """
    Single GraphSAGE layer with typed message passing.

    For each relation type r ∈ {TEAMMATE, OPPONENT, TARGET}, uses a separate
    weight matrix W_r. Aggregation: mean over neighbors of each type,
    then max-pooling across types.
    """

    def __init__(self, in_dim: int, out_dim: int, n_relations: int = 3,
                 aggr: str = 'mean'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_relations = n_relations
        self.aggr = aggr

        # Separate weights per relation type
        self.W_r = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for _ in range(n_relations)
        ])
        self.W_self = nn.Linear(in_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_type: torch.Tensor, edge_weight: torch.Tensor):
        """
        Args:
            x: (n_nodes, in_dim)  node features
            edge_index: (2, n_edges)
            edge_type: (n_edges,)  0/1/2
            edge_weight: (n_edges,)  scalar weight per edge

        Returns:
            out: (n_nodes, out_dim)
        """
        n_nodes = x.shape[0]
        device = x.device

        # Collect messages per relation type
        agg_msgs = torch.zeros(self.n_relations, n_nodes, self.out_dim, device=device)

        for r in range(self.n_relations):
            mask = (edge_type == r)
            if mask.sum() == 0:
                continue

            ei_r = edge_index[:, mask]        # (2, n_r)
            ew_r = edge_weight[mask].view(-1, 1)  # (n_r, 1)
            src, dst = ei_r[0], ei_r[1]

            # Message: W_r @ x[src], weighted by edge_weight
            msg = self.W_r[r](x[src]) * ew_r  # (n_r, out_dim)

            # Aggregate (mean) per destination node
            # Use scatter_add
            n_msgs = torch.zeros(n_nodes, self.out_dim, device=device)
            n_msgs.scatter_add_(0, dst.unsqueeze(1).expand(-1, self.out_dim), msg)

            # Count per destination for mean
            count = torch.zeros(n_nodes, 1, device=device)
            count.scatter_add_(0, dst.unsqueeze(1), ew_r)

            agg_msgs[r] = n_msgs / (count + 1e-6)

        # Combine across relation types: max-pooling
        # (n_relations, n_nodes, out_dim) → (n_nodes, out_dim)
        out = torch.max(agg_msgs, dim=0)[0]

        # Add self-loop transformation
        out = out + self.W_self(x)
        out = self.ln(out)
        out = F.relu(out)

        return out


# ---------------------------------------------------------------------------
# Full GRE module
# ---------------------------------------------------------------------------

class GraphRelationalEncoder(nn.Module):
    """
    Full Graph Relational Encoder (GRE) as described in BCA paper §4.2.

    Architecture:
      - MLP_embed: (19,) → (d_model,)  — per-agent feature embedding
      - GraphSAGE Layer 1: Mean aggregation over typed neighbors
      - GraphSAGE Layer 2: Max-pooling with edge weighting
      - Output: concat(self_embed, mean_pool_all) → (2 * d_model,)
    """

    def __init__(self, per_agent_dim: int = 19, d_model: int = 128,
                 n_layers: int = 2, n_relations: int = 3):
        super().__init__()
        self.per_agent_dim = per_agent_dim
        self.d_model = d_model

        # Per-agent feature embedding
        self.embed = nn.Sequential(
            nn.Linear(per_agent_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

        # GraphSAGE layers
        self.layers = nn.ModuleList([
            TypedGraphSAGE(d_model, d_model, n_relations)
            for _ in range(n_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor, n_red: int, n_blue: int,
                positions: Optional[torch.Tensor] = None,
                fire_targets: Optional[torch.Tensor] = None):
        """
        Args:
            obs: (batch, n_red, obs_dim)  flattened observations
            n_red, n_blue: team sizes
            positions: (batch, n_all, 3)  aircraft positions (optional, extracted from obs if None)
            fire_targets: (batch, n_red)  current fire targets (optional)

        Returns:
            embeddings: (batch, n_red, 2 * d_model)  — concatenated [self, global_mean]
            attn_stats: dict with edge type statistics (for logging)
        """
        batch = obs.shape[0]
        n_all = n_red + n_blue

        # Unpack observations to per-agent features
        # obs: (batch, n_red, n_all * per_agent_dim [+ tactical_dim])
        # Env appends tactical_dim trailing features (see env._get_obs); the
        # graph consumes only the leading n_all * per_agent_dim block.
        # → x_all: (batch, n_all, per_agent_dim)
        obs_graph = obs[..., : n_all * self.per_agent_dim]
        x_flat = obs_graph.reshape(batch, n_red, n_all, self.per_agent_dim)
        x_all = x_flat[:, 0, :, :]  # use red[0]'s view of all agents

        # Embed per-agent features
        x_emb = self.embed(x_all)  # (batch, n_all, d_model)

        # Build graph (batch all nodes together for efficient processing)
        # Flatten to (batch * n_all, d_model)
        x_flat_emb = x_emb.view(batch * n_all, self.d_model)

        # Extract positions for edge weighting (from obs, dims 0-2 of each agent slot)
        if positions is None:
            # Extract from obs: each agent slot has rel_pos(3) — but that's relative.
            # For absolute positions, we need env state. Pass positions as argument.
            # Temporary: use zeros (edge weights will be uniform)
            positions = torch.zeros(batch, n_all, 3, device=obs.device)

        edge_index, edge_type, edge_weight = build_typed_edges(
            positions, n_red, n_blue, fire_targets
        )

        # If no edges (degenerate case), skip message passing
        if edge_index.shape[1] == 0:
            h = x_flat_emb
        else:
            # GraphSAGE layers
            h = x_flat_emb
            for layer in self.layers:
                h = layer(h, edge_index, edge_type, edge_weight)

        # Reshape back to (batch, n_all, d_model)
        h = h.view(batch, n_all, self.d_model)

        # Output: concat(self_embed, global_mean)
        # For each red agent i: [h[i], mean(h[all])]
        global_mean = h.mean(dim=1, keepdim=True).expand(-1, n_all, -1)
        out_all = torch.cat([h, global_mean], dim=-1)  # (batch, n_all, 2*d_model)

        # Only return embeddings for red agents (n_red of them)
        out_red = out_all[:, :n_red, :]  # (batch, n_red, 2*d_model)

        # Attention stats for logging
        attn_stats = {
            'n_edges': edge_index.shape[1] if edge_index.shape[1] > 0 else 0,
            'n_teammate': (edge_type == 0).sum().item() if len(edge_type) > 0 else 0,
            'n_opponent': (edge_type == 1).sum().item() if len(edge_type) > 0 else 0,
            'n_target': (edge_type == 2).sum().item() if len(edge_type) > 0 else 0,
        }

        return out_red, attn_stats


# ---------------------------------------------------------------------------
# Integration wrapper: replaces ActorNetwork's MLP backbone with GRE
# ---------------------------------------------------------------------------

class GREActor(nn.Module):
    """
    Actor network with GRE backbone.

    Replaces the MLP (256→128) in standard MAPPO with:
      GRE (graph relational encoder) → MLP policy head

    Output: (mean, log_std) for continuous actions,
            optionally (mean, log_std, fire_logits) for hybrid.
    """

    def __init__(self, n_red: int, n_blue: int, action_dim: int,
                 d_model: int = 128, n_fire_targets: int = 0):
        super().__init__()
        self.n_red = n_red
        self.n_blue = n_blue
        self.action_dim = action_dim
        self.d_model = d_model

        per_agent_dim = 19  # hardcoded from env.py

        # GRE backbone
        self.gre = GraphRelationalEncoder(
            per_agent_dim=per_agent_dim,
            d_model=d_model,
            n_layers=2,
        )

        # Policy head (same as original but input is 2*d_model)
        self.fc1 = nn.Linear(2 * d_model, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)

        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Linear(128, action_dim)

        self.has_discrete = (n_fire_targets > 0)
        if self.has_discrete:
            self.fire_logits = nn.Linear(128, n_fire_targets)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.zeros_(self.mean.bias)

    def forward(self, obs: torch.Tensor, positions: Optional[torch.Tensor] = None,
                fire_targets: Optional[torch.Tensor] = None):
        """
        Args:
            obs: (batch, obs_dim) or (batch, n_red, obs_dim)
                 If (batch, obs_dim), assume n_red=1 view.
        """
        # Handle both (batch, obs_dim) and (batch, n_red, obs_dim)
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # (batch, 1, obs_dim)

        batch = obs.shape[0]
        n_red = obs.shape[1]

        gre_out, attn_stats = self.gre(
            obs, self.n_red, self.n_blue, positions, fire_targets
        )  # (batch, n_red, 2*d_model)

        # Flatten for MLP: (batch * n_red, 2*d_model)
        h = gre_out.view(-1, 2 * self.d_model)
        h = F.relu(self.ln1(self.fc1(h)))
        h = F.relu(self.ln2(self.fc2(h)))

        mean = torch.tanh(self.mean(h))
        log_std = torch.clamp(self.log_std(h), -5, 2)

        mean = mean.view(batch, n_red, self.action_dim)
        log_std = log_std.view(batch, n_red, self.action_dim)

        if self.has_discrete:
            fire_logits = self.fire_logits(h).view(batch, n_red, -1)
            return mean, log_std, fire_logits, attn_stats

        return mean, log_std, attn_stats

    def get_action(self, obs, deterministic=False, action_mask=None,
                   positions=None, fire_targets=None):
        """Sample action. Compatible with original ActorNetwork interface."""
        if self.has_discrete:
            mean, log_std, fire_logits, attn_stats = self.forward(
                obs, positions, fire_targets
            )
            # ... (same as original ActorNetwork)
            # For brevity, delegate to a helper
            return self._sample_hybrid(
                mean, log_std, fire_logits, deterministic, action_mask
            ), attn_stats
        else:
            mean, log_std, attn_stats = self.forward(obs, positions, fire_targets)
            std = log_std.exp()
            if deterministic:
                action = mean
            else:
                action = mean + std * torch.randn_like(mean)
            log_prob = -0.5 * ((action - mean) / (std + 1e-6)) ** 2 \
                       - log_std - 0.5 * np.log(2 * np.pi)
            return action, log_prob.sum(dim=-1, keepdim=True), attn_stats

    def _sample_hybrid(self, mean, log_std, fire_logits, deterministic, action_mask):
        """Sample hybrid continuous+discrete action."""
        # ... (implementation similar to original ActorNetwork)
        # This is a placeholder — full implementation needed
        raise NotImplementedError("_sample_hybrid: implement based on red_policy.py")


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import numpy as np

    n_red, n_blue = 3, 3
    per_agent_dim = 19
    obs_dim = per_agent_dim * (n_red + n_blue)
    batch = 2

    obs = torch.randn(batch, n_red, obs_dim)
    positions = torch.randn(batch, n_red + n_blue, 3)

    gre = GraphRelationalEncoder(per_agent_dim=per_agent_dim, d_model=128)
    out, stats = gre(obs, n_red, n_blue, positions)
    print(f"GRE output shape: {out.shape}")  # (2, 3, 256)
    print(f"Attention stats: {stats}")
