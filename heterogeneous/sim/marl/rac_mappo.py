"""
sim/marl/rac_mappo.py
=====================
The proposed algorithm: RAC-MAPPO (Role-aware Attention-Critic MAPPO).

Two architectural additions over the MAPPO baseline, both motivated by
heterogeneous-formation air combat:

1. Role-conditioned SHARED actor. One policy network serves every agent;
   a learned role embedding (fighter / UCAV / ...) is concatenated with
   the observation. Roles share tactical regularities (pursuit, evasion,
   fire discipline) while the embedding lets the same network express
   platform-specific envelopes (9 g fighter vs 3.5 g turboprop). This
   halves sample complexity relative to per-role actors and is what lets
   the UCAV benefit from fighter combat experience (and vice versa).

2. Set-attention centralised critic. The critic embeds each agent's
   observation, runs multi-head self-attention over the agent set
   (permutation-invariant, extensible to team-size changes), mean-pools
   and regresses the team value. Attention weights expose WHO is driving
   the value estimate (interpretability hook for the paper).

Everything else (GAE, PPO clip, hybrid action heads, training loop
contract) is inherited from MAPPO — the experiment isolates these two
mechanisms as the contribution.
"""

import numpy as np
import torch
import torch.nn as nn

from .mappo import MAPPO, _mlp


class SharedRoleActor(nn.Module):
    """Single actor over [obs ; role_embedding] (embedding optional).

    With ``pk_gate=True`` the observation is expected to carry the
    predicted single-shot Pk against the fire-control track in its last
    dimension (MarlEnv pk_feature). A learnable per-role threshold then
    gates the fire logit:  logit_fire += W_GATE * (pk - theta_role).
    Remote cues can still inform manoeuvre decisions through the rest of
    the network, but launches below the learned Pk threshold are
    suppressed (the Pk-aware cue gate of the paper's remedy path).
    """

    W_GATE = 20.0

    def __init__(self, obs_dim, n_roles, emb_dim=16, hidden=128,
                 role_emb=True, pk_gate=False):
        super().__init__()
        self.role_emb = role_emb
        self.pk_gate = pk_gate
        self.emb = nn.Embedding(n_roles, emb_dim) if role_emb else None
        self.body = _mlp([obs_dim + (emb_dim if role_emb else 0),
                          hidden, hidden])
        self.mu = nn.Linear(hidden, 2)
        self.log_std = nn.Parameter(-0.5 * torch.ones(2))
        self.mode = nn.Linear(hidden, 3)
        self.fire = nn.Linear(hidden, 2)
        if pk_gate:
            self.theta = nn.Parameter(0.55 * torch.ones(n_roles))

    def forward(self, obs, role_idx):
        pk = obs[..., -1] if self.pk_gate else None
        if self.role_emb:
            obs = torch.cat([obs, self.emb(role_idx)], dim=-1)
        h = self.body(obs)
        # numerical hardening: bounded mean/log-std keep the Normal
        # well-posed even when a rare extreme observation arrives
        mu = torch.clamp(self.mu(h), -4.0, 4.0)
        log_std = torch.clamp(self.log_std, -4.0, 1.5)
        fl = self.fire(h)
        if self.pk_gate:
            shift = self.W_GATE * (pk - self.theta[role_idx])
            fl = torch.stack([fl[..., 0], fl[..., 1] + shift], dim=-1)
        return mu, log_std, self.mode(h), fl


class SetAttentionCritic(nn.Module):
    """Critic over the agent observation SET (concat input, split inside).

    Input: (B, n_agents * obs_dim) — the same concatenation the baseline
    critic receives, so the training loop is unchanged.
    """

    def __init__(self, obs_dim, n_agents, n_roles, emb_dim=16,
                 d_model=128, n_heads=4, hidden=256, role_emb=True):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.role_emb = role_emb
        self.emb = nn.Embedding(n_roles, emb_dim) if role_emb else None
        self.role_of = None                      # set by RACMAPPO
        self.inp = nn.Linear(obs_dim + (emb_dim if role_emb else 0),
                             d_model)
        self.pre_norm = nn.LayerNorm(d_model)   # pre-LN: stabilises attn
        self.attn = nn.MultiheadAttention(d_model, n_heads,
                                          batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(*_mlp([d_model, hidden]),
                                  nn.Linear(hidden, 1))

    def forward(self, gs):
        B = gs.shape[0]
        x = gs.view(B, self.n_agents, self.obs_dim)
        if self.role_emb:
            roles = torch.as_tensor(self.role_of, dtype=torch.long,
                                    device=gs.device).unsqueeze(0).expand(
                                        B, -1)
            x = torch.cat([x, self.emb(roles)], dim=-1)
        h = self.pre_norm(self.inp(x))
        a, _w = self.attn(h, h, h)
        h = self.norm(h + a)
        return self.head(h.mean(dim=1)).squeeze(-1)


class RACMAPPO(MAPPO):
    name = 'rac_mappo'

    def __init__(self, agent_roles: dict, obs_dim: int, n_agents: int,
                 shared_actor: bool = True, attn_critic: bool = True,
                 role_emb: bool = True, pk_gate: bool = False, **kw):
        # build without the parent's networks, then replace them
        super().__init__(agent_roles, obs_dim, n_agents,
                         centralized=True, **kw)
        self.roles_all = sorted(set(agent_roles.values()))
        self.role_idx = {aid: self.roles_all.index(agent_roles[aid])
                         for aid in self.agent_ids}
        n_roles = len(self.roles_all)
        self.shared_actor_flag = shared_actor
        if shared_actor:
            self.shared_actor = SharedRoleActor(obs_dim, n_roles,
                                                role_emb=role_emb,
                                                pk_gate=pk_gate).to(
                self.device)
            self.actors = {'shared': self.shared_actor}
            self.agent_roles = {aid: 'shared' for aid in self.agent_ids}
            self.roles = ['shared']
            self.opt_pi = torch.optim.Adam(self.shared_actor.parameters(),
                                           lr=3e-4)
        # else: keep the parent's per-role actors and optimiser — the
        # contribution is then isolated to the set-attention critic
        self._true_roles = self.role_idx            # aid -> embedding idx
        if attn_critic:
            self.critic = SetAttentionCritic(obs_dim, n_agents, n_roles,
                                             role_emb=role_emb).to(
                self.device)
            self.critic.role_of = [self.role_idx[aid] for aid in
                                   self.agent_ids]
            self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=5e-4)
        # else: keep the parent's plain MLP critic on the concatenated
        # global state — isolates the set-attention critic's contribution

    # ------------------------------------------------------------------
    def _pi_forward(self, aid, obs_t):
        if not self.shared_actor_flag:
            return super()._pi_forward(aid, obs_t)
        ridx = torch.full((obs_t.shape[0],), self._true_roles[aid],
                          dtype=torch.long, device=obs_t.device)
        return self.shared_actor(obs_t, ridx)

    def gate_theta(self):
        """Learned per-role Pk thresholds (None when the gate is off)."""
        if getattr(self.shared_actor, 'pk_gate', False):
            return self.shared_actor.theta.detach().cpu().numpy().tolist()
        return None

    # attention critic receives the concatenated global state directly;
    # default _v_forward (centralised) applies.
