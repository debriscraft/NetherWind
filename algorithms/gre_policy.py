"""
gre_policy.py
==============
GREPolicy: MAPPO + GRE actor (Graph Relational Encoder replacing MLP actor).

This is the minimal BCA implementation — only replaces the actor architecture,
keeps MAPPO's single-critic PPO training. Used to validate that GRE works
before adding BVD (dual critics) and PBC (opponent scheduling).

Usage:
    from gre_policy import GREPolicy
    policy = GREPolicy(obs_dim=..., action_dim=..., n_agents=..., n_red=..., n_blue=..., n_fire_targets=...)

    # Same interface as MAPPOPolicy:
    #   policy.select_actions(obs, action_mask)
    #   policy.store_transition(...)
    #   policy.compute_gae(next_obs, next_done)
    #   policy.update(next_obs, next_done)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, List

from .gre_encoder import GraphRelationalEncoder
from .mappo import MAPPOPolicy


class GREPolicy(MAPPOPolicy):
    """
    MAPPO with GRE actor (Graph Relational Encoder).

    Inherits from MAPPOPolicy and replaces self.actor (MLP) with GRE-based actor.
    Keeps single critic and standard PPO training (no BVD, no PBC).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_agents: int,
        n_red: int,
        n_blue: int,
        n_fire_targets: int = 0,
        lr: float = 1e-4,
        gamma: float = 0.95,
        gae_lambda: float = 0.97,
        clip_epsilon: float = 0.2,
        entropy_coeff: float = 0.05,
        value_coeff: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        mini_batch_size: int = 64,
        gre_d_model: int = 128,
    ):
        # Initialize parent MAPPOPolicy (creates MLP actor + critic)
        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            n_fire_targets=n_fire_targets,
            lr=lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_epsilon=clip_epsilon,
            entropy_coeff=entropy_coeff,
            value_coeff=value_coeff,
            max_grad_norm=max_grad_norm,
            ppo_epochs=ppo_epochs,
            mini_batch_size=mini_batch_size,
        )

        self.n_red = n_red
        self.n_blue = n_blue
        self.gre_d_model = gre_d_model
        per_agent_dim = 19

        # ---- Replace MLP actor with GRE + policy head ----
        # Keep self.critic (MLP, unchanged)
        # Replace self.actor with GRE-based modules

        self.gre = GraphRelationalEncoder(
            per_agent_dim=per_agent_dim,
            d_model=gre_d_model,
            n_layers=2,
        ).to(self.device)

        # Policy head (after GRE)
        self.policy_fc1 = nn.Linear(2 * gre_d_model, 256).to(self.device)
        self.policy_ln1 = nn.LayerNorm(256).to(self.device)
        self.policy_fc2 = nn.Linear(256, 128).to(self.device)
        self.policy_ln2 = nn.LayerNorm(128).to(self.device)
        self.policy_mean = nn.Linear(128, action_dim).to(self.device)
        self.policy_log_std = nn.Linear(128, action_dim).to(self.device)

        if n_fire_targets > 0:
            self.fire_logits = nn.Linear(128, n_fire_targets).to(self.device)

        # ---- Re-initialize optimizers (include GRE params) ----
        actor_params = (
            list(self.gre.parameters()) +
            list(self.policy_fc1.parameters()) +
            list(self.policy_ln1.parameters()) +
            list(self.policy_fc2.parameters()) +
            list(self.policy_ln2.parameters()) +
            list(self.policy_mean.parameters()) +
            list(self.policy_log_std.parameters())
        )
        if n_fire_targets > 0:
            actor_params += list(self.fire_logits.parameters())

        self.actor_optimizer = optim.Adam(actor_params, lr=lr)

        # Remove the old MLP actor (not used anymore)
        self.actor = None

    # -------------------------------------------------------------------
    # Override: select_actions
    # -------------------------------------------------------------------

    def select_actions(self, obs: np.ndarray, action_mask: np.ndarray = None, deterministic: bool = False):
        """
        Select actions using GRE actor.

        Args:
            obs: (n_red, obs_dim) numpy array
                 Each row is one red agent's observation (19 * n_all dims)
            deterministic: if True, use mean action (no sampling). Used for evaluation.

        Returns:
            Same format as MAPPOPolicy.select_actions
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)  # (1, n_red, obs_dim)

        with torch.no_grad():
            # GRE forward
            gre_out, attn_stats = self.gre(obs_t, self.n_red, self.n_blue)

            # Policy head
            batch = gre_out.shape[0]  # 1
            h = gre_out.view(-1, 2 * self.gre_d_model)  # (n_red, 2*d_model)
            h = torch.relu(self.policy_ln1(self.policy_fc1(h)))
            h = torch.relu(self.policy_ln2(self.policy_fc2(h)))

            mean = torch.tanh(self.policy_mean(h))
            log_std = torch.clamp(self.policy_log_std(h), -5, 2)
            std = log_std.exp()

            # Sample or use mean (deterministic)
            if deterministic:
                actions = mean
            else:
                dist = torch.distributions.Normal(mean, std)
                sample = dist.sample()
                actions = torch.tanh(sample)

            # Compute log-prob (always needed for storage during training)
            dist = torch.distributions.Normal(mean, std)
            log_prob = dist.log_prob(actions) - torch.log(1 - actions.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)

            if self.n_fire_targets > 0 and action_mask is not None:
                fire_logits = self.fire_logits(h)
                mask_t = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)
                fire_logits = fire_logits + mask_t
                fire_dist = torch.distributions.Categorical(logits=fire_logits)
                fire_actions = fire_dist.sample()
                fire_log_prob = fire_dist.log_prob(fire_actions).unsqueeze(-1)
                log_prob = log_prob + fire_log_prob

                return (
                    actions.squeeze(0).cpu().numpy(),
                    fire_actions.squeeze(0).cpu().numpy().astype(int),
                    log_prob.squeeze(0).cpu().numpy(),
                )

            return actions.squeeze(0).cpu().numpy(), log_prob.squeeze(0).cpu().numpy()

    # -------------------------------------------------------------------
    # Override: _ppo_update_step (use GRE actor forward)
    # -------------------------------------------------------------------

    def _ppo_update_step(self, obs_t, old_actions_t, old_log_probs_t, advantages, returns, mb_idx,
                         old_fire_actions_t=None):
        """PPO update step with GRE actor."""
        # GRE forward
        # obs_t: (batch, obs_dim) where batch = n_red * mini_batch_size_per_agent
        # Need to reshape to (n_red, mini_batch_size_per_agent, obs_dim) for GRE
        n_mb = obs_t.shape[0] // self.n_agents
        obs_3d = obs_t.view(n_mb, self.n_agents, self.obs_dim)

        gre_out, _ = self.gre(obs_3d, self.n_red, self.n_blue)
        h = gre_out.reshape(-1, 2 * self.gre_d_model)
        # Policy head (missing in original code)
        h = torch.relu(self.policy_ln1(self.policy_fc1(h)))
        h = torch.relu(self.policy_ln2(self.policy_fc2(h)))

        mean = torch.tanh(self.policy_mean(h))
        log_std = torch.clamp(self.policy_log_std(h), -5, 2)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)

        # Continuous log-prob
        new_cont_log_prob = dist.log_prob(old_actions_t) - torch.log(1 - old_actions_t.pow(2) + 1e-6)
        new_cont_log_prob = new_cont_log_prob.sum(dim=-1, keepdim=True)

        if self.n_fire_targets > 0 and old_fire_actions_t is not None:
            fire_logits = self.fire_logits(h)
            new_fire_log_prob = torch.distributions.Categorical(logits=fire_logits).log_prob(
                old_fire_actions_t.squeeze(-1)
            ).unsqueeze(-1)
            new_log_probs = new_cont_log_prob + new_fire_log_prob
            entropy = dist.entropy().sum(dim=-1).mean() + torch.distributions.Categorical(
                logits=fire_logits
            ).entropy().mean()
        else:
            new_log_probs = new_cont_log_prob
            entropy = dist.entropy().sum(dim=-1).mean()

        ratio = torch.exp(new_log_probs - old_log_probs_t)
        mb_advantages = torch.FloatTensor(
            advantages[mb_idx]
        ).to(self.device).reshape(-1, 1)

        surr1 = ratio * mb_advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss (same as MAPPO)
        global_obs_t = obs_t[:n_mb * self.n_agents].reshape(n_mb, -1)
        mb_returns_val = torch.FloatTensor(
            returns[mb_idx][:n_mb].mean(axis=1)
        ).to(self.device).reshape(-1, 1)
        values = self.critic(global_obs_t)
        value_loss = nn.MSELoss()(values, mb_returns_val)

        # Backward
        self.actor_optimizer.zero_grad()
        (policy_loss - self.entropy_coeff * entropy).backward()
        nn.utils.clip_grad_norm_(
            list(self.gre.parameters()) + list(self.policy_fc1.parameters()) +
            list(self.policy_fc2.parameters()) + list(self.policy_mean.parameters()) +
            list(self.policy_log_std.parameters()),
            self.max_grad_norm,
        )
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        return policy_loss.item(), value_loss.item(), entropy.item()

    # -------------------------------------------------------------------
    # Override: save/load (include GRE params)
    # -------------------------------------------------------------------

    def save(self, path: str):
        state = {
            'gre': self.gre.state_dict(),
            'policy_fc1': self.policy_fc1.state_dict(),
            'policy_ln1': self.policy_ln1.state_dict(),
            'policy_fc2': self.policy_fc2.state_dict(),
            'policy_ln2': self.policy_ln2.state_dict(),
            'policy_mean': self.policy_mean.state_dict(),
            'policy_log_std': self.policy_log_std.state_dict(),
            'critic': self.critic.state_dict(),
        }
        if self.n_fire_targets > 0:
            state['fire_logits'] = self.fire_logits.state_dict()
        torch.save(state, path)

    def load(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.gre.load_state_dict(state['gre'])
        self.policy_fc1.load_state_dict(state['policy_fc1'])
        self.policy_ln1.load_state_dict(state['policy_ln1'])
        self.policy_fc2.load_state_dict(state['policy_fc2'])
        self.policy_ln2.load_state_dict(state['policy_ln2'])
        self.policy_mean.load_state_dict(state['policy_mean'])
        self.policy_log_std.load_state_dict(state['policy_log_std'])
        self.critic.load_state_dict(state['critic'])
        if self.n_fire_targets > 0 and 'fire_logits' in state:
            self.fire_logits.load_state_dict(state['fire_logits'])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_gre_policy(obs_dim, action_dim, n_agents, n_red, n_blue, n_fire_targets=0, **kwargs):
    return GREPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_agents=n_agents,
        n_red=n_red,
        n_blue=n_blue,
        n_fire_targets=n_fire_targets,
        **kwargs
    )
