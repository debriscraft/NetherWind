"""
mat.py
=======
MAT (Multi-Agent Transformer) baseline.

Reference: Wen et al., "Multi-Agent Reinforcement Learning with
           Hierarchical Attention", ICLR 2022.

Key features:
  - Transformer-based architecture for multi-agent RL
  - Hierarchical attention: intra-agent and inter-agent attention
  - Centralized critic with Transformer encoder
  - Decentralized actors with attention-based communication

Implementation notes:
  - Uses PyTorch's nn.TransformerEncoder for efficiency
  - Attention over agents and time steps
  - Compatible with continuous action spaces
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, List
import math


class MultiAgentTransformer(nn.Module):
    """
    Transformer-based policy for multi-agent RL.

    Architecture:
      1. Observation embedding (per agent)
      2. Inter-agent attention (Transformer encoder)
      3. Policy head (actor) and value head (critic)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_agents: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coeff: float = 0.01,
    ):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_agents = n_agents
        self.d_model = d_model
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        # PPO update parameters (used by update())
        self.ppo_epochs = 4
        self.mini_batch_size = 64
        self.value_coeff = 0.5
        self.max_grad_norm = 0.5

        # Observation embedding (input LayerNorm is essential: obs contains
        # raw positions in meters (~1e4); without normalization the post-norm
        # transformer diverges within a few updates and NaNs the weights)
        self.input_norm = nn.LayerNorm(obs_dim)
        self.obs_embed = nn.Linear(obs_dim, d_model)

        # Positional encoding (agent ID)
        self.pos_encoding = nn.Parameter(torch.randn(1, n_agents, d_model))

        # Transformer encoder (inter-agent attention); pre-norm for stability
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Actor head (policy)
        self.actor_mean = nn.Linear(d_model, action_dim)
        self.actor_log_std = nn.Linear(d_model, action_dim)

        # Critic head (value)
        self.critic = nn.Linear(d_model * n_agents, 1)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # Storage (PPO-style)
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def forward(self, obs: torch.Tensor):
        """
        Args:
            obs: (batch, n_agents, obs_dim) or (n_agents, obs_dim)
        Returns:
            mean: (batch, n_agents, action_dim)
            log_std: (batch, n_agents, action_dim)
            value: (batch, 1)
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)  # (1, n_agents, obs_dim)

        batch_size = obs.shape[0]

        # Embed observations (normalized first)
        x = self.obs_embed(self.input_norm(obs))  # (batch, n_agents, d_model)

        # Add positional encoding
        x = x + self.pos_encoding

        # Transformer (inter-agent attention)
        x = self.transformer(x)  # (batch, n_agents, d_model)

        # Actor: policy for each agent
        mean = torch.tanh(self.actor_mean(x))  # (batch, n_agents, action_dim)
        log_std = torch.clamp(self.actor_log_std(x), -5, 2)

        # Critic: global value
        x_flat = x.view(batch_size, -1)  # (batch, n_agents * d_model)
        value = self.critic(x_flat)  # (batch, 1)

        return mean, log_std, value

    def select_actions(self, obs: np.ndarray, deterministic: bool = False):
        """
        Select actions for all agents.

        Args:
            obs: (n_agents, obs_dim)
            deterministic: if True, use the policy mean instead of sampling
        Returns:
            actions: (n_agents, action_dim)
            log_probs: (n_agents, 1)
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)  # (1, n_agents, obs_dim)

        with torch.no_grad():
            mean, log_std, value = self.forward(obs_t)
            mean = mean.squeeze(0)  # (n_agents, action_dim)
            log_std = log_std.squeeze(0)

            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            if deterministic:
                actions = mean  # already in [-1,1] due to tanh head
                log_probs = dist.log_prob(actions) - torch.log(1 - actions.pow(2) + 1e-6)
            else:
                sample = dist.sample()
                actions = torch.tanh(sample)
                log_probs = dist.log_prob(sample) - torch.log(1 - actions.pow(2) + 1e-6)
            log_probs = log_probs.sum(dim=-1, keepdim=True)  # (n_agents, 1)

        return actions.cpu().numpy(), log_probs.cpu().numpy()

    def get_values(self, obs: torch.Tensor):
        """Get critic values."""
        _, _, value = self.forward(obs)
        return value

    def store_transition(self, obs, actions, log_probs, rewards, done, value):
        self.states.append(obs)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.rewards.append(rewards)
        self.dones.append(done)
        self.values.append(value)

    def clear_buffer(self):
        """Clear on-policy rollout storage after an update."""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def compute_gae(self, next_obs, next_done):
        """
        Compute Generalized Advantage Estimation for MAT.
        
        MAT uses transformer-based critic, so GAE is computed on the transformer's value estimates.
        """
        n_steps = len(self.states)
        n_agents = self.n_agents
        advantages = np.zeros((n_steps, n_agents), dtype=np.float32)
        
        with torch.no_grad():
            next_obs_t = torch.FloatTensor(next_obs).to(self.device)
            next_value = self.get_values(next_obs_t).item()
        
        last_gae = 0.0
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_val = next_value * (1 - next_done)
            else:
                next_val = self.values[t]
            
            reward = np.mean(self.rewards[t])  # Team reward
            value = self.values[t]
            mask = 1.0 - self.dones[t]
            
            delta = reward + self.gamma * next_val * mask - value
            last_gae = delta + self.gamma * self.gae_lambda * mask * last_gae
            advantages[t] = last_gae  # Same advantage for all agents
        
        returns = advantages + np.array(self.values)[:, np.newaxis]
        return advantages, returns

    def update(self, next_obs, next_done):
        """
        PPO update for MAT (transformer-based policy).
        
        Uses clipped objective like MAPPO.
        """
        advantages, returns = self.compute_gae(next_obs, next_done)

        # Normalize advantages (PPO standard; stabilizes the ratio scale)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # NaN watchdog: snapshot weights; if any mini-batch still manages to
        # corrupt them (non-finite), roll back to this snapshot below.
        watchdog_params = list(self.parameters())
        watchdog_snapshot = [p.detach().clone() for p in watchdog_params]

        all_obs = np.array(self.states)
        all_actions = np.array(self.actions)
        all_log_probs = np.array(self.log_probs)
        n_steps = len(self.rewards)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0
        
        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(n_steps)
            
            for start in range(0, n_steps, self.mini_batch_size):
                end = min(start + self.mini_batch_size, n_steps)
                mb_idx = indices[start:end]
                
                # Transformer expects (batch, n_agents, obs_dim)
                obs_t = torch.FloatTensor(all_obs[mb_idx]).to(self.device)
                old_actions_t = torch.FloatTensor(all_actions[mb_idx]).to(self.device)
                old_log_probs_t = torch.FloatTensor(all_log_probs[mb_idx]).to(self.device)
                
                # Forward pass through transformer
                mean, log_std, values = self.forward(obs_t)

                # NaN guard: skip corrupted mini-batches instead of crashing
                if torch.isnan(mean).any() or torch.isnan(log_std).any() or torch.isnan(values).any():
                    print("[WARN] NaN in MAT forward during update, skipping this mini-batch")
                    continue

                # Compute log probabilities
                std = log_std.exp()
                dist = torch.distributions.Normal(mean, std)

                # Reshape actions to (batch, n_agents, action_dim)
                old_actions_t = old_actions_t.view(-1, self.n_agents, self.action_dim)

                # Tanh-squashed Gaussian: invert stored post-tanh actions to
                # pre-tanh space (also removes the erroneous double tanh in the
                # correction term that destabilized training with NaN weights).
                old_actions_c = torch.clamp(old_actions_t, -0.999999, 0.999999)
                pre_tanh = torch.atanh(old_actions_c)
                log_prob = dist.log_prob(pre_tanh) - torch.log(1 - old_actions_c.pow(2) + 1e-6)
                log_prob = log_prob.sum(dim=-1).mean(dim=-1)  # (batch,)

                # PPO clipped objective
                mb_advantages = torch.FloatTensor(advantages[mb_idx]).to(self.device).mean(dim=-1)  # (batch,)
                old_log_prob = old_log_probs_t.reshape(mb_advantages.shape[0], -1).mean(dim=-1)  # (batch,)
                ratio = torch.exp(torch.clamp(log_prob - old_log_prob, -10.0, 10.0))
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                entropy = dist.entropy().sum(dim=-1).mean()

                # Value loss
                mb_returns = torch.FloatTensor(returns[mb_idx]).to(self.device).mean(dim=-1)  # (batch,)
                value_loss = nn.MSELoss()(values.squeeze(-1), mb_returns)
                
                # Backward (skip non-finite losses so one bad batch cannot
                # poison the Adam moments and NaN the weights)
                loss = policy_loss - self.entropy_coeff * entropy + self.value_coeff * value_loss
                if not torch.isfinite(loss):
                    continue
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1
        
        # NaN watchdog rollback: if any weight became non-finite during the
        # update, restore the pre-update snapshot so training can continue.
        bad = any(not torch.isfinite(p).all() for p in watchdog_params)
        if bad:
            with torch.no_grad():
                for p, s in zip(watchdog_params, watchdog_snapshot):
                    p.copy_(s)
            print("[WARN] NaN/Inf weights after update; rolled back to pre-update snapshot")
        
        self.clear_buffer()
        
        return {
            'policy_loss': total_policy_loss / max(n_updates, 1),
            'value_loss': total_value_loss / max(n_updates, 1),
            'entropy': total_entropy / max(n_updates, 1),
        }


class MATPolicy:
    """Wrapper class for MAT policy (compatible with train.py interface)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_agents: int,
        n_red: int = 3,
        n_blue: int = 3,
        n_fire_targets: int = 0,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        mini_batch_size: int = 64,
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_agents = n_agents
        self.n_red = n_red
        self.action_dim = action_dim

        # Create MAT model
        # NOTE: obs delivered by the env is already PER-AGENT (n_agents, obs_dim);
        # the transformer embeds each agent's obs_dim-dimensional observation.
        self.model = MultiAgentTransformer(
            obs_dim=obs_dim,  # per-agent observation dimension
            action_dim=action_dim,
            n_agents=n_agents,
            lr=lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_epsilon=clip_epsilon,
            entropy_coeff=entropy_coeff,
        ).to(self.device)

        # PPO parameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size

    def select_actions(self, obs: np.ndarray, deterministic: bool = False, **_kwargs):
        # deterministic honoured: eval protocol requires mean actions for all algorithms
        return self.model.select_actions(obs, deterministic=deterministic)

    def get_values(self, obs: torch.Tensor):
        """Return scalar centralized value (float) for GAE bookkeeping."""
        v = self.model.get_values(obs)
        return float(v.flatten()[0].item())

    def store_transition(self, obs, actions, log_probs, rewards, done, value):
        self.model.store_transition(obs, actions, log_probs, rewards, done, value)

    def compute_gae(self, next_obs, next_done):
        self.model.compute_gae(next_obs, next_done)

    def update(self, next_obs, next_done):
        return self.model.update(next_obs, next_done)

    def clear_buffer(self):
        self.model.clear_buffer()

    def save(self, path: str):
        torch.save({'model': self.model.state_dict()}, path)
        print(f"[MAT] Model saved to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        print(f"[MAT] Model loaded from {path}")


if __name__ == '__main__':
    policy = MATPolicy(obs_dim=19 * 6, action_dim=4, n_agents=3)
    print("MAT policy created successfully")
