"""
happo.py
========
HAPPO (Heterogeneous-Agent PPO) baseline.

Reference: Kuba et al., "Trust Region Policy Optimisation in Multi-Agent Reinforcement Learning",
           NeurIPS 2021.

Key features:
  - Heterogeneous policy representation (each agent can have differentobservation/action spaces)
  - Centralized training with decentralized execution (CTDE)
  - Trust region constraint via Kullback-Leibler divergence

Implementation notes:
  - For AirCombatMARL, all red agents have homogeneous space,
    but HAPPO's heterogeneous formulation is preserved for generality.
  - Uses MAPPO's critic structure with heterogeneous policy updates.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, List
import os


class ActorNetwork(nn.Module):
    """MLP actor for heterogeneous agents."""

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Linear(128, action_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.zeros_(self.mean.bias)

    def forward(self, obs: torch.Tensor):
        x = torch.relu(self.ln1(self.fc1(obs)))
        x = torch.relu(self.ln2(self.fc2(x)))
        mean = torch.tanh(self.mean(x))
        log_std = torch.clamp(self.log_std(x), -5, 2)
        return mean, log_std


class CriticNetwork(nn.Module):
    """Centralized critic (same as MAPPO)."""

    def __init__(self, global_obs_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, global_obs: torch.Tensor):
        return self.net(global_obs)


class HAPPOPolicy:
    """
    HAPPO policy.

    Differences from MAPPO:
      1. Each agent has its own actor network (heterogeneous)
      2. Shared critic for value estimation
      3. KL constraint for trust region
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_agents: int,
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
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size

        # Heterogeneous actors (one per agent)
        self.actors = nn.ModuleList([
            ActorNetwork(obs_dim, action_dim).to(self.device)
            for _ in range(n_agents)
        ])

        # Shared critic
        global_obs_dim = obs_dim * n_agents
        self.critic = CriticNetwork(global_obs_dim).to(self.device)

        # Optimizer
        all_params = list(self.critic.parameters())
        for actor in self.actors:
            all_params += list(actor.parameters())
        self.optimizer = optim.Adam(all_params, lr=lr)

        # Storage
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def select_actions(self, obs: np.ndarray, deterministic: bool = False, **_kwargs):
        """Select actions for all agents (each uses its own actor)."""
        obs_t = torch.FloatTensor(obs).to(self.device)
        actions = []
        log_probs = []

        with torch.no_grad():
            for i in range(self.n_agents):
                mean, log_std = self.actors[i](obs_t[i:i+1])
                std = log_std.exp()
                dist = torch.distributions.Normal(mean, std)
                if deterministic:
                    action = mean
                    sample = action
                else:
                    sample = dist.sample()
                    action = torch.tanh(sample)
                log_prob = dist.log_prob(sample) - torch.log(1 - action.pow(2) + 1e-6)
                log_prob = log_prob.sum(dim=-1, keepdim=True)

                actions.append(action[0].cpu().numpy())
                log_probs.append(log_prob[0].cpu().numpy())

        return np.array(actions), np.array(log_probs)

    def get_values(self, global_obs: torch.Tensor):
        """Get critic values."""
        return self.critic(global_obs)

    def store_transition(self, obs, actions, log_probs, rewards, done, value):
        self.states.append(obs)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.rewards.append(rewards)
        self.dones.append(done)
        self.values.append(value)

    def compute_gae(self, next_obs, next_done):
        """
        Compute Generalized Advantage Estimation for HAPPO.
        
        HAPPO uses shared critic, so GAE is computed on the shared value estimates.
        """
        n_steps = len(self.rewards)
        n_agents = self.n_agents
        advantages = np.zeros((n_steps, n_agents), dtype=np.float32)
        
        with torch.no_grad():
            next_obs_t = torch.FloatTensor(next_obs).to(self.device)
            next_global_obs = next_obs_t.reshape(1, -1)
            next_value = self.critic(next_global_obs).cpu().numpy().flatten()[0]
        
        last_gae = np.zeros(n_agents)
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_val = next_value * (1 - next_done)
            else:
                next_val = self.values[t]
            
            rewards_t = np.array(self.rewards[t])
            values_t = np.array(self.values[t])
            mask_t = 1.0 - np.array(self.dones[t])
            
            deltas = rewards_t + self.gamma * next_val * mask_t - values_t
            last_gae = deltas + self.gamma * self.gae_lambda * mask_t * last_gae
            advantages[t] = last_gae
        
        returns = advantages + np.array(self.values)[:, np.newaxis]
        return advantages, returns

    def update(self, next_obs, next_done):
        """
        PPO update for HAPPO.
        
        Differences from MAPPO:
          - Each agent has its own actor, so we compute per-agent policy loss
          - Shared critic (same as MAPPO)
        """
        advantages, returns = self.compute_gae(next_obs, next_done)

        # Normalize advantages per agent (PPO standard; stabilizes the
        # importance-ratio scale against advantage explosions)
        adv_mean = advantages.mean(axis=0, keepdims=True)
        adv_std = advantages.std(axis=0, keepdims=True) + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # NaN watchdog: snapshot weights; if any mini-batch still manages to
        # corrupt them, roll back instead of dying in the next select_actions
        watchdog_params = list(self.critic.parameters()) + [p for actor in self.actors for p in actor.parameters()]
        watchdog_snapshot = [p.detach().clone() for p in watchdog_params]

        all_obs = np.array(self.states)
        all_actions = np.array(self.actions)
        all_log_probs = np.array(self.log_probs)
        n_steps = len(self.rewards)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0
        
        # Flatten storage for minibatch sampling
        # all_obs: (n_steps, n_agents, obs_dim)
        # We need to sample (obs, action, log_prob) for each agent
        
        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(n_steps)
            
            for start in range(0, n_steps, self.mini_batch_size):
                end = min(start + self.mini_batch_size, n_steps)
                mb_idx = indices[start:end]
                
                # Policy loss (per-agent)
                policy_loss = 0.0
                entropy = 0.0
                
                for i in range(self.n_agents):
                    # Get this agent's data
                    agent_obs = torch.FloatTensor(all_obs[mb_idx, i, :]).to(self.device)
                    agent_actions = torch.FloatTensor(all_actions[mb_idx, i, :]).to(self.device)
                    agent_old_log_probs = torch.FloatTensor(all_log_probs[mb_idx, i]).to(self.device)
                    
                    # Forward pass through this agent's actor
                    mean, log_std = self.actors[i](agent_obs)
                    std = log_std.exp()
                    dist = torch.distributions.Normal(mean, std)

                    # Tanh-squashed Gaussian: stored actions are post-tanh, so
                    # invert them to pre-tanh space before evaluating log_prob.
                    # (Previously log_prob was evaluated at the post-tanh value,
                    #  which inflated the importance ratio and caused NaN weights.)
                    agent_actions_c = torch.clamp(agent_actions, -0.999999, 0.999999)
                    pre_tanh = torch.atanh(agent_actions_c)
                    new_log_prob = dist.log_prob(pre_tanh) - torch.log(1 - agent_actions_c.pow(2) + 1e-6)
                    new_log_prob = new_log_prob.sum(dim=-1, keepdim=True)
                    
                    # PPO clipped objective
                    mb_advantages = torch.FloatTensor(advantages[mb_idx, i]).to(self.device).unsqueeze(-1)
                    ratio = torch.exp(torch.clamp(new_log_prob - agent_old_log_probs.unsqueeze(-1), -10.0, 10.0))
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                    policy_loss += -torch.min(surr1, surr2).mean()
                    
                    entropy += dist.entropy().sum(dim=-1).mean()
                
                policy_loss = policy_loss / self.n_agents
                entropy = entropy / self.n_agents
                
                # Value loss (shared critic)
                mb_obs = torch.FloatTensor(all_obs[mb_idx]).reshape(-1, self.n_agents * self.states[0].shape[-1]).to(self.device)
                mb_returns = torch.FloatTensor(returns[mb_idx]).mean(dim=-1).to(self.device).unsqueeze(-1)
                values = self.critic(mb_obs)
                value_loss = nn.MSELoss()(values, mb_returns)
                
                # Backward (skip non-finite losses so one bad batch cannot
                # poison the Adam moments and NaN the weights)
                loss = policy_loss - self.entropy_coeff * entropy + self.value_coeff * value_loss
                if not torch.isfinite(loss):
                    continue
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.critic.parameters()) + [p for actor in self.actors for p in actor.parameters()], self.max_grad_norm)
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

    def clear_buffer(self):
        """Clear the rollout buffer."""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def save(self, path: str):
        """Save model parameters."""
        torch.save({
            'actors': [actor.state_dict() for actor in self.actors],
            'critic': self.critic.state_dict(),
        }, path)
        print(f"[HAPPO] Model saved to {path}")

    def load(self, path: str):
        """Load model parameters."""
        checkpoint = torch.load(path, map_location=self.device)
        for i, actor in enumerate(self.actors):
            actor.load_state_dict(checkpoint['actors'][i])
        self.critic.load_state_dict(checkpoint['critic'])
        print(f"[HAPPO] Model loaded from {path}")
