"""
red_policy.py
=============
Multi-agent RL policies for red team.

Implements:
  - ActorNetwork (MLP):         obs_dim -> 256 -> 128 -> action_dim
  - CriticNetwork:              Centralized value function
  - MAPPOPolicy:                Standard MAPPO with PPO clipping

Paper reference:
  Yu et al., "The Surprising Effectiveness of PPO in Cooperative
  Multi-Agent Games", NeurIPS 2021.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple, List
from collections import deque
import random


# ============================================================
#  Actor Networks
# ============================================================

class ActorNetwork(nn.Module):
    """Standard MLP actor (MAPPO baseline). Supports hybrid continuous+discrete actions."""

    def __init__(self, obs_dim: int, action_dim: int, n_fire_targets: int = 0):
        """
        Args:
            obs_dim: observation dimension
            action_dim: continuous action dimension (flight: pitch, roll, yaw, throttle)
            n_fire_targets: number of discrete fire targets (n_blue+1, last="no fire").
                            0 = no discrete head (backward compatible).
        """
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        # Continuous head (flight controls)
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Linear(128, action_dim)
        # Discrete head (fire target selection)
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

    def forward(self, obs: torch.Tensor):
        """Returns (mean, log_std) or (mean, log_std, fire_logits) if hybrid."""
        x = torch.relu(self.ln1(self.fc1(obs)))
        x = torch.relu(self.ln2(self.fc2(x)))
        mean = torch.tanh(self.mean(x))
        log_std = torch.clamp(self.log_std(x), -5, 2)
        # NaN protection (applied here so all callers are safe)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=1.0, neginf=-1.0)
        log_std = torch.nan_to_num(log_std, nan=-5.0, posinf=2.0, neginf=-5.0)
        if self.has_discrete:
            fire_logits = self.fire_logits(x)
            return mean, log_std, fire_logits
        return mean, log_std

    def get_action(self, obs: torch.Tensor, deterministic: bool = False,
                   action_mask: torch.Tensor = None):
        """
        Sample hybrid action.

        Args:
            obs: (batch, obs_dim)
            deterministic: if True, use mean for continuous actions
            action_mask: (batch, n_fire_targets) large negative values mask invalid targets.
                         Only used when has_discrete=True.

        Returns:
            Without discrete: (cont_action, log_prob)  -- action in [-1,1], log_prob (batch,1)
            With discrete: (cont_action, fire_action, combined_log_prob)
        """
        if self.has_discrete:
            mean, log_std, fire_logits = self.forward(obs)
        else:
            mean, log_std = self.forward(obs)
        
        std = log_std.exp().clamp(min=1e-6)
        dist = torch.distributions.Normal(mean, std)

        if deterministic:
            # Use mean directly (already in [-1,1] due to tanh)
            cont_action = mean
            # Log-prob under distribution (for API compatibility)
            log_prob = dist.log_prob(cont_action) - torch.log(1 - cont_action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        else:
            sample = dist.sample()
            cont_action = torch.tanh(sample)
            log_prob = dist.log_prob(sample) - torch.log(1 - cont_action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)

        if self.has_discrete:
            # Apply action mask: masked logits -> -inf probability
            if action_mask is not None:
                fire_logits = fire_logits + action_mask
            fire_dist = torch.distributions.Categorical(logits=fire_logits)
            if deterministic:
                fire_action = fire_logits.argmax(dim=-1)  # (batch,)
            else:
                fire_action = fire_dist.sample()  # (batch,)
            fire_log_prob = fire_dist.log_prob(fire_action).unsqueeze(-1)
            # Combined log-prob
            total_log_prob = log_prob + fire_log_prob
            return cont_action, fire_action, total_log_prob
        else:
            return cont_action, log_prob

    def evaluate(self, obs: torch.Tensor, action):
        """
        Evaluate log-prob and entropy for given action(s).

        Args:
            obs: (batch, obs_dim)
            action: Tensor (batch, action_dim) if no discrete,
                    or tuple (cont_action, fire_action) if hybrid.
        Returns:
            (log_prob, entropy) both shape (batch, 1)
        """
        if self.has_discrete:
            cont_action, fire_action = action
            mean, log_std, fire_logits = self.forward(obs)
        else:
            cont_action = action
            mean, log_std = self.forward(obs)
        std = log_std.exp().clamp(min=1e-6)
        dist = torch.distributions.Normal(mean, std)

        cont_log_prob = dist.log_prob(cont_action) - torch.log(1 - cont_action.pow(2) + 1e-6)
        cont_log_prob = cont_log_prob.sum(dim=-1, keepdim=True)

        if self.has_discrete:
            fire_dist = torch.distributions.Categorical(logits=fire_logits)
            fire_log_prob = fire_dist.log_prob(fire_action).unsqueeze(-1)
            log_prob = cont_log_prob + fire_log_prob
            entropy = dist.entropy().sum(dim=-1, keepdim=True) + fire_dist.entropy().unsqueeze(-1)
        else:
            log_prob = cont_log_prob
            entropy = dist.entropy().sum(dim=-1, keepdim=True)

        return log_prob, entropy


# ============================================================
#  Critic Network
# ============================================================

class CriticNetwork(nn.Module):
    """Centralized critic (uses all agents' observations concatenated)."""

    def __init__(self, total_obs_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(total_obs_dim, 512)
        self.ln1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 256)
        self.ln2 = nn.LayerNorm(256)
        self.value = nn.Linear(256, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.zeros_(self.value.bias)

    def forward(self, obs: torch.Tensor):
        x = torch.relu(self.ln1(self.fc1(obs)))
        x = torch.relu(self.ln2(self.fc2(x)))
        return self.value(x)


# ============================================================
#  MAPPO Policy (baseline)
# ============================================================

class MAPPOPolicy:
    """Standard MAPPO policy with PPO clipping. Supports hybrid continuous+discrete actions."""

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
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.n_fire_targets = n_fire_targets
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Networks
        self.actor = ActorNetwork(obs_dim, action_dim, n_fire_targets=n_fire_targets).to(self.device)
        self.critic = CriticNetwork(obs_dim * n_agents).to(self.device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Rollout buffer
        self.obs_buf = []
        self.actions_buf = []
        self.fire_actions_buf = []  # discrete fire actions (per-step ints)
        self.log_probs_buf = []
        self.rewards_buf = []
        self.values_buf = []
        self.dones_buf = []

        # Training counter (for SIL scheduling)
        self.update_count = 0

    def select_actions(self, obs: np.ndarray, action_mask: np.ndarray = None, deterministic: bool = False):
        """
        Select actions for all agents.

        Args:
            obs: (n_agents, obs_dim) numpy array
            action_mask: (n_agents, n_fire_targets) numpy array (only used if n_fire_targets>0)
            deterministic: if True, use mean action (no sampling). Used for evaluation.

        Returns:
            Without discrete: (cont_actions_list, log_probs_list)
            With discrete: (cont_actions_list, fire_actions_list, log_probs_list)
        """
        obs_t = torch.FloatTensor(obs).to(self.device)
        if self.n_fire_targets > 0 and action_mask is not None:
            mask_t = torch.FloatTensor(action_mask).to(self.device)
            cont_actions_t, fire_actions_t, log_probs_t = self.actor.get_action(
                obs_t, deterministic=deterministic, action_mask=mask_t)
            cont_actions = cont_actions_t.detach().cpu().numpy()
            fire_actions = fire_actions_t.detach().cpu().numpy().astype(int)
            log_probs = log_probs_t.detach().cpu().numpy()
            return (
                [cont_actions[i] for i in range(self.n_agents)],
                [fire_actions[i] for i in range(self.n_agents)],
                [log_probs[i] for i in range(self.n_agents)],
            )
        else:
            cont_actions_t, log_probs_t = self.actor.get_action(obs_t, deterministic=deterministic)
            cont_actions = cont_actions_t.detach().cpu().numpy()
            log_probs = log_probs_t.detach().cpu().numpy()
            return [cont_actions[i] for i in range(self.n_agents)], [log_probs[i] for i in range(self.n_agents)]

    def store_transition(self, obs, actions, log_probs, rewards, dones, values,
                         fire_actions=None):
        """Store a transition in the rollout buffer."""
        self.obs_buf.append(obs)
        self.actions_buf.append(actions)
        if fire_actions is not None:
            self.fire_actions_buf.append(fire_actions)
        self.log_probs_buf.append(log_probs)
        self.rewards_buf.append(rewards)
        self.dones_buf.append(dones)
        self.values_buf.append(values)

    def compute_gae(self, next_obs, next_done):
        """Compute Generalized Advantage Estimation."""
        n_steps = len(self.rewards_buf)
        n_agents = self.n_agents
        advantages = np.zeros((n_steps, n_agents), dtype=float)

        with torch.no_grad():
            next_obs_t = torch.FloatTensor(next_obs).to(self.device)
            next_global_obs = next_obs_t.reshape(1, -1)
            next_value = self.critic(next_global_obs).cpu().numpy().flatten()[0]

        last_gae = np.zeros(n_agents)
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_val = next_value * (1 - next_done)
            else:
                next_val = self.values_buf[t + 1]

            rewards_t = np.array(self.rewards_buf[t])
            values_t = np.array(self.values_buf[t])
            mask_t = 1.0 - np.array(self.dones_buf[t])

            deltas = rewards_t + self.gamma * next_val * mask_t - values_t
            last_gae = deltas + self.gamma * self.gae_lambda * mask_t * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(self.values_buf)[:, np.newaxis]
        return advantages, returns

    def _ppo_update_step(self, obs_t, old_actions_t, old_log_probs_t, advantages, returns, mb_idx,
                         old_fire_actions_t=None):
        """Single PPO update step (shared by MAPPO and ADAP). Handles hybrid continuous+discrete."""
        if self.n_fire_targets > 0:
            mean, log_std, fire_logits = self.actor.forward(obs_t)
        else:
            mean, log_std = self.actor.forward(obs_t)
        # Double NaN guard (forward already has nan_to_num, but weights may be corrupted)
        if torch.isnan(mean).any() or torch.isnan(log_std).any():
            print("[WARN] NaN in actor output during PPO update, skipping this mini-batch")
            return 0.0, 0.0, 0.0
        std = log_std.exp().clamp(min=1e-6)
        dist = torch.distributions.Normal(mean, std)

        # Continuous log-prob (tanh-squashed Gaussian: invert the stored
        # post-tanh actions to pre-tanh space before evaluating log_prob;
        # previously the ratio was inflated and could NaN the weights)
        old_actions_c = torch.clamp(old_actions_t, -0.999999, 0.999999)
        pre_tanh = torch.atanh(old_actions_c)
        new_cont_log_prob = dist.log_prob(pre_tanh) - torch.log(1 - old_actions_c.pow(2) + 1e-6)
        new_cont_log_prob = new_cont_log_prob.sum(dim=-1, keepdim=True)

        if self.n_fire_targets > 0 and old_fire_actions_t is not None:
            # Discrete log-prob
            fire_dist = torch.distributions.Categorical(logits=fire_logits)
            new_fire_log_prob = fire_dist.log_prob(old_fire_actions_t.squeeze(-1)).unsqueeze(-1)
            new_log_probs = new_cont_log_prob + new_fire_log_prob
            entropy = dist.entropy().sum(dim=-1).mean() + fire_dist.entropy().mean()
        else:
            new_log_probs = new_cont_log_prob
            entropy = dist.entropy().sum(dim=-1).mean()

        ratio = torch.exp(torch.clamp(new_log_probs - old_log_probs_t, -10.0, 10.0))
        mb_advantages = torch.FloatTensor(
            advantages[mb_idx]
        ).to(self.device).reshape(-1, 1)

        surr1 = ratio * mb_advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        n_mb = obs_t.shape[0] // self.n_agents
        global_obs_t = obs_t[:n_mb * self.n_agents].reshape(n_mb, -1)
        mb_returns_val = torch.FloatTensor(
            returns[mb_idx][:n_mb].mean(axis=1)
        ).to(self.device).reshape(-1, 1)
        values = self.critic(global_obs_t)
        value_loss = nn.MSELoss()(values, mb_returns_val)

        # Backward (skip non-finite losses so one bad batch cannot
        # poison the Adam moments and NaN the weights)
        actor_loss = policy_loss - self.entropy_coeff * entropy
        if not torch.isfinite(actor_loss) or not torch.isfinite(value_loss):
            print("[WARN] Non-finite loss in PPO update, skipping this mini-batch")
            return 0.0, 0.0, 0.0
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        return policy_loss.item(), value_loss.item(), entropy.item()

    def update(self, next_obs, next_done):
        """Perform PPO update. Handles hybrid actions when n_fire_targets>0."""
        advantages, returns = self.compute_gae(next_obs, next_done)

        # Normalize advantages (PPO standard; stabilizes the importance-ratio
        # scale). Only used by the PPO branch — the SIL branch draws its own
        # advantages from the SIL replay buffer, so it is unaffected.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # NaN watchdog: snapshot weights; if any mini-batch still manages to
        # corrupt them (non-finite), roll back to this snapshot below.
        watchdog_params = list(self.actor.parameters()) + list(self.critic.parameters())
        watchdog_snapshot = [p.detach().clone() for p in watchdog_params]

        all_obs = np.array(self.obs_buf)
        all_actions = np.array(self.actions_buf)
        all_log_probs = np.array(self.log_probs_buf)

        has_fire = self.n_fire_targets > 0 and len(self.fire_actions_buf) > 0
        if has_fire:
            all_fire_actions = np.array(self.fire_actions_buf)

        n_steps = len(self.rewards_buf)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(n_steps)

            for start in range(0, n_steps, self.mini_batch_size):
                end = min(start + self.mini_batch_size, n_steps)
                mb_idx = indices[start:end]

                obs_t = torch.FloatTensor(all_obs[mb_idx]).to(self.device).reshape(-1, self.obs_dim)
                old_actions_t = torch.FloatTensor(all_actions[mb_idx]).to(self.device).reshape(-1, self.action_dim)
                old_log_probs_t = torch.FloatTensor(all_log_probs[mb_idx]).to(self.device).reshape(-1, 1)

                old_fire_t = None
                if has_fire:
                    old_fire_t = torch.LongTensor(all_fire_actions[mb_idx]).to(self.device).reshape(-1, 1)

                pl, vl, ent = self._ppo_update_step(
                    obs_t, old_actions_t, old_log_probs_t, advantages, returns, mb_idx,
                    old_fire_actions_t=old_fire_t,
                )
                total_policy_loss += pl
                total_value_loss += vl
                total_entropy += ent
                n_updates += 1

        self.update_count += 1

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
            'sil_loss': 0.0,
        }

    def clear_buffer(self):
        """Clear the rollout buffer."""
        self.obs_buf = []
        self.actions_buf = []
        self.fire_actions_buf = []
        self.log_probs_buf = []
        self.rewards_buf = []
        self.values_buf = []
        self.dones_buf = []

    def save(self, path: str):
        """Save model parameters."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }, path)
        print(f"[MAPPO] Model saved to {path}")

    def load(self, path: str):
        """Load model parameters."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        print(f"[MAPPO] Model loaded from {path}")


