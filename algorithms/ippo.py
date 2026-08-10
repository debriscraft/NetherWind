"""
ippo.py
=========
IPPO (Independent PPO) baseline.

Reference: de Witt et al., "Independent PPO", NeurIPS 2020 (Deep RL Workshop).

Key features:
  - Each agent learns independently (no centralized critic)
  - Each agent has its own actor and critic
  - No credit assignment (each agent optimizes its own reward)
  - Simple baseline for multi-agent RL

Implementation:
  - Independent actors and critics (no parameter sharing)
  - Each agent stores its own trajectory
  - PPO update per agent independently
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional, List


class IndependentActor(nn.Module):
    """Actor network for single agent (MLP)."""
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs: torch.Tensor):
        x = F.tanh(self.fc1(obs))
        x = F.tanh(self.fc2(x))
        mean = torch.tanh(self.mean(x))
        log_std = torch.clamp(self.log_std(x), -5, 2)
        return mean, log_std


class IndependentCritic(nn.Module):
    """Critic network for single agent (MLP)."""
    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(self, obs: torch.Tensor):
        return self.net(obs)


class IPPOPolicy:
    """
    Independent PPO policy.
    
    Each agent has its own actor and critic, and learns independently.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_agents: int,
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
        
        # Each agent has independent actor and critic
        self.actors = nn.ModuleList([
            IndependentActor(obs_dim, action_dim).to(self.device)
            for _ in range(n_agents)
        ])
        self.critics = nn.ModuleList([
            IndependentCritic(obs_dim).to(self.device)
            for _ in range(n_agents)
        ])
        
        # Optimizers (one per agent, or shared?)
        # Using per-agent optimizers for true independence
        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=lr)
            for actor in self.actors
        ]
        self.critic_optimizers = [
            optim.Adam(critic.parameters(), lr=lr)
            for critic in self.critics
        ]
        
        # Per-agent storage
        self.states = [[] for _ in range(n_agents)]
        self.actions = [[] for _ in range(n_agents)]
        self.log_probs = [[] for _ in range(n_agents)]
        self.rewards = [[] for _ in range(n_agents)]
        self.dones = [[] for _ in range(n_agents)]
        self.values = [[] for _ in range(n_agents)]
        
    def select_actions(self, obs: np.ndarray):
        """
        Select actions independently for each agent.
        
        Args:
            obs: (n_agents, obs_dim)
        Returns:
            actions: (n_agents, action_dim)
            log_probs: (n_agents, 1)
        """
        obs_t = torch.FloatTensor(obs).to(self.device)
        actions = []
        log_probs = []
        
        for i in range(self.n_agents):
            with torch.no_grad():
                mean, log_std = self.actors[i](obs_t[i:i+1])
                std = log_std.exp()
                dist = torch.distributions.Normal(mean, std)
                sample = dist.sample()
                action = torch.tanh(sample)
                log_prob = dist.log_prob(sample) - torch.log(1 - action.pow(2) + 1e-6)
                log_prob = log_prob.sum(dim=-1, keepdim=True)
                
            actions.append(action[0].cpu().numpy())
            log_probs.append(log_prob[0].cpu().numpy())
            
        return np.array(actions), np.array(log_probs)
        
    def get_values(self, obs: torch.Tensor):
        """
        Get critic values for all agents.
        
        Returns:
            values: (n_agents,) - value for each agent's own critic
        """
        values = []
        for i in range(self.n_agents):
            agent_obs = obs[i].unsqueeze(0)  # (1, obs_dim)
            value = self.critics[i](agent_obs)
            values.append(value.item())
        return np.array(values)
        
    def store_transition(self, obs, actions, log_probs, rewards, done, values):
        """
        Store transition for each agent independently.

        Args:
            obs: (n_agents, obs_dim)
            actions: (n_agents, action_dim)
            log_probs: (n_agents, 1)
            rewards: (n_agents,) array-like or scalar
            done: scalar
            values: (n_agents,) array-like or scalar
        """
        rewards = np.asarray(rewards, dtype=np.float64).flatten()
        values = np.asarray(values, dtype=np.float64).flatten()
        for i in range(self.n_agents):
            self.states[i].append(obs[i])
            self.actions[i].append(actions[i])
            self.log_probs[i].append(log_probs[i])
            self.rewards[i].append(float(rewards[i]) if rewards.size > 1 else float(rewards))
            self.dones[i].append(done)
            self.values[i].append(float(values[i]) if values.size > 1 else float(values))
        
    def compute_gae(self, next_obs, next_done):
        """
        Compute GAE advantages for each agent independently.
        
        Returns:
            advantages: (n_agents, n_steps)
            returns: (n_agents, n_steps)
        """
        n_steps = len(self.rewards[0])  # All agents have same length
        advantages = np.zeros((self.n_agents, n_steps), dtype=np.float32)
        returns = np.zeros((self.n_agents, n_steps), dtype=np.float32)
        
        for i in range(self.n_agents):
            with torch.no_grad():
                next_obs_t = torch.FloatTensor(next_obs[i:i+1]).to(self.device)
                next_value = self.critics[i](next_obs_t).item()
            
            last_gae = 0.0
            for t in reversed(range(n_steps)):
                if t == n_steps - 1:
                    next_val = next_value * (1 - next_done)
                else:
                    next_val = self.values[i][t + 1]
                
                reward = self.rewards[i][t]
                value = self.values[i][t]
                mask = 1.0 - self.dones[i][t]
                
                delta = reward + self.gamma * next_val * mask - value
                last_gae = delta + self.gamma * self.gae_lambda * mask * last_gae
                advantages[i, t] = last_gae
            
            returns[i] = advantages[i] + np.array(self.values[i])
        
        return advantages, returns
        
    def update(self, next_obs, next_done):
        """
        PPO update for each agent independently.
        """
        advantages, returns = self.compute_gae(next_obs, next_done)

        # Normalize advantages per agent (PPO standard; stabilizes the
        # importance-ratio scale against advantage explosions)
        adv_mean = advantages.mean(axis=1, keepdims=True)
        adv_std = advantages.std(axis=1, keepdims=True) + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # NaN watchdog: snapshot weights; if any mini-batch still manages to
        # corrupt them (non-finite), roll back to this snapshot below.
        watchdog_params = [p for actor in self.actors for p in actor.parameters()] + \
                          [p for critic in self.critics for p in critic.parameters()]
        watchdog_snapshot = [p.detach().clone() for p in watchdog_params]

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0
        
        for i in range(self.n_agents):
            # Get this agent's data
            all_obs = np.array(self.states[i])
            all_actions = np.array(self.actions[i])
            all_log_probs = np.array(self.log_probs[i])
            n_steps = len(all_obs)
            
            for _ in range(self.ppo_epochs):
                indices = np.random.permutation(n_steps)
                
                for start in range(0, n_steps, self.mini_batch_size):
                    end = min(start + self.mini_batch_size, n_steps)
                    mb_idx = indices[start:end]
                    
                    obs_t = torch.FloatTensor(all_obs[mb_idx]).to(self.device)
                    old_actions_t = torch.FloatTensor(all_actions[mb_idx]).to(self.device)
                    old_log_probs_t = torch.FloatTensor(all_log_probs[mb_idx]).to(self.device).unsqueeze(-1)
                    
                    # Forward pass
                    mean, log_std = self.actors[i](obs_t)
                    std = log_std.exp()
                    dist = torch.distributions.Normal(mean, std)
                    
                    # Tanh-squashed Gaussian: invert stored post-tanh actions
                    # to pre-tanh space before evaluating log_prob (prevents
                    # inflated importance ratios and NaN weights)
                    old_actions_c = torch.clamp(old_actions_t, -0.999999, 0.999999)
                    pre_tanh = torch.atanh(old_actions_c)
                    new_log_prob = dist.log_prob(pre_tanh) - torch.log(1 - old_actions_c.pow(2) + 1e-6)
                    new_log_prob = new_log_prob.sum(dim=-1, keepdim=True)
                    
                    # PPO clipped objective
                    mb_advantages = torch.FloatTensor(advantages[i, mb_idx]).to(self.device).unsqueeze(-1)
                    ratio = torch.exp(torch.clamp(new_log_prob - old_log_probs_t, -10.0, 10.0))
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    entropy = dist.entropy().sum(dim=-1).mean()
                    
                    # Value loss
                    mb_returns = torch.FloatTensor(returns[i, mb_idx]).to(self.device).unsqueeze(-1)
                    values = self.critics[i](obs_t)
                    value_loss = nn.MSELoss()(values, mb_returns)
                    
                    # Backward (skip non-finite losses so one bad batch cannot
                    # poison the Adam moments and NaN the weights)
                    actor_loss = policy_loss - self.entropy_coeff * entropy
                    if not torch.isfinite(actor_loss) or not torch.isfinite(value_loss):
                        continue
                    self.actor_optimizers[i].zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(self.actors[i].parameters(), self.max_grad_norm)
                    self.actor_optimizers[i].step()
                    
                    self.critic_optimizers[i].zero_grad()
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(self.critics[i].parameters(), self.max_grad_norm)
                    self.critic_optimizers[i].step()
                    
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
        
        # Clear buffers
        for i in range(self.n_agents):
            self.states[i] = []
            self.actions[i] = []
            self.log_probs[i] = []
            self.rewards[i] = []
            self.dones[i] = []
            self.values[i] = []
        
        return {
            'policy_loss': total_policy_loss / max(n_updates, 1),
            'value_loss': total_value_loss / max(n_updates, 1),
            'entropy': total_entropy / max(n_updates, 1),
        }
        
    def save(self, path: str):
        """Save model parameters."""
        torch.save({
            'actors': [actor.state_dict() for actor in self.actors],
            'critics': [critic.state_dict() for critic in self.critics],
        }, path)
        print(f"[IPPO] Model saved to {path}")
        
    def load(self, path: str):
        """Load model parameters."""
        checkpoint = torch.load(path, map_location=self.device)
        for i, actor in enumerate(self.actors):
            actor.load_state_dict(checkpoint['actors'][i])
        for i, critic in enumerate(self.critics):
            critic.load_state_dict(checkpoint['critics'][i])
        print(f"[IPPO] Model loaded from {path}")


if __name__ == '__main__':
    policy = IPPOPolicy(obs_dim=19*6, action_dim=4, n_agents=3)
    print("IPPO policy created successfully")
