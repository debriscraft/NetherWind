"""
sac.py
=======
SAC (Soft Actor-Critic) baseline for continuous control.

Reference: Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy
           Deep Reinforcement Learning with a Stochastic Actor", ICML 2018.

Key features:
  - Maximum entropy RL (encourages exploration)
  - Off-policy (sample efficient)
  - Twin Q-networks (reduces overestimation bias)
  - Automatic temperature tuning (α)

Adaptation for multi-agent:
  - Each agent has independent SAC agent
  - Centralized training with decentralized execution (CTDE)
  - Shared replay buffer (optional) or independent buffers
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional, List, Tuple


class SACActor(nn.Module):
    """Actor network for SAC (outputs mean and log_std)."""
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        # LayerNorm on the input path is essential: obs contains raw
        # meter-scale values (~1e4); without normalization the tanh head
        # saturates and the policy degenerates
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs: torch.Tensor):
        x = F.relu(self.ln1(self.fc1(obs)))
        x = F.relu(self.ln2(self.fc2(x)))
        mean = torch.tanh(self.mean(x))
        log_std = torch.clamp(self.log_std(x), -5, 2)
        return mean, log_std


class SACCritic(nn.Module):
    """Twin Q-networks for SAC."""
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        # Q1 network (LayerNorm: raw meter-scale obs ~1e4)
        self.q1_fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.q1_ln1 = nn.LayerNorm(hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_ln2 = nn.LayerNorm(hidden_dim)
        self.q1_out = nn.Linear(hidden_dim, 1)
        
        # Q2 network
        self.q2_fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.q2_ln1 = nn.LayerNorm(hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_ln2 = nn.LayerNorm(hidden_dim)
        self.q2_out = nn.Linear(hidden_dim, 1)
        
    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        x = torch.cat([obs, action], dim=-1)
        
        q1 = F.relu(self.q1_ln1(self.q1_fc1(x)))
        q1 = F.relu(self.q1_ln2(self.q1_fc2(q1)))
        q1 = self.q1_out(q1)
        
        q2 = F.relu(self.q2_ln1(self.q2_fc1(x)))
        q2 = F.relu(self.q2_ln2(self.q2_fc2(q2)))
        q2 = self.q2_out(q2)
        
        return q1, q2
    
    def q1_forward(self, obs: torch.Tensor, action: torch.Tensor):
        """Only Q1 forward (for actor update)."""
        x = torch.cat([obs, action], dim=-1)
        q1 = F.relu(self.q1_ln1(self.q1_fc1(x)))
        q1 = F.relu(self.q1_ln2(self.q1_fc2(q1)))
        q1 = self.q1_out(q1)
        return q1


class SACPolicy:
    """
    SAC policy for multi-agent air combat.
    
    Each agent has independent SAC agent (independent learning).
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_agents: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        automatic_alpha_tuning: bool = True,
        replay_capacity: int = 100000,
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.automatic_alpha_tuning = automatic_alpha_tuning
        
        # Always initialize alpha (will be updated if automatic_alpha_tuning=True)
        self.alpha = alpha
        
        # Each agent has independent SAC agent
        self.actors = nn.ModuleList([
            SACActor(obs_dim, action_dim).to(self.device)
            for _ in range(n_agents)
        ])
        self.critics = nn.ModuleList([
            SACCritic(obs_dim, action_dim).to(self.device)
            for _ in range(n_agents)
        ])
        self.target_critics = nn.ModuleList([
            SACCritic(obs_dim, action_dim).to(self.device)
            for _ in range(n_agents)
        ])
        
        # Copy weights to target networks
        for i in range(n_agents):
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())
        
        # Optimizers
        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=lr)
            for actor in self.actors
        ]
        self.critic_optimizers = [
            optim.Adam(critic.parameters(), lr=lr)
            for critic in self.critics
        ]
        
        # Alpha (temperature parameter)
        if automatic_alpha_tuning:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.target_entropy = -action_dim  # -dim(action_space)
        else:
            self.alpha = alpha
        
        # Replay buffers (one per agent, or shared?)
        # Using shared buffer for simplicity
        self.replay_buffer = []
        self.replay_capacity = replay_capacity
        self._pending = None       # one-step pending transition (see store_transition)
        self.entropy_coeff = 0.0   # interface compatibility with train.py
        
    def select_actions(self, obs: np.ndarray, evaluate: bool = False,
                       deterministic: bool = False, **_kwargs):
        """
        Select actions for all agents.

        Args:
            obs: (n_agents, obs_dim)
            evaluate/deterministic: If True, use mean action (no sampling)
        Returns:
            (actions (n_agents, action_dim), log_probs (n_agents, 1) zeros)
        """
        evaluate = evaluate or deterministic
        obs_t = torch.FloatTensor(obs).to(self.device)
        actions = []

        for i in range(self.n_agents):
            with torch.no_grad():
                mean, log_std = self.actors[i](obs_t[i:i+1])

                if evaluate:
                    action = mean
                else:
                    std = log_std.exp()
                    dist = torch.distributions.Normal(mean, std)
                    sample = dist.sample()
                    action = torch.tanh(sample)

            actions.append(action[0].cpu().numpy())

        log_probs = np.zeros((self.n_agents, 1), dtype=np.float32)
        return np.array(actions, dtype=np.float32), log_probs

    def store_transition(self, obs, actions, log_probs=None, rewards=None,
                         done=0.0, values=None, next_obs=None, **_kwargs):
        """Store transition in replay buffer.

        Compatible with train.py's call signature
        store_transition(obs, actions, log_probs, rewards, done, value).
        Since train.py does not deliver next_obs, the previous transition is
        committed when the next observation arrives (one-step pending);
        terminal transitions are committed immediately with done=1.
        """
        obs = np.asarray(obs, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        team_r = float(np.sum(rewards))

        if next_obs is not None:
            self._append(obs, actions, team_r, np.asarray(next_obs, dtype=np.float32), float(done))
            return

        if getattr(self, '_pending', None) is not None:
            p_obs, p_act, p_r = self._pending
            self._append(p_obs, p_act, p_r, obs, 0.0)
            self._pending = None

        if done:
            self._append(obs, actions, team_r, obs, 1.0)
        else:
            self._pending = (obs, actions, team_r)

    def _append(self, obs, actions, team_r, next_obs, done):
        self.replay_buffer.append((
            obs.reshape(-1),
            actions.reshape(-1),
            team_r,             # Team reward
            next_obs.reshape(-1),
            float(done),
        ))
        # Limit buffer size
        if len(self.replay_buffer) > self.replay_capacity:
            self.replay_buffer.pop(0)
            
    def sample_replay_buffer(self, batch_size: int):
        """Sample batch from replay buffer."""
        if len(self.replay_buffer) < batch_size:
            return None
        
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]
        
        states = torch.FloatTensor([b[0] for b in batch]).to(self.device)
        actions = torch.FloatTensor([b[1] for b in batch]).to(self.device)
        rewards = torch.FloatTensor([b[2] for b in batch]).to(self.device).unsqueeze(-1)
        next_states = torch.FloatTensor([b[3] for b in batch]).to(self.device)
        dones = torch.FloatTensor([b[4] for b in batch]).to(self.device).unsqueeze(-1)
        
        return states, actions, rewards, next_states, dones
        
    def update(self, *args, gradient_steps: int = 100, batch_size: int = 256, **kwargs):
        """train.py calls update(next_obs, next_done) once per episode;
        SAC runs `gradient_steps` mini-batch updates instead."""
        # NaN watchdog: snapshot weights; if TD divergence still manages to
        # corrupt them (non-finite), roll back to this snapshot below.
        watchdog_params = [p for a in self.actors for p in a.parameters()] + \
                          [p for c in self.critics for p in c.parameters()]
        if self.automatic_alpha_tuning:
            watchdog_params = watchdog_params + [self.log_alpha]
        watchdog_snapshot = [p.detach().clone() for p in watchdog_params]

        actor_l, critic_l, alpha_l = [], [], []
        for _ in range(gradient_steps):
            if len(self.replay_buffer) < batch_size:
                break
            out = self._update_once(batch_size)
            if out is None:
                break  # non-finite loss skipped; watchdog below decides
            actor_l.append(out['actor_loss'])
            critic_l.append(out['critic_loss'])
            alpha_l.append(out['alpha_loss'])

        # NaN watchdog rollback (also re-sync target critics + alpha)
        bad = any(not torch.isfinite(p).all() for p in watchdog_params)
        if bad:
            with torch.no_grad():
                for p, s in zip(watchdog_params, watchdog_snapshot):
                    p.copy_(s)
                for i in range(self.n_agents):
                    self.target_critics[i].load_state_dict(self.critics[i].state_dict())
                if self.automatic_alpha_tuning:
                    self.alpha = self.log_alpha.exp().detach()
            print("[WARN] NaN/Inf weights after SAC update; rolled back to pre-update snapshot")

        return {
            'actor_loss': float(np.mean(actor_l)) if actor_l else 0.0,
            'critic_loss': float(np.mean(critic_l)) if critic_l else 0.0,
            'alpha_loss': float(np.mean(alpha_l)) if alpha_l else 0.0,
            'policy_loss': float(np.mean(actor_l)) if actor_l else 0.0,
            'value_loss': float(np.mean(critic_l)) if critic_l else 0.0,
        }

    def _update_once(self, batch_size: int = 256):
        """
        Single SAC mini-batch update.

        1. Sample batch from replay buffer
        2. Update critics (MSE loss with entropy term)
        3. Update actor (maximize Q - α * entropy)
        4. Update α (temperature parameter)
        5. Soft update target networks
        """
        if len(self.replay_buffer) < batch_size:
            return {'actor_loss': 0.0, 'critic_loss': 0.0, 'alpha_loss': 0.0}
        
        # Sample batch
        batch = self.sample_replay_buffer(batch_size)
        if batch is None:
            return {'actor_loss': 0.0, 'critic_loss': 0.0, 'alpha_loss': 0.0}
        
        states, actions, rewards, next_states, dones = batch
        
        # Reshape states to (batch, n_agents, obs_dim)
        obs_dim = states.shape[1] // self.n_agents
        states = states.view(-1, self.n_agents, obs_dim)
        next_states = next_states.view(-1, self.n_agents, obs_dim)
        actions = actions.view(-1, self.n_agents, self.action_dim)
        
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_alpha_loss = 0.0
        
        for i in range(self.n_agents):
            # Get this agent's data
            agent_states = states[:, i, :]
            agent_actions = actions[:, i, :]
            next_agent_states = next_states[:, i, :]
            
            # Compute target Q-values
            with torch.no_grad():
                next_mean, next_log_std = self.actors[i](next_agent_states)
                next_std = next_log_std.exp()
                next_dist = torch.distributions.Normal(next_mean, next_std)
                next_sample = next_dist.sample()
                next_log_prob = next_dist.log_prob(next_sample) - torch.log(1 - torch.tanh(next_sample).pow(2) + 1e-6)
                next_log_prob = next_log_prob.sum(dim=-1, keepdim=True)
                
                next_q1, next_q2 = self.target_critics[i](next_agent_states, torch.tanh(next_sample))
                next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_prob
                
                target_q = rewards + self.gamma * (1 - dones) * next_q
                # Clamp TD target: the entropy bonus (-alpha*log_prob) is
                # unbounded and, combined with bootstrapping, produced
                # exponential Q divergence (loss ~x90/50eps -> inf -> frozen
                # updates). Episode returns in this env are within ~+-150,
                # so +-300 is generous headroom and lossless in practice.
                target_q = torch.clamp(target_q, -300.0, 300.0)
            
            # Update critics
            current_q1, current_q2 = self.critics[i](agent_states, agent_actions)
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            
            # Skip non-finite losses so a diverging TD target cannot poison
            # the Adam moments and NaN the weights (watchdog in update())
            if not torch.isfinite(critic_loss):
                return None
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critics[i].parameters(), 5.0)
            self.critic_optimizers[i].step()
            
            # Update actor
            mean, log_std = self.actors[i](agent_states)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            sample = dist.sample()
            log_prob = dist.log_prob(sample) - torch.log(1 - torch.tanh(sample).pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            
            q1 = self.critics[i].q1_forward(agent_states, torch.tanh(sample))
            actor_loss = (self.alpha * log_prob - q1).mean()
            
            if not torch.isfinite(actor_loss):
                return None
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actors[i].parameters(), 5.0)
            self.actor_optimizers[i].step()
            
            # Update alpha
            if self.automatic_alpha_tuning:
                alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
                
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                
                self.alpha = self.log_alpha.exp()
                total_alpha_loss += alpha_loss.item()
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            
            # Soft update target networks
            for target_param, param in zip(self.target_critics[i].parameters(), self.critics[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'actor_loss': total_actor_loss / self.n_agents,
            'critic_loss': total_critic_loss / self.n_agents,
            'alpha_loss': total_alpha_loss / self.n_agents,
        }
        
    def save(self, path: str):
        """Save model parameters."""
        torch.save({
            'actors': [actor.state_dict() for actor in self.actors],
            'critics': [critic.state_dict() for critic in self.critics],
        }, path)
        print(f"[SAC] Model saved to {path}")
        
    def load(self, path: str):
        """Load model parameters."""
        checkpoint = torch.load(path, map_location=self.device)
        for i, actor in enumerate(self.actors):
            actor.load_state_dict(checkpoint['actors'][i])
        for i, critic in enumerate(self.critics):
            critic.load_state_dict(checkpoint['critics'][i])
        print(f"[SAC] Model loaded from {path}")


if __name__ == '__main__':
    policy = SACPolicy(obs_dim=19*6, action_dim=4, n_agents=3)
    print("SAC policy created successfully")
