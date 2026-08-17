"""
maddpg.py
=========
MADDPG (Multi-Agent Deep Deterministic Policy Gradient) baseline.

Reference: Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments",
           NIPS 2017.

Key features:
  - Actor-critic architecture with deterministic policy
  - Centralized training with decentralized execution (CTDE)
  - Each agent has its own actor and critic
  - Critic uses global state information

Implementation notes (v3.0, rewritten for the AAG-SIL paper experiment framework):
  - Compatible with train.py's on-policy-style driver interface:
      select_actions(obs) -> (actions, log_probs)
      store_transition(obs, actions, log_probs, rewards, done, values)
      update(next_obs, next_done) -> dict of losses
  - Replay buffer stores (s, a, r, s', done) via a one-step pending mechanism,
    because train.py delivers transitions without next_obs.
  - Correct TD target: y = r + gamma * (1 - done) * Q_target(s', a').
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class MADDPGActor(nn.Module):
    """Actor network (decentralized, uses local observation)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        # LayerNorm on the input path is essential: obs contains raw meter-scale
        # values (~1e4); without normalization tanh saturates and the
        # deterministic policy collapses to constant extreme actions
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor):
        x = F.relu(self.ln1(self.fc1(obs)))
        x = F.relu(self.ln2(self.fc2(x)))
        action = torch.tanh(self.fc3(x))  # Bound to [-1, 1]
        return action


class MADDPGCritic(nn.Module):
    """Critic network (centralized, uses global state)."""

    def __init__(self, global_obs_dim: int, global_action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(global_obs_dim + global_action_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, global_obs: torch.Tensor, global_actions: torch.Tensor):
        x = torch.cat([global_obs, global_actions], dim=-1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        q_value = self.fc3(x)
        return q_value


class MADDPGPolicy:
    """
    MADDPG policy for multi-agent air combat (CTDE).

    Each red agent has its own actor and critic; critics take the global
    state and joint action as input.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_agents: int,
        n_fire_targets: int = 0,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.01,
        noise_std: float = 0.1,
        replay_capacity: int = 100000,
        warmup_steps: int = 1000,
        gradient_steps: int = 100,
        batch_size: int = 256,
        max_grad_norm: float = 0.5,
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.replay_capacity = replay_capacity
        self.warmup_steps = warmup_steps
        self.gradient_steps = gradient_steps
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.entropy_coeff = 0.0  # interface compatibility

        self.global_obs_dim = obs_dim * n_agents
        self.global_action_dim = action_dim * n_agents

        self.actors = nn.ModuleList([
            MADDPGActor(obs_dim, action_dim).to(self.device)
            for _ in range(n_agents)
        ])
        self.target_actors = nn.ModuleList([
            MADDPGActor(obs_dim, action_dim).to(self.device)
            for _ in range(n_agents)
        ])
        self.critics = nn.ModuleList([
            MADDPGCritic(self.global_obs_dim, self.global_action_dim).to(self.device)
            for _ in range(n_agents)
        ])
        self.target_critics = nn.ModuleList([
            MADDPGCritic(self.global_obs_dim, self.global_action_dim).to(self.device)
            for _ in range(n_agents)
        ])
        for i in range(n_agents):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())

        self.actor_optimizers = [optim.Adam(a.parameters(), lr=actor_lr) for a in self.actors]
        self.critic_optimizers = [optim.Adam(c.parameters(), lr=critic_lr) for c in self.critics]

        # Replay buffer of (global_s, global_a, team_r, global_s', done)
        self.replay_buffer = []
        # One-step pending transition (train.py does not deliver next_obs)
        self._pending = None

    # ------------------------------------------------------------------
    # train.py interface
    # ------------------------------------------------------------------

    def select_actions(self, obs: np.ndarray, add_noise: bool = True,
                       deterministic: bool = False, **_kwargs):
        """
        Args:
            obs: (n_agents, obs_dim)
            deterministic: if True, no exploration noise (evaluation)
        Returns:
            (actions (n_agents, action_dim), log_probs (n_agents, 1) zeros)
        """
        obs_t = torch.FloatTensor(obs).to(self.device)
        actions = []
        with torch.no_grad():
            for i in range(self.n_agents):
                action = self.actors[i](obs_t[i:i + 1])[0]
                if add_noise and not deterministic:
                    action = action + torch.randn_like(action) * self.noise_std
                    action = torch.clamp(action, -1.0, 1.0)
                actions.append(action.cpu().numpy())
        log_probs = np.zeros((self.n_agents, 1), dtype=np.float32)
        return np.array(actions, dtype=np.float32), log_probs

    def store_transition(self, obs, actions, log_probs=None, rewards=None,
                         done=0.0, values=None, **_kwargs):
        """
        train.py-style storage. Because next_obs is not delivered, the
        previous transition is committed when the next observation arrives;
        terminal transitions are committed immediately with done=1.
        """
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        actions = np.asarray(actions, dtype=np.float32).reshape(-1)
        team_r = float(np.sum(rewards))

        if self._pending is not None:
            p_s, p_a, p_r = self._pending
            self._append(p_s, p_a, p_r, obs, 0.0)
            self._pending = None

        if done:
            # Terminal: next_state unused (masked by (1 - done))
            self._append(obs, actions, team_r, obs, 1.0)
        else:
            self._pending = (obs, actions, team_r)

    def _append(self, s, a, r, s2, done):
        self.replay_buffer.append((s, a, r, s2, float(done)))
        if len(self.replay_buffer) > self.replay_capacity:
            self.replay_buffer.pop(0)

    def update(self, *args, **kwargs):
        """train.py calls update(next_obs, next_done) once per episode;
        we run `gradient_steps` mini-batch DDPG updates instead."""
        if len(self.replay_buffer) < max(self.batch_size, self.warmup_steps):
            return {'actor_loss': 0.0, 'critic_loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0}

        # NaN watchdog: snapshot weights; if TD divergence still manages to
        # corrupt them (non-finite), roll back to this snapshot below.
        watchdog_params = [p for a in self.actors for p in a.parameters()] + \
                          [p for c in self.critics for p in c.parameters()]
        watchdog_snapshot = [p.detach().clone() for p in watchdog_params]

        actor_losses, critic_losses = [], []
        for _ in range(self.gradient_steps):
            al, cl = self._update_once()
            if al is None:
                break
            actor_losses.append(al)
            critic_losses.append(cl)

        # NaN watchdog rollback
        bad = any(not torch.isfinite(p).all() for p in watchdog_params)
        if bad:
            with torch.no_grad():
                for p, s in zip(watchdog_params, watchdog_snapshot):
                    p.copy_(s)
                # keep target networks consistent with the restored weights
                for i in range(self.n_agents):
                    self.target_actors[i].load_state_dict(self.actors[i].state_dict())
                    self.target_critics[i].load_state_dict(self.critics[i].state_dict())
            print("[WARN] NaN/Inf weights after MADDPG update; rolled back to pre-update snapshot")

        return {
            'actor_loss': float(np.mean(actor_losses)) if actor_losses else 0.0,
            'critic_loss': float(np.mean(critic_losses)) if critic_losses else 0.0,
            'policy_loss': float(np.mean(actor_losses)) if actor_losses else 0.0,
            'value_loss': float(np.mean(critic_losses)) if critic_losses else 0.0,
        }

    # ------------------------------------------------------------------
    # Core DDPG update
    # ------------------------------------------------------------------

    def _sample(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return None
        idx = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in idx]
        states = torch.FloatTensor(np.array([b[0] for b in batch])).to(self.device)
        actions = torch.FloatTensor(np.array([b[1] for b in batch])).to(self.device)
        rewards = torch.FloatTensor([b[2] for b in batch]).to(self.device).unsqueeze(-1)
        next_states = torch.FloatTensor(np.array([b[3] for b in batch])).to(self.device)
        dones = torch.FloatTensor([b[4] for b in batch]).to(self.device).unsqueeze(-1)
        return states, actions, rewards, next_states, dones

    def _update_once(self):
        batch = self._sample(self.batch_size)
        if batch is None:
            return None, None
        states, actions, rewards, next_states, dones = batch

        with torch.no_grad():
            next_actions = []
            for i in range(self.n_agents):
                s_i = next_states[:, i * self.obs_dim:(i + 1) * self.obs_dim]
                next_actions.append(self.target_actors[i](s_i))
            next_actions = torch.cat(next_actions, dim=-1)

        critic_losses = []
        for i in range(self.n_agents):
            with torch.no_grad():
                target_q = rewards + self.gamma * (1.0 - dones) * \
                    self.target_critics[i](next_states, next_actions)
            q = self.critics[i](states, actions)
            critic_loss = F.mse_loss(q, target_q)
            # Skip non-finite losses so a diverging TD target cannot poison
            # the Adam moments and NaN the weights
            if not torch.isfinite(critic_loss):
                return None, None
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critics[i].parameters(), self.max_grad_norm)
            self.critic_optimizers[i].step()
            critic_losses.append(critic_loss.item())

        actor_losses = []
        for i in range(self.n_agents):
            pred_actions = []
            for j in range(self.n_agents):
                s_j = states[:, j * self.obs_dim:(j + 1) * self.obs_dim]
                if j == i:
                    pred_actions.append(self.actors[i](s_j))
                else:
                    with torch.no_grad():
                        pred_actions.append(self.target_actors[j](s_j))
            pred_actions = torch.cat(pred_actions, dim=-1)
            actor_loss = -self.critics[i](states, pred_actions).mean()
            # Skip non-finite losses (critic may be mid-divergence)
            if not torch.isfinite(actor_loss):
                return None, None
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actors[i].parameters(), self.max_grad_norm)
            self.actor_optimizers[i].step()
            actor_losses.append(actor_loss.item())

        for i in range(self.n_agents):
            for tp, p in zip(self.target_actors[i].parameters(), self.actors[i].parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for tp, p in zip(self.target_critics[i].parameters(), self.critics[i].parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return float(np.mean(actor_losses)), float(np.mean(critic_losses))

    # ------------------------------------------------------------------
    # Misc interface
    # ------------------------------------------------------------------

    def get_values(self, global_obs: torch.Tensor):
        """Q-values from all critics (diagnostics only)."""
        with torch.no_grad():
            if global_obs.dim() == 1:
                global_obs = global_obs.unsqueeze(0)
            actions = [self.actors[i](global_obs[:, i * self.obs_dim:(i + 1) * self.obs_dim])
                       for i in range(self.n_agents)]
            global_actions = torch.cat(actions, dim=-1)
            q_values = [c(global_obs, global_actions) for c in self.critics]
            return torch.cat(q_values, dim=-1)

    def clear_buffer(self):
        pass

    def save(self, path: str):
        torch.save({
            'actors': [a.state_dict() for a in self.actors],
            'critics': [c.state_dict() for c in self.critics],
        }, path)
        print(f"[MADDPG] Model saved to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        for a, sd in zip(self.actors, ckpt['actors']):
            a.load_state_dict(sd)
        for c, sd in zip(self.critics, ckpt['critics']):
            c.load_state_dict(sd)
        for i in range(self.n_agents):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())
        print(f"[MADDPG] Model loaded from {path}")


if __name__ == '__main__':
    policy = MADDPGPolicy(obs_dim=19 * 6 + 8, action_dim=4, n_agents=3)
    print("MADDPG policy created successfully")
