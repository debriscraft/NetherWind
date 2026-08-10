"""
bca_policy.py
==============
BCA Policy: GRE (Graph Relational Encoder) + BVD (Bilateral Value Decomposition)
           + PBC (Population-Based Curriculum) training integration.

Implements BCAPolicy class that extends the MAPPO interface with:
  - GRE actor (replaces MLP actor)
  - Dual critics (V_coop, V_comp) for BVD
  - PBC opponent scheduling (Phase 1 adaptive, Phase 2 fixed p4)

Usage:
  policy = BCAPolicy(obs_dim=..., action_dim=..., n_agents=..., n_red=..., n_blue=...)
  # Training loop same as MAPPOPolicy, but compute_gae/update use dual advantages
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple, List

# Import GRE module
from .gre_encoder import GraphRelationalEncoder

# Import base policy for interface reference
from .mappo import MAPPOPolicy


# ---------------------------------------------------------------------------
# Reward Normalizer (running mean/std)
# ---------------------------------------------------------------------------

class RewardNormalizer:
    """
    Maintain running mean/std of rewards, normalize to zero-mean unit-variance.
    Critical for preventing critic output explosion in MARL.
    """
    def __init__(self, eps=1e-8, clip_range=10.0):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4
        self.eps = eps
        self.clip_range = clip_range

    def update(self, rewards: np.ndarray):
        """Update running statistics with a batch of rewards."""
        batch_mean = np.mean(rewards)
        batch_var = np.var(rewards)
        batch_count = len(rewards)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, rewards: np.ndarray):
        """Normalize rewards using current statistics."""
        self.update(rewards)
        std = np.sqrt(self.var + self.eps)
        normalized = (rewards - self.mean) / std
        return np.clip(normalized, -self.clip_range, self.clip_range)

    def state_dict(self):
        """Return state for checkpointing."""
        return {'mean': self.mean, 'var': self.var, 'count': self.count}

    def load_state_dict(self, state):
        """Load state from checkpoint."""
        self.mean = state['mean']
        self.var = state['var']
        self.count = state['count']


class ObsNormalizer:
    """
    Maintain running mean/std of observations, normalize to zero-mean unit-variance.
    Critical for preventing observation explosion and improving training stability in MARL.
    """
    def __init__(self, obs_dim: int, eps=1e-8, clip_range=10.0):
        self.obs_dim = obs_dim
        self.mean = np.zeros(obs_dim)
        self.var = np.ones(obs_dim)
        self.count = 1e-4
        self.eps = eps
        self.clip_range = clip_range

    def update(self, obs: np.ndarray):
        """Update running statistics with a batch of observations."""
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        
        batch_mean = np.mean(obs, axis=0)
        batch_var = np.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, obs: np.ndarray, update_stats=True):
        """
        Normalize observations using current statistics.
        
        Args:
            obs: (batch, obs_dim) or (obs_dim,) numpy array
            update_stats: if True, update running statistics (should be True during training)
        
        Returns:
            normalized observations (same shape as input)
        """
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
            single = True
        else:
            single = False
        
        if update_stats:
            self.update(obs)
        
        std = np.sqrt(self.var + self.eps)
        normalized = (obs - self.mean) / std
        normalized = np.clip(normalized, -self.clip_range, self.clip_range)
        
        if single:
            return normalized.reshape(-1)
        return normalized

    def state_dict(self):
        """Return state for checkpointing."""
        return {
            'mean': self.mean.copy(),
            'var': self.var.copy(),
            'count': self.count
        }

    def load_state_dict(self, state):
        """Load state from checkpoint."""
        self.mean = state['mean'].copy()
        self.var = state['var'].copy()
        self.count = state['count']


# ---------------------------------------------------------------------------
# Dual Critic Network (for BVD)
# ---------------------------------------------------------------------------

class DualCriticNetwork(nn.Module):
    """
    Dual centralized critic: V_coop + V_comp.

    Input: global state (concatenated observations of all agents)
    Output: (V_coop, V_comp) — two scalar value estimates
    """

    def __init__(self, global_obs_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(global_obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )
        self.head_coop = nn.Linear(hidden_dim // 2, 1)
        self.head_comp = nn.Linear(hidden_dim // 2, 1)

    def forward(self, global_obs: torch.Tensor):
        """
        Args:
            global_obs: (batch, global_obs_dim)  concatenated observations
        Returns:
            v_coop: (batch, 1)
            v_comp: (batch, 1)
        """
        h = self.shared(global_obs)
        v_coop = self.head_coop(h)
        v_comp = self.head_comp(h)
        return v_coop, v_comp


# ---------------------------------------------------------------------------
# BCA Policy
# ---------------------------------------------------------------------------

class BCAPolicy:
    """
    BCA Policy: MAPPO + GRE actor + BVD dual advantage + PBC scheduling.

    Extends MAPPOPolicy interface with:
      - GRE-based actor (Graph Relational Encoder)
      - Dual critics (V_coop, V_comp)
      - Bilateral advantage: A = A_coop + lambda_comp * A_comp
      - PBC opponent scheduling (stub for now, full implementation in train.py)
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
        bvd_lambda_comp: float = 0.5,
        bvd_k_samples: int = 5,
        use_gre: bool = True,   # Set False for ablation (w/o GRE)
        use_bvd: bool = True,   # Set False for ablation (w/o BVD)
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.n_red = n_red
        self.n_blue = n_blue
        self.n_fire_targets = n_fire_targets
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        # Annealable log-std ceiling (log space). Default 0.0 => std_max = 1.0.
        # train.py may call set_std_max_logit() to anneal exploration late in training.
        self.std_max_logit = 0.0
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.bvd_lambda_comp = bvd_lambda_comp
        self.bvd_k_samples = bvd_k_samples
        self.gre_d_model = gre_d_model  # Save for update() method
        self.use_gre = use_gre
        self.use_bvd = use_bvd

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        per_agent_dim = 19  # hardcoded from env.py

        # ---- Actor (conditional on use_gre) ----
        if self.use_gre:
            # GRE-based actor (relation-aware)
            self.gre = GraphRelationalEncoder(
                per_agent_dim=per_agent_dim,
                d_model=gre_d_model,
                n_layers=2,
            ).to(self.device)
            self.actor_feature_dim = 2 * gre_d_model
        else:
            # MLP-based actor (ablation: w/o GRE)
            self.mlp_actor = nn.Sequential(
                nn.Linear(obs_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
            ).to(self.device)
            self.actor_feature_dim = 256

        # Policy head
        self.policy_fc1 = nn.Linear(self.actor_feature_dim, 256).to(self.device)
        self.policy_ln1 = nn.LayerNorm(256).to(self.device)
        self.policy_fc2 = nn.Linear(256, 128).to(self.device)
        self.policy_ln2 = nn.LayerNorm(128).to(self.device)
        self.policy_mean = nn.Linear(128, action_dim).to(self.device)
        self.policy_log_std = nn.Linear(128, action_dim).to(self.device)

        if n_fire_targets > 0:
            self.fire_logits = nn.Linear(128, n_fire_targets).to(self.device)

        # ---- Critics (conditional on use_bvd) ----
        global_obs_dim = obs_dim * n_red
        if self.use_bvd:
            # Dual critics (BVD: Bilateral Advantage Decomposition)
            self.critic_coop = nn.Sequential(
                nn.Linear(global_obs_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            ).to(self.device)
            self.critic_comp = nn.Sequential(
                nn.Linear(global_obs_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            ).to(self.device)
        else:
            # Single critic (ablation: w/o BVD)
            self.critic_coop = nn.Sequential(
                nn.Linear(global_obs_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            ).to(self.device)
            self.critic_comp = None

        # ---- Optimizers ----
        actor_params = []
        if self.use_gre:
            actor_params += list(self.gre.parameters())
        actor_params += (
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
        self.critic_coop_optimizer = optim.Adam(self.critic_coop.parameters(), lr=lr)
        if self.use_bvd:
            self.critic_comp_optimizer = optim.Adam(self.critic_comp.parameters(), lr=lr)
        else:
            self.critic_comp_optimizer = None

        # ---- Rollout buffers ----
        self.obs_buf = []
        self.actions_buf = []
        self.fire_actions_buf = []
        self.log_probs_buf = []
        self.rewards_coop_buf = []
        self.rewards_comp_buf = []
        self.values_coop_buf = []
        self.values_comp_buf = []
        self.dones_buf = []

        # ---- Reward normalizer (prevents critic explosion) ----
        self.reward_normalizer = RewardNormalizer(clip_range=10.0)
        
        # ---- Observation normalizer (prevents observation explosion) ----
        self.obs_normalizer = ObsNormalizer(obs_dim=obs_dim * n_red, clip_range=10.0)

        # ---- Training counter ----
        self.update_count = 0

        # ---- Initialize weights ----
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for all Linear layers in sub-modules."""
        # Initialize GRE weights (only if use_gre)
        if self.use_gre:
            for m in self.gre.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    nn.init.zeros_(m.bias)
        # Initialize policy head weights
        for m in [self.policy_fc1, self.policy_ln1, self.policy_fc2,
                   self.policy_ln2, self.policy_mean, self.policy_log_std]:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Special init for policy_mean (smaller gain)
        nn.init.orthogonal_(self.policy_mean.weight, gain=0.01)
        nn.init.zeros_(self.policy_mean.bias)
        # Initialize dual critics
        for critic in [self.critic_coop, self.critic_comp]:
            if critic is not None:
                for m in critic.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                        nn.init.zeros_(m.bias)
        # Initialize fire_logits if exists
        if self.n_fire_targets > 0:
            nn.init.orthogonal_(self.fire_logits.weight, gain=0.01)
            nn.init.zeros_(self.fire_logits.bias)

    def set_std_max_logit(self, v: float):
        """Set the current log-std ceiling (called from the training loop)."""
        self.std_max_logit = float(v)

    def set_lr(self, lr: float):
        """Set learning rate on all optimizers (consolidation-phase decay)."""
        for opt in [self.actor_optimizer, self.critic_coop_optimizer,
                    self.critic_comp_optimizer]:
            if opt is not None:
                for pg in opt.param_groups:
                    pg['lr'] = float(lr)

    # -------------------------------------------------------------------
    # Actor forward
    # -------------------------------------------------------------------

    def _actor_forward(self, obs: torch.Tensor, positions: Optional[torch.Tensor] = None):
        """
        Actor forward pass through GRE + policy head.

        Args:
            obs: (batch, obs_dim) or (batch, n_red, obs_dim)
            positions: (batch, n_red+n_blue, 3) — aircraft positions (optional)

        Returns:
            mean: (batch, n_red, action_dim)
            log_std: (batch, n_red, action_dim)
            fire_logits: (batch, n_red, n_fire_targets) if n_fire_targets > 0
            attn_stats: dict (for logging)
        """
        # Handle input shape
        if obs.dim() == 2:
            # (batch, obs_dim) — assume batch = n_red * rollout_length
            # Reshape to (batch_per_agent, n_red, obs_dim)
            batch = obs.shape[0]
            # This is tricky — during training, obs is flattened
            # For now, assume obs is already (n_steps * n_red, obs_dim)
            # and we need to reshape to (n_steps, n_red, obs_dim)
            # This is handled in the training loop
            # (batch, obs_dim) — flatten from (batch_per_agent, n_red, obs_dim)
            # Reshape to (batch_per_agent, n_red, obs_dim) then merge batch dims
            batch_flat = obs.shape[0]
            # Assume flatten was (T * n_red, obs_dim) → (T, n_red, obs_dim)
            # Actually: during training rollout, select_actions is called per-step
            # with obs = (n_red, obs_dim), which becomes (1, n_red, obs_dim) after unsqueeze
            # This 2D path is for direct calls with (n_red, obs_dim)
            if batch_flat == self.n_red:
                obs = obs.unsqueeze(0)  # (1, n_red, obs_dim)
            # else: flattened buffer — will be reshaped in update(), not here

        # GRE expects (batch, n_red, obs_dim)
        # Actually, GRE.forward expects (batch, n_red, obs_dim) where batch is the number of parallel envs
        # During training, we flatten to (batch * n_red, obs_dim)
        # Let me fix this later

        # For now, assume obs is (n_red, obs_dim) — single step
        if obs.dim() == 2 and obs.shape[0] == self.n_red:
            obs = obs.unsqueeze(0)  # (1, n_red, obs_dim)

        # Actor forward (conditional on use_gre)
        batch = obs.shape[0] if obs.dim() == 3 else 1
        attn_stats = None  # only populated on the GRE path
        if self.use_gre:
            # GRE forward
            gre_out, attn_stats = self.gre(obs, self.n_red, self.n_blue, positions)
            h = gre_out.view(-1, self.actor_feature_dim)
        else:
            # MLP forward (ablation: w/o GRE)
            # Flatten per-agent obs: (batch, n_red, obs_dim) -> (batch * n_red, obs_dim)
            mlp_in = obs.reshape(-1, obs.shape[-1])
            h = self.mlp_actor(mlp_in)

        # Policy head
        h = torch.relu(self.policy_ln1(self.policy_fc1(h)))
        h = torch.relu(self.policy_ln2(self.policy_fc2(h)))

        mean = self.policy_mean(h)  # raw (unsquashed) mean; tanh applied at sampling only
        log_std = torch.clamp(self.policy_log_std(h), -4, self.std_max_logit)

        mean = mean.view(batch, self.n_red, self.action_dim)
        log_std = log_std.view(batch, self.n_red, self.action_dim)

        if self.n_fire_targets > 0:
            fire_logits = self.fire_logits(h).view(batch, self.n_red, -1)
            return mean, log_std, fire_logits, attn_stats

        return mean, log_std, attn_stats

    def select_actions(self, obs: np.ndarray, action_mask: np.ndarray = None,
                       positions: np.ndarray = None, deterministic: bool = False,
                       update_obs_stats: bool = True):
        """
        Select actions for all red agents.

        Args:
            obs: (n_red, obs_dim) numpy array
            action_mask: (n_red, n_fire_targets) numpy array
            positions: (n_red+n_blue, 3) numpy array (optional)
            deterministic: if True, use mean action (for evaluation)
            update_obs_stats: if True, update observation normalizer statistics (training mode)

        Returns:
            cont_actions_list, log_probs_list  (or with fire_actions if n_fire_targets > 0)
        """
        # Normalize observations (prevent observation explosion)
        obs_flat = obs.reshape(1, -1)  # (1, obs_dim * n_red)
        obs_normalized = self.obs_normalizer.normalize(obs_flat, update_stats=update_obs_stats)
        obs_normalized = obs_normalized.reshape(obs.shape)
        
        obs_t = torch.FloatTensor(obs_normalized).unsqueeze(0).to(self.device)  # (1, n_red, obs_dim)
        pos_t = None
        if positions is not None:
            pos_t = torch.FloatTensor(positions).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.n_fire_targets > 0:
                mean, log_std, fire_logits, attn_stats = self._actor_forward(obs_t, pos_t)
                # --- NaN guard ---
                if torch.isnan(mean).any() or torch.isnan(log_std).any():
                    print(f"[FATAL] NaN detected in actor output! mean={mean.mean().item():.4f}, log_std={log_std.mean().item():.4f}")
                    # Fallback: zero-mean, small std (safe random action)
                    mean = torch.zeros_like(mean)
                    log_std = torch.ones_like(log_std) * (-2.0)
                # --- end NaN guard ---
                if deterministic:
                    cont_actions = torch.tanh(mean)
                    dist = torch.distributions.Normal(mean, log_std.exp())
                    cont_log_prob = dist.log_prob(mean) - torch.log(1 - cont_actions.pow(2) + 1e-6)
                else:
                    dist = torch.distributions.Normal(mean, log_std.exp())
                    cont_sample = dist.sample()
                    cont_actions = torch.tanh(cont_sample)
                    cont_log_prob = dist.log_prob(cont_sample) - torch.log(1 - cont_actions.pow(2) + 1e-6)
                cont_log_prob = cont_log_prob.sum(dim=-1, keepdim=True)

                # Discrete fire actions
                if action_mask is not None:
                    mask_t = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)
                    fire_logits = fire_logits + mask_t
                fire_dist = torch.distributions.Categorical(logits=fire_logits)
                if deterministic:
                    fire_actions = torch.argmax(fire_logits, dim=-1)
                else:
                    fire_actions = fire_dist.sample()
                fire_log_prob = fire_dist.log_prob(fire_actions).unsqueeze(-1)

                log_prob = cont_log_prob + fire_log_prob

                return (
                    cont_actions.squeeze(0).cpu().numpy(),
                    fire_actions.squeeze(0).cpu().numpy().astype(int),
                    log_prob.squeeze(0).cpu().numpy()
                )
            else:
                mean, log_std, attn_stats = self._actor_forward(obs_t, pos_t)
                # --- NaN guard ---
                if torch.isnan(mean).any() or torch.isnan(log_std).any():
                    print(f"[FATAL] NaN in select_actions (no fire)! mean_nan={torch.isnan(mean).any()}, log_std_nan={torch.isnan(log_std).any()}")
                    mean = torch.zeros_like(mean)
                    log_std = torch.ones_like(log_std) * (-2.0)
                # --- end NaN guard ---
                if deterministic:
                    actions = torch.tanh(mean)
                    dist = torch.distributions.Normal(mean, log_std.exp())
                    log_prob = dist.log_prob(mean) - torch.log(1 - actions.pow(2) + 1e-6)
                else:
                    dist = torch.distributions.Normal(mean, log_std.exp())
                    sample = dist.sample()
                    actions = torch.tanh(sample)
                    log_prob = dist.log_prob(sample) - torch.log(1 - actions.pow(2) + 1e-6)
                log_prob = log_prob.sum(dim=-1, keepdim=True)

                return actions.squeeze(0).cpu().numpy(), log_prob.squeeze(0).cpu().numpy()

    # -------------------------------------------------------------------
    # Store transition (with reward normalization)
    # -------------------------------------------------------------------

    def store_transition(self, obs, actions, log_probs, rewards_coop, rewards_comp,
                         dones, values_coop, values_comp, fire_actions=None):
        """Store a transition in the rollout buffer. Normalizes rewards and observations to prevent critic explosion."""
        # Normalize observations (prevent observation explosion)
        obs_array = np.array(obs)
        obs_flat = obs_array.reshape(1, -1)  # (1, obs_dim * n_red)
        obs_normalized = self.obs_normalizer.normalize(obs_flat, update_stats=True)
        obs_normalized = obs_normalized.reshape(obs_array.shape)
        
        # Normalize rewards_coop (cooperative reward)
        rewards_coop_arr = np.array(rewards_coop).flatten()
        rewards_coop_norm = self.reward_normalizer.normalize(rewards_coop_arr)

        # rewards_comp is already zero-mean (deviation), clip to [-10, 10]
        rewards_comp_arr = np.array(rewards_comp).flatten()
        rewards_comp_clipped = np.clip(rewards_comp_arr, -10.0, 10.0)

        self.obs_buf.append(obs_normalized.tolist())
        self.actions_buf.append(actions)
        if fire_actions is not None:
            self.fire_actions_buf.append(fire_actions)
        self.log_probs_buf.append(log_probs)
        self.rewards_coop_buf.append(rewards_coop_norm.tolist())
        self.rewards_comp_buf.append(rewards_comp_clipped.tolist())
        self.values_coop_buf.append(values_coop)
        self.values_comp_buf.append(values_comp)
        self.dones_buf.append(dones)

    # -------------------------------------------------------------------
    # Get values (for training loop)
    # -------------------------------------------------------------------

    def get_values(self, obs_t: torch.Tensor):
        """
        Get dual value estimates for GAE computation.

        Args:
            obs_t: (n_red, obs_dim) tensor on device (MAY BE UNNORMALIZED)

        Returns:
            value_coop: float
            value_comp: float (0.0 if use_bvd=False)
        """
        # Normalize observations (prevent observation explosion)
        obs_np = obs_t.cpu().numpy().reshape(1, -1)
        obs_norm = self.obs_normalizer.normalize(obs_np, update_stats=False)  # Don't update stats during value eval
        obs_norm_t = torch.FloatTensor(obs_norm).reshape(obs_t.shape).to(self.device)
        
        global_obs = obs_norm_t.reshape(1, -1)
        with torch.no_grad():
            value_coop_raw = self.critic_coop(global_obs).cpu().numpy().flatten()[0]
            value_coop = np.clip(value_coop_raw, -20.0, 20.0)  # Safety clip
            if self.use_bvd and self.critic_comp is not None:
                value_comp_raw = self.critic_comp(global_obs).cpu().numpy().flatten()[0]
                value_comp = np.clip(value_comp_raw, -20.0, 20.0)  # Safety clip
            else:
                value_comp = 0.0
        return value_coop, value_comp

    # -------------------------------------------------------------------
    # Clear buffers (after update)
    # -------------------------------------------------------------------

    def clear_buffer(self):
        """Clear rollout buffers after PPO update."""
        self.obs_buf.clear()
        self.actions_buf.clear()
        self.fire_actions_buf.clear()
        self.log_probs_buf.clear()
        self.rewards_coop_buf.clear()
        self.rewards_comp_buf.clear()
        self.values_coop_buf.clear()
        self.values_comp_buf.clear()
        self.dones_buf.clear()

    # -------------------------------------------------------------------
    # eval / train mode switches (for proper train/eval control)
    # -------------------------------------------------------------------

    def eval(self):
        """Set all sub-modules to eval mode."""
        if self.use_gre:
            self.gre.eval()
        for m in [self.policy_fc1, self.policy_ln1, self.policy_fc2,
                   self.policy_ln2, self.policy_mean, self.policy_log_std]:
            m.eval()
        self.critic_coop.eval()
        if self.critic_comp is not None:
            self.critic_comp.eval()
        if self.n_fire_targets > 0:
            self.fire_logits.eval()

    def train(self, mode=True):
        """Set all sub-modules to train mode."""
        if self.use_gre:
            self.gre.train(mode)
        for m in [self.policy_fc1, self.policy_ln1, self.policy_fc2,
                   self.policy_ln2, self.policy_mean, self.policy_log_std]:
            m.train(mode)
        self.critic_coop.train(mode)
        if self.critic_comp is not None:
            self.critic_comp.train(mode)
        if self.n_fire_targets > 0:
            self.fire_logits.train(mode)
        return self

    # -------------------------------------------------------------------
    # GAE for dual advantages (BVD)
    # -------------------------------------------------------------------

    def compute_bilateral_gae(self, next_obs, next_done):
        """
        Compute bilateral GAE: A_coop and A_comp separately.

        Returns:
            advantages_coop: (n_steps, n_red)
            advantages_comp: (n_steps, n_red)
            returns_coop: (n_steps, n_red)
            returns_comp: (n_steps, n_red)
        """
        n_steps = len(self.rewards_coop_buf)
        n_agents = self.n_red
        gamma = self.gamma
        lam = self.gae_lambda

        advantages_coop = np.zeros((n_steps, n_agents))
        advantages_comp = np.zeros((n_steps, n_agents))

        # Convert to tensor for value prediction
        with torch.no_grad():
            next_obs_t = torch.FloatTensor(next_obs).to(self.device)
            next_global_obs = next_obs_t.reshape(1, -1)
            next_value_coop = self.critic_coop(next_global_obs).cpu().numpy().flatten()[0]
            next_value_coop = np.clip(next_value_coop, -20.0, 20.0)  # Safety clip
            if self.use_bvd and self.critic_comp is not None:
                next_value_comp = self.critic_comp(next_global_obs).cpu().numpy().flatten()[0]
                next_value_comp = np.clip(next_value_comp, -20.0, 20.0)  # Safety clip
            else:
                next_value_comp = 0.0

        last_gae_coop = np.zeros(n_agents)
        last_gae_comp = np.zeros(n_agents)

        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_val_coop = next_value_coop * (1 - next_done)
                next_val_comp = next_value_comp * (1 - next_done)
            else:
                next_val_coop = np.clip(self.values_coop_buf[t + 1], -20.0, 20.0)
                next_val_comp = np.clip(self.values_comp_buf[t + 1], -20.0, 20.0)

            rewards_coop_t = np.array(self.rewards_coop_buf[t])
            rewards_comp_t = np.array(self.rewards_comp_buf[t])
            values_coop_t = np.clip(np.array(self.values_coop_buf[t]), -20.0, 20.0)
            values_comp_t = np.clip(np.array(self.values_comp_buf[t]), -20.0, 20.0)
            mask_t = 1.0 - np.array(self.dones_buf[t])

            delta_coop = rewards_coop_t + gamma * next_val_coop * mask_t - values_coop_t
            delta_comp = rewards_comp_t + gamma * next_val_comp * mask_t - values_comp_t

            last_gae_coop = delta_coop + gamma * lam * mask_t * last_gae_coop
            last_gae_comp = delta_comp + gamma * lam * mask_t * last_gae_comp

            advantages_coop[t] = last_gae_coop
            advantages_comp[t] = last_gae_comp

        # advantages_coop, advantages_comp are already computed above
        # Normalize advantages (standard PPO trick for stability)
        adv_coop_mean = advantages_coop.mean()
        adv_coop_std = advantages_coop.std() + 1e-8
        adv_comp_mean = advantages_comp.mean()
        adv_comp_std = advantages_comp.std() + 1e-8

        advantages_coop = (advantages_coop - adv_coop_mean) / adv_coop_std
        advantages_comp = (advantages_comp - adv_comp_mean) / adv_comp_std

        # Clip advantages to prevent policy update explosion (BCA fix)
        ADV_CLIP = 5.0
        advantages_coop = np.clip(advantages_coop, -ADV_CLIP, ADV_CLIP)
        advantages_comp = np.clip(advantages_comp, -ADV_CLIP, ADV_CLIP)

        returns_coop = advantages_coop + np.array(self.values_coop_buf)[:, np.newaxis]
        returns_comp = advantages_comp + np.array(self.values_comp_buf)[:, np.newaxis]

        # Clip returns to prevent critic explosion
        returns_coop = np.clip(returns_coop, -20.0, 20.0)
        returns_comp = np.clip(returns_comp, -20.0, 20.0)

        return advantages_coop, advantages_comp, returns_coop, returns_comp

    # -------------------------------------------------------------------
    # PPO update with bilateral advantage
    # -------------------------------------------------------------------

    def update(self, next_obs, next_done):
        """
        PPO update with bilateral advantage.

        Computes A = A_coop + lambda_comp * A_comp.
        Dual critics (V_coop, V_comp) each learn from the same reward signal
        but with independent weights, decomposing value through the graph structure.
        """
        # 1. Compute bilateral GAE
        A_coop, A_comp, R_coop, R_comp = self.compute_bilateral_gae(next_obs, next_done)
        A_bilateral = A_coop + self.bvd_lambda_comp * A_comp
        # Clip bilateral advantage to prevent PL explosion (BCA fix)
        A_bilateral = np.clip(A_bilateral, -5.0, 5.0)

        # 2. Stack buffers → (T, N, ...) shaped
        all_obs_3d = np.array(self.obs_buf)            # (T, N, obs_dim)
        all_actions_3d = np.array(self.actions_buf)     # (T, N, action_dim)
        all_log_probs_3d = np.array(self.log_probs_buf) # (T, N, 1)

        T, N = all_obs_3d.shape[:2]  # T = rollout steps, N = n_red

        # 3. Prepare tensors (keep 3D shape, index by timestep)
        obs_3d = torch.FloatTensor(all_obs_3d).to(self.device)           # (T, N, obs_dim)
        actions_3d = torch.FloatTensor(all_actions_3d).to(self.device)    # (T, N, action_dim)
        old_log_probs_3d = torch.FloatTensor(all_log_probs_3d).to(self.device)  # (T, N, 1)
        advantages_2d = torch.FloatTensor(A_bilateral).to(self.device)    # (T, N)
        if torch.isnan(advantages_2d).any():
            print(f"[WARN] NaN in advantages_2d! before update loop")
            # Debug: print A_bilateral stats
            print(f"  A_bilateral range: [{np.nanmin(A_bilateral):.2f}, {np.nanmax(A_bilateral):.2f}]")
            print(f"  A_coop has NaN: {np.any(np.isnan(A_coop))}")
            print(f"  A_comp has NaN: {np.any(np.isnan(A_comp))}")
            return {'actor_loss': 0.0, 'value_bvd_coop_loss': 0.0,
                    'value_bvd_comp_loss': 0.0, 'entropy': 0.0}

        # Global obs for critic input: (T, obs_dim * N)
        global_obs_all = torch.FloatTensor(all_obs_3d.reshape(T, -1)).to(self.device)

        # For critic targets: mean over agents per timestep
        R_coop_global = R_coop.mean(axis=1)   # (T,)
        R_comp_global = R_comp.mean(axis=1)   # (T,)

        # 4. PPO epochs
        indices = np.arange(T)
        update_info = {'actor_loss': 0.0, 'value_bvd_coop_loss': 0.0,
                       'value_bvd_comp_loss': 0.0, 'entropy': 0.0}
        n_updates = 0

        for epoch in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, T, self.mini_batch_size):
                mb_idx_t = indices[start:start + self.mini_batch_size]
                mb_size_t = len(mb_idx_t)

                # --- Actor: forward (conditional on use_gre) ---
                obs_mb_3d = obs_3d[mb_idx_t]  # (mb_size_t, N, obs_dim)
                if self.use_gre:
                    gre_out, _ = self.gre(obs_mb_3d, self.n_red, self.n_blue)
                    if torch.isnan(gre_out).any():
                        print(f"[WARN] NaN in gre_out at epoch {epoch}, batch {start}")
                        continue
                    # gre_out: (mb_size_t, N, 2*d_model)
                    h = gre_out.reshape(-1, 2 * self.gre_d_model)
                else:
                    # MLP actor (ablation: w/o GRE)
                    # obs_mb_3d: (mb_size_t, N, obs_dim)
                    # MLP expects: (mb_size_t * N, obs_dim * N)
                    mb_size_t = obs_mb_3d.shape[0]
                    N = obs_mb_3d.shape[1]
                    mlp_in = obs_mb_3d.reshape(mb_size_t * N, -1)  # (192, 342) if obs_dim=114, N=3
                    h = self.mlp_actor(mlp_in)  # (mb_size_t * N, 256)
                h = torch.relu(self.policy_ln1(self.policy_fc1(h)))
                if torch.isnan(h).any():
                    print(f"[WARN] NaN in h after fc1 at epoch {epoch}, batch {start}")
                    continue
                h = torch.relu(self.policy_ln2(self.policy_fc2(h)))
                if torch.isnan(h).any():
                    print(f"[WARN] NaN in h after fc2 at epoch {epoch}, batch {start}")
                    continue

                mean = self.policy_mean(h)  # raw mean, consistent with select_actions
                if torch.isnan(mean).any():
                    print(f"[WARN] NaN in mean at epoch {epoch}, batch {start}")
                    continue
                log_std = torch.clamp(self.policy_log_std(h), -4, self.std_max_logit)
                if torch.isnan(log_std).any():
                    print(f"[WARN] NaN in log_std at epoch {epoch}, batch {start}")
                    continue
                std = log_std.exp()
                if torch.isnan(std).any():
                    print(f"[WARN] NaN in std at epoch {epoch}, batch {start}")
                    continue
                dist = torch.distributions.Normal(mean, std)

                # Flatten agent dim for ratio computation
                mb_actions = actions_3d[mb_idx_t].reshape(-1, self.action_dim)
                mb_old_log_probs = old_log_probs_3d[mb_idx_t].reshape(-1, 1)
                mb_adv = advantages_2d[mb_idx_t].reshape(-1, 1)

                # Log-prob of old actions. Stored actions are post-tanh (a=tanh(u));
                # recover the pre-tanh sample u = atanh(a) so dist.log_prob is
                # evaluated on the same variable as in select_actions (ratio consistency).
                mb_a = mb_actions.clamp(-1 + 1e-6, 1 - 1e-6)
                mb_u = torch.atanh(mb_a)
                new_log_prob = dist.log_prob(mb_u) - torch.log(
                    1 - mb_a.pow(2) + 1e-6)
                new_log_prob = new_log_prob.sum(dim=-1, keepdim=True)

                # Ratio + clipped surrogate
                ratio = torch.exp(new_log_prob - mb_old_log_probs)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon,
                                    1 + self.clip_epsilon) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Entropy
                entropy = dist.entropy().sum(dim=-1).mean()

                # --- Dual critic loss (only if use_bvd) ---
                global_obs_mb = global_obs_all[mb_idx_t]  # (mb_size_t, obs_dim * N)
                pred_coop = self.critic_coop(global_obs_mb)   # (mb_size_t, 1)
                pred_coop = torch.clamp(pred_coop, -20.0, 20.0)  # Prevent explosion
                returns_coop_t = torch.FloatTensor(
                    R_coop_global[mb_idx_t]
                ).to(self.device).reshape(-1, 1)
                value_bvd_coop_loss = nn.MSELoss()(pred_coop, returns_coop_t)

                if self.use_bvd and self.critic_comp is not None:
                    pred_comp = self.critic_comp(global_obs_mb)   # (mb_size_t, 1)
                    pred_comp = torch.clamp(pred_comp, -20.0, 20.0)  # Prevent explosion
                    returns_comp_t = torch.FloatTensor(
                        R_comp_global[mb_idx_t]
                    ).to(self.device).reshape(-1, 1)
                    value_bvd_comp_loss = nn.MSELoss()(pred_comp, returns_comp_t)
                else:
                    value_bvd_comp_loss = torch.tensor(0.0, device=self.device)

                # --- NaN check (safety) ---
                # Check each component
                if torch.isnan(policy_loss).any():
                    print(f"[WARN] NaN in policy_loss at epoch {epoch}, batch {start}")
                    continue
                if torch.isnan(value_bvd_coop_loss).any():
                    print(f"[WARN] NaN in value_bvd_coop_loss at epoch {epoch}, batch {start}")
                    continue
                if self.use_bvd and self.critic_comp is not None and torch.isnan(value_bvd_comp_loss).any():
                    print(f"[WARN] NaN in value_bvd_comp_loss at epoch {epoch}, batch {start}")
                    continue

                # --- Gradient updates ---
                self.actor_optimizer.zero_grad()
                (policy_loss - self.entropy_coeff * entropy).backward()
                
                # Enhanced gradient clipping: clip ALL actor parameters
                actor_params = []
                for param_group in self.actor_optimizer.param_groups:
                    actor_params.extend(param_group['params'])
                actor_grad_norm = nn.utils.clip_grad_norm_(actor_params, self.max_grad_norm)
                
                # Debug: log gradient norm (every 100 updates)
                if self.update_count % 100 == 0 and n_updates == 0:
                    print(f"  [Gradient] actor_grad_norm={actor_grad_norm.item():.4f}, max_grad_norm={self.max_grad_norm}")
                
                self.actor_optimizer.step()

                self.critic_coop_optimizer.zero_grad()
                value_bvd_coop_loss.backward()
                critic_coop_params = []
                for param_group in self.critic_coop_optimizer.param_groups:
                    critic_coop_params.extend(param_group['params'])
                critic_coop_grad_norm = nn.utils.clip_grad_norm_(critic_coop_params, self.max_grad_norm)
                self.critic_coop_optimizer.step()

                if self.use_bvd and self.critic_comp_optimizer is not None:
                    self.critic_comp_optimizer.zero_grad()
                    value_bvd_comp_loss.backward()
                    critic_comp_params = []
                    for param_group in self.critic_comp_optimizer.param_groups:
                        critic_comp_params.extend(param_group['params'])
                    critic_comp_grad_norm = nn.utils.clip_grad_norm_(critic_comp_params, self.max_grad_norm)
                    self.critic_comp_optimizer.step()

                update_info['actor_loss'] += policy_loss.item()
                update_info['value_bvd_coop_loss'] += value_bvd_coop_loss.item()
                update_info['value_bvd_comp_loss'] += value_bvd_comp_loss.item()
                update_info['entropy'] += entropy.item()
                n_updates += 1

        # 5. Average & clear buffer
        if n_updates > 0:
            for k in update_info:
                update_info[k] /= n_updates

        self.clear_buffer()
        self.update_count += 1

        return update_info

    # -------------------------------------------------------------------
    # Save/Load
    # -------------------------------------------------------------------

    def save(self, path: str):
        """Save model checkpoint."""
        state = {
            'gre': self.gre.state_dict() if self.use_gre else None,
            'policy_fc1': self.policy_fc1.state_dict(),
            'policy_ln1': self.policy_ln1.state_dict(),
            'policy_fc2': self.policy_fc2.state_dict(),
            'policy_ln2': self.policy_ln2.state_dict(),
            'policy_mean': self.policy_mean.state_dict(),
            'policy_log_std': self.policy_log_std.state_dict(),
            'critic_coop': self.critic_coop.state_dict(),
            'reward_normalizer': self.reward_normalizer.state_dict(),
            'obs_normalizer': self.obs_normalizer.state_dict(),
        }
        if self.use_bvd and self.critic_comp is not None:
            state['critic_comp'] = self.critic_comp.state_dict()
        if self.n_fire_targets > 0:
            state['fire_logits'] = self.fire_logits.state_dict()
        torch.save(state, path)

    def load(self, path: str):
        """Load model checkpoint."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        if self.use_gre and state.get('gre') is not None:
            self.gre.load_state_dict(state['gre'])
        self.policy_fc1.load_state_dict(state['policy_fc1'])
        self.policy_ln1.load_state_dict(state['policy_ln1'])
        self.policy_fc2.load_state_dict(state['policy_fc2'])
        self.policy_ln2.load_state_dict(state['policy_ln2'])
        self.policy_mean.load_state_dict(state['policy_mean'])
        self.policy_log_std.load_state_dict(state['policy_log_std'])
        self.critic_coop.load_state_dict(state['critic_coop'])
        if 'critic_comp' in state and self.critic_comp is not None:
            self.critic_comp.load_state_dict(state['critic_comp'])
        if self.n_fire_targets > 0 and 'fire_logits' in state:
            self.fire_logits.load_state_dict(state['fire_logits'])
        # Load normalizer states (with backward compatibility)
        if 'reward_normalizer' in state:
            self.reward_normalizer.load_state_dict(state['reward_normalizer'])
        if 'obs_normalizer' in state:
            self.obs_normalizer.load_state_dict(state['obs_normalizer'])


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_bca_policy(
    obs_dim: int,
    action_dim: int,
    n_agents: int,
    n_red: int,
    n_blue: int,
    n_fire_targets: int = 0,
    **kwargs
):
    """Create a BCAPolicy instance."""
    return BCAPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_agents=n_agents,
        n_red=n_red,
        n_blue=n_blue,
        n_fire_targets=n_fire_targets,
        **kwargs
    )
