"""
RandomPolicy
============
Uniform random action baseline. Provides the same interface as MAPPOPolicy
so it can be trained/evaluated through the standard train.py pipeline
(no learning takes place; all update/save operations are no-ops).
"""

import numpy as np
import torch

from .mappo import CriticNetwork


class RandomPolicy:
    """Samples continuous actions uniformly from [-1, 1]^action_dim."""

    def __init__(self, obs_dim: int, action_dim: int, n_agents: int,
                 n_fire_targets: int = 0, **_kwargs):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.n_fire_targets = n_fire_targets
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Dummy critic so train.py's value-estimation calls do not crash
        self.critic = CriticNetwork(obs_dim * n_agents).to(self.device)
        self.entropy_coeff = 0.0

    def select_actions(self, obs, action_mask=None, deterministic=False):
        cont = np.random.uniform(-1.0, 1.0, size=(self.n_agents, self.action_dim)).astype(np.float32)
        log_probs = np.zeros((self.n_agents, 1), dtype=np.float32)
        if self.n_fire_targets > 0:
            fire = np.random.randint(0, self.n_fire_targets, size=self.n_agents)
            return ([cont[i] for i in range(self.n_agents)],
                    [int(fire[i]) for i in range(self.n_agents)],
                    [log_probs[i] for i in range(self.n_agents)])
        return ([cont[i] for i in range(self.n_agents)],
                [log_probs[i] for i in range(self.n_agents)])

    def store_transition(self, *args, **kwargs):
        pass

    def store_episode_transition(self, *args, **kwargs):
        pass

    def flush_episode_to_sil(self, *args, **kwargs):
        pass

    def set_episode(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}

    def clear_buffer(self):
        pass

    def save(self, path: str):
        pass  # nothing to save

    def load(self, path: str):
        pass  # nothing to load
