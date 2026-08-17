"""
Algorithms package for AirCombatMARL.

This package contains all RL algorithms:
  - Baselines: MAPPO, HAPPO, IPPO, MADDPG, SAC, MAT
  - Proposed: BCA (GRE + BVD + PBC)
  - Ablations: GRE

Usage:
    from algorithms import MAPPO, HAPPO, BCA
    policy = MAPPO(obs_dim=..., action_dim=...)
"""

from .mappo import MAPPOPolicy as MAPPO
from .happo import HAPPOPolicy as HAPPO
from .ippo import IPPOPolicy as IPPO
from .maddpg import MADDPGPolicy as MADDPG
from .sac import SACPolicy as SAC
from .mat import MATPolicy as MAT
try:
    from .sao_mat import SAOMATPolicy as SAO_MAT
except ImportError:
    SAO_MAT = None  # excluded from release
from .bca import BCAPolicy as BCA
from .gre_policy import GREPolicy as GRE
from .random_policy import RandomPolicy as Random

__all__ = [
    'MAPPO',   # Baseline 1: Centralized critic (Yu et al., NeurIPS 2021)
    'HAPPO',   # Baseline 2: Heterogeneous PPO (Kuba et al., NeurIPS 2021)
    'IPPO',    # Baseline 3: Independent PPO (de Witt et al., NeurIPS 2020)
    'MADDPG',  # Baseline 4: Multi-agent DDPG (Lowe et al., NIPS 2017)
    'SAC',     # Baseline 5: Soft Actor-Critic (Haarnoja et al., ICML 2018)
    'MAT',     # Baseline 6: Multi-agent Transformer (Wen et al., ICLR 2022)
    'SAO_MAT', # Proposed: Sequential Action-Order MAT + anchored SIL
    'Random',  # Baseline 0: Uniform random policy (lower bound)
    'BCA',     # Proposed: Bilateral Credit Assignment (GRE + BVD + PBC)
    'GRE',     # Ablation: GRE only (w/o BVD + PBC)
]


def create_policy(algorithm: str, obs_dim: int, action_dim: int, n_agents: int,
                  n_red: int = 3, n_blue: int = 3, n_fire_targets: int = 0,
                  lr: float = 3e-4, gamma: float = 0.99, gae_lambda: float = 0.95,
                  clip_epsilon: float = 0.2, entropy_coeff: float = 0.01,
                  value_coeff: float = 0.5, max_grad_norm: float = 0.5,
                  ppo_epochs: int = 4, mini_batch_size: int = 64,
                  **kwargs):
    """
    Factory function to create policy based on algorithm name.

    Args:
        algorithm: 'mappo' | 'happo' | 'ippo' | 'maddpg' | 'sac' | 'mat' | 'bca' | 'gre'
        obs_dim: observation dimension (per agent)
        action_dim: action dimension (continuous)
        n_agents: number of red agents
        n_red: number of red aircraft
        n_blue: number of blue aircraft
        n_fire_targets: number of discrete fire targets (0 = no discrete head)
        lr: learning rate
        gamma: discount factor
        gae_lambda: GAE lambda
        clip_epsilon: PPO clip epsilon
        entropy_coeff: entropy coefficient
        value_coeff: value loss coefficient
        max_grad_norm: max gradient norm
        ppo_epochs: PPO update epochs
        mini_batch_size: mini-batch size
        **kwargs: algorithm-specific parameters

    Returns:
        policy object
    """
    algo = algorithm.lower()

    if algo == 'random':
        return Random(
            obs_dim=obs_dim, action_dim=action_dim, n_agents=n_agents,
            n_fire_targets=n_fire_targets,
        )

    if algo == 'mappo':
        return MAPPO(
            obs_dim=obs_dim, action_dim=action_dim, n_agents=n_agents,
            n_fire_targets=n_fire_targets, lr=lr, gamma=gamma,
            gae_lambda=gae_lambda, clip_epsilon=clip_epsilon,
            entropy_coeff=entropy_coeff, value_coeff=value_coeff,
            max_grad_norm=max_grad_norm, ppo_epochs=ppo_epochs,
            mini_batch_size=mini_batch_size
        )

    elif algo == 'happo':
        return HAPPO(
            obs_dim=obs_dim, action_dim=action_dim, n_agents=n_agents,
            n_fire_targets=n_fire_targets, lr=lr, gamma=gamma,
            gae_lambda=gae_lambda, clip_epsilon=clip_epsilon,
            entropy_coeff=entropy_coeff, value_coeff=value_coeff,
            max_grad_norm=max_grad_norm, ppo_epochs=ppo_epochs,
            mini_batch_size=mini_batch_size
        )

    elif algo == 'ippo':
        return IPPO(
            obs_dim=obs_dim, action_dim=action_dim, n_agents=n_agents,
            lr=lr, gamma=gamma, gae_lambda=gae_lambda,
            clip_epsilon=clip_epsilon, entropy_coeff=entropy_coeff
        )

    elif algo == 'maddpg':
        return MADDPG(
            obs_dim=obs_dim, action_dim=action_dim, n_agents=n_agents,
            n_fire_targets=n_fire_targets, actor_lr=lr, critic_lr=lr, gamma=gamma
        )

    elif algo == 'sac':
        return SAC(
            obs_dim=obs_dim, action_dim=action_dim, n_agents=n_agents,
            lr=lr, gamma=gamma
        )

    elif algo == 'mat':
        return MAT(
            obs_dim=obs_dim, action_dim=action_dim, n_agents=n_agents,
            n_red=n_red, n_blue=n_blue, n_fire_targets=n_fire_targets,
            lr=lr, gamma=gamma, gae_lambda=gae_lambda,
            clip_epsilon=clip_epsilon, entropy_coeff=entropy_coeff,
            value_coeff=value_coeff, max_grad_norm=max_grad_norm,
            ppo_epochs=ppo_epochs, mini_batch_size=mini_batch_size
        )

    elif algo == 'sao_mat':
        return SAO_MAT(
            obs_dim=obs_dim, action_dim=action_dim, n_agents=n_agents,
            n_red=n_red, n_blue=n_blue, n_fire_targets=n_fire_targets,
            lr=lr, gamma=gamma, gae_lambda=gae_lambda,
            clip_epsilon=clip_epsilon, entropy_coeff=entropy_coeff,
            value_coeff=value_coeff, max_grad_norm=max_grad_norm,
            ppo_epochs=ppo_epochs, mini_batch_size=mini_batch_size,
            order_coeff=kwargs.get('order_coeff', 0.1),
            sil_lambda=kwargs.get('sil_lambda', 0.1),
            sil_threshold=kwargs.get('sil_threshold', 0.0),
            sil_capacity=kwargs.get('sil_capacity', 30000),
            sil_update_interval=kwargs.get('sil_update_interval', 20),
            sil_advantage_clip=kwargs.get('sil_advantage_clip', 10.0),
            sil_anchor_beta=kwargs.get('sil_anchor_beta', 0.02),
            sil_anchor_tau=kwargs.get('sil_anchor_tau', 0.995),
        )

    elif algo == 'bca':
        return BCA(
            obs_dim=obs_dim, action_dim=action_dim, n_agents=n_agents,
            n_red=n_red, n_blue=n_blue, n_fire_targets=n_fire_targets,
            lr=lr, gamma=gamma, gae_lambda=gae_lambda,
            clip_epsilon=clip_epsilon, entropy_coeff=entropy_coeff,
            value_coeff=value_coeff, max_grad_norm=max_grad_norm,
            ppo_epochs=ppo_epochs, mini_batch_size=mini_batch_size,
            **kwargs
        )

    elif algo == 'gre':
        return GRE(
            obs_dim=obs_dim, action_dim=action_dim, n_agents=n_agents,
            n_red=n_red, n_blue=n_blue, n_fire_targets=n_fire_targets,
            lr=lr, gamma=gamma, gae_lambda=gae_lambda,
            clip_epsilon=clip_epsilon, entropy_coeff=entropy_coeff,
            value_coeff=value_coeff, max_grad_norm=max_grad_norm,
            ppo_epochs=ppo_epochs, mini_batch_size=mini_batch_size
        )

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

