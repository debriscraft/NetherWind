"""
rewards/__init__.py
===================
Reward function registry for AirCombatMARL.

Usage:
    from rewards import get_reward_fn
    reward_fn = get_reward_fn('tactical', env)
"""

from rewards.base_reward import BaseReward
from rewards.tactical_reward import TacticalReward

_REGISTRY = {
    'base': BaseReward,
    'tactical': TacticalReward,
}


def get_reward_fn(name: str, env):
    """
    Get a reward function instance by name.

    Args:
        name: 'base' | 'tactical'
        env: CombatEnv instance (passed to reward fn for state access)

    Returns:
        Reward function instance
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown reward function: {name}. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name](env)


def list_reward_fns():
    """Return list of available reward function names."""
    return list(_REGISTRY.keys())
