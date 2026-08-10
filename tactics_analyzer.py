"""
Tactics Analyzer for AirCombatMARL
====================================
Analyzes emergent tactical behaviors during evaluation.

Behaviors tracked:
1. Lure/Decoy: One agent draws enemy attention while others flank
2. Flanking: Agents approach enemy from multiple angles
3. Role Differentiation: Agents consistently take different roles
4. Coordinated Attack: Multiple agents engage simultaneously

Usage:
    from tactics_analyzer import analyze_episode, print_tactics_report
    
    # During evaluation, collect trajectories
    trajectories = []  # List of (positions_red, positions_blue, actions, rewards)
    
    # After evaluation, analyze
    metrics = analyze_episode(trajectories)
    print_tactics_report(metrics)
"""

import numpy as np
from typing import List, Dict, Tuple


def _filter_alive(positions: np.ndarray) -> np.ndarray:
    """
    Filter out dead agents from positions array.
    
    Args:
        positions: (n_agents, 3) array, may contain NaN for dead agents
        
    Returns:
        (n_alive, 3) array of positions for alive agents only
    """
    alive_mask = ~np.isnan(positions).any(axis=1)
    return positions[alive_mask]


def compute_spatial_spread(positions: np.ndarray) -> float:
    """
    Compute spatial spread of agent positions.
    
    Args:
        positions: (n_agents, 3) array of agent positions (may contain NaN for dead)
        
    Returns:
        float: spatial spread (standard deviation of pairwise distances)
    """
    positions = _filter_alive(positions)
    
    if len(positions) <= 1:
        return 0.0
    
    # Pairwise distances
    dists = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dists.append(np.linalg.norm(positions[i] - positions[j]))
    
    return np.std(dists)


def compute_flanking_score(positions_red: np.ndarray, 
                          positions_blue: np.ndarray) -> float:
    """
    Compute flanking score: how well red agents surround blue agents.
    
    High score = agents approaching from multiple angles.
    Low score = all agents approaching from same direction.
    
    Args:
        positions_red: (n_red, 3) array (may contain NaN for dead)
        positions_blue: (n_blue, 3) array (may contain NaN for dead)
        
    Returns:
        float: flanking score [0, 1] where 1 = perfect surround
    """
    red_alive = _filter_alive(positions_red)
    blue_alive = _filter_alive(positions_blue)
    
    if len(red_alive) <= 1 or len(blue_alive) == 0:
        return 0.0
    
    # Use centroid of blue team as reference
    blue_center = np.mean(blue_alive, axis=0)
    
    # Compute angles of red agents relative to blue center
    # Project to 2D (top-down view)
    angles = []
    for pos in red_alive:
        dx = pos[0] - blue_center[0]
        dy = pos[1] - blue_center[1]
        angle = np.arctan2(dy, dx)
        angles.append(angle)
    
    angles = np.array(angles)
    
    # Flanking = angular spread
    # Perfect flank: angles spread evenly around circle
    if len(angles) == 1:
        return 0.0
    
    # Angular diff (wrapped)
    angle_diff = np.abs(angles[:, None] - angles[None, :])
    angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
    angular_spread = np.mean(angle_diff)
    
    # Normalize: max spread = pi (opposite sides)
    flanking_score = angular_spread / np.pi
    
    return flanking_score


def detect_lure_behavior(positions_red: np.ndarray,
                         positions_blue: np.ndarray,
                         step: int,
                         team_center_history: List[np.ndarray]) -> Tuple[bool, float]:
    """
    Detect lure/decoy behavior: one agent separates from team.
    
    Args:
        positions_red: (n_red, 3) array (may contain NaN for dead)
        positions_blue: (n_blue, 3) array (unused, kept for API compatibility)
        step: current step
        team_center_history: history of team center positions
        
    Returns:
        (bool, float): (lure_detected, lure_score)
    """
    positions_red = _filter_alive(positions_red)
    
    if len(positions_red) <= 1:
        return False, 0.0
    
    # Compute team center
    team_center = np.mean(positions_red, axis=0)
    
    # Compute distance of each agent from team center
    dists_from_center = [np.linalg.norm(pos - team_center) for pos in positions_red]
    
    # Lure = one agent far from center, others close
    dists_sorted = sorted(dists_from_center)
    if len(dists_sorted) >= 2:
        lure_score = (dists_sorted[-1] - dists_sorted[0]) / (np.mean(dists_sorted) + 1e-8)
        
        # Detected if: one agent > 1.5x farther than closest agent
        lure_detected = lure_score > 1.5
    else:
        lure_detected = False
        lure_score = 0.0
    
    return lure_detected, lure_score


def compute_role_differentiation(action_history: List[List[np.ndarray]]) -> float:
    """
    Compute role differentiation score.
    
    High score = agents consistently take different actions (specialization).
    Low score = all agents behave similarly (no role emergence).
    
    Args:
        action_history: List of List of actions, shape (n_steps, n_agents, action_dim)
        
    Returns:
        float: role differentiation score [0, 1]
    """
    if len(action_history) < 10:
        return 0.0
    
    n_agents = len(action_history[0])
    action_dim = len(action_history[0][0])
    
    # Compute mean action for each agent across episode
    agent_mean_actions = []
    for agent_idx in range(n_agents):
        agent_actions = [step_actions[agent_idx] for step_actions in action_history]
        mean_action = np.mean(agent_actions, axis=0)
        agent_mean_actions.append(mean_action)
    
    agent_mean_actions = np.array(agent_mean_actions)  # (n_agents, action_dim)
    
    # Pairwise difference between agents' mean actions
    diffs = []
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            diff = np.linalg.norm(agent_mean_actions[i] - agent_mean_actions[j])
            diffs.append(diff)
    
    # Normalize by action space range (assuming actions in [-1, 1])
    max_diff = 2.0 * np.sqrt(action_dim)  # Max possible diff
    role_score = np.mean(diffs) / (max_diff + 1e-8)
    
    return min(role_score, 1.0)


def analyze_episode(trajectories: List[Dict]) -> Dict:
    """
    Analyze a single evaluation episode for tactical behaviors.
    
    Args:
        trajectories: List of dicts, each containing:
            - 'positions_red': (n_red, 3) numpy array (NaN if dead)
            - 'positions_blue': (n_blue, 3) numpy array (NaN if dead)
            - 'actions_red': List of (n_red,) action arrays (optional)
            - 'step': step number
            
    Returns:
        Dict with tactical metrics
    """
    if len(trajectories) == 0:
        return {}
    
    n_red = trajectories[0]['positions_red'].shape[0]  # Always n_red (with NaN for dead)
    
    # Collect metrics over time
    flanking_scores = []
    spatial_spreads = []
    lure_scores = []
    lure_detected_steps = 0
    
    positions_red_history = []  # Store team centers
    action_history = []
    
    for traj in trajectories:
        pos_red = traj['positions_red']  # (n_red, 3), NaN if dead
        pos_blue = traj['positions_blue']  # (n_blue, 3), NaN if dead
        
        # Flanking
        f_score = compute_flanking_score(pos_red, pos_blue)
        flanking_scores.append(f_score)
        
        # Spatial spread
        spread = compute_spatial_spread(pos_red)
        spatial_spreads.append(spread)
        
        # Lure detection
        lure_detected, lure_score = detect_lure_behavior(
            pos_red, pos_blue, traj['step'], positions_red_history
        )
        lure_scores.append(lure_score)
        if lure_detected:
            lure_detected_steps += 1
        
        # Track team center (using alive agents only)
        red_alive = _filter_alive(pos_red)
        if len(red_alive) > 0:
            positions_red_history.append(np.mean(red_alive, axis=0))
        else:
            positions_red_history.append(np.zeros(3))
        
        if 'actions_red' in traj:
            action_history.append(traj['actions_red'])
    
    # Compile results
    results = {
        'flanking_score_mean': np.mean(flanking_scores),
        'flanking_score_max': np.max(flanking_scores),
        'spatial_spread_mean': np.mean(spatial_spreads),
        'spatial_spread_max': np.max(spatial_spreads),
        'lure_score_mean': np.mean(lure_scores),
        'lure_detected_ratio': lure_detected_steps / len(trajectories),
        'n_red': n_red,
        'episode_length': len(trajectories),
    }
    
    # Role differentiation (if enough data)
    if len(action_history) >= 10:
        results['role_differentiation'] = compute_role_differentiation(action_history)
    else:
        results['role_differentiation'] = 0.0
    
    return results


def analyze_evaluation(eval_trajectories: List[List[Dict]]) -> Dict:
    """
    Analyze all evaluation episodes.
    
    Args:
        eval_trajectories: List of trajectories, one per episode.
                          Each element is a List of trajectory dicts (one per step).
                        
    Returns:
        Dict with aggregated tactical metrics across all episodes.
    """
    if len(eval_trajectories) == 0:
        return {}
    
    all_metrics = []
    for ep_trajs in eval_trajectories:
        ep_metrics = analyze_episode(ep_trajs)
        all_metrics.append(ep_metrics)
    
    # Aggregate
    aggregated = {
        'flanking_score_mean': np.mean([m.get('flanking_score_mean', 0) for m in all_metrics]),
        'flanking_score_std': np.std([m.get('flanking_score_mean', 0) for m in all_metrics]),
        'spatial_spread_mean': np.mean([m.get('spatial_spread_mean', 0) for m in all_metrics]),
        'lure_detected_ratio': np.mean([m.get('lure_detected_ratio', 0) for m in all_metrics]),
        'lure_score_mean': np.mean([m.get('lure_score_mean', 0) for m in all_metrics]),
        'role_differentiation_mean': np.mean([m.get('role_differentiation', 0) for m in all_metrics]),
        'n_episodes': len(eval_trajectories),
    }
    
    return aggregated


def print_tactics_report(metrics: Dict):
    """Print a human-readable tactics report."""
    print("\n  " + "=" * 50)
    print("  TACTICAL BEHAVIOR ANALYSIS")
    print("  " + "=" * 50)
    
    if 'flanking_score_mean' in metrics:
        print(f"  Flanking Score:     {metrics['flanking_score_mean']:.3f} +/- {metrics.get('flanking_score_std', 0):.3f}")
        print(f"    (1.0 = perfect surround, 0.0 = single direction)")
    
    if 'spatial_spread_mean' in metrics:
        print(f"  Spatial Spread:     {metrics['spatial_spread_mean']:.1f}m (mean)")
        print(f"    (higher = more dispersed formation)")
    
    if 'lure_detected_ratio' in metrics:
        print(f"  Lure Behavior:      {metrics['lure_detected_ratio']*100:.1f}% of steps")
        print(f"    (agent separation detected)")
    
    if 'lure_score_mean' in metrics:
        print(f"  Lure Score:         {metrics['lure_score_mean']:.3f}")
        print(f"    (higher = more distinct lure vs attack roles)")
    
    if 'role_differentiation_mean' in metrics:
        print(f"  Role Differentiation: {metrics['role_differentiation_mean']:.3f}")
        print(f"    (higher = more specialized agent roles)")
    
    print("  " + "=" * 50)
    
    # Interpretation
    print("\n  INTERPRETATION:")
    if metrics.get('flanking_score_mean', 0) > 0.5:
        print("  ✓ Good flanking behavior detected")
    else:
        print("  ✗ Limited flanking (agents tend to cluster)")
    
    if metrics.get('lure_detected_ratio', 0) > 0.2:
        print("  ✓ Lure/decoy behavior emerging")
    else:
        print("  ✗ No clear lure behavior")
    
    if metrics.get('role_differentiation_mean', 0) > 0.3:
        print("  ✓ Agent role specialization detected")
    else:
        print("  ✗ Agents behave similarly (no role emergence)")
    
    print()


def save_trajectory_for_visualization(trajectories: List[Dict],
                                      filepath: str,
                                      episode_idx: int,
                                      result: str):
    """
    Save trajectory data for later visualization (3D plotting, paper figures).
    
    Args:
        trajectories: List of trajectory dicts (one per step)
        filepath: output .npz file path
        episode_idx: episode index
        result: 'win', 'loss', or 'draw'
    """
    # Extract arrays (always n_red x 3, with NaN for dead agents)
    positions_red = np.array([t['positions_red'] for t in trajectories])  # (n_steps, n_red, 3)
    positions_blue = np.array([t['positions_blue'] for t in trajectories])  # (n_steps, n_blue, 3)
    
    # Save
    np.savez(
        filepath,
        positions_red=positions_red,
        positions_blue=positions_blue,
        episode_idx=episode_idx,
        result=result,
    )
    print(f"  [TACTICS] Trajectory saved: {filepath}")


if __name__ == '__main__':
    # Quick test
    print("Tactics Analyzer module loaded.")
    print("Usage: from tactics_analyzer import analyze_evaluation, print_tactics_report")
