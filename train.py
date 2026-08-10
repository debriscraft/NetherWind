"""
train.py
========
Training loop for AirCombatMARL.

Supports:
  - Baselines: MAPPO, HAPPO, IPPO, MADDPG, SAC, MAT
  - Proposed: BCA (GRE + BAD + PBC)
  - Ablation: GRE (w/o BAD + PBC)
  - TensorBoard logging
  - CSV reward logging
  - Model checkpointing
  - Interactive prompt mode (PyCharm "Run" friendly)
  - 3D real-time visualization (--visualize)

Usage (command line):
  python train.py                                    # interactive prompt (PyCharm)
  python train.py --algorithm mappo                   # MAPPO baseline
  python train.py --algorithm bca --n_red 3 --n_blue 3 --episodes 1000
  python train.py --visualize --episodes 50           # with 3D viz
  python train.py --resume models/bca_ep100.pt
  python train.py --lr 1e-4 --gamma 0.99 --entropy 0.01

When no command-line arguments are given (e.g., PyCharm "Run"),
enters interactive config mode where you type parameters at prompts.
"""

import os
import sys
import csv
import time
import argparse
import numpy as np
from datetime import datetime
import torch
import yaml  # For experiment config files
import concurrent.futures  # For parallel environment stepping

# Suppress OpenMP duplicate lib warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from env import CombatEnv
from algorithms import create_policy, MAPPO, GRE, BCA, HAPPO, IPPO, MADDPG, SAC, MAT
from missile import fire_decisions_red, DETECT_RANGE


def build_action_masks(env: CombatEnv):
    """
    Build action masks for red agents' fire decisions (RL-controlled firing).

    Returns:
        (n_red, n_blue + 1) float numpy array:
          - Columns 0..n_blue-1: 0.0 if valid target, -1e9 if invalid
          - Column n_blue: always 0.0 ("no fire" always valid)

    Invalid conditions: agent dead, target dead, out of range (>8000m),
                        no missiles left, on cooldown.
    """
    n_red = env.n_red
    n_blue = env.n_blue
    n_options = n_blue + 1
    MASK_VALUE = -1e9
    masks = np.zeros((n_red, n_options), dtype=np.float32)

    for i, ac in enumerate(env.red_aircraft):
        if not ac.alive:
            masks[i, :] = MASK_VALUE
            masks[i, -1] = 0.0
            continue
        for j, bac in enumerate(env.blue_aircraft):
            if not bac.alive:
                masks[i, j] = MASK_VALUE
            elif env.red_missiles_left[i] <= 0:
                masks[i, j] = MASK_VALUE
            elif env.red_cooldowns[i] > 0:
                masks[i, j] = MASK_VALUE
            else:
                dist_val = np.linalg.norm(ac.position - bac.position)
                if dist_val > 8000.0:
                    masks[i, j] = MASK_VALUE
        masks[i, -1] = 0.0  # no-fire always valid
    return masks


def build_arg_parser():
    """Build argument parser for training."""
    p = argparse.ArgumentParser(
        description='AirCombatMARL Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Algorithm ----
    p.add_argument('--algorithm', type=str, default='mappo',
                   choices=['mappo', 'happo', 'ippo', 'maddpg', 'sac', 'mat', 'sao_mat', 'gre', 'bca', 'random'],
                   help='RL algorithm: mappo (baseline), happo, ippo, maddpg, sac, mat, sao_mat (proposed), gre (ablation), bca, random (lower bound)')
    p.add_argument('--use_gre', action='store_true', default=True,
                   help='Use GRE actor (set False for ablation w/o GRE)')
    p.add_argument('--no_gre', action='store_false', dest='use_gre',
                   help='Disable GRE actor (ablation w/o GRE)')
    p.add_argument('--use_bvd', action='store_true', default=True,
                   help='Use BAD dual critic (set False for ablation w/o BAD)')
    p.add_argument('--no_bvd', action='store_false', dest='use_bvd',
                   help='Disable BAD dual critic (ablation w/o BAD)')
    p.add_argument('--bvd_lambda_comp', type=float, default=0.5,
                   help='BVD: weight of the competitive advantage stream (default 0.5)')
    p.add_argument('--pbc_tau_up', type=float, default=0.5,
                   help='PBC: promote difficulty when windowed win rate exceeds this (default 0.5)')
    p.add_argument('--pbc_tau_down', type=float, default=0.3,
                   help='PBC: demote difficulty when windowed win rate falls below this (default 0.3)')
    p.add_argument('--use_pbc', action='store_true', default=True,
                   help='Use PBC curriculum (set False for ablation w/o PBC)')
    p.add_argument('--no_pbc', action='store_false', dest='use_pbc',
                   help='Disable PBC curriculum (ablation w/o PBC)')
    p.add_argument('--sil_lambda', type=float, default=0.1,
                   help='SAO-MAT: SIL loss weight lambda')
    p.add_argument('--sil_threshold', type=float, default=0.0,
                   help='SAO-MAT: SIL advantage threshold')
    p.add_argument('--sil_capacity', type=int, default=30000,
                   help='SAO-MAT: SIL buffer capacity')
    p.add_argument('--sil_update_interval', type=int, default=20,
                   help='SAO-MAT: SIL update interval (episodes)')
    p.add_argument('--sil_advantage_clip', type=float, default=10.0,
                   help='SAO-MAT: SIL advantage tanh clip scale')
    p.add_argument('--sao_order_coeff', type=float, default=0.1,
                   help='SAO-MAT: weight of the action-order subtask loss')
    p.add_argument('--sil_anchor_beta', type=float, default=0.02,
                   help='SAO-MAT: KL weight toward EMA anchor policy on elite states (0 = off)')
    p.add_argument('--lr_final', type=float, default=None,
                   help='BCA: if set, linearly decay lr from --lr to this value over the consolidation phase (last 20%% of episodes)')
    p.add_argument('--sil_anchor_tau', type=float, default=0.995,
                   help='SAO-MAT: EMA decay for the anchor policy')

    # ---- Environment ----
    p.add_argument('--n_red', type=int, default=3,
                   help='Number of red aircraft (1-10)')
    p.add_argument('--n_blue', type=int, default=3,
                   help='Number of blue aircraft (1-10)')
    p.add_argument('--episodes', type=int, default=500,
                   help='Number of training episodes')
    p.add_argument('--rollout_steps', type=int, default=800,
                   help='Max rollout steps per episode (now matches MAX_EPISODE_STEPS=800)')
    p.add_argument('--num_envs', type=int, default=1,
                   help='Number of parallel environments (1=serial, >1=parallel rollout collection)')
    p.add_argument('--blue_difficulty', type=str, default='combat',
                   choices=['passive', 'easy', 'maneuver', 'combat', 'static'],
                   help='Blue team difficulty (passive/easy/maneuver/combat/static) for curriculum learning')
    p.add_argument('--reward_fn', type=str, default='base',
                   choices=['base', 'tactical'],
                   help='Reward shaping function: base (v2.1 original) | tactical (v3.0 improved)')

    # ---- PPO hyperparameters ----
    p.add_argument('--lr', type=float, default=3e-4,
                   help='Learning rate')
    p.add_argument('--gamma', type=float, default=0.99,
                   help='Discount factor')
    p.add_argument('--gae_lambda', type=float, default=0.95,
                   help='GAE lambda')
    p.add_argument('--clip_epsilon', type=float, default=0.1,
                   help='PPO clip epsilon (default 0.1 for stability)')
    p.add_argument('--entropy', type=float, default=0.01,
                   help='Entropy coefficient')
    p.add_argument('--max_grad_norm', type=float, default=0.3,
                   help='Max gradient norm (default 0.3 for stability)')
    p.add_argument('--ppo_epochs', type=int, default=4,
                   help='PPO update epochs')
    p.add_argument('--mini_batch_size', type=int, default=128,
                   help='Mini-batch size (default 128 for stability)')

    # ---- Checkpoint ----
    p.add_argument('--resume', type=str, default=None,
                   help='Resume from checkpoint file')
    p.add_argument('--save_interval', type=int, default=10,
                   help='Save checkpoint every N episodes')
    p.add_argument('--models_dir', type=str, default='models',
                   help='Directory for model checkpoints')
    p.add_argument('--logs_dir', type=str, default='logs',
                   help='Directory for CSV logs')

    # ---- Misc ----
    p.add_argument('--config', type=str, default=None,
                   help='Path to experiment config file (YAML)')
    p.add_argument('--run_name', type=str, default=None,
                   help='Run identifier used as file-name prefix (defaults to --algorithm; '
                        'use to distinguish ablation variants, e.g. sao_no_order)')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed (ignored if --n_seeds > 1)')
    p.add_argument('--n_seeds', type=int, default=1,
                   help='Number of seeds for statistical rigor (mean+/-std)')
    p.add_argument('--base_seed', type=int, default=42,
                   help='Base seed for multi-seed runs')
    p.add_argument('--eval_interval', type=int, default=50,
                   help='Evaluate every N episodes (independent evaluation)')
    p.add_argument('--eval_episodes', type=int, default=50,
                   help='Number of episodes for independent evaluation (increased from 20 for stability)')
    p.add_argument('--eval_deterministic', action='store_true',
                   help='Use deterministic (greedy) actions for evaluation (default: stochastic)')
    p.add_argument('--eval_difficulty', type=str, default='combat',
                   choices=['passive', 'easy', 'maneuver', 'combat'],
                   help='Blue team difficulty for evaluation (default: combat, was hardcoded)')
    p.add_argument('--no_tb', action='store_true',
                   help='Disable TensorBoard logging')
    p.add_argument('--quiet', action='store_true',
                   help='Reduce console output (only summary lines)')
    p.add_argument('--visualize', action='store_true',
                   help='Enable 3D visualization during training (SLOWS training significantly)')
    p.add_argument('--record_trajectories', action='store_true',
                   help='Save per-episode replay JSONs (large disk usage; off by default)')
    p.add_argument('--fire_rl', action='store_true',
                   help='Enable RL-controlled weapon firing (hybrid action space)')
    p.add_argument('--std_anneal', action='store_true',
                   help='Anneal log-std ceiling linearly (log space) from 0.0 to -1.6 '
                        '(std 1.0 -> 0.2) over training; locks the policy into its best '
                        'mode late and prevents end-of-training oscillation (BCA only)')

    return p


def load_config(config_path: str) -> dict:
    """
    Load experiment config from YAML file.

    Args:
        config_path: path to YAML config file
    Returns:
        dict: config dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def apply_config_to_args(args, config: dict):
    """
    Apply config dictionary to args.

    Args:
        args: argparse.Namespace
        config: config dictionary from YAML
    """
    # Environment
    if 'env' in config:
        for key, value in config['env'].items():
            if hasattr(args, key):
                setattr(args, key, value)

    # PPO hyperparameters
    if 'ppo' in config:
        for key, value in config['ppo'].items():
            key = key.replace('-', '_')  # yaml keys might use hyphens
            if hasattr(args, key):
                setattr(args, key, value)

    # GRE-specific args
    if args.algorithm in ['gre', 'bca'] and 'gre' in config:
        pass  # GRE-specific args would be passed to policy constructor

    # BVD-specific args
    if args.algorithm == 'bca' and 'bvd' in config:
        pass  # BVD-specific args would be passed to policy constructor

    # PBC-specific args
    if args.algorithm == 'bca' and 'pbc' in config:
        pass  # PBC-specific args would be passed to env


def judge_result_at_truncation(info, done):
    """Judge episode result, handling early truncation."""
    if done:
        result = info.get('result', 'ongoing')
    else:
        ra, ba = info['red_alive'], info['blue_alive']
        if ra == 0:
            result = 'loss'
        elif ba == 0:
            result = 'win'
        elif ra > ba:
            result = 'win'
        elif ra < ba:
            result = 'loss'
        else:
            result = 'draw'
    return result


def evaluate_policy(policy, args, n_episodes: int = 20):
    """
    Independent evaluation of policy (fixed opponent, no curriculum).

    Args:
        policy: policy object (MAPPO, BCA, etc.)
        args: command-line arguments
        n_episodes: number of evaluation episodes

    Returns:
        dict: {win_rate, avg_return, results, tactics}
    """
    from tactics_analyzer import analyze_evaluation, print_tactics_report, save_trajectory_for_visualization

    # Create eval env with configurable difficulty (was hardcoded 'combat')
    eval_env = CombatEnv(
        n_red=args.n_red,
        n_blue=args.n_blue,
        blue_difficulty=args.eval_difficulty,
        debug_print=False,
        reward_fn=getattr(args, 'reward_fn', 'base'),
    )
    is_bca = (args.algorithm == 'bca')

    wins, losses, draws = 0, 0, 0
    returns = []
    total_kills_red = 0   # enemies destroyed by red
    total_kills_blue = 0  # red aircraft destroyed (= friendly losses)
    all_episode_trajectories = []  # For tactics analysis
    episode_results = []  # Track result of each episode

    for ep in range(n_episodes):
        obs = eval_env.reset()
        episode_return = 0.0
        done = False
        step = 0
        episode_trajectory = []  # Collect positions for this episode

        while not done and step < args.rollout_steps:
            # Same obs hygiene as training: non-finite -> sanitize, then
            # clip astronomical-but-finite values that would overflow
            # float32 inside the policy's first Linear layer.
            if not np.isfinite(obs).all():
                obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            obs = np.clip(obs, -1e6, 1e6)
            # Action selection for evaluation (configurable via --eval_deterministic)
            with torch.no_grad():
                # select_actions expects numpy array, returns (cont_actions_list, log_probs_list)
                try:
                    cont_actions_list, _ = policy.select_actions(obs, deterministic=args.eval_deterministic)
                except TypeError:
                    # Some baseline policies do not accept the deterministic kwarg
                    cont_actions_list, _ = policy.select_actions(obs)

            # Collect trajectory data (positions) - always record all agents (pad with NaN if dead)
            red_positions = np.array([ac.position.copy() if ac.alive else np.full(3, np.nan) for ac in eval_env.red_aircraft])
            blue_positions = np.array([ac.position.copy() if ac.alive else np.full(3, np.nan) for ac in eval_env.blue_aircraft])
            episode_trajectory.append({
                'positions_red': red_positions,  # (n_red, 3), NaN if dead
                'positions_blue': blue_positions,  # (n_blue, 3), NaN if dead
                'step': step,
            })

            # cont_actions_list is already a list of per-agent actions

            # Rule-based fire is handled INSIDE env.step (red_fire_actions=None),
            # identical to training: single launch per opportunity, with the
            # Pk model applied. (Previously eval fired manually here AND let
            # env.step fire again without hit_pk — double missiles, no Pk,
            # which systematically inflated every algorithm's eval win rate.)
            next_obs, rewards, done, info = eval_env.step(cont_actions_list, None)
            episode_return += np.mean(rewards)
            obs = next_obs
            step += 1

        result = judge_result_at_truncation(info, done)
        if result == 'win':
            wins += 1
        elif result == 'loss':
            losses += 1
        else:
            draws += 1
        total_kills_red += info.get('kills_red', 0)
        total_kills_blue += info.get('kills_blue', 0)
        returns.append(episode_return)
        all_episode_trajectories.append(episode_trajectory)
        episode_results.append(result)

    win_rate = wins / n_episodes * 100
    avg_return = np.mean(returns)
    # KLR: enemy kills / friendly losses (denominator floored at 1)
    klr = total_kills_red / max(total_kills_blue, 1)

    # ---- Tactical behavior analysis ----
    tactics_metrics = {}
    if args.algorithm in ('bca', 'gre'):  # Only analyze for our algorithms
        try:
            tactics_metrics = analyze_evaluation(all_episode_trajectories)
            print_tactics_report(tactics_metrics)
        except Exception as e:
            print(f"  [TACTICS] Analysis failed: {e}")

    # ---- Save sample trajectories for paper figures ----
    tactics_dir = os.path.join(args.logs_dir, 'tactics')
    os.makedirs(tactics_dir, exist_ok=True)
    for ep_idx, (traj, result) in enumerate(zip(all_episode_trajectories, episode_results)):
        if result == 'win' and ep_idx < 3:  # Save first 3 wins as examples
            traj_path = os.path.join(tactics_dir, f'{args.algorithm}_win_ep{ep_idx}.npz')
            save_trajectory_for_visualization(traj, traj_path, ep_idx, result)

    return {
        'win_rate': win_rate,
        'avg_return': avg_return,
        'klr': klr,
        'kills_red': total_kills_red,
        'kills_blue': total_kills_blue,
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'tactics': tactics_metrics,  # Add tactics metrics to return value
    }


def run_episode(env, policy, rollout_steps, is_bca=False, visualizer=None, viz_interval=5,
                use_fire_rl=False, record_trajectory=False, traj_dir='trajectories',
                episode_num=None, algorithm='mappo'):
    """
    Run one training episode.

    Args:
        env: CombatEnv instance OR VectorizedCombatEnv
        policy: policy object (MAPPO, BCA, SAO-MAT, etc.)
        rollout_steps: max steps per episode
        is_bca: if True, use BCA dual-value storage
        visualizer: Visualizer3D instance (optional)
        viz_interval: update visualization every N steps
        use_fire_rl: if True, use RL hybrid actions for weapon firing
        record_trajectory: if True, save full replay to JSON
        traj_dir: directory to save trajectory files
        episode_num: current episode number (for filename)
        algorithm: algorithm name (for filename)

    Returns:
        (episode_rewards, info, step, obs_final, done_final)
        For VectorizedCombatEnv: returns COMBINED stats across all envs.
    """
    # ---- Dispatch to vectorized version if needed ----
    if isinstance(env, VectorizedCombatEnv):
        return _run_episode_vectorized(env, policy, rollout_steps, is_bca, visualizer, viz_interval, use_fire_rl)

    # ---- Original single-env logic ----
    # Enable trajectory recording
    if record_trajectory:
        env.record_replay = True
        if not hasattr(env, 'replay_data'):
            env.replay_data = []
        env.replay_data.clear()

    obs = env.reset()
    episode_rewards = [0.0] * env.n_red
    done = False
    step = 0

    while not done and step < rollout_steps:
        # Defensive: never let a non-finite observation reach the policy
        # (a departed flight model is sanitized at source in jsbsim_aircraft,
        # but belt-and-suspenders for 20+ hour unattended batch runs)
        if not np.isfinite(obs).all():
            print(f"[WARN] Non-finite observation at step {step}, sanitizing")
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        # Magnitude hygiene: finite-but-astronomical values (e.g. a division
        # by a near-zero closing rate producing ~1e30) overflow float32 in
        # the first Linear layer -> inf -> LayerNorm -> NaN mean, which kills
        # select_actions with healthy weights. Physical obs are <= ~1e5, so
        # clipping at 1e6 is lossless and applies identically to all algos.
        obs = np.clip(obs, -1e6, 1e6)

        # ---- Action selection ----
        if use_fire_rl:
            action_masks = build_action_masks(env)
            cont_actions, fire_actions, log_probs = policy.select_actions(obs, action_mask=action_masks)
            cont_actions_list = [cont_actions[i] for i in range(env.n_red)]
            fire_actions_list = [int(fire_actions[i]) for i in range(env.n_red)]
            log_probs_list = [log_probs[i] for i in range(env.n_red)]
        else:
            actions, log_probs = policy.select_actions(obs)
            cont_actions_list = [actions[i] for i in range(env.n_red)]
            fire_actions_list = None
            log_probs_list = [log_probs[i] for i in range(env.n_red)]

        # Get global value for GAE
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).to(policy.device)
            if is_bca:
                value_coop, value_comp = policy.get_values(obs_t)
            elif algorithm == 'ippo':
                # IPPO: per-agent independent critics
                value = policy.get_values(obs_t)
            elif algorithm in ('mat', 'sao_mat'):
                # MAT / SAO-MAT: transformer-based centralized value
                value = float(policy.get_values(obs_t))
            elif algorithm in ('maddpg', 'sac', 'random'):
                # Off-policy baselines: value not used by store_transition
                value = 0.0
            else:
                # MAPPO / HAPPO: centralized critic over global state
                global_obs = obs_t.reshape(1, -1)
                value = policy.critic(global_obs).cpu().numpy().flatten()[0]

        # Rule-based fire (only when NOT using RL fire)
        if not use_fire_rl:
            fire_launches = fire_decisions_red(
                env.red_aircraft, env.blue_aircraft,
                env.red_missiles_left,
                [ac.alive for ac in env.blue_aircraft]
            )
            for shooter_idx, target_idx in fire_launches:
                if env.red_missiles_left[shooter_idx] > 0:
                    ac = env.red_aircraft[shooter_idx]
                    env.missile_mgr.launch(
                        ac.position, ac.velocity,
                        target_idx + env.n_red,
                        shooter_team='red', shooter_idx=shooter_idx
                    )
                    env.red_missiles_left[shooter_idx] -= 1

        # Step environment
        step_fire_actions = fire_actions_list if use_fire_rl else None
        next_obs, rewards, done, info = env.step(cont_actions_list, step_fire_actions)

        # Store transition for PPO rollout buffer
        if is_bca:
            # BCA: dual rewards
            rewards_array = np.array(rewards)
            rewards_coop = rewards  # team reward
            rewards_comp = rewards_array - np.mean(rewards_array)  # individual deviation
            policy.store_transition(
                obs, cont_actions_list, log_probs_list,
                rewards_coop.tolist() if isinstance(rewards_coop, np.ndarray) else rewards_coop,
                rewards_comp.tolist() if isinstance(rewards_comp, np.ndarray) else rewards_comp,
                float(done), value_coop, value_comp,
            )
        elif use_fire_rl:
            policy.store_transition(
                obs, cont_actions_list, log_probs_list, rewards,
                float(done), value, fire_actions=fire_actions_list
            )
        else:
            policy.store_transition(
                obs, cont_actions_list, log_probs_list, rewards,
                float(done), value
            )

        # ---- Visualization ----
        if visualizer is not None and step % viz_interval == 0:
            visualizer.update(env)

        obs = next_obs
        for i in range(env.n_red):
            episode_rewards[i] += rewards[i]
        step += 1

    # ---- Save trajectory replay ----
    if record_trajectory and episode_num is not None:
        os.makedirs(traj_dir, exist_ok=True)
        from datetime import datetime
        ts = datetime.now().strftime('%H%M%S')
        fname = f'{algorithm}_ep{episode_num}_{ts}.json'
        fpath = os.path.join(traj_dir, fname)
        try:
            env.save_replay(fpath)
        except Exception as e:
            print(f'  [WARN] Failed to save trajectory: {e}')

    return episode_rewards, info, step, obs, float(done)


class VectorizedCombatEnv:
    """
    Vectorized CombatEnv: runs N environments in parallel using ThreadPoolExecutor.

    Usage:
        envs = VectorizedCombatEnv(n_red=3, n_blue=3, n_envs=4)
        obs_list = envs.reset()  # List of N observations
        # Each obs is (n_red, obs_dim)
        actions_list = [...]  # List of N action lists
        next_obs_list, rewards_list, dones_list, infos_list = envs.step(actions_list)

    NOTE: JSBSim may not be thread-safe. If crashes occur, set use_threads=False
          to fall back to serial stepping.
    """

    def __init__(self, n_red, n_blue, n_envs, blue_difficulty='combat', use_threads=True, reward_fn='base'):
        self.n_red = n_red
        self.n_blue = n_blue
        self.n_envs = n_envs
        self.use_threads = use_threads
        self.envs = [CombatEnv(n_red=n_red, n_blue=n_blue,
                        blue_difficulty=blue_difficulty,
                        debug_print=True,
                        reward_fn=reward_fn)
                      for _ in range(n_envs)]
        self.obs_dim = self.envs[0].obs_dim
        self.action_dim = self.envs[0].action_dim

    def reset(self):
        """Reset all environments. Returns list of observations."""
        if self.use_threads:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_envs) as executor:
                results = list(executor.map(lambda env: env.reset(), self.envs))
            return results
        else:
            return [env.reset() for env in self.envs]

    def step(self, actions_list):
        """
        Step all environments in parallel.

        Args:
            actions_list: List of N action lists, each of length n_red.
                         Each action can be (cont_actions, fire_actions) or just cont_actions.

        Returns:
            (obs_list, rewards_list, dones_list, infos_list)
            - obs_list: List of N observations, each (n_red, obs_dim)
            - rewards_list: List of N reward lists, each (n_red,)
            - dones_list: List of N booleans
            - infos_list: List of N info dicts
        """
        assert len(actions_list) == self.n_envs, f"Expected {self.n_envs} action lists, got {len(actions_list)}"

        if self.use_threads:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_envs) as executor:
                # Unpack actions for each env
                futures = []
                for i, actions in enumerate(actions_list):
                    if isinstance(actions, tuple) and len(actions) == 2:
                        # (cont_actions_list, fire_actions_list)
                        future = executor.submit(self.envs[i].step, actions[0], actions[1])
                    else:
                        # cont_actions_list only
                        future = executor.submit(self.envs[i].step, actions)
                    futures.append(future)
                results = [f.result() for f in futures]
        else:
            results = []
            for i, actions in enumerate(actions_list):
                if isinstance(actions, tuple) and len(actions) == 2:
                    results.append(self.envs[i].step(actions[0], actions[1]))
                else:
                    results.append(self.envs[i].step(actions))

        obs_list, rewards_list, dones_list, infos_list = zip(*results)
        return list(obs_list), list(rewards_list), list(dones_list), list(infos_list)

    def set_difficulty(self, difficulty):
        """Set difficulty for all environments."""
        for env in self.envs:
            env.set_difficulty(difficulty)


def _run_episode_vectorized(env, policy, rollout_steps, is_bca=False, visualizer=None, viz_interval=5,
                             use_fire_rl=False):
    """
    Run one training episode with vectorized (parallel) environments.

    Args:
        env: VectorizedCombatEnv instance
        policy: policy object (MAPPO, BCA, SAO-MAT, etc.)
        rollout_steps: max steps per episode PER ENV
        is_bca: if True, use BCA dual-value storage
        visualizer: Visualizer3D instance (optional, uses env.envs[0])
        viz_interval: update visualization every N steps
        use_fire_rl: if True, use RL hybrid actions for weapon firing

    Returns:
        (episode_rewards, info, total_steps, final_obs, final_done)
        - episode_rewards: Combined across all envs
        - info: Combined info dict (from first done env)
        - total_steps: Total steps across all envs
        - final_obs: Final observation (from first done env)
        - final_done: Whether ALL envs are done
    """
    obs_list = env.reset()  # List of N observations, each (n_red, obs_dim)
    episode_rewards = [0.0] * env.n_red
    total_steps = 0
    all_dones = [False] * env.n_envs
    infos = [None] * env.n_envs

    while not all(all_dones) and total_steps < rollout_steps * env.n_envs:
        # ---- Action selection for ALL active envs ----
        actions_list = []  # List of actions for active envs only
        active_indices = []

        for i in range(env.n_envs):
            if all_dones[i]:
                continue
            active_indices.append(i)

            # Get value for GAE (optional, done inside policy.update())
            if use_fire_rl:
                action_masks = build_action_masks(env.envs[i])
                cont_actions, fire_actions, log_probs = policy.select_actions(obs_list[i], action_mask=action_masks)
                actions_list.append((cont_actions, fire_actions))
            else:
                actions, log_probs = policy.select_actions(obs_list[i])
                actions_list.append(actions)

        if not actions_list:
            break

        # ---- Step ALL active envs in parallel ----
        next_obs_list, rewards_list, dones_list, infos_list = env.step(actions_list)

        # ---- Store transitions from ALL active envs ----
        for idx, i in enumerate(active_indices):
            if is_bca:
                # BCA: dual rewards
                rewards_array = np.array(rewards_list[idx])
                rewards_coop = rewards_list[idx]
                rewards_comp = rewards_array - np.mean(rewards_array)
                policy.store_transition(
                    obs_list[i],
                    actions_list[idx][0] if use_fire_rl else actions_list[idx],
                    [0.0] * env.n_red,  # log_probs (simplified)
                    rewards_coop.tolist(),
                    rewards_comp.tolist(),
                    float(dones_list[idx]),
                    0.0,  # value_coop (simplified)
                    0.0,  # value_comp (simplified)
                )
            else:
                # MAPPO: single reward
                policy.store_transition(
                    obs_list[i],
                    actions_list[idx][0] if use_fire_rl else actions_list[idx],
                    [0.0] * env.n_red,  # log_probs (simplified)
                    rewards_list[idx],
                    float(dones_list[idx]),
                    0.0,  # value (simplified)
                )

            # ---- Update tracking ----
            for j in range(env.n_red):
                episode_rewards[j] += rewards_list[idx][j]
            total_steps += 1
            if dones_list[idx]:
                all_dones[i] = True
            infos[i] = infos_list[idx]

        # ---- Update observations ----
        for idx, i in enumerate(active_indices):
            obs_list[i] = next_obs_list[idx]

    # ---- Return combined stats ----
    # Find first done env's info/obs
    first_done_idx = next((i for i in range(env.n_envs) if all_dones[i]), 0)
    final_obs = obs_list[first_done_idx]
    final_done = all(all_dones)
    combined_info = infos[first_done_idx] if infos[first_done_idx] else {}

    return episode_rewards, combined_info, total_steps, final_obs, final_done


def interactive_config():

    """
    Prompt user for training configuration interactively.
    Used when train.py is launched from PyCharm (no command-line args).
    Returns a list of strings suitable for argparse.parse_args().
    """
    print("=" * 60)
    print("  AirCombatMARL - Interactive Configuration")
    print("  (Press Enter to accept defaults shown in [brackets])")
    print("=" * 60)

    def ask(prompt_text, default):
        val = input(f"  {prompt_text} [{default}]: ").strip()
        return val if val else str(default)

    # ---- Algorithm ----
    algo = ask("Algorithm (mappo/happo/ippo/maddpg/sac/mat/sao_mat/gre/bca)", "mappo").lower()
    if algo not in ('mappo', 'happo', 'ippo', 'maddpg', 'sac', 'mat', 'sao_mat', 'gre', 'bca'):
        print(f"  -> Unknown, falling back to mappo")
        algo = 'mappo'

    # ---- Environment ----
    n_red = ask("Number of red aircraft (1-10)", 3)
    n_blue = ask("Number of blue aircraft (1-10)", 3)
    episodes = ask("Training episodes", 500)
    rollout = ask("Rollout steps per episode", 800)
    blue_diff = ask("Blue difficulty (passive/easy/maneuver/combat/static)", "combat").lower()
    if blue_diff not in ('passive', 'easy', 'maneuver', 'combat', 'static'):
        print(f"  -> Unknown, falling back to combat")
        blue_diff = 'combat'
    reward_fn = ask("Reward function (base/tactical)", "base").lower()
    if reward_fn not in ('base', 'tactical'):
        print(f"  -> Unknown, falling back to base")
        reward_fn = 'base'

    # ---- PPO hyperparams (quick) ----
    print("  --- Hyperparameters (press Enter for defaults) ---")
    lr_default = "1e-4" if algo in ('gre', 'bca') else "3e-4"
    entropy_default = "0.01"
    lr = ask("Learning rate", lr_default)
    entropy = ask("Entropy coefficient", entropy_default)

    # ---- Visualization ----
    viz = input("  Enable 3D visualization? [y/N]: ").strip().lower()
    visualize = viz.startswith('y')

    # ---- Misc ----
    seed = ask("Random seed", 42)
    quiet = input("  Quiet mode (less console output)? [y/N]: ").strip().lower()
    is_quiet = quiet.startswith('y')

    # Build argument list
    cmd_args = [
        '--algorithm', algo,
        '--n_red', str(n_red),
        '--n_blue', str(n_blue),
        '--episodes', str(episodes),
        '--rollout_steps', str(rollout),
        '--lr', str(lr),
        '--entropy', str(entropy),
        '--seed', str(seed),
        '--blue_difficulty', blue_diff,
        '--reward_fn', reward_fn,
    ]
    if visualize:
        cmd_args.append('--visualize')
    if is_quiet:
        cmd_args.append('--quiet')

    print("=" * 60)
    print(f"  Starting: {algo.upper()} {n_red}v{n_blue}, {episodes}ep, "
          f"lr={lr}, ent={entropy}")
    if visualize:
        print("  3D Visualization: ENABLED")
    print("=" * 60)

    return cmd_args


def main():
    parser = build_arg_parser()

    # PyCharm-friendly: if no command-line args, enter interactive mode
    if len(sys.argv) <= 1:
        cmd_args = interactive_config()
        args = parser.parse_args(cmd_args)
    else:
        args = parser.parse_args()

    # Clamp values
    args.n_red = max(1, min(10, args.n_red))
    args.n_blue = max(1, min(10, args.n_blue))

    # Load config file (if provided)
    if args.config:
        config = load_config(args.config)
        apply_config_to_args(args, config)
        print(f"  Loaded config: {args.config}")

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---- Banner (once, before seed loop) ----
    algo_name = args.algorithm.upper()
    run_name = args.run_name if args.run_name else args.algorithm
    print("=" * 60)
    print(f"  AirCombatMARL - {algo_name} Training")
    print("=" * 60)
    print(f"  Config: {args.n_red}v{args.n_blue}, {args.episodes} episodes, "
          f"rollout={args.rollout_steps} steps")
    print(f"  Blue difficulty: {args.blue_difficulty}")
    print(f"  Algorithm: {algo_name}  (run: {run_name})")
    print(f"  Hyperparams: lr={args.lr}, gamma={args.gamma}, "
          f"gae_lambda={args.gae_lambda}, entropy={args.entropy}")
    if args.n_seeds > 1:
        print(f"  Multi-seed: {args.n_seeds} seeds (base={args.base_seed})")

    # ---- Per-seed results (for aggregation) ----
    per_seed_stats = []
    win_rate_history = []  # For sample efficiency tracking
    ep50_reached_at = None  # Episode where win rate first reaches 50%

    for seed_idx in range(args.n_seeds):
        current_seed = args.base_seed + seed_idx
        is_multi = args.n_seeds > 1

        if is_multi:
            print(f"\n  {'=' * 56}")
            print(f"  Seed {seed_idx + 1}/{args.n_seeds} (seed={current_seed})")
            print(f"  {'=' * 56}")

        # Set seed
        np.random.seed(current_seed)
        torch.manual_seed(current_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(current_seed)

        # ---- Setup env ----
        if args.num_envs > 1:
            env = VectorizedCombatEnv(
                n_red=args.n_red, n_blue=args.n_blue,
                n_envs=args.num_envs,
                blue_difficulty=args.blue_difficulty,
                use_threads=True,
                reward_fn=args.reward_fn,
            )
            print(f"  Using {args.num_envs} parallel environments (VectorizedCombatEnv)")
        else:
            env = CombatEnv(n_red=args.n_red, n_blue=args.n_blue,
                            blue_difficulty=args.blue_difficulty,
                            debug_print=True,
                            reward_fn=args.reward_fn)
        obs_dim = env.obs_dim
        action_dim = env.action_dim
        n_agents = args.n_red

        # ---- PBC: Initialize curriculum scheduler (for BCA algorithm) ----
        use_pbc = (args.algorithm == 'bca') and args.use_pbc  # Honor --no_pbc flag
        if use_pbc:
            # PBC phases: Phase 1 (80% eps) adaptive, Phase 2 (20% eps) fixed combat
            pbc_phase1_episodes = int(0.8 * args.episodes)
            pbc_current_difficulty = 'passive'  # PBC MUST start from easiest opponent
            pbc_win_rate_window = []  # track recent win rate
            pbc_window_size = 20  # window size for win rate calculation
            print(f"  PBC: enabled (Phase 1: episodes 1-{pbc_phase1_episodes}, Phase 2: episodes {pbc_phase1_episodes+1}-{args.episodes})")
            # Actually set environment to passive difficulty
            env.set_difficulty('passive')
            print(f"  PBC: initial difficulty set to 'passive'")

        if not is_multi:
            print(f"  obs_dim={obs_dim}, action_dim={action_dim}, n_agents={n_agents}")

        # ---- Setup visualizer (3D real-time) ----
        visualizer = None
        if args.visualize:
            from visualizer import Visualizer3D
            visualizer = Visualizer3D(n_red=args.n_red, n_blue=args.n_blue)
            visualizer.set_episode_info(0, args.episodes)
            if not is_multi:
                print("  3D Visualization: ENABLED (may slow training)")

        # ---- Setup policy ----
        is_gre = (args.algorithm == 'gre')
        is_bca = (args.algorithm == 'bca')
        is_sao = (args.algorithm == 'sao_mat')
        use_fire_rl = args.fire_rl
        n_fire_targets = env.n_blue + 1 if use_fire_rl else 0

        # Build kwargs for create_policy
        policy_kwargs = {
            'algorithm': args.algorithm,
            'obs_dim': obs_dim,
            'action_dim': action_dim,
            'n_agents': n_agents,
            'n_red': args.n_red,
            'n_blue': args.n_blue,
            'n_fire_targets': n_fire_targets,
            'lr': args.lr,
            'gamma': args.gamma,
            'gae_lambda': args.gae_lambda,
            'clip_epsilon': args.clip_epsilon,
            'entropy_coeff': args.entropy,
            'value_coeff': 0.5,
            'max_grad_norm': args.max_grad_norm,
            'ppo_epochs': args.ppo_epochs,
            'mini_batch_size': args.mini_batch_size,
            'use_gre': args.use_gre,
            'use_bvd': args.use_bvd,
        }

        # BCA-specific hyperparameters
        if args.algorithm == 'bca':
            policy_kwargs['bvd_lambda_comp'] = args.bvd_lambda_comp

        # SAO-MAT-specific hyperparameters (order learning + anchored SIL)
        if args.algorithm == 'sao_mat':
            policy_kwargs.update({
                'order_coeff': args.sao_order_coeff,
                'sil_lambda': args.sil_lambda,
                'sil_threshold': args.sil_threshold,
                'sil_capacity': args.sil_capacity,
                'sil_update_interval': args.sil_update_interval,
                'sil_advantage_clip': args.sil_advantage_clip,
                'sil_anchor_beta': args.sil_anchor_beta,
                'sil_anchor_tau': args.sil_anchor_tau,
            })

        policy = create_policy(**policy_kwargs)

        if not is_multi:
            print(f"  Device: {policy.device}")

        # ---- LR warning for BCA (stability) ----
        if args.algorithm == 'bca' and args.lr > 1e-4:
            print(f"  [WARN] BCA with lr={args.lr:.1e} > 1e-4 may cause PL fluctuation.")
            print(f"         Recommended: --lr 1e-4")

        # Resume from checkpoint (seed 0 only)
        start_episode = 1
        if args.resume and seed_idx == 0:
            policy.load(args.resume)
            try:
                fname = os.path.basename(args.resume)
                # Fix: use regex to extract episode number after "ep"
                import re
                match = re.search(r'_ep(\d+)\.pt$', fname)
                if match:
                    start_episode = int(match.group(1)) + 1
                else:
                    # Fallback: try to extract last number from filename
                    numbers = re.findall(r'\d+', fname)
                    if numbers:
                        start_episode = int(numbers[-1]) + 1
            except Exception as e:
                print(f"  [WARN] Could not parse episode from filename: {e}, starting from 1")
                start_episode = 1
            print(f"  Resuming from episode {start_episode}")

        # ---- Setup TensorBoard (per-seed) ----
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        use_tb = not args.no_tb
        writer = None
        if use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                if is_multi:
                    tb_dir = f"runs/{run_name}_{args.n_red}v{args.n_blue}_seed{current_seed}_{timestamp}"
                else:
                    tb_dir = f"runs/{run_name}_{args.n_red}v{args.n_blue}_{timestamp}"
                writer = SummaryWriter(log_dir=tb_dir)
                if not is_multi:
                    print(f"  TensorBoard: {tb_dir}")
            except ImportError:
                use_tb = False
                if not is_multi:
                    print("  TensorBoard not available, skipping.")

        # ---- Setup CSV logging (per-seed) ----
        os.makedirs(args.logs_dir, exist_ok=True)
        if is_multi:
            csv_path = f"{args.logs_dir}/{run_name}_{args.n_red}v{args.n_blue}_seed{current_seed}_{timestamp}.csv"
            eval_csv_path = f"{args.logs_dir}/{run_name}_{args.n_red}v{args.n_blue}_eval_seed{current_seed}_{timestamp}.csv"
        else:
            csv_path = f"{args.logs_dir}/{run_name}_{args.n_red}v{args.n_blue}_{timestamp}.csv"
            eval_csv_path = f"{args.logs_dir}/{run_name}_{args.n_red}v{args.n_blue}_eval_{timestamp}.csv"
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        header = ['episode', 'reward_mean', 'reward_max', 'reward_min',
                  'red_alive', 'blue_alive', 'result', 'steps', 'duration',
                  'red_hp_mean', 'blue_hp_mean', 'red_bullets', 'blue_bullets',
                  'missile_launches_red', 'missile_hits_red', 'missile_hit_rate_red',
                  'missile_launches_blue', 'missile_hits_blue', 'missile_hit_rate_blue',
                  'kills_red', 'kills_blue',
                  'bullet_hits_red', 'bullet_hits_blue']
        csv_writer.writerow(header)

        # ---- Eval metrics CSV (independent evaluation history) ----
        eval_csv_file = open(eval_csv_path, 'w', newline='')
        eval_csv_writer = csv.writer(eval_csv_file)
        eval_csv_writer.writerow(['episode', 'win_rate', 'klr', 'avg_return',
                                  'wins', 'losses', 'draws'])
        last_eval = None

        # ---- Setup models dir ----
        os.makedirs(args.models_dir, exist_ok=True)

        # ---- Training loop ----
        cum_wins = 0
        cum_losses = 0
        cum_draws = 0
        best_win_rate = 0.0
        best_model_path = None

        if not is_multi:
            print(f"\n  Training started...")
            print("-" * 60)

        for episode in range(start_episode, args.episodes + 1):
            t_start = time.time()

            # Reset 3D visualizer for new episode
            if visualizer is not None:
                visualizer.reset_trails()
                visualizer.set_episode_info(episode, args.episodes)

            # ---- std annealing (BCA): linear in log space, 1.0 -> 0.2 ----
            if args.std_anneal and hasattr(policy, 'set_std_max_logit'):
                frac = min(1.0, episode / args.episodes)
                policy.set_std_max_logit(0.0 + frac * (-1.6 - 0.0))

            # ---- lr decay (BCA): linear over the consolidation phase ----
            if is_bca and args.lr_final is not None and hasattr(policy, 'set_lr'):
                decay_start = int(0.8 * args.episodes)
                if episode > decay_start:
                    frac = (episode - decay_start) / max(1, args.episodes - decay_start)
                    new_lr = args.lr + (args.lr_final - args.lr) * min(1.0, frac)
                    policy.set_lr(new_lr)
                    if episode == decay_start + 1 or episode == args.episodes:
                        print(f"  [LR] ep{episode}: lr decayed to {new_lr:.2e}", flush=True)

            # ---- PBC: Dynamic difficulty scheduling (for BCA only) ----
            if use_pbc:
                # Determine phase
                if episode <= pbc_phase1_episodes:
                    # Phase 1: Adaptive scheduling
                    if len(pbc_win_rate_window) >= pbc_window_size:
                        recent_win_rate = sum(pbc_win_rate_window[-pbc_window_size:]) / pbc_window_size
                        # Increase difficulty if win rate > tau_up
                        if recent_win_rate > args.pbc_tau_up and pbc_current_difficulty != 'combat':
                            diff_levels = ['passive', 'easy', 'maneuver', 'combat']
                            current_idx = diff_levels.index(pbc_current_difficulty)
                            pbc_current_difficulty = diff_levels[min(current_idx + 1, 3)]
                            env.set_difficulty(pbc_current_difficulty)
                            print(f"  [PBC] ep{episode} win={recent_win_rate:.2f} -> Increased difficulty to: {pbc_current_difficulty}", flush=True)
                        # Decrease difficulty if win rate < tau_down
                        elif recent_win_rate < args.pbc_tau_down and pbc_current_difficulty != 'passive':
                            diff_levels = ['passive', 'easy', 'maneuver', 'combat']
                            current_idx = diff_levels.index(pbc_current_difficulty)
                            pbc_current_difficulty = diff_levels[max(current_idx - 1, 0)]
                            env.set_difficulty(pbc_current_difficulty)
                            print(f"  [PBC] ep{episode} win={recent_win_rate:.2f} -> Decreased difficulty to: {pbc_current_difficulty}", flush=True)
                else:
                    # Phase 2: Fixed combat difficulty
                    if pbc_current_difficulty != 'combat':
                        pbc_current_difficulty = 'combat'
                        env.set_difficulty(pbc_current_difficulty)
                        print(f"  [PBC] ep{episode} Phase 2 started: fixed difficulty to combat", flush=True)

            # Run one episode
            episode_rewards, info, step_count, final_obs, final_done = run_episode(
                env, policy, args.rollout_steps, is_bca=is_bca,
                visualizer=visualizer, viz_interval=5, use_fire_rl=use_fire_rl,
                record_trajectory=args.record_trajectories,
                traj_dir=os.path.join(args.logs_dir, 'trajectories'),
                episode_num=episode, algorithm=args.algorithm,
            )

            # ---- PPO update ----
            update_info = policy.update(final_obs, final_done)

            # ---- Statistics ----
            t_elapsed = time.time() - t_start
            r_mean = np.mean(episode_rewards)
            r_max = np.max(episode_rewards)
            r_min = np.min(episode_rewards)

            result = judge_result_at_truncation(info, final_done)

            if result == 'win':
                cum_wins += 1
            elif result == 'loss':
                cum_losses += 1
            else:
                cum_draws += 1

            # ---- PBC: Update win rate tracking (for BCA only) ----
            if use_pbc:
                pbc_win_rate_window.append(1 if result == 'win' else 0)
                if len(pbc_win_rate_window) > pbc_window_size * 2:
                    pbc_win_rate_window.pop(0)

            result_map = {'win': '[WIN] ', 'loss': '[LOSS]', 'draw': '[DRAW]'}
            result_tag = result_map.get(result, '[?   ]')

            # ---- Track best model ----
            current_win_rate = cum_wins / (episode - start_episode + 1)
            if current_win_rate > best_win_rate and episode >= 5:
                best_win_rate = current_win_rate
                if is_multi:
                    best_model_path = f"{args.models_dir}/{run_name}_seed{current_seed}_best.pt"
                else:
                    best_model_path = f"{args.models_dir}/{run_name}_best.pt"
                policy.save(best_model_path)

            # ---- CSV logging ----
            row = [
                episode,
                f"{r_mean:.2f}",
                f"{r_max:.2f}",
                f"{r_min:.2f}",
                info['red_alive'], info['blue_alive'],
                result,
                step_count,
                f"{t_elapsed:.1f}",
                f"{info.get('red_hp_mean', 0):.1f}",
                f"{info.get('blue_hp_mean', 0):.1f}",
                info.get('red_bullets', 0),
                info.get('blue_bullets', 0),
                info.get('missile_launches_red', 0),
                info.get('missile_hits_red', 0),
                f"{info.get('missile_hit_rate_red', 0):.1f}",
                info.get('missile_launches_blue', 0),
                info.get('missile_hits_blue', 0),
                f"{info.get('missile_hit_rate_blue', 0):.1f}",
                info.get('kills_red', 0),
                info.get('kills_blue', 0),
                info.get('bullet_hits_red', 0),
                info.get('bullet_hits_blue', 0),
            ]
            csv_writer.writerow(row)
            csv_file.flush()

            # ---- TensorBoard ----
            if use_tb:
                writer.add_scalar('reward/mean', r_mean, episode)
                writer.add_scalar('reward/max', r_max, episode)
                writer.add_scalar('reward/min', r_min, episode)
                writer.add_scalar('train/policy_loss', update_info.get('policy_loss', update_info.get('actor_loss', 0)), episode)
                if is_bca:
                    writer.add_scalar('train/value_bvd_coop_loss', update_info.get('value_bvd_coop_loss', 0.0), episode)
                    writer.add_scalar('train/value_bvd_comp_loss', update_info.get('value_bvd_comp_loss', 0.0), episode)
                else:
                    writer.add_scalar('train/value_loss', update_info.get('value_loss', 0.0), episode)
                writer.add_scalar('train/entropy', update_info.get('entropy', 0.0), episode)
                # SAO-MAT: SIL + order-subtask metrics
                if is_sao:
                    writer.add_scalar('sil/loss', update_info.get('sil_loss', 0.0), episode)
                    writer.add_scalar('sil/buffer_size', update_info.get('sil_buffer', 0), episode)
                    writer.add_scalar('order/loss', update_info.get('order_loss', 0.0), episode)
                writer.add_scalar('stats/red_alive', info['red_alive'], episode)
                writer.add_scalar('stats/blue_alive', info['blue_alive'], episode)
                writer.add_scalar('stats/win_rate', cum_wins / episode * 100, episode)
                result_code = 1 if result == 'win' else (-1 if result == 'loss' else 0)
                writer.add_scalar('stats/result', result_code, episode)
                # Sample efficiency tracking
                current_wr = cum_wins / episode * 100
                win_rate_history.append(current_wr)
                if ep50_reached_at is None and current_wr >= 50.0:
                    ep50_reached_at = episode
                writer.add_scalar('stats/red_hp_mean', info.get('red_hp_mean', 0), episode)
                writer.add_scalar('stats/blue_hp_mean', info.get('blue_hp_mean', 0), episode)
                writer.add_scalar('stats/red_bullets', info.get('red_bullets', 0), episode)
                writer.add_scalar('stats/blue_bullets', info.get('blue_bullets', 0), episode)
                writer.add_scalar('tactical/red_missile_launches', info.get('missile_launches_red', 0), episode)
                writer.add_scalar('tactical/red_missile_hits', info.get('missile_hits_red', 0), episode)
                writer.add_scalar('tactical/red_missile_hit_rate', info.get('missile_hit_rate_red', 0), episode)
                writer.add_scalar('tactical/blue_missile_launches', info.get('missile_launches_blue', 0), episode)
                writer.add_scalar('tactical/blue_missile_hits', info.get('missile_hits_blue', 0), episode)
                writer.add_scalar('tactical/blue_missile_hit_rate', info.get('missile_hit_rate_blue', 0), episode)
                writer.add_scalar('tactical/red_kills', info.get('kills_red', 0), episode)
                writer.add_scalar('tactical/blue_kills', info.get('kills_blue', 0), episode)
                writer.add_scalar('tactical/red_bullet_hits', info.get('bullet_hits_red', 0), episode)
                writer.add_scalar('tactical/blue_bullet_hits', info.get('bullet_hits_blue', 0), episode)

            # ---- Console output ----
            if not args.quiet or episode % max(1, args.episodes // 20) == 0:
                pl = update_info.get('policy_loss', update_info.get('actor_loss', 0.0))
                if is_bca:
                    vl = update_info.get('value_bvd_coop_loss', 0.0)
                    vl2 = update_info.get('value_bvd_comp_loss', 0.0)
                    vl_str = f"VL_bvd_coop={vl:.4f} VL_bvd_comp={vl2:.4f}"
                else:
                    vl = update_info.get('value_loss', 0.0)
                    vl_str = f"VL={vl:.4f}"

                line = (
                    f"  Ep {episode:4d}/{args.episodes} | "
                    f"R={r_mean:+8.2f} [{r_min:+.1f}, {r_max:+.1f}] | "
                    f"Red={info['red_alive']}/{args.n_red}({info.get('red_hp_mean',0):.0f}hp) "
                    f"Blue={info['blue_alive']}/{args.n_blue}({info.get('blue_hp_mean',0):.0f}hp) | "
                    f"Blt:R{info.get('red_bullets',0)}/B{info.get('blue_bullets',0)} | "
                    f"{result_tag} | WR={cum_wins/episode*100:5.1f}% | "
                    f"PL={pl:.4f} "
                    f"{vl_str}"
                )
                # SAO-MAT: add SIL info to console output
                if is_sao:
                    sil_loss = update_info.get('sil_loss', 0.0)
                    sil_buf = update_info.get('sil_buffer', 0)
                    line += f" | SIL_loss={sil_loss:.4f} SIL_buf={sil_buf}"
                line += f" | {t_elapsed:.1f}s"
                print(line)

            # ---- Independent Evaluation (every N episodes) ----
            if episode % args.eval_interval == 0 and episode > 0:
                eval_result = evaluate_policy(policy, args, n_episodes=args.eval_episodes)
                eval_win_rate = eval_result['win_rate']
                eval_avg_return = eval_result['avg_return']
                eval_klr = eval_result.get('klr', 0.0)
                tactics = eval_result.get('tactics', {})

                # Log eval metrics to CSV (for paper tables/curves)
                eval_csv_writer.writerow([
                    episode, f"{eval_win_rate:.2f}", f"{eval_klr:.3f}",
                    f"{eval_avg_return:.2f}",
                    eval_result['wins'], eval_result['losses'], eval_result['draws'],
                ])
                eval_csv_file.flush()
                last_eval = eval_result

                # Main eval output
                eval_line = f"  [EVAL] Ep {episode}: Win Rate={eval_win_rate:.1f}%, "
                eval_line += f"KLR={eval_klr:.3f}, "
                eval_line += f"Avg Return={eval_avg_return:.2f}, "
                eval_line += f"W/L/D={eval_result['wins']}/{eval_result['losses']}/{eval_result['draws']} "
                eval_line += f"[difficulty={args.eval_difficulty}, deterministic={args.eval_deterministic}]"

                # Add tactics summary (if available)
                if tactics:
                    eval_line += f"\n         Tactics: "
                    eval_line += f"Flank={tactics.get('flanking_score_mean', 0):.2f}, "
                    eval_line += f"Spread={tactics.get('spatial_spread_mean', 0):.0f}m, "
                    eval_line += f"Lure={tactics.get('lure_detected_ratio', 0)*100:.0f}%, "
                    eval_line += f"Role={tactics.get('role_differentiation_mean', 0):.2f}"

                print(eval_line)

                # Log to TensorBoard
                if use_tb:
                    writer.add_scalar('eval/win_rate', eval_win_rate, episode)
                    writer.add_scalar('eval/avg_return', eval_avg_return, episode)
                    writer.add_scalar('eval/klr', eval_klr, episode)

                    # Log tactics metrics
                    if tactics:
                        writer.add_scalar('eval/flanking_score', tactics.get('flanking_score_mean', 0), episode)
                        writer.add_scalar('eval/spatial_spread', tactics.get('spatial_spread_mean', 0), episode)
                        writer.add_scalar('eval/lure_detected_ratio', tactics.get('lure_detected_ratio', 0), episode)
                        writer.add_scalar('eval/role_differentiation', tactics.get('role_differentiation_mean', 0), episode)

                # Save best model based on eval win rate
                if not hasattr(args, 'best_eval_win_rate'):
                    args.best_eval_win_rate = 0.0
                if eval_win_rate > args.best_eval_win_rate:
                    args.best_eval_win_rate = eval_win_rate
                    best_eval_path = f"{args.models_dir}/{run_name}_best_eval.pt"
                    policy.save(best_eval_path)
                    print(f"  [EVAL] New best eval model saved: {best_eval_path}")

            # Save checkpoint
            if episode % args.save_interval == 0:
                if is_multi:
                    ckpt_path = f"{args.models_dir}/{run_name}_seed{current_seed}_ep{episode}.pt"
                else:
                    ckpt_path = f"{args.models_dir}/{run_name}_ep{episode}.pt"
                policy.save(ckpt_path)

        # ---- Per-seed cleanup ----
        csv_file.close()
        eval_csv_file.close()
        if use_tb:
            writer.close()
        if visualizer is not None:
            visualizer.close()

        # Save final model
        if is_multi:
            final_path = f"{args.models_dir}/{run_name}_seed{current_seed}_final.pt"
        else:
            final_path = f"{args.models_dir}/{run_name}_final.pt"
        policy.save(final_path)

        # ---- Per-seed summary ----
        neff = args.episodes - start_episode + 1
        seed_wr = cum_wins / neff * 100
        # Compute sample efficiency metrics (manual trapezoidal rule for AUC)
        if len(win_rate_history) >= 2:
            auc = float(np.sum((np.array(win_rate_history[1:]) + np.array(win_rate_history[:-1])) / 2))
        else:
            auc = 0.0
        per_seed_stats.append({
            'seed': current_seed,
            'wins': cum_wins,
            'losses': cum_losses,
            'draws': cum_draws,
            'win_rate': seed_wr,
            'ep50': ep50_reached_at if ep50_reached_at else -1,
            'auc': auc,
            'eval_win_rate': last_eval['win_rate'] if last_eval else float('nan'),
            'eval_klr': last_eval.get('klr', float('nan')) if last_eval else float('nan'),
            'eval_return': last_eval['avg_return'] if last_eval else float('nan'),
            'csv_path': csv_path,
            'eval_csv_path': eval_csv_path,
            'best_model': best_model_path,
        })

        if is_multi:
            print(f"  Seed {current_seed}: {cum_wins}W/{cum_losses}L/{cum_draws}D | WR={seed_wr:.1f}%")

    # ======== Aggregate summary (after all seeds) ========
    print("\n" + "=" * 60)
    print(f"  Training complete! ({args.n_seeds} seed{'s' if args.n_seeds > 1 else ''})")
    print("=" * 60)

    if args.n_seeds > 1:
        wr_list = [s['win_rate'] for s in per_seed_stats]
        print(f"  Win Rate (mean +/- std): {np.mean(wr_list):.1f}% +/- {np.std(wr_list):.1f}%")
        eval_wr_list = [s['eval_win_rate'] for s in per_seed_stats]
        eval_klr_list = [s['eval_klr'] for s in per_seed_stats]
        eval_ret_list = [s['eval_return'] for s in per_seed_stats]
        print(f"  Final Eval WR:  {np.nanmean(eval_wr_list):.1f}% +/- {np.nanstd(eval_wr_list):.1f}%")
        print(f"  Final Eval KLR: {np.nanmean(eval_klr_list):.3f} +/- {np.nanstd(eval_klr_list):.3f}")
        print(f"  Final Eval Ret: {np.nanmean(eval_ret_list):.1f} +/- {np.nanstd(eval_ret_list):.1f}")
        print(f"  Per-seed: ", end="")
        for s in per_seed_stats:
            print(f"seed{s['seed']}={s['win_rate']:.1f}%", end="  ")
        print()

        # Save aggregate CSV
        agg_path = f"{args.logs_dir}/{run_name}_{args.n_red}v{args.n_blue}_aggregate_{timestamp}.csv"
        with open(agg_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['seed', 'wins', 'losses', 'draws', 'win_rate_pct', 'ep50', 'auc',
                        'eval_win_rate', 'eval_klr', 'eval_return',
                        'csv_path', 'eval_csv_path', 'best_model'])
            for s in per_seed_stats:
                w.writerow([s['seed'], s['wins'], s['losses'], s['draws'],
                            f"{s['win_rate']:.1f}", s['ep50'], f"{s['auc']:.1f}",
                            f"{s['eval_win_rate']:.2f}", f"{s['eval_klr']:.3f}",
                            f"{s['eval_return']:.1f}",
                            s['csv_path'], s['eval_csv_path'], s['best_model'] or ''])
        # Statistical significance: 95% CI + t-test (vs MAPPO baseline if available)
        if args.n_seeds > 1:
            wr_list = [s['win_rate'] for s in per_seed_stats]
            ep50_list = [s['ep50'] for s in per_seed_stats if s['ep50'] > 0]
            auc_list = [s['auc'] for s in per_seed_stats]
            import scipy.stats as stats
            ci_low, ci_high = stats.t.interval(0.95, len(wr_list)-1, loc=np.mean(wr_list), scale=stats.sem(wr_list))
            print(f"  Win Rate: {np.mean(wr_list):.1f}% [{ci_low:.1f}, {ci_high:.1f}] (95% CI)")
            if ep50_list:
                print(f"  Sample Efficiency (ep50): {np.mean(ep50_list):.0f} +/- {np.std(ep50_list):.0f} eps")
            print(f"  AUC: {np.mean(auc_list):.0f} +/- {np.std(auc_list):.0f}")
        print(f"  Aggregate CSV: {agg_path}")
    else:
        s = per_seed_stats[0]
        print(f"  Episodes: {neff}  |  "
              f"Results: {s['wins']}W / {s['losses']}L / {s['draws']}D  |  "
              f"Win rate: {s['win_rate']:.1f}%")
        if s['best_model']:
            print(f"  Best model: {s['best_model']}")
        print(f"  Final model: {final_path}")
        print(f"  CSV log: {s['csv_path']}")

    if use_tb or not args.no_tb:
        print(f"  TensorBoard: tensorboard --logdir=runs")
    print("=" * 60)


if __name__ == '__main__':
    main()
