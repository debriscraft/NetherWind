"""
pbc.py
======
Population-Based Curriculum (PBC) for BCA.

Implements the PBC two-phase training schedule from the BCA paper (§4.3):
  - Phase 1 (0-80% episodes): Adaptive opponent selection
  - Phase 2 (80-100% episodes): Fixed opponent p4 (Rule-Combat)

Also includes:
  - Opponent pool management (p1-p7)
  - Adaptive scheduling based on win rate
  - Diversity regularization loss
"""

import numpy as np
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Opponent Pool Definition (matches paper Table 2)
# ---------------------------------------------------------------------------

OPPONENT_POOL = {
    1: {'name': 'Rule-Passive', 'type': 'rule', 'difficulty': 'passive', 'weapons': False},
    2: {'name': 'Rule-Mild', 'type': 'rule', 'difficulty': 'easy', 'weapons': False},
    3: {'name': 'Rule-Evasive', 'type': 'rule', 'difficulty': 'maneuver', 'weapons': False},
    4: {'name': 'Rule-Combat', 'type': 'rule', 'difficulty': 'combat', 'weapons': True},
    5: {'name': 'Snapshot-100', 'type': 'snapshot', 'epoch': 100, 'weapons': True},
    6: {'name': 'Snapshot-200', 'type': 'snapshot', 'epoch': 200, 'weapons': True},
    7: {'name': 'Snapshot-300', 'type': 'snapshot', 'epoch': 300, 'weapons': True},
}


# ---------------------------------------------------------------------------
# PBC Scheduler
# ---------------------------------------------------------------------------

class PBCScheduler:
    """
    PBC two-phase training scheduler.

    Phase 1 (0-80% episodes): Adaptive opponent selection
    Phase 2 (80-100% episodes): Fixed opponent p4 (Rule-Combat)
    """

    def __init__(
        self,
        total_episodes: int,
        phase1_ratio: float = 0.8,
        tau_up: float = 0.65,
        tau_down: float = 0.35,
        eval_interval: int = 20,
        hysteresis: int = 2,
        pool_size: int = 7,
    ):
        self.total_episodes = total_episodes
        self.phase1_end = int(total_episodes * phase1_ratio)
        self.tau_up = tau_up
        self.tau_down = tau_down
        self.eval_interval = eval_interval
        self.hysteresis = hysteresis
        self.pool_size = pool_size

        # Current state
        self.current_opponent = 1  # Start with easiest opponent
        self.current_phase = 1  # Phase 1 or 2
        self.win_rates = []  # History of win rates
        self.eval_count = 0
        self.consecutive_up = 0
        self.consecutive_down = 0

    def get_opponent(self, episode: int) -> int:
        """
        Get current opponent index based on training episode.

        Returns:
            opponent_idx: 1-7 (index into OPPONENT_POOL)
        """
        if episode < self.phase1_end:
            # Phase 1: Adaptive scheduling
            return self.current_opponent
        else:
            # Phase 2: Fixed p4 (Rule-Combat)
            self.current_phase = 2
            return 4

    def update(self, episode: int, win_rate: float) -> int:
        """
        Update opponent based on recent win rate (Phase 1 only).

        Args:
            episode: Current episode number
            win_rate: Win rate over last eval_interval episodes

        Returns:
            new_opponent: Updated opponent index (or current if no change)
        """
        if episode >= self.phase1_end:
            # Phase 2: No opponent update
            return self.current_opponent

        # Phase 1: Adaptive scheduling
        self.win_rates.append(win_rate)
        self.eval_count += 1

        if self.eval_count < self.hysteresis:
            # Not enough evaluations for hysteresis
            return self.current_opponent

        # Check last `hysteresis` evaluations
        recent_rates = self.win_rates[-self.hysteresis:]

        if all(r > self.tau_up for r in recent_rates):
            # Advance opponent (increase difficulty)
            self.consecutive_up += 1
            self.consecutive_down = 0
            if self.consecutive_up >= self.hysteresis:
                self.current_opponent = min(self.current_opponent + 1, self.pool_size)
                self.consecutive_up = 0
        elif all(r < self.tau_down for r in recent_rates):
            # Retreat opponent (decrease difficulty)
            self.consecutive_down += 1
            self.consecutive_up = 0
            if self.consecutive_down >= self.hysteresis:
                self.current_opponent = max(self.current_opponent - 1, 1)
                self.consecutive_down = 0
        else:
            # Reset counters
            self.consecutive_up = 0
            self.consecutive_down = 0

        return self.current_opponent

    def get_phase(self, episode: int) -> int:
        """Return current phase (1 or 2) based on episode number."""
        return 1 if episode < self.phase1_end else 2


# ---------------------------------------------------------------------------
# Diversity Regularization Loss
# ---------------------------------------------------------------------------

def diversity_loss(embeddings: np.ndarray, lambda_div: float = 0.01) -> float:
    """
    Compute diversity regularization loss to prevent policy collapse.

    Args:
        embeddings: (n_red, d_model) GRE embeddings
        lambda_div: Regularization coefficient

    Returns:
        loss: Scalar diversity loss (negative = encourage diversity)
    """
    n_red = embeddings.shape[0]
    if n_red < 2:
        return 0.0

    # Pairwise distance matrix
    dist_matrix = np.zeros((n_red, n_red))
    for i in range(n_red):
        for j in range(n_red):
            if i != j:
                dist_matrix[i, j] = np.linalg.norm(embeddings[i] - embeddings[j])

    # Loss = -mean(dist_matrix) (negative to encourage larger distances)
    loss = -lambda_div * np.mean(dist_matrix)

    return loss
