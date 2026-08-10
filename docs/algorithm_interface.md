# Writing a Custom Algorithm

External researchers can plug their own MARL algorithm into the benchmark by
adding one file to `algorithms/` and registering it in
`algorithms/__init__.py`. No other code changes are needed.

## The policy interface

`train.py` drives training through a small policy-object interface. Your
class must provide:

| Method | Signature | Semantics |
|--------|-----------|-----------|
| `select_actions` | `(obs: np.ndarray) -> (actions, log_probs, values)` | `obs` has shape `(n_agents, obs_dim)`. Returns per-agent actions in `[-1, 1]^4`, their log-probabilities (or `None` for off-policy), and value estimates (or `None`). For deterministic evaluation the training loop calls the same method with `deterministic=True` if your signature accepts it. |
| `store_transition` | `(obs, actions, log_probs, rewards, done, values)` | Called once per environment step with the full transition. |
| `update` | `(next_obs, next_done) -> dict | None` | Called at rollout boundaries; run your gradient steps here. Return a dict of scalars for logging (e.g. `{"loss": ...}`), or `None`. |
| `get_values` | `(obs: torch.Tensor) -> torch.Tensor` | Centralized or per-agent values; used by the critic diagnostics. |
| `save` / `load` | `(path: str)` | Checkpoint I/O. `train.py` saves every `--save_interval` episodes as `models/<run_name>_ep<N>.pt` and relies on `load` for `--resume`. |

Optional hooks (called only when present, via `hasattr`):

- `set_lr(lr)` — learning-rate scheduling.
- `set_std_max_logit(x)` — exploration-noise annealing.
- `device` attribute — `torch.device` used by the training loop for tensors.

## Observation and action spaces

- **Observation** per agent: `19 * (n_red + n_blue) + 12` floats
  (3v3: 126; 5v5: 202). Layout: 19 features per aircraft (position, velocity,
  attitude, speed, alive flag, team id, engagement geometry) for every
  aircraft on both teams, followed by 12 tactical summary features
  (force ratio, threat counts, centroid offsets).
- **Action** per agent: 4 continuous commands in `[-1, 1]` —
  `[pitch_cmd, roll_cmd, yaw_cmd, throttle]`, mapped to the JSBSim
  elevator/aileron/rudder/throttle channels. Weapon release is governed by
  the compiled rule-based fire control, identical for both teams; learned
  behavior is maneuver only.

## Registration

```python
# algorithms/my_algo.py
class MyAlgoPolicy:
    def __init__(self, obs_dim, action_dim, n_agents, lr=3e-4, **kw): ...

# algorithms/__init__.py
from .my_algo import MyAlgoPolicy as MY_ALGO
# and add a branch in create_policy(...):  elif algo == 'my_algo': ...
```

Then train with:

```bash
python train.py --algorithm my_algo --n_red 3 --n_blue 3 --episodes 1000 \
    --blue_difficulty combat --reward_fn base --run_name my_run --seed 42
```

## Evaluation protocol (please keep it identical)

For comparable numbers, evaluate with the bundled protocol rather than ad-hoc
rollouts:

```bash
# 1. build the SWA checkpoint (average of ep600..ep1000 checkpoints)
python swa_build_rev3.py   # edit FAMILIES for your run name

# 2. paired deterministic re-evaluation (common random numbers across algorithms)
python reeval_deterministic.py --runs my_algo --algo my_algo --suffix swa \
    --episodes 100 --out results/my_algo_swa.csv
```

The paired protocol fixes the episode initializations (`--seed_base 20000`),
so win-rate differences are not confounded by sampling luck.
