"""
sim/envs/entity_tokens.py
=========================
paper08 Phase 2: entity-set observation builder (variable team size).

The legacy fixed-length vector (marl_env._obs, 63 dims with 3 hard-coded
foe slots and 2 teammate slots) cannot represent a formation whose size
was not seen at training time -- THAT rigidity is precisely what the
paper08 architecture removes. Here every agent's local observation is a
SET of entity tokens:

    { self } ∪ { allies } ∪ { foe tracks } ∪ { inbound missiles }

padded to a fixed maximum count with a boolean mask, so the tensor shape
is constant for batching while the VALID token count varies with n and
with partial observability (foe tokens exist only for sensor tracks).

Token layout (TOKEN_DIM = 21):
  [0:4]   type one-hot: self / ally / foe / missile
  [4:7]   relative position (entity - own)  / 20000 m      (0 for self)
  [7:10]  relative velocity (entity - own)  / 300 m/s      (0 for self)
  [10]    range / 30000 m
  [11]    closing rate / 500 m/s
  [12]    entity speed / 300 m/s
  [13]    entity hp / 100            (self/ally only; else 0)
  [14]    valid/fresh flag           (fresh track, alive ally, live missile)
  [15]    sin(psi)                   (self only)
  [16]    cos(psi)                   (self only)
  [17]    phi   (roll, rad)          (self only)
  [18]    theta (pitch, rad)         (self only)
  [19]    missiles left / 4          (self only)
  [20]    specific energy / 20000 m  (self only)

Geometric attention-bias features (threat geometry per foe token, used by
the ANA-MAPPO encoder in Phase 3) are computed FROM this layout, so the
bias is a pure function of the tokens -- no hidden side channels.

Invariants this module guarantees (unit-tested in tests/test_entity_tokens.py):
  * the token MULTISET is independent of dict iteration order;
  * dead allies / expired tracks / dead ownship flip mask bits only,
    never resize the array.
"""

import numpy as np

G = 9.80665

TOKEN_DIM = 21
TYPE_SELF, TYPE_ALLY, TYPE_FOE, TYPE_MSL = 0, 1, 2, 3

# slot counts inside one agent's token array (padded)
def token_capacity(max_n: int, max_threats: int = 2) -> int:
    """1 self + (max_n-1) allies + max_n foes + max_threats missiles."""
    return 1 + (max_n - 1) + max_n + max_threats


def _kin(ac_pos, ac_vel, pos, vel):
    rel = (pos - ac_pos) / 20000.0
    rv = (vel - ac_vel) / 300.0
    rng = float(np.linalg.norm(pos - ac_pos))
    closing = -float(np.dot(pos - ac_pos, vel - ac_vel)) / max(rng, 1.0)
    return rel, rv, rng / 30000.0, closing / 500.0


def build_tokens(m, team_members, pic, threats, missiles_left,
                 max_n: int = 6, max_threats: int = 2):
    """Build the (tokens, mask) pair for one agent.

    Args:
        m:              member dict of THIS agent ('ac', 'hp', 'id', 'idx')
        team_members:   list of member dicts of THIS agent's team (any order)
        pic:            {track_idx: track} this agent's (fused) track picture
        threats:        missile objects threatening this agent's side
        missiles_left:  remaining missile count for this agent
        max_n:          padding size for ally/foe groups
        max_threats:    padding size for missile group

    Returns:
        tokens [token_capacity, TOKEN_DIM] float32, mask [capacity] float32
        (1.0 = valid token). A dead ownship yields mask all-zero.
    """
    cap = token_capacity(max_n, max_threats)
    tok = np.zeros((cap, TOKEN_DIM), dtype=np.float32)
    mask = np.zeros(cap, dtype=np.float32)
    ac = m['ac']
    if not ac.alive:
        return tok, mask

    ac_pos, ac_vel = ac.position, ac.velocity

    # --- self token (slot 0) -------------------------------------------
    alt = -ac_pos[2]
    es = alt + ac.speed ** 2 / (2 * G)
    t0 = tok[0]
    t0[TYPE_SELF] = 1.0
    t0[12] = ac.speed / 300.0
    t0[13] = m['hp'] / 100.0
    t0[14] = 1.0
    t0[15] = np.sin(ac.psi)
    t0[16] = np.cos(ac.psi)
    t0[17] = ac.phi
    t0[18] = ac.theta
    t0[19] = missiles_left / 4.0
    t0[20] = es / 20000.0
    mask[0] = 1.0

    # --- ally tokens (slots 1 .. max_n-1) -------------------------------
    allies = [x for x in team_members if x['id'] != m['id'] and x['ac'].alive]
    allies.sort(key=lambda x: x['id'])          # canonical order
    for k, x in enumerate(allies[:max_n - 1]):
        s = 1 + k
        rel, rv, rng, clo = _kin(ac_pos, ac_vel,
                                 x['ac'].position, x['ac'].velocity)
        tk = tok[s]
        tk[TYPE_ALLY] = 1.0
        tk[4:7] = rel
        tk[7:10] = rv
        tk[10] = rng
        tk[11] = clo
        tk[12] = x['ac'].speed / 300.0
        tk[13] = x['hp'] / 100.0
        tk[14] = 1.0
        mask[s] = 1.0

    # --- foe tokens (slots max_n .. 2*max_n-1), from sensor tracks ------
    base_f = max_n
    tracks = sorted(pic.items(), key=lambda kv: kv[0])   # canonical order
    for k, (tidx, trk) in enumerate(tracks[:max_n]):
        s = base_f + k
        rel, rv, rng, clo = _kin(ac_pos, ac_vel, trk['pos'],
                                 trk.get('vel', np.zeros(3)))
        tk = tok[s]
        tk[TYPE_FOE] = 1.0
        tk[4:7] = rel
        tk[7:10] = rv
        tk[10] = rng
        tk[11] = clo
        tk[12] = float(np.linalg.norm(trk.get('vel', np.zeros(3)))) / 300.0
        tk[14] = 1.0 if trk['fresh'] else 0.0
        mask[s] = 1.0

    # --- missile tokens (last max_threats slots) ------------------------
    base_m = 2 * max_n
    mine = [t for t in threats if t.target is ac]
    mine.sort(key=lambda t: float(np.linalg.norm(ac_pos - t.position)))
    for k, t in enumerate(mine[:max_threats]):
        s = base_m + k
        rel, rv, rng, clo = _kin(ac_pos, ac_vel, t.position, t.velocity)
        tk = tok[s]
        tk[TYPE_MSL] = 1.0
        tk[4:7] = rel
        tk[7:10] = rv
        tk[10] = rng
        tk[11] = clo
        tk[12] = float(np.linalg.norm(t.velocity)) / 700.0
        tk[14] = 1.0
        mask[s] = 1.0

    return tok, mask
