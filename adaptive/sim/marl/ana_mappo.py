"""
sim/marl/ana_mappo.py
=====================
paper08 Phase 3: CA-MAPPO (Cardinality-Adaptive MAPPO).

NOTE on naming: the paper refers to the method as CA-MAPPO
(Cardinality-Adaptive MAPPO); the implementation keeps the original
module/class names (ana_mappo / ANAMAPPO) for compatibility with the
released logs and checkpoints. They are the same algorithm.

Architecture
------------
Actor  : entity tokens -> linear embed -> L x Transformer encoder layers
         WITHOUT positional encoding (permutation equivariance by
         construction) with a KEY-DEPENDENT geometric attention bias
         (tokens are egocentric, so range/closing/type of the key token
         is a sufficient statistic for the bias). The shared per-agent
         head decodes the hybrid action from the SELF token (slot 0).
         Parameter count is independent of team size n; FLOPs ~ O(K^2)
         in the token count K.

Critic : same encoder (separate parameters) -> masked mean over tokens
         -> per-agent slot embedding -> learned-seed set attention over
         slots (permutation INVARIANT) -> concat count feature
         (n_alive / max_n) -> MLP -> V. One scalar V per env, shared by
         all alive agents (standard centralised-critic usage).

Training extras
---------------
  * per-scale advantage normalisation (kills the 1/sqrt(n) return-noise
    dilution measured in paper07);
  * dead-agent loss masking (episode-internal attrition);
  * optional token-dropout ("mask training", ablation switch);
  * PPO clip + GAE + Huber value loss, same stabilisers as the validated
    MAPPO baseline (sim/marl/mappo.py).

Action interface is IDENTICAL to MAPPO (cont[2], mode, fire) so the
training loop, evaluator and env need no action-side changes.
"""

import numpy as np
import torch
import torch.nn as nn

from ..envs.entity_tokens import TOKEN_DIM

GEO_FEAT_DIM = 8     # type one-hot(4) + range + closing + speed + fresh


def _geo_features(tokens):
    """Key-token geometric features [B, K, 8] from raw tokens [B, K, D]."""
    return torch.cat([
        tokens[..., 0:4],          # type one-hot
        tokens[..., 10:11],        # range / 30000
        tokens[..., 11:12],        # closing / 500
        tokens[..., 12:13],        # speed
        tokens[..., 14:15],        # fresh / alive flag
    ], dim=-1)


class AttnLayer(nn.Module):
    """Pre-norm MHA + FFN with additive key-dependent bias and key mask."""

    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(),
                                 nn.Linear(4 * d, d))

    def forward(self, h, bias, key_mask):
        """h [B,K,d]; bias [B,heads,K] (key-dependent); key_mask [B,K]."""
        B, K, d = h.shape
        x = self.norm1(h)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, K, self.h, d // self.h).transpose(1, 2)
        k = k.view(B, K, self.h, d // self.h).transpose(1, 2)
        v = v.view(B, K, self.h, d // self.h).transpose(1, 2)
        logits = q @ k.transpose(-2, -1) / np.sqrt(d // self.h)
        logits = logits + bias.unsqueeze(2)          # [B,h,1,K] broadcast
        neg = (~key_mask.bool()).view(B, 1, 1, K)
        logits = logits.masked_fill(neg, -1e9)
        attn = torch.softmax(logits, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)       # all-masked rows -> 0
        h = h + self.out((attn @ v).transpose(1, 2).reshape(B, K, d))
        h = h + self.ffn(self.norm2(h))
        return h


class EntityEncoder(nn.Module):
    def __init__(self, d=128, layers=3, heads=4, geo_bias=True):
        super().__init__()
        self.inp = nn.Linear(TOKEN_DIM, d)
        self.geo = (nn.Sequential(nn.Linear(GEO_FEAT_DIM, 32), nn.Tanh(),
                                  nn.Linear(32, heads))
                    if geo_bias else None)
        self.layers = nn.ModuleList(
            [AttnLayer(d, heads) for _ in range(layers)])

    def forward(self, tokens, mask):
        """tokens [B,K,D], mask [B,K] (1=valid) -> h [B,K,d]."""
        B, K, _ = tokens.shape
        if self.geo is not None:
            bias = self.geo(_geo_features(tokens)).permute(0, 2, 1)
        else:
            bias = torch.zeros(B, self.layers[0].h, K, device=tokens.device)
        h = self.inp(tokens)
        for lyr in self.layers:
            h = lyr(h, bias, mask)
        return h


class ANAActor(nn.Module):
    """Shared per-agent hybrid-action head over the self-token (slot 0)."""

    def __init__(self, d=128, layers=3, heads=4, geo_bias=True):
        super().__init__()
        self.enc = EntityEncoder(d, layers, heads, geo_bias)
        self.mu = nn.Linear(d, 2)
        self.log_std = nn.Parameter(-0.5 * torch.ones(2))
        self.mode = nn.Linear(d, 3)
        self.fire = nn.Linear(d, 2)

    def forward(self, tokens, mask):
        h = self.enc(tokens, mask)
        h0 = h[:, 0]                                  # self token
        return self.mu(h0), self.log_std, self.mode(h0), self.fire(h0)


class UPDeTActor(nn.Module):
    """UPDeT-faithful actor: parameter-shared transformer over PER-AGENT
    observation vectors (each agent's token set flattened to one fixed-size
    vector, zero-padded), no positional encoding, no geometric bias.
    Shared per-agent heads decode from each agent's own embedding, so the
    parameter count is likewise independent of team size.  This is the
    closest executable realisation of the native UPDeT input interface
    (per-agent feature vectors) in this environment; only the action head
    is adapted (hybrid continuous instead of SMAC discrete)."""

    def __init__(self, d=128, layers=3, heads=4, token_cap=14):
        super().__init__()
        self.inp = nn.Linear(TOKEN_DIM * token_cap, d)
        self.layers = nn.ModuleList(
            [AttnLayer(d, heads) for _ in range(layers)])
        self.heads_n = heads
        self.mu = nn.Linear(d, 2)
        self.log_std = nn.Parameter(-0.5 * torch.ones(2))
        self.mode = nn.Linear(d, 3)
        self.fire = nn.Linear(d, 2)

    def forward(self, flat, agent_mask):
        """flat [B,S,K*D], agent_mask [B,S] (1=alive) -> per-slot dists."""
        B, S, _ = flat.shape
        h = self.inp(flat)
        bias = torch.zeros(B, self.heads_n, S, device=flat.device)
        for lyr in self.layers:
            h = lyr(h, bias, agent_mask)
        return self.mu(h), self.log_std, self.mode(h), self.fire(h)


class SetCritic(nn.Module):
    """Permutation-invariant centralised critic over all agents' tokens."""

    def __init__(self, d=128, layers=2, heads=4, set_attn=True):
        super().__init__()
        self.enc = EntityEncoder(d, layers, heads)
        self.set_attn = set_attn
        if set_attn:
            self.seed = nn.Parameter(torch.randn(1, 1, d) * 0.02)
            self.slot_attn = nn.MultiheadAttention(d, heads,
                                                   batch_first=True)
        self.head = nn.Sequential(nn.Linear(d + 1, 256), nn.Tanh(),
                                  nn.Linear(256, 256), nn.Tanh(),
                                  nn.Linear(256, 1))

    def forward(self, tokens, mask):
        """tokens [B,S,K,D], mask [B,S,K] -> V [B]."""
        B, S, K, D = tokens.shape
        h = self.enc(tokens.reshape(B * S, K, D),
                     mask.reshape(B * S, K))          # [B*S,K,d]
        m = mask.reshape(B * S, K, 1)
        slot = (h * m).sum(1) / m.sum(1).clamp(min=1.0)   # [B*S,d]
        slot = slot.reshape(B, S, -1)
        slot_valid = (mask.sum(-1) > 0)               # [B,S]
        if self.set_attn:
            q = self.seed.expand(B, -1, -1)
            pooled, _ = self.slot_attn(
                q, slot, slot,
                key_padding_mask=~slot_valid.bool())
            pooled = torch.nan_to_num(pooled.squeeze(1), nan=0.0)
        else:
            sv = slot_valid.float().unsqueeze(-1)
            pooled = (slot * sv).sum(1) / sv.sum(1).clamp(min=1.0)
        count = slot_valid.float().sum(1, keepdim=True) / S
        return self.head(torch.cat([pooled, count], -1)).squeeze(-1)


class ANAMAPPO:
    name = 'ana_mappo'

    def __init__(self, max_n=6, token_cap=14, lr=3e-4, lr_v=1e-3,
                 gamma=0.995, gae_lambda=0.95, clip=0.2, ent_coef=0.005,
                 epochs=4, minibatch_size=512, max_grad_norm=0.5,
                 device=None, seed=0, geo_bias=True, set_critic=True,
                 mask_train_p=0.0, d_model=128, per_scale_norm=True,
                 updet=False):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.gamma, self.lam, self.clip = gamma, gae_lambda, clip
        self.ent_coef, self.epochs = ent_coef, epochs
        self.mb, self.mgn = minibatch_size, max_grad_norm
        self.max_n, self.cap = max_n, token_cap
        self.mask_train_p = mask_train_p
        self.per_scale_norm = per_scale_norm
        self.updet = updet
        self.device = device or ('cuda' if torch.cuda.is_available()
                                 else 'cpu')
        if updet:
            self.actor = UPDeTActor(d_model,
                                    token_cap=self.cap).to(self.device)
        else:
            self.actor = ANAActor(d_model, geo_bias=geo_bias).to(self.device)
        self.critic = SetCritic(d_model,
                                set_attn=set_critic).to(self.device)
        self.opt_pi = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=lr_v)
        self.rollout = []

    # ------------------------------------------------------------------
    def _dropout_tokens(self, tok, msk):
        """Mask-training: randomly drop valid non-self tokens (training
        only) so the policy is robust to missing tracks / attrition."""
        if self.mask_train_p <= 0:
            return msk
        drop = (np.random.rand(*msk.shape) < self.mask_train_p)
        drop[..., 0] = False                          # never drop self
        return msk * (~drop).astype(np.float32)

    def act(self, tok, msk, deterministic=False, training=False):
        """tok [E,S,K,D], msk [E,S,K] -> actions [E,S] tuples + info."""
        if training:
            msk = self._dropout_tokens(tok, msk)
        E, S, K, D = tok.shape
        t_t = torch.as_tensor(tok, dtype=torch.float32,
                              device=self.device).reshape(E * S, K, D)
        m_t = torch.as_tensor(msk, dtype=torch.float32,
                              device=self.device).reshape(E * S, K)
        with torch.no_grad():
            if self.updet:
                flat = torch.as_tensor(tok, dtype=torch.float32,
                                       device=self.device).reshape(
                                           E, S, K * D)
                amask = (torch.as_tensor(msk, dtype=torch.float32,
                                         device=self.device).sum(-1) > 0
                         ).float()
                mu, log_std, ml, fl = self.actor(flat, amask)
                mu = mu.reshape(E * S, 2)
                ml = ml.reshape(E * S, 3)
                fl = fl.reshape(E * S, 2)
            else:
                mu, log_std, ml, fl = self.actor(t_t, m_t)
            if deterministic:
                cont = torch.tanh(mu)
                mode = torch.argmax(ml, -1)
                fire = torch.argmax(fl, -1)
                logp = torch.zeros(E * S, device=self.device)
            else:
                std = log_std.exp()
                raw = mu + std * torch.randn_like(mu)
                cont = torch.tanh(raw)
                logp = torch.distributions.Normal(mu, std).log_prob(
                    raw).sum(-1)
                logp = logp - torch.log(1 - cont.pow(2) + 1e-6).sum(-1)
                md = torch.distributions.Categorical(logits=ml)
                fd = torch.distributions.Categorical(logits=fl)
                mode, fire = md.sample(), fd.sample()
                logp = logp + md.log_prob(mode) + fd.log_prob(fire)
            v = self.critic(torch.as_tensor(tok, dtype=torch.float32,
                                            device=self.device),
                            torch.as_tensor(msk, dtype=torch.float32,
                                            device=self.device))
        c = cont.cpu().numpy().reshape(E, S, 2)
        mo = mode.cpu().numpy().reshape(E, S)
        fi = fire.cpu().numpy().reshape(E, S)
        acts = [[(c[e, s], int(mo[e, s]), int(fi[e, s])) for s in range(S)]
                for e in range(E)]
        info = dict(logp=logp.cpu().numpy().reshape(E, S),
                    value=v.cpu().numpy())
        return acts, info

    # ------------------------------------------------------------------
    def store(self, tok, msk, alive, scale, acts, info, rew, done):
        self.rollout.append(dict(
            tok=np.asarray(tok), msk=np.asarray(msk),
            alive=np.asarray(alive, dtype=np.float32),
            scale=np.asarray(scale, dtype=np.int64),
            cont=np.asarray([[a[0] for a in row] for row in acts],
                            dtype=np.float32),
            mode=np.asarray([[a[1] for a in row] for row in acts],
                            dtype=np.int64),
            fire=np.asarray([[a[2] for a in row] for row in acts],
                            dtype=np.int64),
            logp=np.asarray(info['logp'], dtype=np.float32),
            value=np.asarray(info['value'], dtype=np.float32),
            rew=np.asarray(rew, dtype=np.float32),
            done=np.asarray(done, dtype=np.float32),
        ))

    # ------------------------------------------------------------------
    def update(self, last_value):
        if not self.rollout:
            return {}
        T = len(self.rollout)
        E, S = self.rollout[0]['rew'].shape
        val = np.stack([tr['value'] for tr in self.rollout])      # [T,E]
        done = np.stack([tr['done'] for tr in self.rollout])

        adv = np.zeros((T, E, S), dtype=np.float32)
        lastgae = np.zeros((E, S), dtype=np.float32)
        for t in reversed(range(T)):
            nextval = last_value if t == T - 1 else val[t + 1]
            nonterm = (1.0 - done[t])[:, None]
            delta = (self.rollout[t]['rew']
                     + self.gamma * nextval[:, None] * nonterm
                     - val[t][:, None])
            lastgae = delta + self.gamma * self.lam * nonterm * lastgae
            adv[t] = lastgae
        ret = adv + val[:, :, None]

        # per-scale advantage normalisation ------------------------------
        # (ablatable via per_scale_norm=False: advantages are then
        # standardised globally over the mixed batch)
        scale = np.stack([tr['scale'] for tr in self.rollout])    # [T,E]
        if self.per_scale_norm:
            for n in np.unique(scale):
                idx = np.where(scale == n)
                sel = adv[idx[0], idx[1], :]
                adv[idx[0], idx[1], :] = (sel - sel.mean()) / (sel.std() + 1e-8)
        else:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # per-env return target: mean over alive slots (V is env-level)
        alive3 = np.stack([tr['alive'] for tr in self.rollout])   # [T,E,S]
        ret_env = (ret * alive3).sum(-1) / np.clip(
            alive3.sum(-1), 1, None)                              # [T,E]
        ret_env_b = ret_env.reshape(-1)

        # flatten to per-slot samples [T*E*S, ...] (slot-major: the flat
        # index // S recovers the flat env index t*E + e)
        tok_all = np.stack([tr['tok'] for tr in self.rollout])  # [T,E,S,K,D]
        msk_all = np.stack([tr['msk'] for tr in self.rollout])  # [T,E,S,K]
        K_, D_ = tok_all.shape[-2:]
        tok_b = tok_all.reshape(-1, K_, D_)
        msk_b = msk_all.reshape(-1, K_)
        alive_b = np.stack([tr['alive'] for tr in self.rollout]).reshape(-1)
        cont_b = np.stack([tr['cont'] for tr in self.rollout]).reshape(-1, 2)
        mode_b = np.stack([tr['mode'] for tr in self.rollout]).reshape(-1)
        fire_b = np.stack([tr['fire'] for tr in self.rollout]).reshape(-1)
        logp_b = np.stack([tr['logp'] for tr in self.rollout]).reshape(-1)
        adv_b = adv.reshape(-1)
        tok_env = tok_all                                        # [T,E,S,K,D]
        msk_env = msk_all
        N = tok_b.shape[0]

        stats = dict(policy_loss=0.0, value_loss=0.0, entropy=0.0, n=0)
        idx_all = np.arange(N)
        for _ in range(self.epochs):
            np.random.shuffle(idx_all)
            for s in range(0, N, self.mb):
                mb = idx_all[s:s + self.mb]
                keep = alive_b[mb] > 0.5
                if keep.sum() < 2:
                    continue
                mb = mb[keep]
                o = torch.as_tensor(tok_b[mb], dtype=torch.float32,
                                    device=self.device)
                mk = torch.as_tensor(msk_b[mb], dtype=torch.float32,
                                     device=self.device)
                c = torch.as_tensor(cont_b[mb], dtype=torch.float32,
                                    device=self.device)
                mo = torch.as_tensor(mode_b[mb], dtype=torch.long,
                                     device=self.device)
                fi = torch.as_tensor(fire_b[mb], dtype=torch.long,
                                     device=self.device)
                old_lp = torch.as_tensor(logp_b[mb], dtype=torch.float32,
                                         device=self.device)
                ad = torch.as_tensor(adv_b[mb], dtype=torch.float32,
                                     device=self.device)
                ad = torch.nan_to_num(ad, nan=0.0, posinf=0.0, neginf=0.0)

                if self.updet:
                    # transformer attends across agents, so forward whole
                    # env rows, then gather the sampled per-slot rows
                    env_rows_a = np.unique(mb // S)
                    pos = np.searchsorted(env_rows_a, mb // S)
                    slot = mb % S
                    te4 = torch.as_tensor(
                        tok_env.reshape(-1, *tok_env.shape[2:])[env_rows_a],
                        dtype=torch.float32, device=self.device)
                    me4 = torch.as_tensor(
                        msk_env.reshape(-1, *msk_env.shape[2:])[env_rows_a],
                        dtype=torch.float32, device=self.device)
                    Bp = te4.shape[0]
                    flat = te4.reshape(Bp, S, K_ * D_)
                    amask = (me4.sum(-1) > 0).float()
                    mu_a, log_std, ml_a, fl_a = self.actor(flat, amask)
                    pos_t = torch.as_tensor(pos, dtype=torch.long,
                                            device=self.device)
                    slot_t = torch.as_tensor(slot, dtype=torch.long,
                                             device=self.device)
                    mu = mu_a[pos_t, slot_t]
                    ml = ml_a[pos_t, slot_t]
                    fl = fl_a[pos_t, slot_t]
                else:
                    mu, log_std, ml, fl = self.actor(o, mk)
                std = log_std.exp()
                if not (torch.isfinite(mu).all()
                        and torch.isfinite(std).all()):
                    continue
                raw = torch.atanh(c.clamp(-0.999999, 0.999999))
                lp = torch.distributions.Normal(mu, std).log_prob(
                    raw).sum(-1)
                lp = lp - torch.log(1 - c.pow(2) + 1e-6).sum(-1)
                md = torch.distributions.Categorical(logits=ml)
                fd = torch.distributions.Categorical(logits=fl)
                lp = lp + md.log_prob(mo) + fd.log_prob(fi)
                ent = (md.entropy() + fd.entropy()).mean()
                ratio = torch.exp((lp - old_lp).clamp(-20.0, 20.0))
                s1 = ratio * ad
                s2 = torch.clamp(ratio, 1 - self.clip,
                                 1 + self.clip) * ad
                pi_loss = -torch.min(s1, s2).mean() - self.ent_coef * ent
                if not torch.isfinite(pi_loss):
                    continue
                self.opt_pi.zero_grad()
                pi_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.mgn)
                self.opt_pi.step()

                # critic minibatch at env level (row index = t*E + e)
                env_rows = np.unique(mb // S)
                te = torch.as_tensor(
                    tok_env.reshape(-1, *tok_env.shape[2:])[env_rows],
                    dtype=torch.float32, device=self.device)
                me = torch.as_tensor(
                    msk_env.reshape(-1, *msk_env.shape[2:])[env_rows],
                    dtype=torch.float32, device=self.device)
                r = torch.as_tensor(ret_env_b[env_rows],
                                    dtype=torch.float32, device=self.device)
                r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
                v = self.critic(te, me)
                v_loss = nn.functional.smooth_l1_loss(v, r)
                if torch.isfinite(v_loss):
                    self.opt_v.zero_grad()
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(),
                                             self.mgn)
                    self.opt_v.step()

                stats['policy_loss'] += float(pi_loss)
                stats['value_loss'] += float(v_loss)
                stats['entropy'] += float(ent)
                stats['n'] += 1

        self.rollout = []
        n = max(stats.pop('n'), 1)
        return {k: v / n for k, v in stats.items()}

    # ------------------------------------------------------------------
    def save(self, path):
        torch.save(self.actor.state_dict(), path + '.actor.pt')
        torch.save(self.critic.state_dict(), path + '.critic.pt')

    def load(self, path, strict=True):
        for mod, suffix in ((self.actor, '.actor.pt'),
                            (self.critic, '.critic.pt')):
            sd = torch.load(path + suffix, map_location=self.device)
            if strict:
                mod.load_state_dict(sd)
            else:
                res = mod.load_state_dict(sd, strict=False)
                print(f'[load strict=False] {suffix[1:]} '
                      f'missing={list(res.missing_keys)} '
                      f'unexpected={list(res.unexpected_keys)}')
