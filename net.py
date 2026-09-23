"""
MAPPO actor-critic architecture for cooperative ghost pursuit.

GhostActor:  FiLM-modulated CNN trunk feeding two action heads --
             a pointer head that self-attends over the live heuristic candidate set and scores each
             candidate directly, and a spatial head that nominates cells no allocator rule proposed.
GhostCritic: CNN over the omniscient global state + MLP over the joint ghost vectors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from obs import SPATIAL_CH, MAX_H, MAX_W, VEC_DIM, CRITIC_VEC_DIM, GLOBAL_SPATIAL_CH, MAX_CANDIDATES, CAND_FEAT_DIM

SPEED_FLOOR   = float(__import__("os").environ.get("SPEED_FLOOR", "0.75"))
SPEED_PRIOR_A = 8.0
SPEED_PRIOR_B = -6.0
GATE_PRIOR_LOGIT = -3.0
RL_MAX_DEVIATION = 1.05
CAND_DIM   = 128
CAND_HEADS = 4
CAND_LAYERS = 2

def speed_to_mult(throttle):            #throttle in [0,1] -> speed multiplier in [SPEED_FLOOR, 1.0]
    return SPEED_FLOOR + (1.0 - SPEED_FLOOR) * float(throttle)

def mult_to_throttle(mult):             #inverse of speed_to_mult, for behaviour-cloning targets
    return (float(mult) - SPEED_FLOOR) / (1.0 - SPEED_FLOOR)

class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, cond_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.bn1   = nn.GroupNorm(8, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.bn2   = nn.GroupNorm(8, c_out)
        self.skip  = (nn.Sequential(nn.Conv2d(c_in, c_out, 1), nn.GroupNorm(8, c_out)) if c_in != c_out else nn.Identity())
        self.film = FiLM(cond_dim, c_out) if cond_dim else None

    def forward(self, x, cond=None):
        r = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)))
        if self.film is not None and cond is not None:
            x = self.film(x, cond)
        x = self.bn2(self.conv2(x))
        return F.relu(x + r)

class FiLM(nn.Module):
    """Feature-wise Linear Modulation: gamma * feat + β."""
    def __init__(self, cond_dim, n_channels):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, n_channels)
        self.beta  = nn.Linear(cond_dim, n_channels)

    def forward(self, spatial, cond):
        #spatial: (B, C, H, W)   cond: (B, cond_dim)
        g = self.gamma(cond).unsqueeze(-1).unsqueeze(-1)   #(B, C, 1, 1)
        b = self.beta(cond).unsqueeze(-1).unsqueeze(-1)
        #clamp gamma to [-10, 10] to prevent FiLM from causing activation explosion
        g = g.clamp(-10.0, 10.0)
        return g * spatial + b

def _sample_k(flat, base_invalid, K: int):
    """K sequential categorical picks without replacement over a masked logit row.

    Once every remaining option is -inf the row falls back to uniform over the originally-valid options,
    which is what keeps a ghost with fewer live options than K from producing a NaN distribution.
    """
    flat = torch.nan_to_num(flat, nan=float('-inf')).clone()
    idxs, lps = [], []
    for _ in range(K):
        all_inf = (torch.isinf(flat) & (flat < 0)).all(dim=1, keepdim=True)
        fallback = torch.where(base_invalid, torch.full_like(flat, float('-inf')), torch.zeros_like(flat))
        flat = torch.where(all_inf, fallback, flat)
        dist = torch.distributions.Categorical(logits=flat)
        i = dist.sample()
        idxs.append(i)
        lps.append(dist.log_prob(i))
        flat = flat.scatter(1, i.unsqueeze(1), float('-inf'))
    return torch.stack(idxs, 1), torch.stack(lps, 1)

def _eval_k(flat, base_invalid, actions):
    """Re-score stored picks under the same sequential-without-replacement scheme as _sample_k."""
    flat = torch.nan_to_num(flat, nan=float('-inf')).clone()
    lps, ents = [], []
    for k in range(actions.shape[1]):
        all_inf = (torch.isinf(flat) & (flat < 0)).all(dim=1, keepdim=True)
        fallback = torch.where(base_invalid, torch.full_like(flat, float('-inf')), torch.zeros_like(flat))
        flat = torch.where(all_inf, fallback, flat)
        dist = torch.distributions.Categorical(logits=flat)
        lps.append(dist.log_prob(actions[:, k]))
        ents.append(dist.entropy())
        hit = torch.zeros_like(flat, dtype=torch.bool).scatter(1, actions[:, k].unsqueeze(1), True)
        flat = torch.where(hit, float('-inf'), flat)
    return torch.stack(lps, 1), torch.stack(ents, 1)

def _confidence_above_uniform(logits, valid):
    """Map a masked logit row to [0,1] CBBA scores: 0 = no better than guessing, 1 = all mass on one option."""
    p = torch.softmax(torch.nan_to_num(logits, nan=float('-inf')), dim=1)
    n_valid = valid.sum(dim=1, keepdim=True).clamp(min=2).to(p.dtype)
    out = (torch.log(p * n_valid + 1e-12) / torch.log(n_valid)).clamp(0.0, 1.0)
    return torch.nan_to_num(out, nan=0.0)

class CandidateBlock(nn.Module):
    """Pre-norm self-attention over the candidate set: lets a ghost score a task in the context of every
    other task on offer and of what its peers have already claimed, rather than one cell at a time."""
    def __init__(self, d=CAND_DIM, heads=CAND_HEADS):
        super().__init__()
        self.ln1  = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2  = nn.LayerNorm(d)
        self.ff   = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x, key_padding_mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + torch.nan_to_num(a, nan=0.0)
        return x + self.ff(self.ln2(x))

class GhostActor(nn.Module):
    def __init__(self, vec_dim: int = VEC_DIM):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(SPATIAL_CH, 64, 7, padding=3), nn.GroupNorm(8, 64), nn.ReLU())
        self.vec_mlp = nn.Sequential(nn.Linear(vec_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU())
        self.res1 = ResBlock(64, 128, cond_dim=128)
        self.res2 = ResBlock(128, 128, cond_dim=128)
        self.res3 = ResBlock(128, 128, cond_dim=128)
        #1×1 conv to logit map (combines 128 local spatial channels + 128 global context channels)
        self.head = nn.Conv2d(256, 1, 1)
        #pointer head: each candidate is (its own features, the trunk features at its target cell, global context)
        self.cand_enc = nn.Sequential(nn.Linear(CAND_FEAT_DIM + 128 + 256, CAND_DIM), nn.LayerNorm(CAND_DIM), nn.GELU(),
                                      nn.Linear(CAND_DIM, CAND_DIM), nn.LayerNorm(CAND_DIM), nn.GELU())
        self.cand_blocks = nn.ModuleList([CandidateBlock() for _ in range(CAND_LAYERS)])
        self.cand_head = nn.Linear(CAND_DIM, 1)
        nn.init.zeros_(self.cand_head.weight)
        nn.init.zeros_(self.cand_head.bias)
        self.speed_head = nn.Sequential(nn.Linear(256, 64), nn.LayerNorm(64), nn.ReLU(), nn.Linear(64, 2))
        self.dir_head = nn.Sequential(nn.Linear(256, 64), nn.LayerNorm(64), nn.ReLU(), nn.Linear(64, 2))
        self.gate_head = nn.Sequential(nn.Linear(256, 64), nn.LayerNorm(64), nn.ReLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.speed_head[-1].weight)
        self.speed_head[-1].bias.data = torch.tensor([SPEED_PRIOR_A, SPEED_PRIOR_B])
        nn.init.zeros_(self.dir_head[-1].weight)
        nn.init.zeros_(self.dir_head[-1].bias)
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, GATE_PRIOR_LOGIT)

    def encode(self, spatial, vector):
        x = self.stem(spatial)
        vec = self.vec_mlp(vector)
        x = self.res1(x, cond=vec)
        x = self.res2(x, cond=vec)
        x = self.res3(x, cond=vec)
        pool = F.adaptive_avg_pool2d(x, 1).flatten(1)   #(B, 128)
        return x, pool, vec

    def logits_from_features(self, feats, pool, mask):
        #feats: (B, 128, H, W), pool: (B, 128), mask: (B, H, W) bool
        H, W = feats.shape[2], feats.shape[3]
        pool_expanded = pool.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        combined = torch.cat([feats, pool_expanded], dim=1)  #(B, 256, H, W)
        logits = self.head(combined).squeeze(1)              #(B, H, W)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        m = mask.to(logits.dtype)
        n_valid = m.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)
        logits = logits - (logits * m).sum(dim=(1, 2), keepdim=True) / n_valid
        logits = logits.masked_fill(~mask, float('-inf'))
        return logits

    def candidate_logits(self, feats, pool, vec, cand_feat, cand_cell, cand_mask):
        """
        cand_feat : (B, M, CAND_FEAT_DIM)   tabular features of each live candidate
        cand_cell : (B, M) long             flattened r*W+c of each candidate's target, for feature gathering
        cand_mask : (B, M) bool             True where a real candidate sits

        Returns (B, M) logits masked to the live set, and the padding-safe mask actually used.
        """
        B, C, H, W = feats.shape
        M = cand_feat.shape[1]
        flat = feats.reshape(B, C, H * W)
        idx = cand_cell.clamp(0, H * W - 1).unsqueeze(1).expand(B, C, M)
        local = flat.gather(2, idx).permute(0, 2, 1)                      #(B, M, 128)
        ctx = torch.cat([pool, vec], dim=1).unsqueeze(1).expand(-1, M, -1)  #(B, M, 256)
        x = self.cand_enc(torch.cat([cand_feat.to(local.dtype), local, ctx], dim=-1))
        #a ghost with no candidates at all would make every key padded, and MHA returns NaN for such a row
        safe = cand_mask | (~cand_mask.any(dim=1, keepdim=True) & (torch.arange(M, device=cand_mask.device) == 0))
        for blk in self.cand_blocks:
            x = blk(x, key_padding_mask=~safe)
        logits = torch.nan_to_num(self.cand_head(x).squeeze(-1), nan=0.0, posinf=0.0, neginf=0.0)
        #centre over the live set before masking, the same anchoring the spatial head needs
        m = safe.to(logits.dtype)
        n_valid = m.sum(dim=1, keepdim=True).clamp(min=1.0)
        logits = logits - (logits * m).sum(dim=1, keepdim=True) / n_valid
        return logits.masked_fill(~safe, float('-inf')), safe

    def forward(self, spatial, vector, mask, cand_feat, cand_cell, cand_mask, K_cand=3, K_novel=1):
        """
        Returns
        -------
        cand_idx    : (B, K_cand)  long  — indices into the ghost's own candidate list
        cand_lp     : (B, K_cand)
        cand_scores : (B, M)             — confidence above uniform in [0,1], the CBBA nomination score
        novel_idx   : (B, K_novel) long  — flattened cell indices for off-menu waypoints
        novel_lp    : (B, K_novel)
        novel_scores: (B, H, W)
        pool, vec   : (B, 128) each      — critic tokens
        speed / direction / gate and their log-probs, each (B, 1)
        cand_logits : (B, M)             — masked candidate logits (15th element), for the COMA baseline
        """
        feats, pool, vec = self.encode(spatial, vector)
        c_logits, safe = self.candidate_logits(feats, pool, vec, cand_feat, cand_cell, cand_mask)
        cand_scores = _confidence_above_uniform(c_logits, safe)
        cand_idx, cand_lp = _sample_k(c_logits, ~safe, K_cand)
        s_logits = self.logits_from_features(feats, pool, mask)
        B0 = s_logits.shape[0]
        flat_s = s_logits.view(B0, -1)
        novel_scores = _confidence_above_uniform(flat_s, torch.isfinite(flat_s)).view_as(s_logits)
        novel_idx, novel_lp = _sample_k(flat_s, ~mask.reshape(B0, -1), K_novel)
        tok = torch.cat([pool, vec], dim=1)
        dist_speed = self._speed_dist(tok)
        speed = torch.clamp(dist_speed.sample(), 1e-3, 1.0 - 1e-3)
        speed_lp = dist_speed.log_prob(speed)
        dist_dir = self._dir_dist(tok)
        direction = torch.clamp(dist_dir.sample(), 1e-3, 1.0 - 1e-3)
        dir_lp = dist_dir.log_prob(direction)
        dist_gate = self._gate_dist(tok)
        gate = dist_gate.sample()
        gate_lp = dist_gate.log_prob(gate)
        return (cand_idx, cand_lp, cand_scores, novel_idx, novel_lp, novel_scores, pool, vec,
                speed.unsqueeze(1), speed_lp.unsqueeze(1), direction.unsqueeze(1), dir_lp.unsqueeze(1),
                gate.unsqueeze(1), gate_lp.unsqueeze(1), c_logits)

    def _speed_dist(self, tok):
        params = torch.clamp(F.softplus(self.speed_head(tok)) + 1.5, min=1.5, max=12.0)
        return torch.distributions.Beta(params[:, 0], params[:, 1])

    def _dir_dist(self, tok):
        params = torch.clamp(F.softplus(self.dir_head(tok)) + 2.0, min=2.0, max=12.0)
        return torch.distributions.Beta(params[:, 0], params[:, 1])

    def _gate_dist(self, tok):
        return torch.distributions.Bernoulli(logits=self.gate_head(tok).squeeze(-1).clamp(-8.0, 8.0))

    def evaluate_actions(self, spatial, vector, mask, cand_feat, cand_cell, cand_mask,
                         cand_actions, novel_actions, speeds, directions=None, gates=None):
        """
        Re-computes log-probs and entropy for *stored* actions. One forward pass per PPO micro-batch.

        Returns
        -------
        logprobs     : (B, K_cand + K_novel + 3) — per-head, so PPO can clip and measure KL head by head
        entropy      : (B,)  — every head summed
        cand_ent     : (B,)  — candidate picks only, the exploration-bonus target
        cand_ent_norm: (B,)  — cand_ent normalised by its own maximum; stage-invariant, so one entropy
                               target holds whether a ghost is choosing among 4 candidates or 19
        novel_ent    : (B,)  — off-menu head only, kept alive by its own small entropy bonus
        pool, vec    : critic tokens
        cand_logits  : (B, M)    — candidate BC target lives on this
        flat_logits  : (B, H*W)  — spatial BC target lives on this
        speed_params : (B, 2)
        """
        feats, pool, vec = self.encode(spatial, vector)
        c_logits, safe = self.candidate_logits(feats, pool, vec, cand_feat, cand_cell, cand_mask)
        cand_lp, cand_ents = _eval_k(c_logits, ~safe, cand_actions)
        s_logits = self.logits_from_features(feats, pool, mask)
        B = spatial.shape[0]
        base_invalid = ~mask.reshape(B, -1)
        flat_clean = torch.nan_to_num(s_logits.view(B, -1), nan=float('-inf'))
        all_inf_clean = (torch.isinf(flat_clean) & (flat_clean < 0)).all(dim=1, keepdim=True)
        fallback_clean = torch.where(base_invalid, torch.full_like(flat_clean, float('-inf')), torch.zeros_like(flat_clean))
        flat_clean = torch.where(all_inf_clean, fallback_clean, flat_clean)
        novel_lp, novel_ents = _eval_k(flat_clean, base_invalid, novel_actions)
        tok = torch.cat([pool, vec], dim=1)
        dist_speed = self._speed_dist(tok)
        speeds = torch.clamp(speeds.squeeze(-1), 1e-3, 1.0 - 1e-3)
        speed_lp = dist_speed.log_prob(speeds)
        speed_ent = dist_speed.entropy()
        if directions is not None:
            dist_dir = self._dir_dist(tok)
            directions = torch.clamp(directions.squeeze(-1), 1e-3, 1.0 - 1e-3)
            dir_lp = dist_dir.log_prob(directions)
            dir_ent = dist_dir.entropy()
        else:
            dir_lp = torch.zeros_like(speed_lp)
            dir_ent = torch.zeros_like(speed_ent)
        if gates is not None:
            dist_gate = self._gate_dist(tok)
            gate_lp = dist_gate.log_prob(gates.squeeze(-1))
            gate_ent = dist_gate.entropy()
        else:
            gate_lp = torch.zeros_like(speed_lp)
            gate_ent = torch.zeros_like(speed_ent)
        logprobs = torch.cat([cand_lp, novel_lp, speed_lp.unsqueeze(1), dir_lp.unsqueeze(1), gate_lp.unsqueeze(1)], dim=1)
        cand_ent = cand_ents.sum(1)
        novel_ent = novel_ents.sum(1)
        entropy = cand_ent + novel_ent + speed_ent + dir_ent + gate_ent
        n_live = safe.sum(dim=1).to(cand_ent.dtype)
        max_ent = sum(torch.log((n_live - k).clamp(min=1.0)) for k in range(cand_actions.shape[1]))
        cand_ent_norm = cand_ent / max_ent.clamp(min=1e-3)
        speed_params = torch.stack([dist_speed.concentration1, dist_speed.concentration0], dim=1)
        return logprobs, entropy, cand_ent, cand_ent_norm, novel_ent, pool, vec, c_logits, flat_clean, speed_params

CAND_Q_DIM = 32

class GhostCritic(nn.Module):
    """Centralised action-value critic Q_i(s, c): joint state (omniscient map + every ghost's vector) plus
    the features of the candidate ghost i chose. With a candidate set of at most MAX_CANDIDATES this makes
    the COMA counterfactual baseline  b_i = sum_c pi_i(c|s) Q_i(s,c)  cheap: the spatial stem and the vector
    MLP run once, only the small head repeats per candidate.

    Runs 13-17 used V(s) of the joint state, so a ghost's advantage was 'team outcome minus team value':
    noise with respect to which candidate that ghost picked. 779 stage-3 updates moved the pointer head
    27.7 L2 without changing its sharpness or the kill rate — a random walk. Q(s,c) - b(s) isolates the
    marginal value of the pick itself."""
    def __init__(self, vec_dim: int = CRITIC_VEC_DIM):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(GLOBAL_SPATIAL_CH, 64, 7, padding=3), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.ReLU())
        self.vec_mlp = nn.Sequential(
            nn.Linear(vec_dim, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU())
        self.cand_mlp = nn.Sequential(
            nn.Linear(CAND_FEAT_DIM, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, CAND_Q_DIM), nn.LayerNorm(CAND_Q_DIM), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(128 + 128 + CAND_Q_DIM, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 1))

    def encode_spatial(self, spatial):
        x = self.stem(spatial)
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    def forward_from_pool(self, pool, vector, cand_feat=None):
        """Q for ONE candidate per row. cand_feat (B, CAND_FEAT_DIM); None gives the no-candidate value."""
        vec = self.vec_mlp(vector)
        if cand_feat is None:
            cand_feat = torch.zeros(pool.shape[0], CAND_FEAT_DIM, device=pool.device, dtype=pool.dtype)
        c = self.cand_mlp(cand_feat.to(pool.dtype))
        return self.head(torch.cat([pool, vec, c], dim=-1))

    def q_all(self, pool, vector, cand_feat, cand_mask):
        """Q for EVERY candidate. pool (B,128), vector (B,V), cand_feat (B,M,F), cand_mask (B,M) bool.
        Returns (B, M) with -inf-free zeros on padded slots (mask them yourself)."""
        B, M, _ = cand_feat.shape
        vec = self.vec_mlp(vector)
        c = self.cand_mlp(cand_feat.reshape(B * M, -1).to(pool.dtype)).reshape(B, M, -1)
        sv = torch.cat([pool, vec], dim=-1).unsqueeze(1).expand(B, M, -1)
        q = self.head(torch.cat([sv, c], dim=-1)).squeeze(-1)
        return torch.nan_to_num(q, nan=0.0) * cand_mask.to(q.dtype)

    def forward(self, spatial, vector, cand_feat=None):
        pool = self.encode_spatial(spatial)
        return self.forward_from_pool(pool, vector, cand_feat)

def gate_for_eval(critic, gsp_padded, cve, t_cf, t_cm, c_logits, pick0, margin: float = float(__import__("os").environ.get("GATE_MARGIN", "0.02")), cbc=None):
    """Execute-gate for the evaluation paths (probe, test.py, live game), sharing the trainer's rule.

    gsp_padded : (GLOBAL_SPATIAL_CH, H, W) numpy   the omniscient map for this env-step
    cve        : (N, CRITIC_VEC_DIM) numpy         from obs.build_cve
    t_cf, t_cm : (N, M, F) / (N, M) tensors        candidate features and live-mask
    c_logits   : (N, M) tensor                     the actor's masked candidate logits (15th output)
    pick0      : (N,) long tensor                  the actor's first pick
    cbc        : (N, M) numpy, optional          the heuristic's own candidate scores; enables the vs-heuristic gate
    Returns (use_pick bool numpy (N,), edge numpy (N,)) where edge = Q(pick) - Q(heuristic's choice) when cbc is given.
    """
    dev = next(critic.parameters()).device
    with torch.inference_mode():
        g = torch.as_tensor(gsp_padded, dtype=torch.float32, device=dev).unsqueeze(0)
        pool = critic.encode_spatial(g).expand(t_cf.shape[0], -1)
        q_all = critic.q_all(pool, torch.as_tensor(cve, dtype=torch.float32, device=dev),
                             t_cf.to(dev), t_cm.to(dev))
        cm_d = t_cm.to(dev); p0 = pick0.to(dev)
        ref = heuristic_ref_idx(cbc, cm_d, p0) if cbc is not None else None
        _b, adv, gate = counterfactual_gate(q_all, c_logits.to(dev), cm_d, p0, margin, ref)
        if ref is not None:
            M = q_all.shape[1]
            edge = q_all.gather(1, p0.clamp(0, M - 1).unsqueeze(1)).squeeze(1) - q_all.gather(1, ref.clamp(0, M - 1).unsqueeze(1)).squeeze(1)
        else:
            edge = adv
    return gate.cpu().numpy(), edge.float().cpu().numpy()

def heuristic_ref_idx(cbc, cand_mask, fallback_idx):
    """Index of the candidate the HEURISTIC would execute: the argmax of its own scores (the BC target)
    over live slots. Rows with no positive heuristic score fall back to `fallback_idx` (edge becomes 0)."""
    c = torch.as_tensor(cbc, dtype=torch.float32, device=cand_mask.device)
    masked = torch.where(cand_mask, c, torch.full_like(c, float('-inf')))
    ref = masked.argmax(dim=1)
    has = torch.isfinite(masked.max(dim=1).values) & (masked.max(dim=1).values > 0)
    return torch.where(has, ref, fallback_idx.to(ref.dtype))

def counterfactual_gate(q_all, cand_logits, cand_mask, pick_idx, margin: float = 0.0, ref_idx=None):
    probs = torch.softmax(torch.nan_to_num(cand_logits, nan=float('-inf')), dim=1) * cand_mask.to(q_all.dtype)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
    b = (probs * q_all).sum(dim=1)
    M = q_all.shape[1]
    q_pick = q_all.gather(1, pick_idx.clamp(0, M - 1).unsqueeze(1)).squeeze(1)
    adv = q_pick - b
    if ref_idx is None:
        return b, adv, adv > margin
    q_ref = q_all.gather(1, ref_idx.clamp(0, M - 1).unsqueeze(1)).squeeze(1)
    return b, adv, (q_pick - q_ref) > margin

PREDICTOR_IN_DIM = 19
PREDICTOR_HIDDEN_DIM = 32

class MovementPredictor(nn.Module):
    """Controller-agnostic opponent velocity and transition predictor."""
    def __init__(self, in_dim: int = PREDICTOR_IN_DIM, hidden_dim: int = PREDICTOR_HIDDEN_DIM):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRUCell(in_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(nn.Linear(hidden_dim, 32), nn.GELU(), nn.Linear(32, 2))

    def forward(self, x, hx=None, base_vel=None):
        #x: (B, in_dim), hx: (B, hidden_dim), base_vel: optional (B, 2)
        if hx is None:
            hx = torch.zeros(x.shape[0], self.hidden_dim, device=x.device, dtype=x.dtype)
        hx = self.gru(x, hx)
        out = self.head(self.ln(hx))
        if base_vel is not None: out = out + base_vel
        return out, hx

    def forward_sequence(self, x_seq, hx=None, base_vel_seq=None):
        """Unroll GRU over sequence (B, T, D) for BPTT training."""
        B, T, _ = x_seq.shape
        if hx is None:
            hx = torch.zeros(B, self.hidden_dim, device=x_seq.device, dtype=x_seq.dtype)
        out_list = []
        for t in range(T):
            hx = self.gru(x_seq[:, t], hx)
            out_t = self.head(self.ln(hx))
            if base_vel_seq is not None:
                out_t = out_t + base_vel_seq[:, t]
            out_list.append(out_t)
        return torch.stack(out_list, dim=1), hx