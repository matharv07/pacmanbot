"""
MAPPO actor-critic architecture for cooperative ghost pursuit.

GhostActor:  Spatial FiLM-modulated CNN yielding full 2D logit maps for sequential
             waypoint picking without replacement, plus a discrete sprint-prior speed head.
GhostCritic: Global spatial CNN + joint vector MLP evaluating the global team state V(s).
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from obs import SPATIAL_CH, MAX_H, MAX_W, VEC_DIM, CRITIC_VEC_DIM, GLOBAL_SPATIAL_CH

SPEED_VALUES = [1.0, 0.88, 0.75]
SPEED_FLOOR  = float(__import__("os").environ.get("SPEED_FLOOR", "0.0"))

def speed_to_mult(val) -> float:
    if isinstance(val, torch.Tensor):
        val = val.item()
    if isinstance(val, (int, np.integer)) and val in (0, 1, 2):
        return [1.0, 0.88, 0.75][val]
    try:
        return float(np.clip(float(val), 0.0, 1.0))
    except Exception:
        return 1.0

def mult_to_throttle(mult):
    try:
        return float(np.clip(float(mult), 0.0, 1.0))
    except Exception:
        return 1.0

speed_idx_to_mult = speed_to_mult

class ResBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.bn1   = nn.GroupNorm(8, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.bn2   = nn.GroupNorm(8, c_out)
        self.skip  = (nn.Sequential(nn.Conv2d(c_in, c_out, 1), nn.GroupNorm(8, c_out)) if c_in != c_out else nn.Identity())

    def forward(self, x):
        r = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + r)

class FiLM(nn.Module):
    """Feature-wise Linear Modulation: gamma * feat + beta."""
    def __init__(self, cond_dim, n_channels):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, n_channels)
        self.beta  = nn.Linear(cond_dim, n_channels)

    def forward(self, spatial, cond):
        # spatial: (B, C, H, W)   cond: (B, cond_dim)
        g = self.gamma(cond).unsqueeze(-1).unsqueeze(-1)
        b = self.beta(cond).unsqueeze(-1).unsqueeze(-1)
        g = g.clamp(-10.0, 10.0)
        return g * spatial + b

def _sample_k(flat, base_invalid, K: int):
    """K sequential categorical picks without replacement over a masked logit row."""
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
    """Re-score stored picks under the sequential-without-replacement scheme."""
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

class GhostActor(nn.Module):
    def __init__(self, vec_dim: int = VEC_DIM):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(SPATIAL_CH, 64, 7, padding=3), nn.GroupNorm(8, 64), nn.ReLU())
        self.res1 = ResBlock(64, 128)
        self.res2 = ResBlock(128, 128)
        self.res3 = ResBlock(128, 128)
        self.vec_mlp = nn.Sequential(nn.Linear(vec_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU())
        self.film = FiLM(128, 128)
        self.head = nn.Conv2d(128, 1, 1)
        self.speed_mu = nn.Sequential(nn.Linear(256, 64), nn.LayerNorm(64), nn.ReLU(), nn.Linear(64, 1))
        self.speed_log_std = nn.Parameter(torch.tensor([-1.2]))  # initial std ~= 0.30
        nn.init.orthogonal_(self.speed_mu[-1].weight, gain=0.01)
        nn.init.constant_(self.speed_mu[-1].bias, 3.0)  # sigmoid(3.0) ~= 0.953 (nominally 1.0 full speed)

    def encode(self, spatial, vector):
        x = self.stem(spatial)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        vec = self.vec_mlp(vector)
        x = self.film(x, vec)
        pool = F.adaptive_avg_pool2d(x, 1).flatten(1)  #(B, 128)
        return x, pool, vec

    def logits_from_features(self, feats, mask=None):
        logits = self.head(feats).squeeze(1)  #(B, H, W)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        if mask is not None:
            if mask.dim() == 4:
                mask = mask.squeeze(1)
            if mask.dtype != torch.bool:
                mask = mask > 0.5
            logits = logits.masked_fill(~mask, float('-inf'))
        return logits

    def forward(self, spatial, vector, mask=None, *args, K=3, **kwargs):
        """
        Returns
        -------
        sel_idx      : (B, K) long — sampled waypoint indices
        sel_lp       : (B, K)      — log-prob of waypoint selections
        scores       : (B, H, W)   — sigmoid confidence map for CBBA
        pool         : (B, 128)    — critic token
        vec          : (B, 128)    — vector embedding token
        speed_idx    : (B,) long   — discrete speed action index [0, 1, 2]
        speed_lp     : (B,)        — log-prob of speed choice
        speed_logits : (B, 3)      — speed logits
        """
        if 'K_cand' in kwargs:
            K = kwargs['K_cand']
        if mask is None:
            mask = (spatial[:, 0] == 0)
        else:
            if mask.dim() == 4:
                mask = mask.squeeze(1)
            if mask.dtype != torch.bool:
                mask = mask > 0.5
        feats, pool, vec = self.encode(spatial, vector)
        logits = self.logits_from_features(feats, mask)
        scores = torch.sigmoid(logits)
        B = spatial.shape[0]
        base_invalid = ~mask.reshape(B, -1)
        flat = logits.view(B, -1).clone()
        flat = torch.nan_to_num(flat, nan=float('-inf'))
        sel_idx, sel_lp = _sample_k(flat, base_invalid, K)
        tok = torch.cat([pool, vec], dim=1)
        mu = torch.sigmoid(self.speed_mu(tok)).squeeze(-1)  #(B,)
        std = torch.exp(torch.clamp(self.speed_log_std, -3.0, 0.5))
        speed_dist = torch.distributions.Normal(mu, std)
        raw_speed = speed_dist.sample()
        speed_act = torch.clamp(raw_speed, 0.0, 1.0)
        speed_lp = speed_dist.log_prob(raw_speed)
        return (sel_idx, sel_lp, scores, pool, vec, speed_act, speed_lp, mu)

    def evaluate_actions(self, spatial, vector, mask=None, actions=None, speed_actions=None, *args, **kwargs):
        """
        Re-computes log-probs and entropy for stored actions in PPO.
        """
        if mask is None:
            mask = (spatial[:, 0] == 0)
        else:
            if mask.dim() == 4:
                mask = mask.squeeze(1)
            if mask.dtype != torch.bool:
                mask = mask > 0.5
        feats, pool, vec = self.encode(spatial, vector)
        logits = self.logits_from_features(feats, mask)
        B = spatial.shape[0]
        base_invalid = ~mask.reshape(B, -1)
        flat_clean = torch.nan_to_num(logits.view(B, -1), nan=float('-inf'))
        all_inf_clean = (torch.isinf(flat_clean) & (flat_clean < 0)).all(dim=1, keepdim=True)
        fallback_clean = torch.where(base_invalid, torch.full_like(flat_clean, float('-inf')), torch.zeros_like(flat_clean))
        flat_clean = torch.where(all_inf_clean, fallback_clean, flat_clean)
        spatial_lp, spatial_ents = _eval_k(flat_clean, base_invalid, actions)
        spatial_lp_sum = spatial_lp.sum(1)
        spatial_ent_sum = spatial_ents.sum(1)
        tok = torch.cat([pool, vec], dim=1)
        mu = torch.sigmoid(self.speed_mu(tok)).squeeze(-1)
        std = torch.exp(torch.clamp(self.speed_log_std, -3.0, 0.5))
        speed_dist = torch.distributions.Normal(mu, std)
        if speed_actions.ndim > 1:
            speed_actions = speed_actions.squeeze(-1)
        speed_actions_clamped = speed_actions.float().clamp(0.0, 1.0)
        speed_lp = speed_dist.log_prob(speed_actions_clamped)
        speed_ent = speed_dist.entropy()
        logprobs = spatial_lp_sum + speed_lp
        entropy  = spatial_ent_sum + 0.1 * speed_ent
        return logprobs, entropy, pool, vec, flat_clean, mu, spatial_lp_sum, speed_lp

class GhostCritic(nn.Module):
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
        self.head = nn.Sequential(
            nn.Linear(128 + 128, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 1))

    def encode_spatial(self, spatial):
        x = self.stem(spatial)
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    def forward_from_pool(self, pool, vector):
        vec = self.vec_mlp(vector)
        tokens = torch.cat([pool, vec], dim=-1)
        return self.head(tokens).squeeze(-1)

    def forward(self, spatial, vector):
        pool = self.encode_spatial(spatial)
        return self.forward_from_pool(pool, vector)

def gate_for_eval(*args, **kwargs):
    """Pass-through gate: executes 100% of RL spatial picks directly into CBBA."""
    n = 1
    if args and hasattr(args[0], '__len__'):
        n = len(args[0])
    elif 'cve' in kwargs and hasattr(kwargs['cve'], 'shape'):
        n = kwargs['cve'].shape[0]
    return np.ones(n, dtype=bool), np.zeros(n, dtype=np.float32)

def counterfactual_gate(*args, **kwargs):
    b = torch.tensor(0.0)
    adv = torch.tensor(0.0)
    gate = torch.tensor(True)
    return b, adv, gate

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
        if hx is None:
            hx = torch.zeros(x.shape[0], self.hidden_dim, device=x.device, dtype=x.dtype)
        hx = self.gru(x, hx)
        out = self.head(self.ln(hx))
        if base_vel is not None:
            out = out + base_vel
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