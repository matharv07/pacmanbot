"""
MAPPO actor-critic architecture for cooperative ghost pursuit.

GhostActor:  FiLM-modulated CNN - sequential categorical waypoint sampler
GhostCritic: Sequence-agnostic multi-head self-attention centralised value head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from obs import SPATIAL_CH, MAX_H, MAX_W, VEC_DIM, CRITIC_VEC_DIM, GLOBAL_SPATIAL_CH

SPEED_FLOOR   = 0.55
SPEED_PRIOR_A = 2.6
SPEED_PRIOR_B = -2.0
GATE_PRIOR_LOGIT = -1.2
RL_MAX_DEVIATION = 1.05

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
        #continuous speed head (alpha, beta for Beta distribution over the [0,1] throttle)
        #the throttle is remapped to SPEED_FLOOR..1.0 of the ghost speed cap in worker.py
        self.speed_head = nn.Sequential(nn.Linear(256, 64), nn.LayerNorm(64), nn.ReLU(), nn.Linear(64, 2))
        #continuous tactical steering head (alpha, beta) — a RESIDUAL rotation of the
        #heuristic heading, not an absolute angle, so an untrained head is a no-op
        self.dir_head = nn.Sequential(nn.Linear(256, 64), nn.LayerNorm(64), nn.ReLU(), nn.Linear(64, 2))
        #binary hijack gate: does the policy take over micro-navigation this step?
        self.gate_head = nn.Sequential(nn.Linear(256, 64), nn.LayerNorm(64), nn.ReLU(), nn.Linear(64, 1))
        #priors: start near the speed cap, steer straight, and defer to the heuristic controller
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
        logits = logits.masked_fill(~mask, float('-inf'))
        return logits

    def forward(self, spatial, vector, mask, K=3):
        """
        Returns
        -------
        indices  : (B, K) long — flattened cell indices
        logprobs : (B, K)      — log-prob of each sequential pick
        scores   : (B, H, W)   — independent sigmoid for CBBA
        pool     : (B, 128)    — spatial pool for critic token
        vec      : (B, 128)    — vector embedding for critic token
        speed    : (B, 1)      — sampled continuous speed [0, 1]
        speed_lp : (B, 1)      — log prob of sampled speed
        """
        feats, pool, vec = self.encode(spatial, vector)
        logits = self.logits_from_features(feats, pool, mask)
        #independent sigmoid scores for CBBA
        scores = torch.sigmoid(logits)
        B = spatial.shape[0]
        base_invalid = ~mask.reshape(B, -1)
        flat = logits.view(B, -1).clone()
        flat = torch.nan_to_num(flat, nan=float('-inf'))
        sel_idx, sel_lp = [], []
        for _ in range(K):
            inf_mask = torch.isinf(flat) & (flat < 0)
            all_inf = inf_mask.all(dim=1, keepdim=True)
            fallback = torch.where(base_invalid, torch.full_like(flat, float('-inf')), torch.zeros_like(flat))
            flat = torch.where(all_inf, fallback, flat)
            dist = torch.distributions.Categorical(logits=flat)
            idx  = dist.sample()
            sel_idx.append(idx)
            sel_lp.append(dist.log_prob(idx))
            flat.scatter_(1, idx.unsqueeze(1), float('-inf'))
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
        return (torch.stack(sel_idx, 1), torch.stack(sel_lp, 1), scores, pool, vec, speed.unsqueeze(1), speed_lp.unsqueeze(1), 
                direction.unsqueeze(1), dir_lp.unsqueeze(1), gate.unsqueeze(1), gate_lp.unsqueeze(1))

    def _speed_dist(self, tok):
        params = torch.clamp(F.softplus(self.speed_head(tok)) + 1.5, min=1.5, max=12.0)
        return torch.distributions.Beta(params[:, 0], params[:, 1])

    def _dir_dist(self, tok):
        params = torch.clamp(F.softplus(self.dir_head(tok)) + 2.0, min=2.0, max=12.0)
        return torch.distributions.Beta(params[:, 0], params[:, 1])

    def _gate_dist(self, tok):
        return torch.distributions.Bernoulli(logits=self.gate_head(tok).squeeze(-1).clamp(-8.0, 8.0))

    def evaluate_actions(self, spatial, vector, mask, actions, speeds, directions=None, gates=None):
        """
        Re-computes log-probs and entropy for *stored* action indices.
        Used inside the PPO update loop (single forward pass).

        Parameters
        ----------
        actions    : (B, K) long — previously sampled flattened indices
        speeds     : (B, 1) float — previously sampled speed throttles
        directions : (B, 1) float, optional — previously sampled steering residuals
        gates      : (B, 1) float, optional — previously sampled hijack gate (0/1)

        Returns
        -------
        logprobs    : (B, K+3)      — per-head log-probs: K cell picks, speed, direction, gate
        entropy     : (B,)          — mean entropy across K steps + speed + direction entropy
        pool        : (B, 128)      — spatial pool token
        vec         : (B, 128)      — vector embedding token
        flat_logits : (B, H*W)      — reusable for BC loss (NOT detached)
        speed_params: (B, 2)        — (alpha, beta) for BC loss
        """
        feats, pool, vec = self.encode(spatial, vector)
        logits = self.logits_from_features(feats, pool, mask)
        B = spatial.shape[0]
        base_invalid = ~mask.reshape(B, -1)
        flat_clean = torch.nan_to_num(
            logits.view(B, -1), nan=float('-inf'))
        inf_mask_clean = torch.isinf(flat_clean) & (flat_clean < 0)
        all_inf_clean = inf_mask_clean.all(dim=1, keepdim=True)
        fallback_clean = torch.where(base_invalid, torch.full_like(flat_clean, float('-inf')), torch.zeros_like(flat_clean))
        flat_clean = torch.where(all_inf_clean, fallback_clean, flat_clean)
        flat = flat_clean.clone()
        lp_list, ent_list = [], []
        K = actions.shape[1]
        for k in range(K):
            inf_mask = torch.isinf(flat) & (flat < 0)
            all_inf = inf_mask.all(dim=1, keepdim=True)
            fallback = torch.where(base_invalid, torch.full_like(flat, float('-inf')), torch.zeros_like(flat))
            flat = torch.where(all_inf, fallback, flat)
            dist = torch.distributions.Categorical(logits=flat)
            lp_list.append(dist.log_prob(actions[:, k]))
            ent_list.append(dist.entropy())
            mask_k = torch.zeros_like(flat, dtype=torch.bool)
            mask_k.scatter_(1, actions[:, k].unsqueeze(1), True)
            flat = torch.where(mask_k, float('-inf'), flat)
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
        #per-head log-probs (K cell picks, speed, direction, gate) so PPO can clip and measure KL
        #head by head. Summing into one joint log-prob let a single low-probability tail sample
        #(2nd/3rd sequential pick, Beta tail) blow up the whole ratio and inflate approx_kl
        logprobs = torch.cat([torch.stack(lp_list, 1), speed_lp.unsqueeze(1), dir_lp.unsqueeze(1), gate_lp.unsqueeze(1)], dim=1)
        entropy  = torch.stack(ent_list, 1).sum(1) + speed_ent + dir_ent + gate_ent
        speed_params = torch.stack([dist_speed.concentration1, dist_speed.concentration0], dim=1)
        return logprobs, entropy, pool, vec, flat_clean, speed_params

class GhostCritic(nn.Module):
    #Independent CNN-based Critic: evaluates each ghost state
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
        return self.head(tokens)

    def forward(self, spatial, vector):
        pool = self.encode_spatial(spatial)
        return self.forward_from_pool(pool, vector)

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