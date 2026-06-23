"""
MAPPO actor-critic architecture for cooperative ghost pursuit.

GhostActor:  FiLM-modulated CNN - sequential categorical waypoint sampler
GhostCritic: Sequence-agnostic multi-head self-attention centralised value head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from obs import SPATIAL_CH, MAX_H, MAX_W, VEC_DIM, CRITIC_VEC_DIM, GLOBAL_SPATIAL_CH

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
        self.res1 = ResBlock(64, 128)
        self.res2 = ResBlock(128, 128)
        self.res3 = ResBlock(128, 128)
        self.vec_mlp = nn.Sequential(nn.Linear(vec_dim, 256), nn.LayerNorm(256), nn.GELU(),
                                     nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU(),
                                     nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU())
        #vector context modulates spatial features
        self.film = FiLM(128, 128)
        #1×1 conv to logit map
        self.head = nn.Conv2d(128, 1, 1)

    def encode(self, spatial, vector):
        x = self.stem(spatial)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        vec = self.vec_mlp(vector)
        x = self.film(x, vec)
        pool = F.adaptive_avg_pool2d(x, 1).flatten(1)   #(B, 128)
        return x, pool, vec

    def logits_from_features(self, feats, mask):
        #feats: (B, 128, H, W),  mask: (B, H, W) bool
        logits = self.head(feats).squeeze(1)              #(B, H, W)
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
        """
        feats, pool, vec = self.encode(spatial, vector)
        logits = self.logits_from_features(feats, mask)
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
        return (torch.stack(sel_idx, 1), torch.stack(sel_lp, 1), scores, pool, vec)

    def evaluate_actions(self, spatial, vector, mask, actions):
        """
        Re-computes log-probs and entropy for *stored* action indices.
        Used inside the PPO update loop (single forward pass).

        Parameters
        ----------
        actions : (B, K) long — previously sampled flattened indices

        Returns
        -------
        logprobs    : (B,)          — sum of log-probs for the K actions
        entropy     : (B,)          — mean entropy across K steps
        pool        : (B, 128)      — spatial pool token
        vec         : (B, 128)      — vector embedding token
        flat_logits : (B, H*W)      — reusable for BC loss (NOT detached)
        """
        feats, pool, vec = self.encode(spatial, vector)
        logits = self.logits_from_features(feats, mask)
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
        logprobs = torch.stack(lp_list, 1).sum(1)    #(B,)
        entropy  = torch.stack(ent_list, 1).sum(1)   #(B,)
        return logprobs, entropy, pool, vec, flat_clean

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