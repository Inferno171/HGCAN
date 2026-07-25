"""
models/entity_type_head.py                                           [HGCAN]
Entity-level (cross-body) TYPE head — "Revision B".

WHY
---
Mean pooling collapses a body's [N_ent, emb] entity matrix into a single [emb]
vector BEFORE the type head runs. Whether a bore and a shaft are coaxial, which
face is the bore wall at all -- all of that is averaged away. That loss sits
directly upstream of the model's dominant residual error (Revolute predicted as
Rigid), because a fastened cylinder and a rotating cylinder differ in exactly the
per-entity detail that pooling destroys.

This module keeps the PRE-POOL entity matrix for the type decision only:

    S = (H_i Wq)(H_j Wk)^T / sqrt(d)        [P, k, k]

one score per (entity of body i, entity of body j) pair. Two readouts come off S:
a soft attention context (order-invariant), and hard scalars (max / logsumexp /
mean). They produce a DELTA on the pooled type logits.

DESIGN CHOICES THAT MATTER
--------------------------
1. ADDITIVE, GATED. The module outputs `delta` added to the existing pooled type
   logits, scaled by a learned scalar `gate` INITIALISED TO ZERO. At step 0 the
   model is bit-for-bit the pooled baseline, so enabling this cannot corrupt a
   run before it has learned anything, and the ablation is a clean A/B.

2. NO CHANGE TO PairHead. The pooled path (existence, type, DOF, CAD features)
   is untouched. This module never sees the head's internals.

3. ORDER-INVARIANT. A joint (A,B) is the joint (B,A), so the readouts use
   sum/|diff| of the two contexts and symmetric scalars -- swapping i and j gives
   the identical delta, matching the pooled head's existing symmetry.

4. TOP-K BOUNDED. S is quadratic in entities. A learned saliency score selects at
   most `topk` entities per body, so S is at most [P, k, k] regardless of whether
   a part has 30 faces or 3,000. Bodies with fewer than k entities are padded and
   masked (no cost, no distortion).

5. NO TRAIN/INFERENCE SKEW. The delta is computed for EVERY candidate pair, not
   only for positives or gate survivors, so the type head behaves identically in
   both regimes. At k=48 the cost is a few hundred small matmuls per assembly.

Aux outputs (S, selection, saliency) are returned for the anchor losses in
models/anchor_loss.py -- the same S is where the recorded joint anchors attach.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.constants import NUM_JOINT_TYPES

NEG = -1e4          # finite mask value; -inf would produce NaN on all-pad rows


def topk_entities_per_body(saliency, ent_to_occ, num_occ, k):
    """Pick up to k entities per body, ranked by `saliency` (descending).

    Vectorised via a double sort: sort by saliency, then STABLE-sort by body, so
    entities come out grouped by body and saliency-ordered inside each group.
    The within-group rank then falls out of a cumulative-count offset.

    Returns
        idx  [num_occ, k] long  entity indices (clamped to 0 where padded)
        mask [num_occ, k] bool  True where the slot holds a real entity
    """
    device = saliency.device
    n = saliency.numel()
    if n == 0 or num_occ == 0:
        return (torch.zeros((num_occ, k), dtype=torch.long, device=device),
                torch.zeros((num_occ, k), dtype=torch.bool, device=device))

    perm = torch.argsort(saliency, descending=True)
    try:
        perm2 = torch.sort(ent_to_occ[perm], stable=True).indices
    except TypeError:                                  # very old torch
        perm2 = torch.argsort(ent_to_occ[perm])
    order = perm[perm2]                                # grouped by body, salient-first

    bodies = ent_to_occ[order]
    counts = torch.bincount(bodies, minlength=num_occ)
    offsets = torch.cumsum(counts, 0) - counts
    rank = torch.arange(order.numel(), device=device) - offsets[bodies]

    keep = rank < k
    idx = torch.full((num_occ, k), -1, dtype=torch.long, device=device)
    idx[bodies[keep], rank[keep]] = order[keep]
    mask = idx >= 0
    return idx.clamp(min=0), mask


def _masked_mean(x, mask, dim=1):
    """Mean over `dim`, ignoring padded slots. x [.., L, D], mask [.., L]."""
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(dim) / m.sum(dim).clamp(min=1.0)


class EntityCrossAttention(nn.Module):
    """Cross-body entity attention producing a delta on the pooled type logits."""

    def __init__(self, emb=64, proj=64, topk=48, dropout=0.1,
                 n_types=NUM_JOINT_TYPES):
        super().__init__()
        self.topk = int(topk)
        self.proj = int(proj)

        self.q = nn.Linear(emb, proj)
        self.k = nn.Linear(emb, proj)
        self.v = nn.Linear(emb, proj)
        self.sal = nn.Linear(emb, 1)             # entity saliency, for top-k

        # 2*proj (sum, |diff| of the two contexts) + 3 symmetric scalars
        self.mix = nn.Sequential(
            nn.Linear(2 * proj + 3, proj), nn.ReLU(), nn.Dropout(dropout),
        )
        self.type_delta = nn.Linear(proj, n_types)
        # zero-init gate => at initialisation this module is a no-op and the
        # network is EXACTLY the pooled baseline.
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, h_ent, ent_to_occ, num_occ, pair_index):
        """
        h_ent       [N_ent, emb]  PRE-POOL Level-1 entity embeddings
        ent_to_occ  [N_ent]       body index per entity
        num_occ     int           number of bodies in this assembly
        pair_index  [2, P]        candidate pairs

        Returns (delta_type_logits [P, n_types], aux dict)
            aux["S"]        [P, k, k]        masked compatibility scores
            aux["sel_idx"]  [num_occ, k]     selected entity indices
            aux["sel_mask"] [num_occ, k]     validity of those slots
            aux["saliency"] [N_ent]          raw saliency logits
            aux["pair_index"]                echoed, for the anchor loss
        """
        P = pair_index.size(1)
        n_types = self.type_delta.out_features
        sal = self.sal(h_ent).squeeze(-1)                       # [N_ent]

        if P == 0 or h_ent.numel() == 0:
            z = h_ent.new_zeros
            return z((P, n_types)), {
                "S": z((P, self.topk, self.topk)),
                "sel_idx": torch.zeros((num_occ, self.topk), dtype=torch.long,
                                       device=h_ent.device),
                "sel_mask": torch.zeros((num_occ, self.topk), dtype=torch.bool,
                                        device=h_ent.device),
                "saliency": sal, "pair_index": pair_index,
            }

        # --- select at most k entities per body (selection itself is detached:
        #     top-k is non-differentiable; `sal` is trained by the saliency BCE) ---
        sel_idx, sel_mask = topk_entities_per_body(
            sal.detach(), ent_to_occ, num_occ, self.topk)

        i, j = pair_index[0], pair_index[1]
        Hi = h_ent[sel_idx[i]]                                  # [P, k, emb]
        Hj = h_ent[sel_idx[j]]
        Mi, Mj = sel_mask[i], sel_mask[j]                       # [P, k]

        Qi, Ki, Vi = self.q(Hi), self.k(Hi), self.v(Hi)
        Qj, Kj, Vj = self.q(Hj), self.k(Hj), self.v(Hj)
        # SYMMETRISED bilinear score. A plain Qi·Kj^T would NOT satisfy
        # S(j,i) == S(i,j)^T -- q and k are different projections -- and the pair
        # delta would then depend on which body was named first, breaking the
        # order invariance the pooled head deliberately guarantees. Averaging the
        # two directions restores it exactly while keeping both projections.
        S = 0.5 * (torch.bmm(Qi, Kj.transpose(1, 2))
                   + torch.bmm(Ki, Qj.transpose(1, 2))) / math.sqrt(self.proj)

        valid = Mi.unsqueeze(2) & Mj.unsqueeze(1)               # [P, k, k]
        S = S.masked_fill(~valid, NEG)

        # --- soft readout: each side attends over the other, then masked-mean ---
        ctx_i = _masked_mean(torch.bmm(torch.softmax(S, dim=2), Vj), Mi)
        ctx_j = _masked_mean(
            torch.bmm(torch.softmax(S.transpose(1, 2), dim=2), Vi), Mj)

        # --- hard readout: symmetric scalars over the whole matrix ---
        flat = S.reshape(P, -1)
        vflat = valid.reshape(P, -1)
        s_max = flat.max(dim=1).values
        s_lse = torch.logsumexp(flat, dim=1)
        s_mean = (flat * vflat).sum(1) / vflat.sum(1).clamp(min=1)

        feat = torch.cat([ctx_i + ctx_j, (ctx_i - ctx_j).abs(),
                          s_max.unsqueeze(-1), s_lse.unsqueeze(-1),
                          s_mean.unsqueeze(-1)], dim=-1)
        delta = self.type_delta(self.mix(feat)) * self.gate

        return delta, {"S": S, "sel_idx": sel_idx, "sel_mask": sel_mask,
                       "saliency": sal, "pair_index": pair_index}
