"""
models/head.py                                                       [HGCAN]
Decoupled pair head with EXPLICIT CAD geometric interaction features.

  exist_logit  [P]      binary: is this pair jointed?   (BCEWithLogits + pos_weight)
  type_logits  [P, 7]   which of 7 joint types           (CE on positives only)
  rot/trans    [P, 3]   DOF auxiliary (regularizer)

Two feature streams fused before the shared encoder:
  (1) LEARNED pair features from context embeddings:
        [h_i+h_j | |h_i-h_j| | h_i*h_j | cos | dist]
  (2) EXPLICIT CAD pair features from per-body world geometry (node_geom) + holes:
        axis_angle, axis_offset, radius_diff, radius_ratio, centroid_dist,
        both_have_axis, hole_radius_diff, through/blind compatibility, ...
Stream (2) directly targets the Rigid<->Revolute<->Cylindrical confusion, which
is geometric (shared axis, matching radius) and hard for generic embeddings.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.constants import NUM_JOINT_TYPES, N_ROT_CLASSES, N_TRANS_CLASSES

N_CAD_FEATS = 9


def learned_pair_features(hi, hj):
    """Order-invariant learned encoding. Width = 3*emb + 2."""
    summ = hi + hj
    diff = (hi - hj).abs()
    prod = hi * hj
    cos = F.cosine_similarity(hi, hj, dim=-1, eps=1e-8).unsqueeze(-1)
    dist = torch.linalg.norm(hi - hj, dim=-1, keepdim=True)
    return torch.cat([summ, diff, prod, cos, dist], dim=-1)


def cad_pair_features(gi, gj, hi_hole, hj_hole):
    """Explicit CAD interaction features from per-body world geometry + holes.
    gi, gj: [P, 8]  = [centroid(3) | axis_dir(3) | radius(1) | has_axis(1)]
    *_hole: [P, 6]  = [has_hole, count, mean_d, max_d, through_ratio, blind_ratio]
    Returns [P, N_CAD_FEATS], all order-invariant."""
    ci, cj = gi[:, 0:3], gj[:, 0:3]
    ai, aj = gi[:, 3:6], gj[:, 3:6]
    ri, rj = gi[:, 6], gj[:, 6]
    hai, haj = gi[:, 7], gj[:, 7]
    both_axis = hai * haj

    # axis angle (0..1): |cos| so parallel==antiparallel; only meaningful if both have axes
    cos_ax = (ai * aj).sum(-1).clamp(-1, 1).abs()
    axis_angle = cos_ax * both_axis

    # centroid distance and axis perpendicular offset
    dvec = ci - cj
    cdist = torch.linalg.norm(dvec, dim=-1)
    # perpendicular distance of centroid-offset from body i's axis (coaxiality signal)
    proj = (dvec * ai).sum(-1, keepdim=True) * ai
    perp = torch.linalg.norm(dvec - proj, dim=-1) * both_axis

    # radius comparison (diameter-match signal for revolute/cylindrical)
    rdiff = (ri - rj).abs()
    rmax = torch.maximum(ri, rj).clamp(min=1e-3)
    rratio = torch.minimum(ri, rj) / rmax * both_axis

    # hole compatibility
    hole_both = hi_hole[:, 0] * hj_hole[:, 0]
    hole_rdiff = (hi_hole[:, 2] - hj_hole[:, 2]).abs()          # mean-diameter diff
    through_compat = (hi_hole[:, 4] * hj_hole[:, 4] +
                      hi_hole[:, 5] * hj_hole[:, 5]) * hole_both  # both-through or both-blind

    feats = torch.stack([
        axis_angle, both_axis, cdist, perp,
        rdiff, rratio, hole_both, hole_rdiff, through_compat,
    ], dim=-1)
    return torch.nan_to_num(feats, 0.0)


class PairHead(nn.Module):
    def __init__(self, emb=64, hidden=128, dropout=0.1, use_cad=True):
        super().__init__()
        self.use_cad = use_cad
        in_feat = 3 * emb + 2 + (N_CAD_FEATS if use_cad else 0)
        self.shared = nn.Sequential(
            nn.Linear(in_feat, hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.exist_head = nn.Linear(hidden, 1)
        self.type_head = nn.Linear(hidden, NUM_JOINT_TYPES)
        self.rot_head = nn.Linear(hidden, N_ROT_CLASSES)
        self.trans_head = nn.Linear(hidden, N_TRANS_CLASSES)

    def pair_representation(self, h_occ, pair_index, node_geom=None, node_hole=None):
        """Build the pair encoding the heads actually see.
        Returns (feat, hidden):
          feat   [P, in_feat]  input pair features (learned + optional CAD)
          hidden [P, hidden]   after the shared MLP = the TYPE-DECISION SPACE
        Used by forward() and by plot_pair_embeddings.py."""
        i, j = pair_index[0], pair_index[1]
        feat = learned_pair_features(h_occ[i], h_occ[j])
        if self.use_cad and node_geom is not None and node_hole is not None:
            cad = cad_pair_features(node_geom[i].float(), node_geom[j].float(),
                                    node_hole[i].float(), node_hole[j].float())
            feat = torch.cat([feat, cad], dim=-1)
        return feat, self.shared(feat)

    def forward(self, h_occ, pair_index, node_geom=None, node_hole=None):
        if pair_index.numel() == 0:
            z = h_occ.new_zeros
            return (z((0,)), z((0, NUM_JOINT_TYPES)),
                    z((0, N_ROT_CLASSES)), z((0, N_TRANS_CLASSES)))
        _, s = self.pair_representation(h_occ, pair_index, node_geom, node_hole)
        return (self.exist_head(s).squeeze(-1),
                self.type_head(s),
                self.rot_head(s),
                self.trans_head(s))
