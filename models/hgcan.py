"""
models/hgcan.py                                                      [HGCAN]
Hierarchical Graph Context Attention Network.

  x_ent ──EntityEncoder(relation-aware GATv2)──▶ entity emb
        ──pool by body node──▶ node emb
        ──fuse 6-dim hole features──▶ node emb
        ──ContextGNN(relation-aware GATv2 over assembly graph)──▶ contextual node emb
        ──PairHead on candidate pairs──▶ (K+1) type + DOF aux

Nodes are B-Rep bodies. Relations used at both levels: entity edge_type
(convexity/incidence) and assembly asm_edge_type (contact/knn/same-occurrence).
Hole features (per body) are fused onto the node before context propagation, so a
part's hole geometry can inform "does this hole fit that part's shaft".

REVISION B (optional, off by default): when model.entity_type_head is true, the
PRE-POOL entity matrix is additionally routed to a cross-body attention module
that emits a DELTA on the pooled type logits. The delta is gated by a zero-init
scalar, so at initialisation the network is bit-for-bit the pooled baseline and
the ablation is a clean A/B. The pooled path -- existence, DOF, CAD features --
is untouched.
"""
import torch
import torch.nn as nn

from models.encoder import EntityEncoder, pool_to_occ
from models.context_gnn import ContextGNN
from models.head import PairHead
from models.entity_type_head import EntityCrossAttention
from models.constants import NUM_ENTITY_RELATIONS, NUM_ASM_RELATIONS, HOLE_DIM


class HGCAN(nn.Module):
    def __init__(self, in_dim, cfg_model):
        super().__init__()
        emb = cfg_model["entity_emb"]
        rel_emb = cfg_model.get("rel_emb", 16)
        self.encoder = EntityEncoder(
            in_dim, emb=emb, layers=cfg_model["entity_layers"],
            heads=cfg_model["heads"], dropout=cfg_model["dropout"],
            num_relations=NUM_ENTITY_RELATIONS, rel_emb=rel_emb,
            split_type_encoders=cfg_model.get("split_type_encoders", False),
        )
        # fuse pooled node embedding with per-body hole features -> emb
        self.hole_fuse = nn.Sequential(
            nn.Linear(emb + HOLE_DIM, emb), nn.ReLU(),
        )
        self.context = ContextGNN(
            emb=emb, layers=cfg_model["context_layers"],
            heads=cfg_model["heads"], dropout=cfg_model["dropout"],
            num_relations=NUM_ASM_RELATIONS, rel_emb=rel_emb,
        )
        self.head = PairHead(emb=emb, hidden=cfg_model["pair_hidden"],
                             dropout=cfg_model["dropout"],
                             use_cad=cfg_model.get("use_cad_features", True))

        # ---- Revision B: entity-level type path (opt-in) ----
        self.entity_type = None
        if cfg_model.get("entity_type_head", False):
            self.entity_type = EntityCrossAttention(
                emb=emb,
                proj=cfg_model.get("entity_proj", emb),
                topk=cfg_model.get("entity_topk", 48),
                dropout=cfg_model["dropout"],
            )
        # aux payload from the most recent forward(), consumed by the anchor
        # losses in train.py. Stashed rather than returned so forward() keeps its
        # 4-tuple contract (evaluate(), sweep_threshold.py, predict.py all rely
        # on it). Safe because training runs one assembly per step.
        self.last_entity_aux = None

    def _trunk(self, data):
        """entities -> pooled bodies -> hole fusion -> context.
        Returns (h_ent, h_geom, h_context); h_ent is the PRE-POOL entity matrix."""
        num_occ = int(data.num_occ) if not torch.is_tensor(data.num_occ) \
            else int(data.num_occ.sum())
        nt = getattr(data, "node_type", None)
        h_ent = self.encoder(data.x_ent.float(), data.ent_edge_index,
                             data.ent_edge_type, node_type=nt)
        h_geom = pool_to_occ(h_ent, data.ent_to_occ, num_occ)
        h = h_geom
        if hasattr(data, "node_hole") and data.node_hole is not None:
            h = self.hole_fuse(torch.cat([h, data.node_hole.float()], dim=-1))
        h_context = self.context(h, data.asm_edge_index, data.asm_edge_type)
        return h_ent, h_geom, h_context

    def _h_occ(self, data):
        """Body-level embeddings only. Kept at 2 return values so embed() and
        embed_pairs() -- and plot_embeddings.py -- are unaffected."""
        _, h_geom, h_context = self._trunk(data)
        return h_geom, h_context

    def forward(self, data):
        self.last_entity_aux = None                  # never reuse a stale payload
        h_ent, _, h_occ = self._trunk(data)
        ng = getattr(data, "node_geom", None)
        nh = getattr(data, "node_hole", None)
        exist_logit, type_logits, rot_logits, trans_logits = self.head(
            h_occ, data.pair_index, node_geom=ng, node_hole=nh)

        if self.entity_type is not None and data.pair_index.numel():
            num_occ = int(data.num_occ) if not torch.is_tensor(data.num_occ) \
                else int(data.num_occ.sum())
            delta, aux = self.entity_type(h_ent, data.ent_to_occ, num_occ,
                                          data.pair_index)
            type_logits = type_logits + delta
            self.last_entity_aux = aux

        return exist_logit, type_logits, rot_logits, trans_logits

    @torch.no_grad()
    def embed(self, data):
        """Return per-BODY embeddings at both levels, for visualization:
          h_geom    [N, emb]  Level-1: pooled entity (face/edge) embedding, geometry only
          h_context [N, emb]  Level-2: after the context GNN sees the neighbourhood
        Separate from forward() so it never affects training/ablation runs.
        """
        return self._h_occ(data)

    @torch.no_grad()
    def embed_pairs(self, data):
        """Return per-PAIR representations for visualizing the TYPE-DECISION space:
          feat   [P, in_feat]  input pair features (learned [3*emb+2] + CAD if enabled)
          hidden [P, hidden]   after head.shared = the space the type head linearly
                               classifies; if types separate anywhere, it is here.
        Pairs follow data.pair_index; labels are data.pair_label (0=NoJoint, 1..7).
        """
        _, h_occ = self._h_occ(data)
        if data.pair_index.numel() == 0:
            z = h_occ.new_zeros
            return z((0, 1)), z((0, 1))
        ng = getattr(data, "node_geom", None)
        nh = getattr(data, "node_hole", None)
        return self.head.pair_representation(h_occ, data.pair_index,
                                             node_geom=ng, node_hole=nh)
