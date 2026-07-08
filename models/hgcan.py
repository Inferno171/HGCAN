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
"""
import torch
import torch.nn as nn

from models.encoder import EntityEncoder, pool_to_occ
from models.context_gnn import ContextGNN
from models.head import PairHead
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

    def _h_occ(self, data):
        """Shared trunk: entities -> pooled bodies -> hole fusion -> context."""
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
        return h_geom, h_context

    def forward(self, data):
        _, h_occ = self._h_occ(data)
        ng = getattr(data, "node_geom", None)
        nh = getattr(data, "node_hole", None)
        return self.head(h_occ, data.pair_index, node_geom=ng, node_hole=nh)

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
