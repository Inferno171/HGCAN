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

    def forward(self, data):
        num_occ = int(data.num_occ) if not torch.is_tensor(data.num_occ) \
            else int(data.num_occ.sum())
        h_ent = self.encoder(data.x_ent.float(), data.ent_edge_index, data.ent_edge_type)
        h_occ = pool_to_occ(h_ent, data.ent_to_occ, num_occ)
        if hasattr(data, "node_hole") and data.node_hole is not None:
            h_occ = self.hole_fuse(torch.cat([h_occ, data.node_hole.float()], dim=-1))
        h_occ = self.context(h_occ, data.asm_edge_index, data.asm_edge_type)
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
        num_occ = int(data.num_occ) if not torch.is_tensor(data.num_occ) \
            else int(data.num_occ.sum())
        h_ent = self.encoder(data.x_ent.float(), data.ent_edge_index, data.ent_edge_type)
        h_geom = pool_to_occ(h_ent, data.ent_to_occ, num_occ)      # Level-1 (pure geometry)
        h = h_geom
        if hasattr(data, "node_hole") and data.node_hole is not None:
            h = self.hole_fuse(torch.cat([h, data.node_hole.float()], dim=-1))
        h_context = self.context(h, data.asm_edge_index, data.asm_edge_type)  # Level-2
        return h_geom, h_context
