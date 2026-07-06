"""
models/encoder.py                                                    [HGCAN]
Level-1 entity encoder: per-entity MLP -> RELATION-AWARE GATv2 over the
face+edge graph, then mean-pool entities to per-occurrence embeddings.

Relation-aware: the edge_type (0 convex / 1 concave / 2 smooth / 3 incidence) is
embedded and fed to GATv2 as edge features, so convexity -- which the extractor
computes but the earlier model ignored -- now influences attention. GATv2 keeps
the attention mechanism (the "Attention" in HGCAN) while using relations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

try:
    from torch_geometric.utils import scatter
    def _mean_pool(x, index, dim_size):
        return scatter(x, index, dim=0, dim_size=dim_size, reduce="mean")
except Exception:
    from torch_scatter import scatter_mean
    def _mean_pool(x, index, dim_size):
        return scatter_mean(x, index, dim=0, dim_size=dim_size)


class RelGATv2(nn.Module):
    """A stack of GATv2 layers conditioned on an embedded edge relation type."""
    def __init__(self, emb, layers, heads, dropout, num_relations, rel_emb=16):
        super().__init__()
        self.rel = nn.Embedding(num_relations, rel_emb)
        self.convs = nn.ModuleList(
            GATv2Conv(emb, emb // heads, heads=heads, dropout=dropout,
                      edge_dim=rel_emb, add_self_loops=True)
            for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(emb) for _ in range(layers))

    def forward(self, h, edge_index, edge_type):
        ea = self.rel(edge_type) if edge_index.numel() else None
        for conv, norm in zip(self.convs, self.norms):
            if edge_index.numel():
                m = conv(h, edge_index, edge_attr=ea)
            else:
                m = h                                   # isolated nodes: identity message
            h = norm(h + F.relu(m))
        return h


class EntityEncoder(nn.Module):
    def __init__(self, in_dim, emb=64, layers=2, heads=4, dropout=0.1,
                 num_relations=4, rel_emb=16):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(in_dim, emb), nn.ReLU(), nn.Linear(emb, emb))
        self.gnn = RelGATv2(emb, layers, heads, dropout, num_relations, rel_emb)
        self.emb = emb

    def forward(self, x_ent, ent_edge_index, ent_edge_type):
        h = self.inp(x_ent)
        return self.gnn(h, ent_edge_index, ent_edge_type)


def pool_to_occ(h_ent, ent_to_occ, num_occ):
    """Entity embeddings [N_ent, emb] -> occurrence embeddings [num_occ, emb]."""
    return _mean_pool(h_ent, ent_to_occ, num_occ)
