"""
models/context_gnn.py                                                [HGCAN]
Level-2 relation-aware "Context" GNN over the assembly occurrence graph
(contact / kNN / parent / child / sibling). Reuses RelGATv2, so contact vs kNN
vs tree relations condition attention -- a contact neighbour and a distant kNN
neighbour are treated differently. This context is what lets HGCAN disambiguate
joint type beyond isolated pairwise geometry.
"""
from models.encoder import RelGATv2
import torch.nn as nn


class ContextGNN(nn.Module):
    def __init__(self, emb=64, layers=2, heads=4, dropout=0.1,
                 num_relations=5, rel_emb=16):
        super().__init__()
        self.gnn = RelGATv2(emb, layers, heads, dropout, num_relations, rel_emb)

    def forward(self, h_occ, asm_edge_index, asm_edge_type):
        return self.gnn(h_occ, asm_edge_index, asm_edge_type)
