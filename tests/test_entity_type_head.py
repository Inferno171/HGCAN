"""
tests/test_entity_type_head.py                                       [HGCAN]
Synthetic-fixture checks for the Revision-B entity type head, before any
full-dataset run. Covers the properties that would silently corrupt results if
wrong: zero-init no-op, order invariance, top-k correctness, anchor/candidate
alignment (including the flipped case), and gradient flow.

Run:  python -m tests.test_entity_type_head
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.entity_type_head import EntityCrossAttention, topk_entities_per_body
from models.anchor_loss import build_anchor_targets, entity_aux_losses


class FakeData:
    """Minimal stand-in for AssemblyPairData carrying only what these modules read."""
    pass


def make_fixture(seed=0, n_bodies=4, ents_per_body=(5, 60, 12, 9), emb=64):
    torch.manual_seed(seed)
    ent_to_occ = torch.cat([torch.full((n,), b, dtype=torch.long)
                            for b, n in enumerate(ents_per_body)])
    n_ent = ent_to_occ.numel()
    h_ent = torch.randn(n_ent, emb)

    pair_index = torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long)

    d = FakeData()
    d.ent_to_occ = ent_to_occ
    d.num_occ = n_bodies
    d.pair_index = pair_index

    starts = torch.cumsum(torch.tensor([0] + list(ents_per_body)), 0)
    # joint 0: bodies (0,1) in the SAME order as pair_index -> aligned
    # joint 1: bodies (2,0) in the OPPOSITE order to pair_index[:,1]=(0,2) -> flipped
    d.joint_occ_pairs = torch.tensor([[0, 2], [1, 0]], dtype=torch.long)
    d.joint_pos_i = [torch.tensor([int(starts[0]) + 1, int(starts[0]) + 2]),
                     torch.tensor([int(starts[2]) + 3])]
    d.joint_pos_j = [torch.tensor([int(starts[1]) + 4]),
                     torch.tensor([int(starts[0]) + 0])]
    d.joint_type = torch.tensor([2, 1], dtype=torch.long)
    return d, h_ent


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    return ok


def main():
    allok = True
    emb, topk = 64, 16
    d, h_ent = make_fixture(emb=emb)
    mod = EntityCrossAttention(emb=emb, proj=32, topk=topk, dropout=0.0)
    mod.eval()

    print("\n1. top-k selection")
    sal = torch.randn(h_ent.size(0))
    idx, mask = topk_entities_per_body(sal, d.ent_to_occ, d.num_occ, topk)
    allok &= check("shape", tuple(idx.shape) == (d.num_occ, topk), str(tuple(idx.shape)))
    # body 0 has 5 entities (< k) -> exactly 5 valid slots; body 1 has 60 -> k valid
    allok &= check("small body keeps all", int(mask[0].sum()) == 5, f"{int(mask[0].sum())}/5")
    allok &= check("large body capped at k", int(mask[1].sum()) == topk,
                   f"{int(mask[1].sum())}/{topk}")
    sel_bodies = d.ent_to_occ[idx[1][mask[1]]]
    allok &= check("selection stays within its own body", bool((sel_bodies == 1).all()))
    picked = sal[idx[1][mask[1]]]
    body1 = sal[d.ent_to_occ == 1]
    allok &= check("picks the most salient", torch.allclose(
        picked.sort(descending=True).values, body1.sort(descending=True).values[:topk]))

    print("\n2. zero-init gate => exact no-op")
    delta, aux = mod(h_ent, d.ent_to_occ, d.num_occ, d.pair_index)
    allok &= check("delta is exactly zero at init", bool((delta == 0).all()),
                   f"max|delta|={float(delta.detach().abs().max()):.2e}")
    allok &= check("delta shape", tuple(delta.shape) == (3, 7), str(tuple(delta.shape)))
    allok &= check("S shape", tuple(aux["S"].shape) == (3, topk, topk),
                   str(tuple(aux["S"].shape)))

    print("\n3. order invariance (joint A-B == joint B-A)")
    with torch.no_grad():
        mod.gate.fill_(1.0)                     # open the gate so delta is non-trivial
    dfwd, _ = mod(h_ent, d.ent_to_occ, d.num_occ, d.pair_index)
    flipped = d.pair_index.flip(0)
    dflip, _ = mod(h_ent, d.ent_to_occ, d.num_occ, flipped)
    allok &= check("delta unchanged when the pair is swapped",
                   torch.allclose(dfwd, dflip, atol=1e-5),
                   f"max diff={float((dfwd - dflip).abs().max()):.2e}")
    allok &= check("delta is non-zero once gated", float(dfwd.abs().max()) > 0)

    print("\n4. padding is masked out of S")
    _, aux2 = mod(h_ent, d.ent_to_occ, d.num_occ, d.pair_index)
    S, m = aux2["S"], aux2["sel_mask"]
    i, j = d.pair_index[0], d.pair_index[1]
    valid = m[i].unsqueeze(2) & m[j].unsqueeze(1)
    allok &= check("all pad cells hold the mask value", bool((S[~valid] <= -1e3).all()))
    allok &= check("no NaN / Inf in S", bool(torch.isfinite(S).all()))

    print("\n5. anchor targets (alignment, incl. the flipped joint)")
    # body 1 has 60 entities, so topk=16 would prune its anchor by chance. Use a
    # selector wide enough to retain everything, isolating the ALIGNMENT logic.
    wide = EntityCrossAttention(emb=emb, proj=32, topk=64, dropout=0.0)
    wide.eval()
    _, auxw = wide(h_ent, d.ent_to_occ, d.num_occ, d.pair_index)
    tg = build_anchor_targets(d, auxw)
    allok &= check("targets built", tg is not None)
    allok &= check("both anchored joints mapped when nothing is pruned",
                   tg["pair_rows"].numel() == 2,
                   f"{tg['pair_rows'].numel()}/2  coverage={tg['coverage']:.2f}")
    allok &= check("mapped to the right candidate pairs",
                   sorted(tg["pair_rows"].tolist()) == [0, 1],
                   str(tg["pair_rows"].tolist()))
    kw = auxw["sel_idx"].size(1)
    cell = tg["cell_mask"][tg["pair_rows"].tolist().index(1)].reshape(kw, kw)
    rows_hit = cell.any(dim=1).nonzero().flatten()
    cols_hit = cell.any(dim=0).nonzero().flatten()
    row_ents = auxw["sel_idx"][0][rows_hit]
    col_ents = auxw["sel_idx"][2][cols_hit]
    allok &= check("flipped joint: ROW side is body 0 (candidate order), not body 2",
                   bool((d.ent_to_occ[row_ents] == 0).all()),
                   f"rows in bodies {sorted(set(d.ent_to_occ[row_ents].tolist()))}")
    allok &= check("flipped joint: COL side is body 2",
                   bool((d.ent_to_occ[col_ents] == 2).all()),
                   f"cols in bodies {sorted(set(d.ent_to_occ[col_ents].tolist()))}")
    # joint 0 has 2 anchors on side i and 1 on side j -> 2x1 = 2 target cells
    cell0 = tg["cell_mask"][tg["pair_rows"].tolist().index(0)]
    allok &= check("multi-entity anchor yields a 2x1 target cell set",
                   int(cell0.sum()) == 2, f"{int(cell0.sum())} cells")

    print("\n5b. coverage degrades when top-k prunes an anchor (by design)")
    _, auxn = mod(h_ent, d.ent_to_occ, d.num_occ, d.pair_index)
    tgn = build_anchor_targets(d, auxn)
    allok &= check("narrow selector reports coverage < 1.0",
                   tgn["coverage"] < 1.0,
                   f"coverage={tgn['coverage']:.2f} at topk={topk} "
                   f"(body 1 has 60 entities)")
    allok &= check("coverage is reported, not silently dropped",
                   "coverage" in tgn and tgn["n_joints"] == 2)

    print("\n6. losses + gradient flow")
    mod.train()
    h = h_ent.clone().requires_grad_(True)
    delta, aux3 = mod(h, d.ent_to_occ, d.num_occ, d.pair_index)
    laux, parts = entity_aux_losses(d, aux3, lambda_anchor=0.5, lambda_saliency=0.2)
    total = delta.pow(2).mean() + laux
    total.backward()
    allok &= check("aux loss is finite", bool(torch.isfinite(laux)),
                   f"anchor={parts['anchor']:.4f} sal={parts['saliency']:.4f} "
                   f"cov={parts['coverage']:.2f}")
    allok &= check("gradient reaches h_ent", h.grad is not None and bool(h.grad.abs().sum() > 0))
    allok &= check("gradient reaches the saliency head",
                   mod.sal.weight.grad is not None and bool(mod.sal.weight.grad.abs().sum() > 0))
    allok &= check("gradient reaches the gate",
                   mod.gate.grad is not None and bool(mod.gate.grad.abs().sum() > 0))

    print("\n7. degenerate inputs")
    d0, h0 = make_fixture(ents_per_body=(3, 3, 3, 3), emb=emb)
    d0.pair_index = torch.zeros((2, 0), dtype=torch.long)
    dz, auxz = mod(h0, d0.ent_to_occ, d0.num_occ, d0.pair_index)
    allok &= check("empty pair set returns empty delta", dz.shape[0] == 0)
    lz, pz = entity_aux_losses(d0, auxz)
    allok &= check("aux loss safe with no pairs", bool(torch.isfinite(lz)))
    empty = torch.zeros((0, emb))
    de, _ = mod(empty, torch.zeros((0,), dtype=torch.long), 0,
                torch.zeros((2, 0), dtype=torch.long))
    allok &= check("empty entity set is safe", de.shape[0] == 0)

    print("\n" + ("ALL CHECKS PASSED" if allok else "SOME CHECKS FAILED"))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
