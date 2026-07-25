"""
models/anchor_loss.py                                                [HGCAN]
Entity-ANCHOR supervision for the cross-body attention matrix S.

WHAT THIS USES THAT THE MODEL PREVIOUSLY IGNORED
------------------------------------------------
build_assembly_dataset.py already extracts, for every ground-truth joint, WHICH
face/edge on each body the joint attaches to, and stores it in the cache as:

    d.joint_occ_pairs [2, J]      the two body indices of each anchored joint
    d.joint_pos_i     list[J]     GLOBAL entity indices of the anchor on side i
    d.joint_pos_j     list[J]     GLOBAL entity indices of the anchor on side j

Nothing in the model has ever consumed these. They are exactly labels for a cell
of S: "row r (a face of body i) should match column c (a face of body j)."

TWO LOSSES
----------
1. ANCHOR (multi-positive cross-entropy on S).
   Each side's anchor is a SET of entities, not a single one, so the target is a
   set of cells and the loss is  -log( sum of probability over those cells )  --
   the standard multi-positive form. Computed only where at least one target cell
   survived top-k selection.

2. SALIENCY (BCE on the selector).
   Top-k selection is non-differentiable, so the anchor loss cannot teach the
   selector to KEEP anchors. This term does: entities that are anchors are
   positive, other entities of participating bodies are negative. Without it,
   anchor coverage is whatever random initialisation happens to give.

The reported `coverage` (fraction of anchored joints whose target survived
selection) is the diagnostic to watch -- if it is low, raise `topk` or
`lambda_saliency` before concluding the anchor loss does not help.

NOTE ON ORDERING: d.joint_occ_pairs stores (ia, ib) in extraction order, while
d.pair_index stores each pair SORTED (i < j). The mapping below handles both
alignments explicitly; getting this wrong would silently transpose the target.
"""
import torch
import torch.nn.functional as F

NEG = -1e4


def _positions_in_row(sel_row, sel_mask_row, wanted):
    """Positions within a [k] selection row at which `wanted` entities appear.

    sel_row      [k] long   selected global entity indices for one body
    sel_mask_row [k] bool   which slots are real
    wanted       [W] long   global entity indices we are looking for
    Returns      [k] bool   True at slots holding one of `wanted`
    """
    if wanted.numel() == 0:
        return torch.zeros_like(sel_mask_row)
    hit = (sel_row.unsqueeze(0) == wanted.unsqueeze(1)).any(dim=0)
    return hit & sel_mask_row


def build_anchor_targets(data, aux):
    """Map cached joint anchors onto cells of S.

    Returns a dict with
        pair_rows  [M] long   index into P (which candidate pair)
        cell_mask  [M, k*k]   bool, True at target cells
        coverage   float      fraction of anchored joints successfully mapped
        n_joints   int
    or None when this assembly carries no usable anchors.
    """
    jop = getattr(data, "joint_occ_pairs", None)
    jpi = getattr(data, "joint_pos_i", None)
    jpj = getattr(data, "joint_pos_j", None)
    if jop is None or jpi is None or jpj is None or jop.numel() == 0:
        return None

    sel_idx, sel_mask = aux["sel_idx"], aux["sel_mask"]
    pair_index = aux["pair_index"]
    device = sel_idx.device
    k = sel_idx.size(1)
    P = pair_index.size(1)
    if P == 0:
        return None

    # candidate pair lookup: (min,max) body pair -> position in pair_index
    pi = pair_index[0].tolist()
    pj = pair_index[1].tolist()
    lookup = {}
    for p in range(P):
        lookup[(min(pi[p], pj[p]), max(pi[p], pj[p]))] = p

    rows, masks = [], []
    n_joints = jop.size(1)
    for t in range(n_joints):
        ia, ib = int(jop[0, t]), int(jop[1, t])
        p = lookup.get((min(ia, ib), max(ia, ib)))
        if p is None:
            continue                                  # anchored joint not a candidate

        a_i = jpi[t].to(device) if torch.is_tensor(jpi[t]) else \
            torch.as_tensor(jpi[t], device=device)
        a_j = jpj[t].to(device) if torch.is_tensor(jpj[t]) else \
            torch.as_tensor(jpj[t], device=device)

        # align: does pair_index[0][p] correspond to side i or side j?
        if int(pair_index[0, p]) == ia:
            row_body, col_body, row_a, col_a = ia, ib, a_i, a_j
        else:
            row_body, col_body, row_a, col_a = ib, ia, a_j, a_i

        r_hit = _positions_in_row(sel_idx[row_body], sel_mask[row_body], row_a)
        c_hit = _positions_in_row(sel_idx[col_body], sel_mask[col_body], col_a)
        if not (r_hit.any() and c_hit.any()):
            continue                                  # anchor pruned by top-k

        cell = (r_hit.unsqueeze(1) & c_hit.unsqueeze(0)).reshape(-1)   # [k*k]
        rows.append(p)
        masks.append(cell)

    if not rows:
        return {"pair_rows": torch.zeros((0,), dtype=torch.long, device=device),
                "cell_mask": torch.zeros((0, k * k), dtype=torch.bool, device=device),
                "coverage": 0.0, "n_joints": n_joints}

    return {"pair_rows": torch.tensor(rows, dtype=torch.long, device=device),
            "cell_mask": torch.stack(masks, 0),
            "coverage": len(rows) / max(n_joints, 1),
            "n_joints": n_joints}


def anchor_ce(S, targets):
    """Multi-positive cross-entropy on the flattened S matrix.

    S       [P, k, k]  masked compatibility scores
    targets output of build_anchor_targets
    """
    if targets is None or targets["pair_rows"].numel() == 0:
        return S.new_zeros(())
    rows = targets["pair_rows"]
    logits = S[rows].reshape(rows.numel(), -1)               # [M, k*k]
    logp = F.log_softmax(logits, dim=1)
    tgt = targets["cell_mask"]
    # -log sum_{cells in target} p   ==  -logsumexp of the target log-probs
    masked = logp.masked_fill(~tgt, NEG)
    return -(torch.logsumexp(masked, dim=1)).mean()


def saliency_bce(saliency, data, aux):
    """Teach the top-k selector to retain anchor entities.

    Positives: entities that are a joint anchor on either side.
    Negatives: the other entities of bodies that participate in an anchored joint
               (restricting to participating bodies keeps the term focused rather
               than drowning in every entity of every unrelated part).
    """
    jop = getattr(data, "joint_occ_pairs", None)
    jpi = getattr(data, "joint_pos_i", None)
    jpj = getattr(data, "joint_pos_j", None)
    if jop is None or jpi is None or jop.numel() == 0 or saliency.numel() == 0:
        return saliency.new_zeros(())

    device = saliency.device
    ent_to_occ = data.ent_to_occ.to(device)
    pos = torch.zeros_like(saliency, dtype=torch.bool)
    bodies = set()
    for t in range(jop.size(1)):
        bodies.add(int(jop[0, t])); bodies.add(int(jop[1, t]))
        for lst in (jpi, jpj):
            a = lst[t]
            a = a.to(device) if torch.is_tensor(a) else torch.as_tensor(a, device=device)
            if a.numel():
                pos[a] = True
    if not bodies:
        return saliency.new_zeros(())

    body_t = torch.tensor(sorted(bodies), device=device)
    in_scope = (ent_to_occ.unsqueeze(1) == body_t.unsqueeze(0)).any(dim=1)
    if not in_scope.any():
        return saliency.new_zeros(())

    logits = saliency[in_scope]
    target = pos[in_scope].float()
    n_pos = target.sum().clamp(min=1.0)
    n_neg = (target.numel() - target.sum()).clamp(min=1.0)
    pw = (n_neg / n_pos).clamp(max=50.0)          # anchors are a tiny minority
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)


def entity_aux_losses(data, aux, lambda_anchor=0.5, lambda_saliency=0.2):
    """Combined entity-level auxiliary loss.

    Returns (loss_tensor, parts_dict). Safe to call when the entity head is off
    (aux is None) or the assembly has no anchors -- returns a zero scalar.
    """
    if aux is None:
        return None, {"anchor": 0.0, "saliency": 0.0, "coverage": 0.0}

    targets = build_anchor_targets(data, aux)
    l_anchor = anchor_ce(aux["S"], targets)
    l_sal = saliency_bce(aux["saliency"], data, aux)
    total = lambda_anchor * l_anchor + lambda_saliency * l_sal
    return total, {"anchor": float(l_anchor.detach()),
                   "saliency": float(l_sal.detach()),
                   "coverage": (targets or {}).get("coverage", 0.0)}
