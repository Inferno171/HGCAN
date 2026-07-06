"""
inspect_cache.py                                                    [HGCAN]
Verify the cached AssemblyPairData is (a) complete and (b) trainable.

It loads cached .pt files and:
  1. prints every field, shape, dtype, and value ranges
  2. checks required fields for the model + loss are present and consistent
  3. dry-runs ONE forward pass + ONE loss + ONE backward through HGCAN
     (proves the cache -> model -> loss path works before a full train run)

Run (from HGCAN/ project root):
  python inspect_cache.py --config configs/base.yaml           # inspect + dry-run a few
  python inspect_cache.py --config configs/base.yaml --id 20193_2da4f0a0
  python inspect_cache.py --config configs/base.yaml --n 5     # inspect first 5
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.constants import JOINT_TYPES_K1, NUM_CLASSES, NUM_ENTITY_RELATIONS, \
    NUM_ASM_RELATIONS, HOLE_DIM


REQUIRED = [
    "x_ent", "ent_edge_index", "ent_edge_type", "ent_to_occ", "node_type",
    "num_occ", "asm_edge_index", "asm_edge_type", "pair_index", "pair_label",
    "node_hole",
]


def fmt(t):
    if not torch.is_tensor(t):
        return f"{type(t).__name__}={t}"
    s = f"{tuple(t.shape)} {str(t.dtype).replace('torch.','')}"
    if t.numel() and t.is_floating_point():
        s += f"  [{t.min():.3g}, {t.max():.3g}]"
    elif t.numel():
        s += f"  [{int(t.min())}, {int(t.max())}]"
    return s


def inspect_one(d, aid):
    print(f"\n{'='*66}\n{aid}\n{'='*66}")
    print(f"  num_occ (bodies)   : {int(d.num_occ)}")
    print(f"  x_ent              : {fmt(d.x_ent)}   (entity features)")
    print(f"  node_type          : {fmt(d.node_type)}  (0=face 1=edge)")
    nf = int((d.node_type == 0).sum()); ne = int((d.node_type == 1).sum())
    print(f"     faces={nf}  edges={ne}")
    print(f"  ent_edge_index     : {fmt(d.ent_edge_index)}")
    print(f"  ent_edge_type      : {fmt(d.ent_edge_type)}  (0cvx 1ccv 2smooth 3incid)")
    print(f"  ent_to_occ         : {fmt(d.ent_to_occ)}  (entity -> body node)")
    print(f"  asm_edge_index     : {fmt(d.asm_edge_index)}")
    print(f"  asm_edge_type      : {fmt(d.asm_edge_type)}  (0contact 1knn 2sameocc)")
    print(f"  node_hole          : {fmt(d.node_hole)}  [has,count,mean_d,max_d,thru,blind]")
    nh = int((d.node_hole[:, 0] > 0).sum()) if d.node_hole.numel() else 0
    print(f"     bodies with holes: {nh}/{int(d.num_occ)}")
    print(f"  pair_index         : {fmt(d.pair_index)}  (candidate body pairs)")
    print(f"  pair_label         : {fmt(d.pair_label)}")
    # label histogram
    if d.pair_label.numel():
        hist = torch.bincount(d.pair_label, minlength=NUM_CLASSES)
        named = {JOINT_TYPES_K1[c]: int(hist[c]) for c in range(NUM_CLASSES) if hist[c] > 0}
        print(f"     labels: {named}")
    print(f"  joint_occ_pairs    : {fmt(getattr(d,'joint_occ_pairs',None))}  (localisation)")
    print(f"  joint_type         : {fmt(getattr(d,'joint_type',None))}")

    # ---- consistency checks ----
    problems = []
    for f in REQUIRED:
        if not hasattr(d, f):
            problems.append(f"MISSING field: {f}")
    N = int(d.num_occ)
    if d.ent_to_occ.numel() and int(d.ent_to_occ.max()) >= N:
        problems.append(f"ent_to_occ max {int(d.ent_to_occ.max())} >= num_occ {N}")
    if d.pair_index.numel() and int(d.pair_index.max()) >= N:
        problems.append(f"pair_index max {int(d.pair_index.max())} >= num_occ {N}")
    if d.ent_edge_index.numel() and int(d.ent_edge_index.max()) >= d.x_ent.size(0):
        problems.append("ent_edge_index out of range vs x_ent")
    if d.ent_edge_type.numel() and int(d.ent_edge_type.max()) >= NUM_ENTITY_RELATIONS:
        problems.append(f"ent_edge_type max >= NUM_ENTITY_RELATIONS ({NUM_ENTITY_RELATIONS})")
    if d.asm_edge_type.numel() and int(d.asm_edge_type.max()) >= NUM_ASM_RELATIONS:
        problems.append(f"asm_edge_type max >= NUM_ASM_RELATIONS ({NUM_ASM_RELATIONS})")
    if d.node_hole.size(-1) != HOLE_DIM:
        problems.append(f"node_hole dim {d.node_hole.size(-1)} != HOLE_DIM {HOLE_DIM}")
    if not torch.isfinite(d.x_ent).all():
        problems.append("x_ent has non-finite values")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--id", default=None, help="inspect one assembly id")
    ap.add_argument("--n", type=int, default=3, help="how many to inspect")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    asm_dir = ROOT / cfg["paths"]["cache_assembly"]
    pts = sorted(asm_dir.glob("*.pt"))
    if not pts:
        print(f"no cached .pt in {asm_dir} -> run build_cache.py first"); return
    print(f"[cache] {len(pts)} cached assemblies in {asm_dir}")

    if args.id:
        pts = [asm_dir / f"{args.id}.pt"]
    else:
        pts = pts[:args.n]

    all_problems = {}
    in_dim = None
    for p in pts:
        try:
            d = torch.load(p, weights_only=False)
        except Exception as e:
            print(f"[load FAIL] {p.name}: {e}"); continue
        probs = inspect_one(d, p.stem)
        if in_dim is None and d.x_ent.numel():
            in_dim = d.x_ent.size(-1)
        if probs:
            all_problems[p.stem] = probs
            print("  PROBLEMS:", *probs, sep="\n    - ")
        else:
            print("  checks: OK")

    # ---- dry-run through the model ----
    print(f"\n{'='*66}\nMODEL DRY-RUN (one forward + loss + backward)\n{'='*66}")
    if in_dim is None:
        print("no entity features found -> cannot build model."); return
    try:
        from models.hgcan import HGCAN
        from models.losses import hgcan_loss
        model = HGCAN(in_dim, cfg["model"])
        print(f"model built: in_dim={in_dim}  params={sum(x.numel() for x in model.parameters()):,}")
        d = torch.load(pts[0], weights_only=False)
        model.train()
        exist_logit, type_logits, rot_logits, trans_logits = model(d)
        print(f"forward OK: exist {tuple(exist_logit.shape)}  type {tuple(type_logits.shape)} "
              f"(expect [{d.pair_index.size(1)}] and [{d.pair_index.size(1)}, 7])")
        loss, parts = hgcan_loss(exist_logit, type_logits, rot_logits, trans_logits,
                                 d.pair_label, type_weights=None, exist_pos_weight=None,
                                 lambda_type=cfg["train"].get("lambda_type", 1.0),
                                 lambda_dof=cfg["train"]["lambda_dof"])
        loss.backward()
        g = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
        print(f"loss OK: total={float(loss):.4f}  exist={parts['exist']:.4f}  "
              f"type={parts['type']:.4f}  dof={parts['dof']:.4f}")
        print(f"backward OK: total grad magnitude = {g:.3g}")
        print("\nVERDICT: cache -> model -> loss -> backward all PASS. Trainable. OK")
    except Exception as e:
        import traceback
        print("MODEL DRY-RUN FAILED:")
        traceback.print_exc()
        print("\nVERDICT: not yet trainable — fix the above.")

    if all_problems:
        print(f"\n[!] {len(all_problems)} assemblies had field problems (see above).")


if __name__ == "__main__":
    main()
