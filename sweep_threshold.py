"""
sweep_threshold.py                                                  [HGCAN]
Evaluate a TRAINED checkpoint at multiple existence thresholds, WITHOUT retraining.

The existence head outputs a probability; the 0.5 default may under- or over-predict
joints. This sweeps thresholds and prints existence P/R/F1 + joint-type accuracy at
each, so you can pick the operating point off the already-trained model.

Run (from HGCAN/ project root, after training):
  python sweep_threshold.py --config configs/base.yaml --ckpt checkpoints/best.pt
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.dataset import HGCANCache, split_ids
from models.hgcan import HGCAN
from models.constants import JOINT_TYPES_K1, NUM_CLASSES, hierarchical_type_pred


@torch.no_grad()
def eval_at_thresholds(model, ds, device, thresholds, hierarchical=False):
    """Collect existence probs + labels once, then score at each threshold."""
    all_prob, all_y, all_typepred = [], [], []
    for i in range(len(ds)):
        d = ds[i].to(device)
        if d.pair_index.numel() == 0:
            continue
        exist_logit, type_logits, rot_logits, trans_logits = model(d)
        all_prob.append(torch.sigmoid(exist_logit).cpu())
        if hierarchical:
            tp = hierarchical_type_pred(rot_logits, trans_logits, type_logits) + 1
        else:
            tp = type_logits.argmax(-1) + 1
        all_typepred.append(tp.cpu())
        all_y.append(d.pair_label.cpu())
    if not all_prob:
        print("no pairs to evaluate."); return
    prob = torch.cat(all_prob); y = torch.cat(all_y); tp = torch.cat(all_typepred)
    te = (y > 0).long()
    n_pos = int(te.sum()); n_neg = int((te == 0).sum())
    print(f"val pairs: {len(y)}  (joints {n_pos}, non-joints {n_neg})\n")
    print(f"{'thr':>5} {'exP':>6} {'exR':>6} {'exF1':>6} {'joint_type_acc':>15}")
    print("-" * 42)
    best = (None, -1)
    for thr in thresholds:
        pe = (prob > thr).long()
        tp_ = int(((pe == 1) & (te == 1)).sum())
        fp_ = int(((pe == 1) & (te == 0)).sum())
        fn_ = int(((pe == 0) & (te == 1)).sum())
        P = tp_ / max(tp_ + fp_, 1)
        R = tp_ / max(tp_ + fn_, 1)
        F1 = 2 * P * R / max(P + R, 1e-9)
        # joint-type accuracy on TRUE joints that were detected at this threshold
        det = (pe == 1) & (te == 1)
        typ_acc = (tp[det] == y[det]).float().mean().item() if int(det.sum()) else 0.0
        flag = ""
        if F1 > best[1]:
            best = (thr, F1); flag = "  <-- best F1"
        print(f"{thr:5.2f} {P:6.3f} {R:6.3f} {F1:6.3f} {typ_acc:15.3f}{flag}")
    print(f"\nbest F1 {best[1]:.3f} at threshold {best[0]:.2f}")
    print("Set train.existence_threshold to this value for reporting / inference.")


@torch.no_grad()
def confusion_at(model, ds, device, thr, hierarchical=False):
    """Full 8x8 confusion + per-class P/R at one threshold (for the report table)."""
    from models.constants import JOINT_TYPES_K1
    NC = NUM_CLASSES
    conf = [[0] * NC for _ in range(NC)]
    for i in range(len(ds)):
        d = ds[i].to(device)
        if d.pair_index.numel() == 0:
            continue
        exist_logit, type_logits, rot_logits, trans_logits = model(d)
        exist_pred = torch.sigmoid(exist_logit) > thr
        if hierarchical:
            type_pred = hierarchical_type_pred(rot_logits, trans_logits, type_logits) + 1
        else:
            type_pred = type_logits.argmax(-1) + 1
        final = torch.where(exist_pred, type_pred, torch.zeros_like(type_pred))
        for p, t in zip(final.tolist(), d.pair_label.tolist()):
            conf[t][p] += 1
    short = ["NoJoint", "Rigid", "Revol", "Slider", "Cylind", "PinSlot", "Planar", "Ball"]
    col = [sum(conf[r][c] for r in range(NC)) for c in range(NC)]
    row = [sum(conf[r]) for r in range(NC)]
    print(f"\nCONFUSION @ threshold {thr}  (rows=TRUE, cols=PRED)")
    print("true\\pred".ljust(9) + "".join(s[:7].rjust(8) for s in short) + "    total")
    for r in range(NC):
        if row[r] == 0:
            continue
        print(short[r].ljust(9) + "".join(str(conf[r][c]).rjust(8) for c in range(NC)) + str(row[r]).rjust(9))
    print("recall".ljust(9) + "".join((f"{conf[r][r]/row[r]:.2f}" if row[r] else "  -  ").rjust(8) for r in range(NC)))
    print("precis".ljust(9) + "".join((f"{conf[c][c]/col[c]:.2f}" if col[c] else "  -  ").rjust(8) for c in range(NC)))
    jc = sum(conf[r][r] for r in range(1, NC)); jt = sum(row[1:])
    if jt:
        print(f"type accuracy on TRUE joints only: {jc/jt:.3f}  over {jt} joints")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--split", default="val", choices=["val", "test"],
                    help="val = carved-from-train (selection set); test = official held-out")
    ap.add_argument("--thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    ap.add_argument("--confusion", type=float, default=None,
                    help="also print full confusion matrix at this threshold")
    ap.add_argument("--split-encoders", choices=["on", "off"], default=None,
                    help="override model.split_type_encoders (on|off)")
    ap.add_argument("--use-cad", choices=["on", "off"], default=None,
                    help="MUST match how the checkpoint was trained (architecture)")
    ap.add_argument("--use-hierarchical", choices=["on", "off"], default=None,
                    help="MUST match how the checkpoint was trained (decode)")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    # overrides MUST match how the checkpoint was trained, or load_state_dict fails
    # on a param-count mismatch (CAD features change the head input width).
    if args.use_cad is not None:
        cfg["model"]["use_cad_features"] = (args.use_cad == "on")
    if args.split_encoders is not None:
        cfg["model"]["split_type_encoders"] = (args.split_encoders == "on")
    if args.use_hierarchical is not None:
        cfg["train"]["use_hierarchical_type"] = (args.use_hierarchical == "on")
    device = cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    asm_dir = ROOT / cfg["paths"]["cache_assembly"]
    split_json = cfg["paths"].get("split_json", "")

    if split_json and Path(split_json).exists():
        from data.dataset import official_splits
        _, val_ids, test_ids = official_splits(
            asm_dir, split_json, cfg["train"].get("val_frac", 0.15), cfg["train"]["seed"])
        ids = test_ids if args.split == "test" else val_ids
    else:
        _, ids = split_ids(asm_dir, cfg["train"]["val_frac"], cfg["train"]["seed"])
        if args.split == "test":
            print("[warn] no split_json -> cannot evaluate official test; using random val")
    eval_ds = HGCANCache(asm_dir, ids)
    if len(eval_ds) == 0:
        print(f"[error] {args.split} split is EMPTY (0 assemblies). "
              f"Check that official split IDs match cache filenames."); return

    d0 = next((eval_ds[i] for i in range(len(eval_ds)) if eval_ds[i].x_ent.numel()), None)
    in_dim = d0.x_ent.size(-1)
    model = HGCAN(in_dim, cfg["model"]).to(device)
    model.load_state_dict(torch.load(ROOT / args.ckpt, weights_only=False, map_location=device))
    model.eval()
    print(f"[sweep] {args.ckpt} on {len(eval_ds)} {args.split.upper()} assemblies  device={device}\n")

    hierarchical = bool(cfg["train"].get("use_hierarchical_type", False))
    if hierarchical:
        print("[mode] HIERARCHICAL type decoding (DOF-factored)")
    thresholds = [float(x) for x in args.thresholds.split(",")]
    eval_at_thresholds(model, eval_ds, device, thresholds, hierarchical)
    if args.confusion is not None:
        confusion_at(model, eval_ds, device, args.confusion, hierarchical)


if __name__ == "__main__":
    main()
