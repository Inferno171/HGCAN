"""
train.py                                                             [HGCAN]
Smoke / base training for HGCAN with DECOUPLED heads.

  detection (existence): BCEWithLogits, pos_weight controls precision/recall
  type:                  CE on positives only, weights = sqrt|linear|none (+ cap)
  DOF:                   auxiliary regularizer on positives (lambda_dof)

Run (from HGCAN/ project root, after build_cache.py):
  python train.py --config configs/base.yaml --limit 20 --epochs 3   # smoke
  python train.py --config configs/base.yaml                          # full
"""
import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.dataset import HGCANCache, split_ids, official_splits
from models.hgcan import HGCAN
from models.losses import hgcan_loss
from models.constants import JOINT_TYPES_K1, JOINT_TYPES_7, NUM_CLASSES, NUM_JOINT_TYPES, hierarchical_type_pred


def infer_in_dim(ds):
    for i in range(len(ds)):
        d = ds[i]
        if d.x_ent.numel():
            return d.x_ent.size(-1)
    raise RuntimeError("no entity features found in cache")


def count_labels(ds):
    cnt = Counter()
    for i in range(len(ds)):
        for c in ds[i].pair_label.tolist():
            cnt[c] += 1
    return cnt


def type_weights(cnt, scheme, cap, device):
    """Weights over the 7 joint types (labels 1..7 -> idx 0..6). Scheme:
    sqrt (tempered), linear (inverse-freq), none. Capped at `cap`."""
    counts = [cnt.get(c + 1, 0) for c in range(NUM_JOINT_TYPES)]  # Rigid..Ball
    total = sum(counts)
    w = torch.ones(NUM_JOINT_TYPES, device=device)
    if scheme == "none" or total == 0:
        return w
    for c in range(NUM_JOINT_TYPES):
        k = max(counts[c], 1)
        if scheme == "sqrt":
            w[c] = math.sqrt(total / k)
        else:  # linear
            w[c] = total / (NUM_JOINT_TYPES * k)
    w = w / w.mean()                 # normalize so mean weight ~1
    w = torch.clamp(w, max=cap)
    return w


def existence_pos_weight(cnt, cfg_val, cap, device):
    """pos_weight for BCE: up-weights the positive (jointed) class."""
    pos = sum(v for k, v in cnt.items() if k > 0)
    neg = cnt.get(0, 0)
    if cfg_val == "none":
        return None
    if cfg_val == "auto":
        pw = neg / max(pos, 1)
    else:
        pw = float(cfg_val)
    return torch.tensor(min(pw, cap), dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate(model, ds, device, threshold, hierarchical=False,
             type_weights=None, exist_pos_weight=None, lam_type=1.0, lam_dof=0.2):
    model.eval()
    correct = Counter(); total = Counter()
    ex_tp = ex_fp = ex_fn = 0
    confusion = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]  # rows=true, cols=pred
    vloss = {"exist": 0.0, "type": 0.0, "dof": 0.0, "total": 0.0}; nseen = 0
    for i in range(len(ds)):
        d = ds[i].to(device)
        if d.pair_index.numel() == 0:
            continue
        exist_logit, type_logits, rot_logits, trans_logits = model(d)
        # validation loss (same objective as training, no grad)
        l, parts = hgcan_loss(exist_logit, type_logits, rot_logits, trans_logits,
                              d.pair_label, type_weights=type_weights,
                              exist_pos_weight=exist_pos_weight,
                              lambda_type=lam_type, lambda_dof=lam_dof)
        vloss["exist"] += parts["exist"]; vloss["type"] += parts["type"]
        vloss["dof"] += parts["dof"]; vloss["total"] += float(l); nseen += 1
        exist_pred = torch.sigmoid(exist_logit) > threshold
        if hierarchical:
            type_pred = hierarchical_type_pred(rot_logits, trans_logits, type_logits) + 1
        else:
            type_pred = type_logits.argmax(-1) + 1                 # 1..7
        final = torch.where(exist_pred, type_pred, torch.zeros_like(type_pred))
        y = d.pair_label
        for p, t in zip(final.tolist(), y.tolist()):
            total[t] += 1
            if p == t:
                correct[t] += 1
            confusion[t][p] += 1                               # rows=true, cols=pred
        pe = exist_pred.long(); te = (y > 0).long()
        ex_tp += int(((pe == 1) & (te == 1)).sum())
        ex_fp += int(((pe == 1) & (te == 0)).sum())
        ex_fn += int(((pe == 0) & (te == 1)).sum())
    per_type = {JOINT_TYPES_K1[c]: round(correct[c] / total[c], 3)
                for c in range(NUM_CLASSES) if total[c] > 0}
    prec = ex_tp / max(ex_tp + ex_fp, 1)
    rec = ex_tp / max(ex_tp + ex_fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    n = max(nseen, 1)
    val_loss = {k: round(v / n, 4) for k, v in vloss.items()}
    return per_type, {"exist_P": round(prec, 3), "exist_R": round(rec, 3),
                      "exist_F1": round(f1, 3)}, confusion, val_loss


def print_confusion(confusion, threshold):
    """Render the 8x8 confusion matrix (rows=true, cols=pred) with short labels,
    plus per-class precision/recall. Class 0 = NoJoint, 1..7 = joint types."""
    short = ["NoJoint", "Rigid", "Revol", "Slider", "Cylind", "PinSlot", "Planar", "Ball"]
    col_tot = [sum(confusion[r][c] for r in range(NUM_CLASSES)) for c in range(NUM_CLASSES)]
    row_tot = [sum(confusion[r]) for r in range(NUM_CLASSES)]
    grand = sum(row_tot)
    print("\n" + "=" * 72)
    print(f"CONFUSION MATRIX  (rows=TRUE, cols=PRED, threshold={threshold})")
    print("=" * 72)
    header = "true\\pred".ljust(9) + "".join(s[:7].rjust(8) for s in short) + "    total"
    print(header)
    for r in range(NUM_CLASSES):
        if row_tot[r] == 0:
            continue
        row = short[r].ljust(9) + "".join(str(confusion[r][c]).rjust(8) for c in range(NUM_CLASSES))
        print(row + str(row_tot[r]).rjust(9))
    print("-" * 72)
    print("recall  ".ljust(9) + "".join(
        (f"{confusion[r][r]/row_tot[r]:.2f}" if row_tot[r] else "  -  ").rjust(8)
        for r in range(NUM_CLASSES)))
    print("precis  ".ljust(9) + "".join(
        (f"{confusion[c][c]/col_tot[c]:.2f}" if col_tot[c] else "  -  ").rjust(8)
        for c in range(NUM_CLASSES)))
    acc = sum(confusion[i][i] for i in range(NUM_CLASSES)) / max(grand, 1)
    print(f"\noverall accuracy (all pairs, incl. NoJoint): {acc:.3f}  over {grand} val pairs")
    # joint-only accuracy (exclude true-NoJoint rows) — the number that matters
    j_correct = sum(confusion[r][r] for r in range(1, NUM_CLASSES))
    j_total = sum(row_tot[1:])
    if j_total:
        print(f"type accuracy on TRUE joints only: {j_correct/j_total:.3f}  over {j_total} joints")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--use-cad", choices=["on", "off"], default=None,
                    help="override model.use_cad_features (on|off)")
    ap.add_argument("--use-hierarchical", choices=["on", "off"], default=None,
                    help="override train.use_hierarchical_type (on|off)")
    ap.add_argument("--tag", default=None,
                    help="label appended to checkpoint/log filenames (keeps ablation runs separate)")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    tc = cfg["train"]
    # CLI overrides so one config file serves all ablation runs (handy on Kaggle)
    if args.use_cad is not None:
        cfg["model"]["use_cad_features"] = (args.use_cad == "on")
    if args.use_hierarchical is not None:
        tc["use_hierarchical_type"] = (args.use_hierarchical == "on")
    device = tc["device"] if torch.cuda.is_available() else "cpu"
    torch.manual_seed(tc["seed"])

    asm_dir = ROOT / cfg["paths"]["cache_assembly"]
    split_json = cfg["paths"].get("split_json", "")
    test_ds = None
    if split_json and Path(split_json).exists():
        train_ids, val_ids, test_ids = official_splits(
            asm_dir, split_json, tc.get("val_frac", 0.15), tc["seed"])
        if args.limit:
            train_ids = train_ids[:args.limit]
            val_ids = val_ids[:max(1, args.limit // 4)]
            test_ids = test_ids[:max(1, args.limit // 4)]
        test_ds = HGCANCache(asm_dir, test_ids)
    else:
        if split_json:
            print(f"[split] WARNING: split_json '{split_json}' not found -> random split")
        train_ids, val_ids = split_ids(asm_dir, tc["val_frac"], tc["seed"])
        if args.limit:
            train_ids = train_ids[:args.limit]
            val_ids = val_ids[:max(1, args.limit // 4)]
    train_ds = HGCANCache(asm_dir, train_ids)
    val_ds = HGCANCache(asm_dir, val_ids)
    print(f"[data] train={len(train_ds)} val={len(val_ds)}"
          + (f" test={len(test_ds)}" if test_ds is not None else "")
          + f" assemblies  device={device}")
    if len(train_ds) == 0:
        print("[data] cache empty -> run build_cache.py first."); return

    in_dim = infer_in_dim(train_ds)
    model = HGCAN(in_dim, cfg["model"]).to(device)
    print(f"[model] in_dim={in_dim}  params={sum(p.numel() for p in model.parameters()):,}")

    cnt = count_labels(train_ds)
    print("[data] class counts:", {JOINT_TYPES_K1[c]: cnt.get(c, 0) for c in range(NUM_CLASSES)})

    scheme = tc.get("class_weights", "sqrt")
    cap = float(tc.get("max_class_weight", 10.0))
    tw = type_weights(cnt, scheme, cap, device)
    epw = existence_pos_weight(cnt, tc.get("existence_pos_weight", "auto"),
                               float(tc.get("max_existence_pos_weight", 10.0)), device)
    thr = float(tc.get("existence_threshold", 0.5))
    print(f"[data] type weights ({scheme}, cap {cap}): "
          f"{ {JOINT_TYPES_7[c]: round(float(tw[c]),2) for c in range(NUM_JOINT_TYPES)} }")
    print(f"[data] existence pos_weight: {float(epw) if epw is not None else None}")

    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    lam_dof = tc["lambda_dof"]; lam_type = tc.get("lambda_type", 1.0); clip = tc["grad_clip"]
    hierarchical = bool(tc.get("use_hierarchical_type", False))
    if hierarchical:
        # rot/trans heads become the PRIMARY type predictors -> train them at full
        # weight (not the 0.2 auxiliary weight). Type head stays for the (1,1) tiebreak.
        lam_dof = max(lam_dof, 1.0)
        print(f"[mode] HIERARCHICAL type (DOF-factored, Design A); dof_weight={lam_dof}")
    else:
        print("[mode] FLAT 7-way type head")
    epochs = args.epochs or tc["epochs"]

    # per-run suffix so ablation runs don't overwrite each other's outputs.
    # auto-derives from the toggles if --tag not given: e.g. cad1_hier0
    suffix = args.tag or f"cad{int(cfg['model'].get('use_cad_features', True))}" \
                          f"_hier{int(tc.get('use_hierarchical_type', False))}"
    ckpt_dir = ROOT / cfg["paths"]["checkpoints"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log = ckpt_dir / f"train_log_{suffix}.jsonl"
    best_f1 = -1.0
    best_path = ckpt_dir / f"best_{suffix}.pt"
    print(f"[run] tag={suffix}  ->  {best_path.name}, {log.name}")

    for ep in range(epochs):
        model.train(); t0 = time.time()
        agg = defaultdict(float); nseen = 0
        for i in range(len(train_ds)):
            d = train_ds[i].to(device)
            if d.pair_index.numel() == 0:
                continue
            opt.zero_grad()
            exist_logit, type_logits, rot_logits, trans_logits = model(d)
            loss, parts = hgcan_loss(exist_logit, type_logits, rot_logits, trans_logits,
                                     d.pair_label, type_weights=tw, exist_pos_weight=epw,
                                     lambda_type=lam_type, lambda_dof=lam_dof)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            for k, v in parts.items():
                agg[k] += v
            nseen += 1

        per_type, ex, confusion, vloss = evaluate(
            model, val_ds, device, thr, hierarchical,
            type_weights=tw, exist_pos_weight=epw, lam_type=lam_type, lam_dof=lam_dof)
        rec = {"epoch": ep, "sec": round(time.time() - t0, 1),
               "train_loss_exist": round(agg["exist"] / max(nseen, 1), 4),
               "train_loss_type": round(agg["type"] / max(nseen, 1), 4),
               "train_loss_dof": round(agg["dof"] / max(nseen, 1), 4),
               "train_loss_total": round((agg["exist"] + lam_type * agg["type"]
                                          + lam_dof * agg["dof"]) / max(nseen, 1), 4),
               "val_loss_exist": vloss["exist"], "val_loss_type": vloss["type"],
               "val_loss_dof": vloss["dof"], "val_loss_total": vloss["total"],
               "existence": ex,
               "val_per_type_recall": per_type}
        print(json.dumps(rec))
        with open(log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        # keep only the BEST checkpoint by val existence F1
        if ex["exist_F1"] > best_f1:
            best_f1 = ex["exist_F1"]
            torch.save(model.state_dict(), best_path)
            print(f"  [checkpoint] new best F1={best_f1:.3f} -> {best_path.name}")

    print(f"[done] best val existence F1 = {best_f1:.3f}  ({best_path})")
    print(f"[done] log in {log}")

    def save_confusion(conf, ex_metrics, per_type_r, split):
        out = {"run": suffix, "split": split, "threshold": thr, "hierarchical": hierarchical,
               "existence": ex_metrics, "per_type_recall": per_type_r,
               "labels": JOINT_TYPES_K1,
               "confusion": conf, "note": "rows=true, cols=pred"}
        p = ckpt_dir / f"confusion_{split}_{suffix}.json"
        with open(p, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {p}")

    # final confusion matrix (reload best checkpoint for the definitive picture)
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, weights_only=False, map_location=device))
        per_type_v, ex_final, confusion, _ = evaluate(
            model, val_ds, device, thr, hierarchical,
            type_weights=tw, exist_pos_weight=epw, lam_type=lam_type, lam_dof=lam_dof)
        print(f"\n[best checkpoint] VAL existence: {ex_final}")
    print("\n### VALIDATION confusion ###")
    print_confusion(confusion, thr)
    save_confusion(confusion, ex_final, per_type_v, "val")

    # held-out TEST set: touched ONCE, on the best checkpoint -> the reported numbers
    if test_ds is not None and len(test_ds) > 0:
        per_type_t, ex_test, conf_test, _ = evaluate(
            model, test_ds, device, thr, hierarchical,
            type_weights=tw, exist_pos_weight=epw, lam_type=lam_type, lam_dof=lam_dof)
        print(f"\n[best checkpoint] TEST existence: {ex_test}")
        print("\n### HELD-OUT TEST confusion (official split) ###")
        print_confusion(conf_test, thr)
        save_confusion(conf_test, ex_test, per_type_t, "test")


if __name__ == "__main__":
    main()
