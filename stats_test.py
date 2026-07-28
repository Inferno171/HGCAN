"""
stats_test.py                                                        [HGCAN]
Significance testing for the ablation ladder.

WHY THIS EXISTS
---------------
Comparing two arms by their headline F1 does not establish that one is better.
With ~11k test pairs and three seeds, a 0.02 F1 gap could be sampling noise. Two
complementary tests are reported:

1. McNEMAR'S TEST on the paired EXISTENCE decisions. Both arms are evaluated on
   the identical pairs, so the predictions are paired: for every pair we know
   whether arm A was right and whether arm B was right. McNemar's looks ONLY at
   the disagreements -- cases where one arm is right and the other wrong -- and
   asks whether the split between "A right / B wrong" and "B right / A wrong" is
   further from 50:50 than chance allows. Pairs both arms get right (or both get
   wrong) carry no information about which is better and are correctly ignored.
   The exact binomial version is used, so it is valid at any discordant count.

2. BOOTSTRAP CONFIDENCE INTERVALS on the metric deltas, resampled at the
   ASSEMBLY level rather than the pair level. Pairs within one assembly are not
   independent -- they share bodies, geometry and a designer -- so pair-level
   resampling would understate the variance and produce CIs that are too narrow.
   Resampling whole assemblies respects that clustering.

Both tests are run per seed and then pooled, so you can see whether an effect is
consistent across initialisations or driven by one lucky run.

USAGE
-----
    # 1. dump per-pair predictions for every checkpoint (needs GPU/CPU + cache)
    python stats_test.py dump --config-dir configs/runs --ckpt-dir checkpoints \\
        --split test --out predictions

    # 2. run the tests (pure CPU, no model needed)
    python stats_test.py test --pred-dir predictions --baseline C_cad_focal

    # or compare two specific arms
    python stats_test.py test --pred-dir predictions --pairs B_cad:C_cad_focal
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


# ───────────────────────────────────────────────────────── dumping predictions
def cmd_dump(args):
    """Run every checkpoint over the split and save per-pair predictions."""
    import torch
    import yaml
    from models.hgcan import HGCAN
    from data.dataset import HGCANCache, split_ids, official_splits

    ROOT = Path(__file__).resolve().parent
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cfgs = sorted(Path(args.config_dir).glob("*.yaml"))
    if not cfgs:
        raise SystemExit(f"no run configs under {args.config_dir}")

    for cfg_path in cfgs:
        tag = cfg_path.stem
        ck = Path(args.ckpt_dir) / f"best_{tag}.pt"
        if not ck.exists():
            print(f"  skip {tag}: no checkpoint")
            continue
        dst = out / f"{tag}_{args.split}.npz"
        if dst.exists() and not args.overwrite:
            print(f"  skip {tag}: already dumped")
            continue

        cfg = yaml.safe_load(cfg_path.read_text())
        tc = cfg["train"]
        dev = "cuda" if torch.cuda.is_available() else "cpu"

        # Mirror train.py's split construction EXACTLY. tc["seed"] (not model_seed)
        # carves the split, so every arm and seed sees the same assemblies in the
        # same order -- which is what makes the predictions PAIRED and McNemar valid.
        asm_dir = ROOT / cfg["paths"]["cache_assembly"]
        split_json = cfg["paths"].get("split_json", "")
        if split_json and Path(split_json).exists():
            train_ids, val_ids, test_ids = official_splits(
                asm_dir, split_json, tc.get("val_frac", 0.15), tc["seed"])
        else:
            train_ids, val_ids = split_ids(asm_dir, tc["val_frac"], tc["seed"])
            test_ids = []
            if args.split == "test":
                raise SystemExit("no official split_json -> there is no test split")
        ids = test_ids if args.split == "test" else val_ids
        ds = HGCANCache(asm_dir, ids)

        in_dim = int(ds[0].x_ent.size(-1))
        model = HGCAN(in_dim, cfg["model"]).to(dev)
        state = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(state["model"] if isinstance(state, dict)
                              and "model" in state else state)
        model.eval()

        asm_id, exist_p, type_p, y = [], [], [], []
        with torch.no_grad():
            for k in range(len(ds)):
                d = ds[k].to(dev)
                if d.pair_index.numel() == 0:
                    continue
                e, t, _, _ = model(d)
                exist_p.append(torch.sigmoid(e).cpu().numpy().ravel())
                type_p.append((t.argmax(-1) + 1).cpu().numpy())      # 1..7
                y.append(d.pair_label.cpu().numpy())
                asm_id.append(np.full(int(d.pair_label.numel()), k, dtype=np.int32))

        np.savez_compressed(
            dst,
            assembly=np.concatenate(asm_id),
            exist_prob=np.concatenate(exist_p).astype(np.float32),
            type_pred=np.concatenate(type_p).astype(np.int8),
            label=np.concatenate(y).astype(np.int8),
        )
        print(f"  dumped {tag}: {len(np.concatenate(y))} pairs -> {dst.name}")


# ───────────────────────────────────────────────────────────────── metrics
def existence_correct(z, thr=0.5):
    """Per-pair boolean: did the existence head get this pair right?"""
    pred = z["exist_prob"] >= thr
    true = z["label"] > 0
    return pred == true, pred, true


def f1_from(pred, true):
    tp = int((pred & true).sum()); fp = int((pred & ~true).sum())
    fn = int((~pred & true).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def macro_recall_from(z, thr=0.5):
    """Macro-averaged recall over joint types, on pairs the gate accepted.
    A true joint the gate rejected counts as a miss -- typing is only reachable
    through the gate, so crediting it would flatter the type head."""
    accepted = z["exist_prob"] >= thr
    recs = []
    for c in range(1, 8):
        m = z["label"] == c
        if not m.any():
            continue
        recs.append(float(((z["type_pred"] == c) & accepted & m).sum() / m.sum()))
    return float(np.mean(recs)) if recs else float("nan")


# ───────────────────────────────────────────────────────────── McNemar's test
def mcnemar(a_ok, b_ok):
    """Exact binomial McNemar on paired correctness vectors.

    b01 = A right, B wrong    b10 = A wrong, B right
    Concordant pairs are uninformative about which arm is better and are dropped.
    """
    b01 = int((a_ok & ~b_ok).sum())
    b10 = int((~a_ok & b_ok).sum())
    n = b01 + b10
    if n == 0:
        return dict(b01=0, b10=0, n_discordant=0, p=1.0, stat=0.0)
    k = min(b01, b10)
    # two-sided exact binomial under H0: p = 0.5
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    p = min(1.0, 2 * tail)
    stat = (abs(b01 - b10) - 1) ** 2 / n          # continuity-corrected chi2
    return dict(b01=b01, b10=b10, n_discordant=n, p=p, stat=stat)


# ───────────────────────────────────────────── assembly-level bootstrap CIs
def bootstrap_delta(za, zb, fn, n_boot=2000, seed=0, thr=0.5):
    """CI on fn(B) - fn(A), resampling ASSEMBLIES with replacement.

    Pairs inside one assembly share bodies and a designer, so they are correlated.
    Resampling pairs would treat them as independent and give CIs that are too
    narrow; resampling assemblies keeps the cluster structure intact.
    """
    assert np.array_equal(za["label"], zb["label"]), \
        "prediction files are not aligned -- same split, same order required"
    asm = za["assembly"]
    uniq = np.unique(asm)
    idx_by_asm = {a: np.flatnonzero(asm == a) for a in uniq}
    rng = np.random.default_rng(seed)

    point = fn(zb, thr) - fn(za, thr)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        sel = np.concatenate([idx_by_asm[a] for a in pick])
        sa = {k: za[k][sel] for k in ("exist_prob", "type_pred", "label")}
        sb = {k: zb[k][sel] for k in ("exist_prob", "type_pred", "label")}
        deltas[b] = fn(sb, thr) - fn(sa, thr)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    # fraction of resamples on the wrong side of zero -> a bootstrap p-value
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return dict(delta=float(point), lo=float(lo), hi=float(hi),
                p=float(min(1.0, p)), n_assemblies=int(uniq.size))


def _f1_metric(z, thr):
    _, pred, true = existence_correct(z, thr)
    return f1_from(pred, true)


# ───────────────────────────────────────────────────────────────── the report
def cmd_test(args):
    pred = Path(args.pred_dir)
    files = sorted(pred.glob(f"*_{args.split}.npz"))
    if not files:
        raise SystemExit(f"no *_{args.split}.npz under {pred}")

    runs = {}
    for f in files:
        tag = f.name[: -len(f"_{args.split}.npz")]
        arm, _, seed = tag.rpartition("_s")
        runs.setdefault(arm or tag, {})[seed or "0"] = np.load(f)
    print(f"loaded {len(files)} runs across {len(runs)} arms: {sorted(runs)}\n")

    if args.pairs:
        pairs = [tuple(p.split(":")) for p in args.pairs.split(",")]
    else:
        base = args.baseline
        if base not in runs:
            raise SystemExit(f"baseline '{base}' not among {sorted(runs)}")
        pairs = [(a, base) for a in sorted(runs) if a != base]

    report = {}
    for a, b in pairs:
        if a not in runs or b not in runs:
            print(f"skip {a} -> {b}: missing arm"); continue
        seeds = sorted(set(runs[a]) & set(runs[b]))
        print("=" * 74)
        print(f"{a}  ->  {b}     ({len(seeds)} matched seed(s): {', '.join(seeds)})")
        print("=" * 74)

        per_seed = []
        for s in seeds:
            za, zb = runs[a][s], runs[b][s]
            aok, _, _ = existence_correct(za, args.thr)
            bok, _, _ = existence_correct(zb, args.thr)
            mc = mcnemar(aok, bok)
            f1 = bootstrap_delta(za, zb, _f1_metric, args.n_boot, seed=int(s or 0), thr=args.thr)
            mr = bootstrap_delta(za, zb, macro_recall_from, args.n_boot, seed=int(s or 0), thr=args.thr)
            per_seed.append(dict(seed=s, mcnemar=mc, f1=f1, macro_recall=mr))

            sig = "***" if mc["p"] < .001 else "**" if mc["p"] < .01 else \
                  "*" if mc["p"] < .05 else "ns"
            print(f"  seed {s}")
            print(f"    McNemar  {a} only right: {mc['b01']:>5}   "
                  f"{b} only right: {mc['b10']:>5}   "
                  f"discordant {mc['n_discordant']:>6}   p={mc['p']:.3g} {sig}")
            print(f"    F1 delta      {f1['delta']:+.4f}   "
                  f"95% CI [{f1['lo']:+.4f}, {f1['hi']:+.4f}]   p={f1['p']:.3g}"
                  f"{'   (excludes 0)' if f1['lo'] * f1['hi'] > 0 else ''}")
            print(f"    macroR delta  {mr['delta']:+.4f}   "
                  f"95% CI [{mr['lo']:+.4f}, {mr['hi']:+.4f}]   p={mr['p']:.3g}"
                  f"{'   (excludes 0)' if mr['lo'] * mr['hi'] > 0 else ''}")

        if len(per_seed) > 1:
            nsig = sum(1 for r in per_seed if r["mcnemar"]["p"] < .05)
            same = len({np.sign(r["f1"]["delta"]) for r in per_seed}) == 1
            print(f"\n  ACROSS SEEDS: McNemar significant in {nsig}/{len(per_seed)}; "
                  f"F1 delta sign {'consistent' if same else 'INCONSISTENT'}")
            if not same:
                print("    -> the effect flips direction between seeds; report as inconclusive.")
        print()
        report[f"{a}->{b}"] = per_seed

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2, default=float))
        print(f"[saved] {args.json_out}")

    print("Reporting notes:")
    print("  * McNemar's uses only DISCORDANT pairs -- agreements carry no information")
    print("    about which arm is better, so a large 'discordant' count means a")
    print("    well-powered test even when the F1 gap looks small.")
    print("  * CIs resample ASSEMBLIES, not pairs, because pairs within an assembly")
    print("    are correlated; pair-level CIs would be too narrow.")
    print("  * A CI that excludes 0 and a McNemar p<0.05 together are what justify")
    print("    calling a difference real.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dump", help="save per-pair predictions for each checkpoint")
    d.add_argument("--config-dir", default="configs/runs")
    d.add_argument("--ckpt-dir", default="checkpoints")
    d.add_argument("--split", default="test", choices=["val", "test"])
    d.add_argument("--out", default="predictions")
    d.add_argument("--overwrite", action="store_true")
    d.set_defaults(func=cmd_dump)

    t = sub.add_parser("test", help="McNemar + bootstrap CIs on dumped predictions")
    t.add_argument("--pred-dir", default="predictions")
    t.add_argument("--split", default="test", choices=["val", "test"])
    t.add_argument("--baseline", default="C_cad_focal",
                   help="arm every other arm is compared against")
    t.add_argument("--pairs", default="",
                   help="explicit comparisons, e.g. B_cad:C_cad_focal,C_cad_focal:E_entity")
    t.add_argument("--thr", type=float, default=0.5)
    t.add_argument("--n-boot", type=int, default=2000)
    t.add_argument("--json-out", default="reports/stats_test.json")
    t.set_defaults(func=cmd_test)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
