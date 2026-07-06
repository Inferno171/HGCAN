"""
plot_training.py                                                    [HGCAN]
Build training/validation figures from the logs train.py writes:
  - checkpoints/train_log.jsonl        (per-epoch losses + val metrics)
  - checkpoints/confusion_val.json     (final val confusion)
  - checkpoints/confusion_test.json    (held-out test confusion)

Produces PNGs in reports/figures/:
  loss_curves.png       train vs val loss (total + per-head)  -> overfitting view
  existence_curve.png   val existence P / R / F1 over epochs
  per_type_recall.png   val per-type recall over epochs
  confusion_test.png     held-out test confusion heatmap
  confusion_val.png      val confusion heatmap

Run (from HGCAN/ project root, after training):
  python plot_training.py --config configs/base.yaml
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_jsonl(p):
    rows = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--tag", default=None,
                    help="run suffix, e.g. cad1_hier0 (matches train_log_<tag>.jsonl). "
                         "If omitted, uses the most recent train_log_*.jsonl.")
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib/numpy not installed. Run: pip install matplotlib numpy")
        sys.exit(1)

    import yaml
    cfg = yaml.safe_load(open(args.config))
    ckpt = ROOT / cfg["paths"]["checkpoints"]
    figs = ROOT / cfg["paths"]["reports"] / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    log_p = ckpt / "train_log.jsonl"
    if args.tag:
        log_p = ckpt / f"train_log_{args.tag}.jsonl"
    elif not log_p.exists():
        # no legacy log -> pick the most recent tagged log
        cands = sorted(ckpt.glob("train_log_*.jsonl"), key=lambda p: p.stat().st_mtime)
        if cands:
            log_p = cands[-1]
            print(f"[auto] using most recent log: {log_p.name}")
    if not log_p.exists():
        print(f"no log at {log_p} -> train first."); sys.exit(1)
    tag = args.tag or (log_p.stem.replace("train_log_", "") if "train_log_" in log_p.stem else "")
    sfx = f"_{tag}" if tag else ""
    rows = load_jsonl(log_p)
    ep = [r["epoch"] for r in rows]

    # ---- 1. loss curves: train vs val (total + per head) ----
    def col(key, default=None):
        return [r.get(key, default) for r in rows]
    have_val = "val_loss_total" in rows[0]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    panels = [("total", "train_loss_total", "val_loss_total"),
              ("existence (BCE)", "train_loss_exist", "val_loss_exist"),
              ("type (CE)", "train_loss_type", "val_loss_type"),
              ("DOF aux", "train_loss_dof", "val_loss_dof")]
    for axp, (name, tk, vk) in zip(axes.ravel(), panels):
        tr = col(tk)
        if any(v is not None for v in tr):
            axp.plot(ep, tr, label="train", color="#1f77b4", lw=2)
        if have_val:
            axp.plot(ep, col(vk), label="val", color="#d62728", lw=2, ls="--")
        axp.set_title(f"{name} loss"); axp.set_xlabel("epoch"); axp.grid(alpha=.3)
        axp.legend(fontsize=8)
    fig.suptitle("HGCAN — training vs validation loss", fontweight="bold")
    fig.tight_layout()
    fig.savefig(figs / f"loss_curves{sfx}.png", dpi=140); plt.close(fig)
    print(f"[fig] loss_curves{sfx}.png")

    # ---- 2. existence P/R/F1 over epochs ----
    exP = [r["existence"]["exist_P"] for r in rows]
    exR = [r["existence"]["exist_R"] for r in rows]
    exF = [r["existence"]["exist_F1"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ep, exP, label="precision", color="#2ca02c", lw=2)
    ax.plot(ep, exR, label="recall", color="#ff7f0e", lw=2)
    ax.plot(ep, exF, label="F1", color="#1f77b4", lw=2.5)
    best = int(np.argmax(exF))
    ax.axvline(ep[best], color="#888", ls=":", lw=1)
    ax.annotate(f"best F1={exF[best]:.3f}\n@epoch {ep[best]}",
                (ep[best], exF[best]), textcoords="offset points", xytext=(8, -18), fontsize=9)
    ax.set_title("Validation — joint existence detection", fontweight="bold")
    ax.set_xlabel("epoch"); ax.set_ylabel("score"); ax.set_ylim(0, 1); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(figs / f"existence_curve{sfx}.png", dpi=140); plt.close(fig)
    print(f"[fig] existence_curve{sfx}.png")

    # ---- 3. per-type recall over epochs ----
    types = set()
    for r in rows:
        types.update(r.get("val_per_type_recall", {}).keys())
    types = [t for t in ["NoJoint", "RigidJointType", "RevoluteJointType",
                         "SliderJointType", "CylindricalJointType",
                         "PinSlotJointType", "PlanarJointType", "BallJointType"] if t in types]
    fig, ax = plt.subplots(figsize=(9, 5))
    for t in types:
        ys = [r.get("val_per_type_recall", {}).get(t, None) for r in rows]
        ax.plot(ep, ys, label=t.replace("JointType", ""), lw=1.8, marker=".", ms=4)
    ax.set_title("Validation — per-type recall", fontweight="bold")
    ax.set_xlabel("epoch"); ax.set_ylabel("recall"); ax.set_ylim(0, 1)
    ax.grid(alpha=.3); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(figs / f"per_type_recall{sfx}.png", dpi=140); plt.close(fig)
    print(f"[fig] per_type_recall{sfx}.png")

    # ---- 4. confusion heatmaps ----
    def plot_conf(js_path, out_name, title):
        if not js_path.exists():
            print(f"[skip] {js_path.name} not found"); return
        d = json.loads(js_path.read_text())
        M = np.array(d["confusion"], float)
        labels = [l.replace("JointType", "") for l in d["labels"]]
        # row-normalize for readability (recall view), keep counts as annotations
        counts = M.copy()
        rown = M.sum(1, keepdims=True); rown[rown == 0] = 1
        Mn = M / rown
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        im = ax.imshow(Mn, cmap="cividis", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        for i in range(len(labels)):
            for j in range(len(labels)):
                if counts[i, j] > 0:
                    ax.text(j, i, int(counts[i, j]), ha="center", va="center",
                            fontsize=7, color="white" if Mn[i, j] < .55 else "black")
        f1 = d.get("existence", {}).get("exist_F1", "?")
        ax.set_title(f"{title}\nexistence F1={f1}  ·  thr={d.get('threshold')}", fontweight="bold", fontsize=11)
        fig.colorbar(im, ax=ax, fraction=.046, label="row-normalized (recall)")
        fig.tight_layout(); fig.savefig(figs / out_name, dpi=140); plt.close(fig)
        print(f"[fig] {figs/out_name}")

    sfx = f"_{tag}" if tag else ""
    plot_conf(ckpt / f"confusion_test_{tag}.json" if tag else ckpt / "confusion_test.json",
              f"confusion_test{sfx}.png", "Held-out TEST confusion")
    plot_conf(ckpt / f"confusion_val_{tag}.json" if tag else ckpt / "confusion_val.json",
              f"confusion_val{sfx}.png", "Validation confusion")

    print(f"\nAll figures -> {figs}")


if __name__ == "__main__":
    main()
