"""
plot_pair_embeddings.py                                             [HGCAN]
Visualize the TYPE-DECISION SPACE: per-PAIR representations, colored by the
pair's true joint type.

Why this exists: body-level embeddings colored by joint type are inherently
muddy -- type is a PAIR property, and one body can sit in several joints. This
script plots the representation the type head actually classifies:

  panel 1  input pair features   [3*emb+2 (+9 CAD)]   what the head receives
  panel 2  head hidden (shared)  [pair_hidden]        what the type head
                                                       LINEARLY separates

If joint types separate anywhere, it is panel 2. If they do not separate even
there, that is a mechanistic explanation for the confusion matrix: the learned
representation does not linearly encode kinematic type.

Metrics (computed on the RAW high-dim vectors, not the 2-D projection):
  * kNN type purity among TRUE joints (k=10)      -- type structure
  * kNN joint/no-joint separability (k=10)        -- detection structure
Rare classes (PinSlot/Planar/Ball) drawn last, larger, black-edged.

Run (flags MUST match the checkpoint's architecture):
  python plot_pair_embeddings.py --config configs/kaggle.yaml \
      --ckpt checkpoints/best_cad.pt --use-cad on --split val --tag cad
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.dataset import HGCANCache, split_ids, official_splits
from models.hgcan import HGCAN
from models.constants import JOINT_TYPES_7


def reduce_2d(X, seed=42):
    """UMAP -> t-SNE -> PCA, whichever is available."""
    try:
        import umap
        return umap.UMAP(n_components=2, random_state=seed).fit_transform(X), "UMAP"
    except Exception:
        pass
    try:
        from sklearn.manifold import TSNE
        perp = int(min(30, max(5, X.shape[0] // 50)))
        return TSNE(n_components=2, random_state=seed, perplexity=perp,
                    init="pca").fit_transform(X), "t-SNE"
    except Exception:
        pass
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=seed).fit_transform(X), "PCA"


def knn_agree(X, y, k=10, mask=None):
    """Mean fraction of a point's k nearest neighbours sharing its label."""
    if mask is not None:
        X, y = X[mask], y[mask]
    n = X.shape[0]
    if n <= k + 1:
        return float("nan")
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nn.kneighbors(X)
    neigh = y[idx[:, 1:]]                    # drop self
    return float((neigh == y[:, None]).mean())


def pair_metrics(F_in, H, L, k=10):
    """kNN metrics on the raw vectors for both spaces."""
    is_joint = (L > 0)
    out = {}
    out["sep_input"] = knn_agree(F_in, is_joint.astype(int), k)
    out["sep_hidden"] = knn_agree(H, is_joint.astype(int), k)
    if is_joint.sum() > k + 1:
        out["type_purity_input"] = knn_agree(F_in, L, k, mask=is_joint)
        out["type_purity_hidden"] = knn_agree(H, L, k, mask=is_joint)
    print(f"[metrics] joint/no-joint kNN separability: "
          f"input={out['sep_input']:.3f}  hidden={out['sep_hidden']:.3f}")
    if "type_purity_hidden" in out:
        print(f"[metrics] type kNN purity (true joints): "
              f"input={out['type_purity_input']:.3f}  "
              f"hidden={out['type_purity_hidden']:.3f}  "
              f"({out['type_purity_hidden']-out['type_purity_input']:+.3f})")
        print("          (chance under the class mix is roughly the majority share; "
              "purity near it = types NOT separated)")
    return out


def scatter_panel(ax, XY, L, title):
    names7 = [t.replace("JointType", "") for t in JOINT_TYPES_7]
    cmap = plt.get_cmap("tab10")
    m0 = L == 0
    ax.scatter(XY[m0, 0], XY[m0, 1], s=8, color="lightgray", alpha=0.5,
               label="no joint", zorder=1)
    rare = [5, 6, 7]                          # PinSlot, Planar, Ball
    for t in range(1, 8):
        if t in rare:
            continue
        m = L == t
        if m.any():
            ax.scatter(XY[m, 0], XY[m, 1], s=15, color=cmap((t - 1) % 10),
                       alpha=0.85, label=names7[t - 1], zorder=3)
    for t in rare:
        m = L == t
        if m.any():
            ax.scatter(XY[m, 0], XY[m, 1], s=55, color=cmap((t - 1) % 10),
                       alpha=0.95, label=names7[t - 1],
                       edgecolors="black", linewidths=0.6, zorder=5)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--split-encoders", choices=["on", "off"], default=None,
                    help="override model.split_type_encoders (on|off)")
    ap.add_argument("--use-cad", choices=["on", "off"], default=None,
                    help="override model.use_cad_features (on|off)")
    ap.add_argument("--use-hierarchical", choices=["on", "off"], default=None,
                    help="accepted for CLI symmetry (does not change the model shape)")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--max-nojoint", type=int, default=3000,
                    help="subsample NoJoint pairs for readability (0 = keep all)")
    ap.add_argument("--max-assemblies", type=int, default=0,
                    help="cap assemblies scanned (0 = all in split)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    if args.use_cad is not None:
        cfg["model"]["use_cad_features"] = (args.use_cad == "on")
    if args.split_encoders is not None:
        cfg["model"]["split_type_encoders"] = (args.split_encoders == "on")

    device = cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    asm_dir = ROOT / cfg["paths"]["cache_assembly"]

    split_json = cfg["paths"].get("split_json") or ""
    if split_json and Path(split_json).exists():
        tr, va, te = official_splits(asm_dir, split_json,
                                     cfg["train"]["val_frac"], cfg["train"]["seed"])
        ids = va if args.split == "val" else te
    else:
        tr, va = split_ids(asm_dir, cfg["train"]["val_frac"], cfg["train"]["seed"])
        ids = va
    ds = HGCANCache(asm_dir, ids)

    d0 = next((ds[i] for i in range(len(ds)) if ds[i].x_ent.numel()), None)
    model = HGCAN(d0.x_ent.size(-1), cfg["model"]).to(device)
    model.load_state_dict(torch.load(ROOT / args.ckpt, weights_only=False,
                                     map_location=device))
    model.eval()
    print(f"[pairs] ckpt={args.ckpt}  split={args.split}  "
          f"assemblies={len(ds)}  device={device}")

    feats, hids, labs = [], [], []
    n_scan = len(ds) if args.max_assemblies <= 0 else min(len(ds), args.max_assemblies)
    for i in range(n_scan):
        d = ds[i].to(device)
        if d.pair_index.numel() == 0:
            continue
        f, h = model.embed_pairs(d)
        feats.append(f.cpu().numpy()); hids.append(h.cpu().numpy())
        labs.append(d.pair_label.cpu().numpy())
    F_in = np.concatenate(feats, 0); H = np.concatenate(hids, 0)
    L = np.concatenate(labs, 0)
    print(f"[pairs] {len(L)} pairs  (joints {(L>0).sum()}, no-joint {(L==0).sum()})  "
          f"input dim={F_in.shape[1]}  hidden dim={H.shape[1]}")

    # subsample NoJoint for a readable plot (metrics use the same subsample so
    # the printed numbers describe exactly what the figure shows)
    rng = np.random.RandomState(args.seed)
    if args.max_nojoint > 0 and (L == 0).sum() > args.max_nojoint:
        keep = np.where(L > 0)[0]
        nz = np.where(L == 0)[0]
        keep = np.concatenate([keep, rng.choice(nz, args.max_nojoint, replace=False)])
        F_in, H, L = F_in[keep], H[keep], L[keep]
        print(f"[pairs] subsampled NoJoint -> {len(L)} pairs plotted")

    pair_metrics(F_in, H, L)

    XY_in, m1 = reduce_2d(F_in, args.seed)
    XY_h, m2 = reduce_2d(H, args.seed)

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7))
    scatter_panel(axes[0], XY_in, L, f"Pair INPUT features  [{m1}]")
    scatter_panel(axes[1], XY_h, L, f"Head hidden = type-decision space  [{m2}]")
    axes[1].legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.suptitle("HGCAN pair embeddings — colored by TRUE joint type "
                 f"({args.split} split, tag={args.tag})", fontsize=13, y=0.98)
    out_dir = ROOT / cfg["paths"]["reports"] / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"pair_embeddings_{args.tag}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
