"""
plot_embeddings.py                                                  [HGCAN]
Visualize per-BODY embeddings from a trained checkpoint in two spaces:
  Level-1 (geometry)  — pooled entity embedding, before the body sees neighbours
  Level-2 (context)   — after the Context GNN attends over the assembly graph

Shows them side by side so you can SEE what the context attention does: if the
two-level design is working, bodies re-organize between the two panels (e.g.
joint-participating bodies pull together after context).

Coloring (--color-by):
  joint_type    dominant joint type among the pairs this body participates in
  participation body is in >=1 true joint (vs isolated part)
  has_axis      body has a usable cylinder axis (from node_geom)
  holes         number of holes on the body (from node_hole)

Dim-reduction: UMAP if installed, else t-SNE, else PCA (auto-fallback).

Run (from HGCAN/ project root, after training):
  python plot_embeddings.py --config configs/base.yaml --ckpt checkpoints/best_cad.pt \
      --use-cad on --split val --color-by joint_type
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.dataset import HGCANCache, official_splits, split_ids
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
        n = X.shape[0]
        perp = max(5, min(30, n // 4))
        return TSNE(n_components=2, random_state=seed, perplexity=perp,
                    init="pca").fit_transform(X), "t-SNE"
    except Exception:
        pass
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=seed).fit_transform(X), "PCA"


def per_body_labels(d, color_by):
    """Return (values, is_categorical) per body node for the requested coloring."""
    N = int(d.num_occ)
    if color_by == "has_axis":
        ng = getattr(d, "node_geom", None)
        if ng is None:
            return np.zeros(N), True
        return ng[:, 7].cpu().numpy().astype(int), True          # has_axis flag
    if color_by == "holes":
        nh = getattr(d, "node_hole", None)
        if nh is None:
            return np.zeros(N), False
        return nh[:, 1].cpu().numpy(), False                     # hole count (continuous)

    # joint-based colorings need the candidate pairs + labels
    pi = d.pair_index.cpu().numpy()          # [2, P]
    pl = d.pair_label.cpu().numpy()          # [P]  0=NoJoint, 1..7=type
    per_node_types = [[] for _ in range(N)]
    for k in range(pi.shape[1]):
        if pl[k] > 0:
            per_node_types[pi[0, k]].append(pl[k])
            per_node_types[pi[1, k]].append(pl[k])

    if color_by == "participation":
        return np.array([1 if per_node_types[i] else 0 for i in range(N)]), True
    # joint_type: dominant type among this body's joints (0 = none)
    out = np.zeros(N, dtype=int)
    for i in range(N):
        if per_node_types[i]:
            out[i] = Counter(per_node_types[i]).most_common(1)[0][0]   # 1..7
    return out, True


def embedding_metrics(G, C, L, color_by, k=10):
    """Quantify L1(geometry) vs L2(context) on the RAW embeddings (not the 2D map).
    Prints and returns a dict. Three complementary measures:
      - kNN joint-type purity: for jointed bodies, fraction of k nearest neighbours
        (in embedding space) sharing the same joint type. Higher = types cluster.
      - joint/no-joint kNN separability: for ALL bodies, fraction of k neighbours on
        the same side of the joint/no-joint divide. Higher = detection-relevant.
      - centroid gap: normalized distance between the jointed and no-joint centroids.
        Higher = jointed bodies sit apart from isolated parts.
    """
    from sklearn.neighbors import NearestNeighbors

    def knn_agree(X, y, mask=None):
        idx_all = np.arange(len(y))
        sel = idx_all if mask is None else idx_all[mask]
        if len(sel) <= k:
            return float("nan")
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
        _, nbr = nn.kneighbors(X[sel])
        agree = []
        for row, i in zip(nbr, sel):
            neigh = row[row != i][:k]
            agree.append(np.mean(y[neigh] == y[i]))
        return float(np.mean(agree))

    def centroid_gap(X, joint_mask):
        a = X[joint_mask].mean(0); b = X[~joint_mask].mean(0)
        scale = X.std(0).mean() + 1e-9
        return float(np.linalg.norm(a - b) / scale)

    joint_mask = L > 0 if color_by in ("joint_type", "participation") else None
    out = {}
    print("\n" + "=" * 60)
    print(f"EMBEDDING METRICS  (k={k}, computed on 64-d vectors)")
    print("=" * 60)

    if color_by == "joint_type" and joint_mask is not None and joint_mask.sum() > k:
        pg = knn_agree(G, L, joint_mask)      # among jointed bodies, type purity
        pc = knn_agree(C, L, joint_mask)
        out["type_purity"] = (pg, pc)
        print(f"kNN joint-TYPE purity (jointed only): "
              f"L1={pg:.3f}  L2={pc:.3f}  ({pc-pg:+.3f})")

    if joint_mask is not None:
        yb = joint_mask.astype(int)
        sg = knn_agree(G, yb); sc = knn_agree(C, yb)
        out["joint_separability"] = (sg, sc)
        print(f"kNN joint/no-joint separability     : "
              f"L1={sg:.3f}  L2={sc:.3f}  ({sc-sg:+.3f})")
        gg = centroid_gap(G, joint_mask); gc = centroid_gap(C, joint_mask)
        out["centroid_gap"] = (gg, gc)
        print(f"jointed vs no-joint centroid gap    : "
              f"L1={gg:.3f}  L2={gc:.3f}  ({gc-gg:+.3f})")
    else:
        # non-joint colorings: purity w.r.t. the categorical label itself
        if color_by in ("has_axis",):
            sg = knn_agree(G, L); sc = knn_agree(C, L)
            out["label_purity"] = (sg, sc)
            print(f"kNN {color_by} purity: L1={sg:.3f}  L2={sc:.3f}  ({sc-sg:+.3f})")

    print("=" * 60)
    print("higher L2 than L1 => the context GNN adds discriminative structure.\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--color-by", default="joint_type",
                    choices=["joint_type", "participation", "has_axis", "holes"])
    ap.add_argument("--split-encoders", choices=["on", "off"], default=None,
                    help="override model.split_type_encoders (on|off)")
    ap.add_argument("--use-cad", choices=["on", "off"], default=None,
                    help="MUST match how the checkpoint was trained")
    ap.add_argument("--use-hierarchical", choices=["on", "off"], default=None)
    ap.add_argument("--max-assemblies", type=int, default=0,
                    help="cap assemblies processed (0 = all in split)")
    ap.add_argument("--tag", default=None, help="suffix for the output PNG")
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. pip install matplotlib"); sys.exit(1)

    import yaml
    cfg = yaml.safe_load(open(args.config))
    if args.use_cad is not None:
        cfg["model"]["use_cad_features"] = (args.use_cad == "on")
    if args.split_encoders is not None:
        cfg["model"]["split_type_encoders"] = (args.split_encoders == "on")
    if args.use_hierarchical is not None:
        cfg["train"]["use_hierarchical_type"] = (args.use_hierarchical == "on")
    device = cfg["train"]["device"] if torch.cuda.is_available() else "cpu"

    asm_dir = ROOT / cfg["paths"]["cache_assembly"]
    split_json = cfg["paths"].get("split_json")
    if split_json and Path(split_json).exists():
        tr, va, te = official_splits(asm_dir, split_json,
                                     cfg["train"].get("val_frac", 0.15), cfg["train"]["seed"])
        ids = {"train": tr, "val": va, "test": te}[args.split]
    else:
        tr, va = split_ids(asm_dir, cfg["train"]["val_frac"], cfg["train"]["seed"])
        ids = {"train": tr, "val": va, "test": va}[args.split]
    ds = HGCANCache(asm_dir, ids)
    if args.max_assemblies:
        ds.paths = ds.paths[:args.max_assemblies]
    print(f"[embed] {len(ds)} {args.split} assemblies  device={device}")

    d0 = next((ds[i] for i in range(len(ds)) if ds[i].x_ent.numel()), None)
    in_dim = d0.x_ent.size(-1)
    model = HGCAN(in_dim, cfg["model"]).to(device)
    model.load_state_dict(torch.load(ROOT / args.ckpt, weights_only=False, map_location=device))
    model.eval()

    geom, ctx, labs = [], [], []
    for i in range(len(ds)):
        d = ds[i]
        if d.pair_index.numel() == 0 or d.x_ent.numel() == 0:
            continue
        d = d.to(device)
        h_geom, h_ctx = model.embed(d)
        geom.append(h_geom.cpu().numpy())
        ctx.append(h_ctx.cpu().numpy())
        lv, is_cat = per_body_labels(d, args.color_by)
        labs.append(lv)
    if not geom:
        print("no bodies to embed."); return
    G = np.concatenate(geom, 0); C = np.concatenate(ctx, 0); L = np.concatenate(labs, 0)
    print(f"[embed] {G.shape[0]} bodies  dim={G.shape[1]}  color-by={args.color_by}")

    # ---- quantitative comparison L1 vs L2 (on the RAW embeddings, not the 2D map) ----
    emb_metrics = embedding_metrics(G, C, L, args.color_by)

    G2, algoG = reduce_2d(G)
    C2, algoC = reduce_2d(C)

    # ---- plot side by side ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    names7 = [t.replace("JointType", "") for t in JOINT_TYPES_7]
    for ax, XY, title, algo in [(axes[0], G2, "Level-1 · geometry (encoder space)", algoG),
                                (axes[1], C2, "Level-2 · context (after Context GNN)", algoC)]:
        if args.color_by == "joint_type":
            cmap = plt.get_cmap("tab10")
            # 0 = no joint (grey) drawn first (bottom)
            mask0 = L == 0
            ax.scatter(XY[mask0, 0], XY[mask0, 1], s=8, c="#cccccc", label="no joint", alpha=.4)
            # common types next, rare types LAST (on top) with edge + bigger marker
            common = [1, 2, 3, 4]       # Rigid, Revolute, Slider, Cylindrical
            rare = [5, 6, 7]            # PinSlot, Planar, Ball
            for t in common:
                m = L == t
                if m.any():
                    ax.scatter(XY[m, 0], XY[m, 1], s=15, color=cmap((t - 1) % 10),
                               label=names7[t - 1], alpha=.8)
            for t in rare:
                m = L == t
                if m.any():
                    ax.scatter(XY[m, 0], XY[m, 1], s=55, color=cmap((t - 1) % 10),
                               label=names7[t - 1], alpha=.95,
                               edgecolors="black", linewidths=0.6, zorder=5)
        elif args.color_by == "participation":
            for v, c, lab in [(0, "#cccccc", "isolated"), (1, "#d62728", "in joint")]:
                m = L == v
                if m.any():
                    ax.scatter(XY[m, 0], XY[m, 1], s=14, c=c, label=lab, alpha=.75)
        elif args.color_by == "has_axis":
            for v, c, lab in [(0, "#4c78a8", "no axis"), (1, "#f58518", "has axis")]:
                m = L == v
                if m.any():
                    ax.scatter(XY[m, 0], XY[m, 1], s=14, c=c, label=lab, alpha=.75)
        else:  # holes (continuous)
            sc = ax.scatter(XY[:, 0], XY[:, 1], s=14, c=L, cmap="viridis", alpha=.8)
            fig.colorbar(sc, ax=ax, fraction=.046, label="hole count")
        ax.set_title(f"{title}\n[{algo}]", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        if args.color_by in ("joint_type", "participation", "has_axis"):
            ax.legend(fontsize=8, markerscale=1.4, loc="best")
    sep = emb_metrics.get("joint_separability")
    subtitle = f"HGCAN body embeddings — colored by {args.color_by}"
    if sep:
        subtitle += f"   |   joint/no-joint kNN separability: L1={sep[0]:.2f} → L2={sep[1]:.2f}"
    fig.suptitle(subtitle, fontweight="bold")
    fig.tight_layout()

    figs = ROOT / cfg["paths"]["reports"] / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    sfx = f"_{args.tag}" if args.tag else ""
    out = figs / f"embeddings_{args.color_by}{sfx}.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"[fig] {out}")


if __name__ == "__main__":
    main()
