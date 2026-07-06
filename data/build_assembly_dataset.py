"""
data/build_assembly_dataset.py                                       [HGCAN]
assembly.json (+ per-body graphs) -> AssemblyPairData for joint
EXISTENCE + TYPE classification, with BODY-LEVEL nodes and hole features.

NODE MODEL (body-level):
  Every placed body instance is a node -- bodies under occurrences AND bodies on
  the root component. This matches how the dataset references geometry: joints,
  contacts and holes ALL key on `body`, so mapping is a direct lookup with no
  occurrence fallback chain. It also fixes mixed assemblies (e.g. a hinged door +
  flap on a root-component box) where a jointed body had no occurrence and was
  previously dropped.

  world pose per body : owning occurrence's chain_transform (identity for root bodies)
  joints  : geometry_or_origin_{one,two}.entity_one.body -> node  (direct)
  contacts: entity_{one,two}.body -> node                          (direct)
  holes   : per-body 6-dim feature [has_hole, count, mean_d, max_d, through, blind]

ASSEMBLY GRAPH relations (Level-2 message passing):
  0 contact | 1 knn | 2 same-occurrence (two bodies of one part instance)

Candidates = contact + kNN (same at train and inference). recall@candidate is the
honest ceiling; a jointed pair outside it is a MISS, not an injected positive.

Probe:
  python -m data.build_assembly_dataset  path/to/<assembly>/assembly.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from data.assembly_graph import (
    tree_depths_and_parents, chain_transform, chain_origin,
    _body_filename, _entity_anchor, _label_point,
    JOINT_TYPES, JT_TO_IDX, KNN_K,
)
from data.step_graph import step_to_graph  # noqa: F401  (default loader)

# --------------------------------------------------------------- (K+1) classes
NO_JOINT = "NoJoint"
JOINT_TYPES_K1 = [NO_JOINT] + list(JOINT_TYPES)
NUM_CLASSES = len(JOINT_TYPES_K1)

# --------------------------------------------------------------- assembly relations
A_CONTACT, A_KNN, A_SAMEOCC = 0, 1, 2
NUM_ASM_RELATIONS = 3

HOLE_DIM = 6


def type_to_class(joint_type):
    j = JT_TO_IDX.get(joint_type)
    return None if j is None else j + 1


class AssemblyPairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key in ("ent_edge_index",):
            return self.x_ent.size(0)
        if key in ("pair_index", "asm_edge_index", "joint_occ_pairs", "ent_to_occ"):
            return self.num_occ
        if key in ("joint_pos_i", "joint_pos_j"):
            return self.x_ent.size(0)
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ("ent_edge_index", "asm_edge_index", "pair_index", "joint_occ_pairs"):
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)


# ============================================================ body-node enumeration
def _enumerate_body_nodes(doc):
    """All placed body instances -> [(body_uuid, owner_occ_or_None)]. Dedup by uuid."""
    occs = doc.get("occurrences") or {}
    root = doc.get("root") or {}
    seen, nodes = set(), []
    for oid in sorted(occs.keys()):
        for buid in ((occs[oid] or {}).get("bodies") or {}):
            if buid not in seen:
                seen.add(buid); nodes.append((buid, oid))
    for buid in (root.get("bodies") or {}):
        if buid not in seen:
            seen.add(buid); nodes.append((buid, None))
    return nodes


# ============================================================ node/entity load
def _load_nodes(doc, body_graph_loader):
    occs = doc.get("occurrences") or {}
    bodies = doc.get("bodies") or {}
    _, parents = tree_depths_and_parents(doc.get("tree", {}))

    body_nodes = _enumerate_body_nodes(doc)
    if len(body_nodes) < 2:
        raise ValueError("fewer than 2 bodies")
    idx = {buid: i for i, (buid, _) in enumerate(body_nodes)}
    N = len(body_nodes)
    owner = [occ for _, occ in body_nodes]

    world_R, world_o = {}, {}
    for i, (buid, occ) in enumerate(body_nodes):
        if occ is not None:
            world_R[i], world_o[i] = chain_transform(occ, parents, occs)
        else:
            world_R[i], world_o[i] = np.eye(3), np.zeros(3)

    ent_x, e_src, e_dst, e_rel = [], [], [], []
    ent_to_occ, ent_axis, ent_valid, node_type = [], [], [], []
    slice_of, occ_samp_pts, occ_samp_eid = {}, {}, {}
    occ_has_geom = []
    m = 0
    for i, (buid, occ) in enumerate(body_nodes):
        start = m
        g = body_graph_loader(buid, _body_filename(buid, bodies.get(buid, {})))
        if g is not None and g.num_nodes > 0:
            f = g.x.size(0)
            ent_x.append(g.x.numpy())
            ei = g.edge_index.numpy() + m
            e_src += ei[0].tolist(); e_dst += ei[1].tolist()
            e_rel += g.edge_type.numpy().tolist()
            ent_to_occ += [i] * f
            ent_axis.append(g.entity_axis.numpy())
            ent_valid.append(g.entity_axis_valid.numpy())
            node_type.append(g.node_type.numpy())
            if hasattr(g, "entity_samples") and g.entity_samples.shape[0]:
                occ_samp_pts[i] = g.entity_samples.numpy()
                occ_samp_eid[i] = g.entity_sample_eid.numpy()
            else:
                occ_samp_pts[i] = np.zeros((0, 3), np.float32)
                occ_samp_eid[i] = np.zeros((0,), np.int64)
            m += f
        else:
            occ_samp_pts[i] = np.zeros((0, 3), np.float32)
            occ_samp_eid[i] = np.zeros((0,), np.int64)
        slice_of[i] = (start, m)
        occ_has_geom.append(m > start)

    if m == 0:
        raise ValueError("no entities loaded for any body")

    # world COM per body (body-local com transformed by owner occurrence pose)
    P = np.zeros((N, 3))
    for i, (buid, occ) in enumerate(body_nodes):
        com = ((bodies.get(buid) or {}).get("physical_properties") or {}).get("center_of_mass")
        if com:
            P[i] = world_R[i] @ np.array([com["x"], com["y"], com["z"]]) + world_o[i]
        else:
            P[i] = world_o[i]

    # ---- per-body WORLD-FRAME geometric summary (for explicit CAD pair features) ----
    # node_geom[i] = [centroid_xyz(3) | dominant_axis_dir_xyz(3) | dominant_radius(1) | has_axis(1)]
    # The dominant axis = the (radius-weighted) mean direction of cylindrical faces /
    # circular edges, transformed to WORLD by the body's occurrence pose so axes from
    # different bodies are directly comparable (angle between them is meaningful).
    ent_axis_all = np.concatenate(ent_axis, 0).astype(np.float32) if ent_axis else np.zeros((0, 6), np.float32)
    ent_valid_all = np.concatenate(ent_valid, 0) if ent_valid else np.zeros((0,), bool)
    x_all = np.concatenate(ent_x, 0) if ent_x else np.zeros((0, 0), np.float32)
    node_type_all = np.concatenate(node_type, 0) if node_type else np.zeros((0,), np.int64)
    node_geom = np.zeros((N, 8), np.float32)
    FACE_R_SLOT = 2 + 12 + 3          # cyl_radius channel in the face block
    EDGE_R_SLOT = 2 + 20 + 9 + 2      # radius channel in the edge block
    for i in range(N):
        node_geom[i, 0:3] = P[i]
        s, e = slice_of[i]
        if e <= s:
            continue
        Ri = world_R[i]
        axes = []      # (weight, world_dir)
        for k in range(s, e):
            if not ent_valid_all[k]:
                continue
            d_local = ent_axis_all[k, 3:6]
            n = np.linalg.norm(d_local)
            if n < 1e-8:
                continue
            d_world = Ri @ (d_local / n)
            is_face = (node_type_all[k] == 0)
            r = float(x_all[k, FACE_R_SLOT]) if is_face else float(x_all[k, EDGE_R_SLOT])
            w = max(r, 0.0) + 1e-3     # weight by radius; small floor so any axis counts
            axes.append((w, d_world, r))
        if not axes:
            continue
        # radius-weighted dominant direction (sign-align to the heaviest axis to avoid
        # +d/-d cancellation), and the dominant (max) radius
        ref = max(axes, key=lambda t: t[0])[1]
        acc = np.zeros(3); wsum = 0.0; rmax = 0.0
        for w, d, r in axes:
            if np.dot(d, ref) < 0:
                d = -d
            acc += w * d; wsum += w; rmax = max(rmax, r)
        if wsum > 0:
            v = acc / wsum
            nv = np.linalg.norm(v)
            if nv > 1e-8:
                node_geom[i, 3:6] = v / nv
                node_geom[i, 6] = rmax
                node_geom[i, 7] = 1.0

    return {
        "body_nodes": body_nodes, "idx": idx, "N": N, "owner": owner,
        "x_ent": np.concatenate(ent_x, 0),
        "ent_edge": (e_src, e_dst, e_rel),
        "ent_to_occ": np.asarray(ent_to_occ, np.int64),
        "entity_axis": np.concatenate(ent_axis, 0).astype(np.float32),
        "entity_valid": np.concatenate(ent_valid, 0),
        "node_type": np.concatenate(node_type, 0),
        "slice_of": slice_of, "world_R": world_R, "world_o": world_o,
        "occ_samp_pts": occ_samp_pts, "occ_samp_eid": occ_samp_eid,
        "occ_has_geom": occ_has_geom, "P": P, "node_geom": node_geom,
    }


# ============================================================ hole features
def _hole_features(doc, idx):
    """Per body node: [has_hole, count, mean_diameter, max_diameter, through_ratio, blind_ratio]."""
    H = np.zeros((len(idx), HOLE_DIM), np.float32)
    per = defaultdict(list)
    for h in (doc.get("holes") or []):
        b = (h or {}).get("body")
        if b in idx:
            per[idx[b]].append((float(h.get("diameter", 0.0) or 0.0), h.get("type") or ""))
    for ni, lst in per.items():
        diams = [min(max(d, 0.0), 1e3) for d, _ in lst]
        types = [t for _, t in lst]
        cnt = len(lst)
        through = sum(1 for t in types if "Through" in t)
        blind = sum(1 for t in types if "Blind" in t)
        H[ni] = [1.0, float(cnt),
                 float(np.mean(diams)) if diams else 0.0,
                 float(np.max(diams)) if diams else 0.0,
                 through / cnt if cnt else 0.0,
                 blind / cnt if cnt else 0.0]
    return H


# ============================================================ candidate pairs
def _contact_pairs(doc, idx):
    """Undirected body pairs sharing a contact (entity.body)."""
    pairs = set()
    contacts = doc.get("contacts") or []
    for c in (contacts.values() if isinstance(contacts, dict) else contacts):
        try:
            a = c["entity_one"]["body"]; b = c["entity_two"]["body"]
        except (KeyError, TypeError):
            continue
        if a in idx and b in idx and idx[a] != idx[b]:
            pairs.add(frozenset((idx[a], idx[b])))
    return pairs


def _knn_pairs(P, occ_has_geom, k):
    geo = [i for i in range(len(occ_has_geom)) if occ_has_geom[i]]
    pairs = set()
    if len(geo) <= 1:
        return pairs
    Pg = np.stack([P[i] for i in geo])
    for a_, ia in enumerate(geo):
        order = np.argsort(np.linalg.norm(Pg - Pg[a_], axis=1))[1:k + 1]
        for b_ in order:
            jb = geo[int(b_)]
            if ia != jb:
                pairs.add(frozenset((ia, jb)))
    return pairs


def _assembly_relations(doc, idx, P, occ_has_geom, owner, knn_k):
    """Level-2 message-passing graph: contact | knn | same-occurrence."""
    asrc, adst, arel = [], [], []
    def add(ia, ib, r):
        if ia != ib:
            asrc.extend([ia, ib]); adst.extend([ib, ia]); arel.extend([r, r])
    for key in _contact_pairs(doc, idx):
        if len(key) == 2:
            i, j = tuple(key); add(i, j, A_CONTACT)
    for key in _knn_pairs(P, occ_has_geom, knn_k):
        if len(key) == 2:
            i, j = tuple(key); add(i, j, A_KNN)
    # same-occurrence: bodies sharing one occurrence instance
    groups = defaultdict(list)
    for ni, occ in enumerate(owner):
        if occ is not None:
            groups[occ].append(ni)
    for members in groups.values():
        for a_ in range(len(members)):
            for b_ in range(a_ + 1, len(members)):
                add(members[a_], members[b_], A_SAMEOCC)
    return asrc, adst, arel


# ============================================================ joint labels
def _side_body(j, side, idx, doc):
    """Resolve one joint side to a body node: prefer entity.body, fall back to a
    single-body occurrence."""
    geo = j.get(f"geometry_or_origin_{side}") or {}
    ent = geo.get("entity_one") or {}
    b = ent.get("body")
    if b and b in idx:
        return idx[b]
    occ = j.get(f"occurrence_{side}")
    if occ:
        bl = list(((doc.get("occurrences") or {}).get(occ) or {}).get("bodies") or {})
        if len(bl) == 1 and bl[0] in idx:
            return idx[bl[0]]
    return None


def _jointed_pairs(doc, idx):
    out = {}
    joints = {**(doc.get("joints") or {}), **(doc.get("as_built_joints") or {})}
    for j in joints.values():
        j = j or {}
        cls = type_to_class((j.get("joint_motion") or {}).get("joint_type"))
        if cls is None:
            continue
        b1 = _side_body(j, "one", idx, doc)
        b2 = _side_body(j, "two", idx, doc)
        if b1 is None or b2 is None or b1 == b2:
            continue
        a1 = _entity_anchor(j.get("geometry_or_origin_one"))
        a2 = _entity_anchor(j.get("geometry_or_origin_two"))
        out[frozenset((b1, b2))] = (cls, (b1, a1), (b2, a2))
    return out


# ============================================================ main builder
def build_pair_dataset(json_path, body_graph_loader, knn_k=KNN_K, easy_per_pos=3, seed=42):
    rng = np.random.RandomState(seed)
    doc = json.loads(Path(json_path).read_text(encoding="utf-8"))
    nd = _load_nodes(doc, body_graph_loader)
    idx, N = nd["idx"], nd["N"]

    contact = {p for p in _contact_pairs(doc, idx) if len(p) == 2}
    knn = {p for p in _knn_pairs(nd["P"], nd["occ_has_geom"], knn_k) if len(p) == 2}
    candidates = {p for p in (contact | knn) if len(p) == 2}
    jointed = {p: v for p, v in _jointed_pairs(doc, idx).items() if len(p) == 2}

    n_joint = len(jointed)
    caught = [p for p in jointed if p in candidates]
    missed = [p for p in jointed if p not in candidates]
    recall = (len(caught) / n_joint) if n_joint else float("nan")

    positives = [p for p in candidates if p in jointed]
    hard_negs = [p for p in (contact - set(jointed))]
    easy_pool = [p for p in ((knn - contact) - set(jointed))]
    n_easy = min(len(easy_pool), max(easy_per_pos * max(len(positives), 1), easy_per_pos))
    easy_idx = rng.choice(len(easy_pool), size=n_easy, replace=False) if easy_pool else []
    easy_negs = [easy_pool[k] for k in np.atleast_1d(easy_idx).astype(int)] if len(easy_pool) else []

    pairs = positives + hard_negs + easy_negs

    pair_i, pair_j, pair_lbl, is_contact, is_knn = [], [], [], [], []
    for p in pairs:
        t = sorted(tuple(p))
        if len(t) != 2:
            continue
        i, j = t
        pair_i.append(i); pair_j.append(j)
        pair_lbl.append(jointed[p][0] if p in jointed else 0)
        is_contact.append(p in contact); is_knn.append(p in knn)

    jo_i, jo_j, jt = [], [], []
    jpos_i, jpos_j, residuals = [], [], []
    for p in positives:
        cls, (ia, a1), (ib, a2) = jointed[p]
        if a1 is None or a2 is None:
            continue
        def side(node_i, anchor):
            s, e = nd["slice_of"][node_i]
            r = _label_point(anchor["point"], nd["world_R"][node_i], nd["world_o"][node_i],
                             nd["occ_samp_pts"][node_i], nd["occ_samp_eid"][node_i],
                             nd["node_type"][s:e], anchor["want_edge"])
            return r, s
        r1, s1 = side(ia, a1)
        r2, s2 = side(ib, a2)
        if r1 is None or r2 is None:
            continue
        _, pos1, d1, f1 = r1
        _, pos2, d2, f2 = r2
        jo_i.append(ia); jo_j.append(ib); jt.append(cls)
        jpos_i.append(torch.tensor([s1 + q for q in pos1], dtype=torch.long))
        jpos_j.append(torch.tensor([s2 + q for q in pos2], dtype=torch.long))
        residuals.append((d1, f1)); residuals.append((d2, f2))

    asrc, adst, arel = _assembly_relations(doc, idx, nd["P"], nd["occ_has_geom"],
                                           nd["owner"], knn_k)
    node_hole = _hole_features(doc, idx)

    e_src, e_dst, e_rel = nd["ent_edge"]
    d = AssemblyPairData(x_ent=torch.from_numpy(nd["x_ent"]).float())
    d.ent_edge_index = (torch.tensor([e_src, e_dst], dtype=torch.long)
                        if e_src else torch.zeros((2, 0), dtype=torch.long))
    d.ent_edge_type = torch.tensor(e_rel, dtype=torch.long) if e_rel else torch.zeros((0,), dtype=torch.long)
    d.ent_to_occ = torch.from_numpy(nd["ent_to_occ"])
    d.node_type = torch.from_numpy(nd["node_type"])
    d.entity_axis = torch.from_numpy(nd["entity_axis"])
    d.entity_axis_valid = torch.from_numpy(nd["entity_valid"])
    d.num_occ = N
    d.occ_has_geom = torch.tensor(nd["occ_has_geom"], dtype=torch.bool)
    d.node_hole = torch.from_numpy(node_hole)
    d.node_geom = torch.from_numpy(nd["node_geom"])   # [N,8] world centroid+axis+radius

    d.asm_edge_index = (torch.tensor([asrc, adst], dtype=torch.long)
                        if asrc else torch.zeros((2, 0), dtype=torch.long))
    d.asm_edge_type = torch.tensor(arel, dtype=torch.long) if arel else torch.zeros((0,), dtype=torch.long)

    d.pair_index = (torch.tensor([pair_i, pair_j], dtype=torch.long)
                    if pair_i else torch.zeros((2, 0), dtype=torch.long))
    d.pair_label = torch.tensor(pair_lbl, dtype=torch.long)
    d.pair_is_contact = torch.tensor(is_contact, dtype=torch.bool)
    d.pair_is_knn = torch.tensor(is_knn, dtype=torch.bool)

    d.joint_occ_pairs = (torch.tensor([jo_i, jo_j], dtype=torch.long)
                         if jo_i else torch.zeros((2, 0), dtype=torch.long))
    d.joint_type = torch.tensor(jt, dtype=torch.long)
    d.joint_pos_i = jpos_i
    d.joint_pos_j = jpos_j
    d.assembly_id = Path(json_path).parent.name
    d.body_uuids = [b for b, _ in nd["body_nodes"]]

    report = {
        "assembly_id": d.assembly_id, "n_occ": N, "n_joints": n_joint,
        "recall_at_candidate": recall, "n_caught": len(caught), "n_missed": len(missed),
        "n_candidates": len(candidates), "n_pairs_kept": len(pairs),
        "n_pos": len(positives), "n_hard_neg": len(hard_negs), "n_easy_neg": len(easy_negs),
        "n_with_holes": int((node_hole[:, 0] > 0).sum()),
        "class_hist": {JOINT_TYPES_K1[c]: int((d.pair_label == c).sum()) for c in range(NUM_CLASSES)},
        "residuals": residuals,
    }
    return d, report


# ============================================================ reporting / gate
def label_quality_report_pairs(reports):
    from collections import Counter
    if not reports:
        print("no assemblies processed."); return
    rec = np.array([r["recall_at_candidate"] for r in reports
                    if not np.isnan(r["recall_at_candidate"])])
    tot_pos = sum(r["n_pos"] for r in reports)
    tot_pairs = sum(r["n_pairs_kept"] for r in reports)
    cls = Counter()
    for r in reports:
        for k, v in r["class_hist"].items():
            cls[k] += v
    res = [x for r in reports for x in r["residuals"]]
    print(f"assemblies          : {len(reports)}")
    if rec.size:
        print(f"recall@candidate    : mean {rec.mean()*100:.1f}%  min {rec.min()*100:.1f}%")
    print(f"pairs kept          : {tot_pairs}  (positives {tot_pos}, "
          f"pos:neg = 1:{(tot_pairs-tot_pos)/max(tot_pos,1):.1f})")
    print(f"bodies with holes   : {sum(r.get('n_with_holes',0) for r in reports)}")
    print("class histogram     :")
    for k in JOINT_TYPES_K1:
        print(f"    {k:22s} {cls.get(k,0)}")
    if res:
        dist = np.array([d for d, _ in res], float)
        print(f"entity residual (mm): p50 {np.percentile(dist,50):.3f}  "
              f"p90 {np.percentile(dist,90):.3f}  within5 {(dist<5.0).mean()*100:.1f}%")
    if rec.size and rec.mean() < 0.90:
        print("VERDICT: recall@candidate < 90% -> widen candidate generator.")
    elif rec.size:
        print("VERDICT: candidate recall healthy -> proceed. Rigid dominates; use class weighting.")


if __name__ == "__main__":
    path = [a for a in sys.argv[1:] if not a.startswith("--")][0]
    base = Path(path).parent
    def loader(buid, fname):
        p = base / fname
        if not p.exists():
            p = base / f"{buid}.step"
        return step_to_graph(str(p)) if p.exists() else None
    data, rep = build_pair_dataset(path, loader)
    print(f"\n[{rep['assembly_id']}] bodies={rep['n_occ']} joints={rep['n_joints']} "
          f"pairs={rep['n_pairs_kept']} pos={rep['n_pos']} holes={rep['n_with_holes']}")
    print(f"recall@candidate={rep['recall_at_candidate']}")
    print("class_hist:", {k: v for k, v in rep["class_hist"].items() if v})
    print()
    label_quality_report_pairs([rep])
