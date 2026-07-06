"""
data/step_graph.py                                                   [HGCAN]
STEP body -> heterogeneous (face + edge) B-Rep graph -> PyG Data.

Design (from our analysis):
  * Faces AND edges are first-class graph nodes. Many revolute/cylindrical joints
    are authored on a circular EDGE (a hole rim), which must exist as a node.
  * ANALYTIC features only (no UV-grids). Surface/curve type, area, EDGE LENGTH,
    and RADIUS are the high-value signals; UV-grids were dropped deliberately.
  * Convexity (convex/concave/smooth) is carried as EDGE-TYPE relations, so the
    relation-aware encoder can use it (the earlier model ignored it).
  * Each entity's analytic AXIS + RADIUS come only from the part's OWN geometry
    (BRepAdaptor), never from assembly.json joint fields -> leakage-safe.

Output PyG Data:
  x                 (N, NODE_FEAT_DIM)  unified face+edge features
  node_type         (N,)   0 = face, 1 = edge
  edge_index        (2, E) relational graph (both directions)
  edge_type         (E,)   0 convex / 1 concave / 2 smooth (face-face)
                           3 incidence (face-edge)
  entity_axis       (N, 6) [loc_xyz | dir_xyz] from the entity's own geometry
  entity_axis_valid (N,)   bool; True where an analytic axis exists
  entity_samples    (S, 3) body-local surface sample points (for label mapping)
  entity_sample_eid (S,)   entity id (0..N-1) each sample belongs to
  n_faces, n_edges

Feature layout (NODE_FEAT_DIM):
  [0:2]   node-type flag           [is_face, is_edge]
  FACE block (zeros on edge nodes):
    surface one-hot(12) | log_area | rel_area | holes(inner loops) | cyl_radius | curv(4)
  EDGE block (zeros on face nodes):
    curve one-hot(9) | log_len | rel_len | radius | closed

Verified occwl/OCC calls are preserved verbatim from the working extractor; only
structure, naming, radius channels, and docs changed. Run the probe first:
  python -m data.step_graph  path/to/<guid>.step
"""
import math
import sys

import numpy as np
import torch
from torch_geometric.data import Data

from occwl.compound import Compound
from occwl.graph import face_adjacency
from occwl.edge_data_extractor import EdgeDataExtractor, EdgeConvexity

ZERO3 = np.zeros(3, np.float32)


class StepGraphError(Exception):
    """Raised when a body cannot be converted. Message = rejection reason."""


# ============================================================ vocabularies
SURF_TYPES = [
    "plane", "cylinder", "cone", "sphere", "torus", "bezier",
    "bspline", "revolution", "extrusion", "offset", "other", "unknown",
]
SURF_TO_IDX = {s: i for i, s in enumerate(SURF_TYPES)}

CURVE_TYPES = [
    "line", "circle", "ellipse", "hyperbola", "parabola",
    "bezier", "bspline", "offset", "other",
]
CURVE_TO_IDX = {c: i for i, c in enumerate(CURVE_TYPES)}

# ---- edge-type relations ----
REL_CONVEX, REL_CONCAVE, REL_SMOOTH, REL_INCIDENCE = 0, 1, 2, 3
NUM_RELATIONS = 4
CONVEXITY_TO_REL = {
    EdgeConvexity.CONVEX: REL_CONVEX,
    EdgeConvexity.CONCAVE: REL_CONCAVE,
    EdgeConvexity.SMOOTH: REL_SMOOTH,
}
SMOOTH_TOL_RADS = 0.0872   # ~5 deg dihedral -> smooth/tangent

# ---- feature layout ----
CURV_SAMPLES = 5           # 5x5 interior UV grid for curvature stats
# face vector = surface one-hot(12) + [log_area, rel_area, holes, cyl_radius](4) + curv(4)
FACE_BLOCK = len(SURF_TYPES) + 4 + 4    # 12 + 4 + 4 = 20
# edge vector = curve one-hot(9) + [log_len, rel_len, radius, closed](4)
EDGE_BLOCK = len(CURVE_TYPES) + 4       # 9 + 4 = 13
NODE_FEAT_DIM = 2 + FACE_BLOCK + EDGE_BLOCK   # 2 + 20 + 13 = 35

SAMPLE_FACE_N = 3          # 3x3 interior points per face
SAMPLE_EDGE_N = 5          # 5 points along each edge


# ============================================================ analytic geometry
def _face_geom(face):
    """Axis + radius from the face's OWN surface (leakage-safe).
    Returns (valid_axis, loc[3], dir[3], radius). radius>0 only for cylinders."""
    try:
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone
        ad = BRepAdaptor_Surface(face.topods_shape())
        t = ad.GetType()
        radius = 0.0
        if t == GeomAbs_Plane:
            ax = ad.Plane().Axis()
        elif t == GeomAbs_Cylinder:
            cyl = ad.Cylinder(); ax = cyl.Axis(); radius = float(cyl.Radius())
        elif t == GeomAbs_Cone:
            ax = ad.Cone().Axis()
        else:
            return False, ZERO3, ZERO3, 0.0
        loc, d = ax.Location(), ax.Direction()
        return (True,
                np.array([loc.X(), loc.Y(), loc.Z()], np.float32),
                np.array([d.X(), d.Y(), d.Z()], np.float32),
                radius)
    except Exception:
        return False, ZERO3, ZERO3, 0.0


def _edge_geom(edge):
    """Axis + radius from the edge's OWN curve. circle -> (centre, normal, r);
    line -> (point, dir, 0). Returns (valid_axis, loc[3], dir[3], radius)."""
    try:
        from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
        from OCC.Core.GeomAbs import GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse
        ad = BRepAdaptor_Curve(edge.topods_shape())
        t = ad.GetType()
        radius = 0.0
        if t == GeomAbs_Circle:
            c = ad.Circle(); ax = c.Axis(); radius = float(c.Radius())
            loc, d = ax.Location(), ax.Direction()
        elif t == GeomAbs_Ellipse:
            c = ad.Ellipse(); ax = c.Axis(); radius = float(c.MajorRadius())
            loc, d = ax.Location(), ax.Direction()
        elif t == GeomAbs_Line:
            ln = ad.Line(); loc, d = ln.Location(), ln.Direction()
        else:
            return False, ZERO3, ZERO3, 0.0
        return (True,
                np.array([loc.X(), loc.Y(), loc.Z()], np.float32),
                np.array([d.X(), d.Y(), d.Z()], np.float32),
                radius)
    except Exception:
        return False, ZERO3, ZERO3, 0.0


# ============================================================ features
def _face_features(face, total_area, cyl_radius):
    """FACE_BLOCK-dim face vector: surface one-hot + size/hole/radius/curvature."""
    onehot = np.zeros(len(SURF_TYPES), np.float32)
    onehot[SURF_TO_IDX.get(face.surface_type(), SURF_TO_IDX["unknown"])] = 1.0

    area = max(face.area(), 0.0)
    log_area = math.log(area + 1e-9)
    rel_area = area / (total_area + 1e-9)
    holes = float(face.num_wires() - 1)               # inner loops = through-holes
    radius = max(min(cyl_radius, 1e3), 0.0)            # cylinder radius (cm), 0 else

    bounds = face.uv_bounds()
    (umin, vmin), (umax, vmax) = bounds.min_point(), bounds.max_point()
    gauss, mean = [], []
    for u in np.linspace(umin, umax, CURV_SAMPLES + 2)[1:-1]:
        for v in np.linspace(vmin, vmax, CURV_SAMPLES + 2)[1:-1]:
            try:
                gauss.append(face.gaussian_curvature((u, v)))
                mean.append(face.mean_curvature((u, v)))
            except Exception:
                pass
    gauss = np.nan_to_num(np.asarray(gauss, np.float32))
    mean = np.nan_to_num(np.asarray(mean, np.float32))
    curv = [
        float(gauss.mean()) if gauss.size else 0.0,
        float(gauss.std()) if gauss.size else 0.0,
        float(mean.mean()) if mean.size else 0.0,
        float(mean.std()) if mean.size else 0.0,
    ]
    curv = [max(-1e3, min(1e3, c)) for c in curv]
    return np.concatenate([onehot, [log_area, rel_area, holes, radius], curv]).astype(np.float32)


def _edge_features(edge, total_len, radius):
    """EDGE_BLOCK-dim edge vector: curve one-hot + length + radius + closed flag."""
    onehot = np.zeros(len(CURVE_TYPES), np.float32)
    try:
        ct = edge.curve_type()
    except Exception:
        ct = "other"
    onehot[CURVE_TO_IDX.get(ct, CURVE_TO_IDX["other"])] = 1.0

    try:
        length = max(float(edge.length()), 0.0)
    except Exception:
        length = 0.0
    log_len = math.log(length + 1e-9)
    rel_len = length / (total_len + 1e-9)
    rad = max(min(radius, 1e3), 0.0)                  # circle/ellipse radius (cm)

    closed = 0.0
    for meth in ("closed_edge", "closed_curve", "closed"):
        if hasattr(edge, meth):
            try:
                closed = float(bool(getattr(edge, meth)())); break
            except Exception:
                pass
    return np.concatenate([onehot, [log_len, rel_len, rad, closed]]).astype(np.float32)


# ============================================================ surface samples
def _face_samples(face):
    try:
        b = face.uv_bounds()
        (umin, vmin), (umax, vmax) = b.min_point(), b.max_point()
        pts = []
        for u in np.linspace(umin, umax, SAMPLE_FACE_N + 2)[1:-1]:
            for v in np.linspace(vmin, vmax, SAMPLE_FACE_N + 2)[1:-1]:
                try:
                    pts.append(np.asarray(face.point((float(u), float(v))), np.float32))
                except Exception:
                    pass
        return np.stack(pts) if pts else np.zeros((0, 3), np.float32)
    except Exception:
        return np.zeros((0, 3), np.float32)


def _edge_samples(edge):
    try:
        from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
        ad = BRepAdaptor_Curve(edge.topods_shape())
        t0, t1 = ad.FirstParameter(), ad.LastParameter()
        pts = []
        for s in np.linspace(t0, t1, SAMPLE_EDGE_N):
            p = ad.Value(float(s))
            pts.append(np.array([p.X(), p.Y(), p.Z()], np.float32))
        return np.stack(pts) if pts else np.zeros((0, 3), np.float32)
    except Exception:
        return np.zeros((0, 3), np.float32)


# ============================================================ edge<->face ancestry
def _edge_face_map(faces):
    """For each TopoDS edge, the face indices it bounds. LINEAR (O(E+F)):
    walk each face's edges once, accumulate parents in a shape-hash-keyed dict."""
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_EDGE
    from OCC.Core.TopoDS import topods
    from occwl.edge import Edge

    edge_rep, edge_faces = {}, {}
    for k, f in enumerate(faces):
        exp = TopExp_Explorer(f.topods_shape(), TopAbs_EDGE)
        while exp.More():
            e = topods.Edge(exp.Current())
            key = e.__hash__()
            if key not in edge_rep:
                edge_rep[key] = e; edge_faces[key] = set()
            edge_faces[key].add(k)
            exp.Next()
    return {ei: (Edge(e), sorted(edge_faces[key]))
            for ei, (key, e) in enumerate(edge_rep.items())}


# ============================================================ main builder
def solid_to_graph(shape) -> Data:
    """occwl Solid/Shell/Compound -> heterogeneous PyG Data (faces + edges)."""
    try:
        nxg = face_adjacency(shape, self_loops=False)
    except RuntimeError as e:
        raise StepGraphError(f"non-manifold: {e}")
    except AssertionError as e:
        # occwl find_left_and_right_faces / is_left_of asserts on healed or
        # imported STEP with inconsistent edge<->face topology. Skip this body.
        raise StepGraphError(f"occwl topology assert in face_adjacency: {e}")
    if nxg is None:
        raise StepGraphError("open/non-manifold shell (face_adjacency returned None)")
    if nxg.number_of_nodes() == 0:
        raise StepGraphError("zero faces")

    faces = [nxg.nodes[i]["face"] for i in sorted(nxg.nodes)]
    F = len(faces)
    total_area = sum(max(f.area(), 0.0) for f in faces)

    feats = np.zeros((F, NODE_FEAT_DIM), np.float32)
    node_type = np.zeros(F, np.int64)
    axis = np.zeros((F, 6), np.float32)
    axis_ok = np.zeros(F, bool)
    face_samp_pts, face_samp_eid = [], []
    for k, f in enumerate(faces):
        ok, loc, d, radius = _face_geom(f)
        feats[k, 0] = 1.0                                   # is_face
        feats[k, 2:2 + FACE_BLOCK] = _face_features(f, total_area, radius)
        axis_ok[k] = ok
        if ok:
            axis[k, :3], axis[k, 3:] = loc, d
        sp = _face_samples(f)
        if sp.shape[0]:
            face_samp_pts.append(sp); face_samp_eid.append(np.full(sp.shape[0], k, np.int64))

    # ---- face-face convexity relations ----
    src, dst, rel = [], [], []
    seen = {}
    for i, j, attrs in nxg.edges(data=True):
        key = (min(i, j), max(i, j))
        if key in seen:
            r = seen[key]
        else:
            # occwl can assert "Edge doesn't belong to face" on healed/imported
            # STEP geometry. One un-classifiable edge must not fail the whole body
            # -> degrade that edge's convexity to SMOOTH (neutral) and continue.
            try:
                ext = EdgeDataExtractor(attrs["edge"], [faces[i], faces[j]], num_samples=10)
                r = (CONVEXITY_TO_REL[ext.edge_convexity(SMOOTH_TOL_RADS)]
                     if ext.good else REL_SMOOTH)
            except Exception:
                r = REL_SMOOTH
            seen[key] = r
        src.append(i); dst.append(j); rel.append(r)
    id2idx = {nid: k for k, nid in enumerate(sorted(nxg.nodes))}
    src = [id2idx[s] for s in src]; dst = [id2idx[d] for d in dst]

    # ---- edges as nodes + face-edge incidence ----
    efmap = _edge_face_map(faces)
    edge_objs, edge_fidx = [], []
    for _, (eobj, fidx) in efmap.items():
        edge_objs.append(eobj); edge_fidx.append(fidx)
    E = len(edge_objs)
    total_len = sum(max(getattr(e, "length", lambda: 0.0)(), 0.0) for e in edge_objs) if E else 0.0

    edge_feats = np.zeros((E, NODE_FEAT_DIM), np.float32)
    edge_axis = np.zeros((E, 6), np.float32)
    edge_axis_ok = np.zeros(E, bool)
    edge_samp_pts, edge_samp_eid = [], []
    for k, e in enumerate(edge_objs):
        ok, loc, d, radius = _edge_geom(e)
        edge_feats[k, 1] = 1.0                              # is_edge
        edge_feats[k, 2 + FACE_BLOCK:] = _edge_features(e, total_len, radius)
        edge_axis_ok[k] = ok
        if ok:
            edge_axis[k, :3], edge_axis[k, 3:] = loc, d
        sp = _edge_samples(e)
        if sp.shape[0]:
            edge_samp_pts.append(sp); edge_samp_eid.append(np.full(sp.shape[0], F + k, np.int64))

    for ei, fidx in enumerate(edge_fidx):
        enode = F + ei
        for fi in fidx:
            src += [fi, enode]; dst += [enode, fi]
            rel += [REL_INCIDENCE, REL_INCIDENCE]

    # ---- assemble ----
    x = np.concatenate([feats, edge_feats], axis=0)
    nt = np.concatenate([node_type, np.ones(E, np.int64)])
    ent_axis = np.concatenate([axis, edge_axis], axis=0)
    ent_ok = np.concatenate([axis_ok, edge_axis_ok], axis=0)
    if not np.isfinite(x).all():
        raise StepGraphError("non-finite node features")

    edge_index = (torch.tensor([src, dst], dtype=torch.long) if src
                  else torch.zeros((2, 0), dtype=torch.long))
    edge_type = (torch.tensor(rel, dtype=torch.long) if rel
                 else torch.zeros((0,), dtype=torch.long))

    data = Data(x=torch.from_numpy(x), edge_index=edge_index, edge_type=edge_type)
    data.node_type = torch.from_numpy(nt)
    data.entity_axis = torch.from_numpy(ent_axis)
    data.entity_axis_valid = torch.from_numpy(ent_ok)
    data.n_faces = F
    data.n_edges = E
    all_sp = face_samp_pts + edge_samp_pts
    all_eid = face_samp_eid + edge_samp_eid
    if all_sp:
        data.entity_samples = torch.from_numpy(np.concatenate(all_sp, 0))
        data.entity_sample_eid = torch.from_numpy(np.concatenate(all_eid, 0))
    else:
        data.entity_samples = torch.zeros((0, 3), dtype=torch.float32)
        data.entity_sample_eid = torch.zeros((0,), dtype=torch.long)
    return data


def _merge_graphs(graphs):
    """Concatenate per-solid graphs into ONE disconnected (islands) graph."""
    if len(graphs) == 1:
        return graphs[0]
    xs, nts, axes, oks = [], [], [], []
    e_src, e_dst, e_rel = [], [], []
    sp_pts, sp_eid = [], []
    n_off = 0; F_tot = E_tot = 0
    for g in graphs:
        n = g.x.size(0)
        xs.append(g.x); nts.append(g.node_type)
        axes.append(g.entity_axis); oks.append(g.entity_axis_valid)
        ei = g.edge_index + n_off
        e_src += ei[0].tolist(); e_dst += ei[1].tolist(); e_rel += g.edge_type.tolist()
        if g.entity_samples.shape[0]:
            sp_pts.append(g.entity_samples); sp_eid.append(g.entity_sample_eid + n_off)
        n_off += n; F_tot += int(g.n_faces); E_tot += int(g.n_edges)
    data = Data(
        x=torch.cat(xs, 0),
        edge_index=(torch.tensor([e_src, e_dst], dtype=torch.long) if e_src
                    else torch.zeros((2, 0), dtype=torch.long)),
        edge_type=(torch.tensor(e_rel, dtype=torch.long) if e_rel
                   else torch.zeros((0,), dtype=torch.long)),
    )
    data.node_type = torch.cat(nts, 0)
    data.entity_axis = torch.cat(axes, 0)
    data.entity_axis_valid = torch.cat(oks, 0)
    data.n_faces = F_tot; data.n_edges = E_tot
    data.entity_samples = torch.cat(sp_pts, 0) if sp_pts else torch.zeros((0, 3))
    data.entity_sample_eid = (torch.cat(sp_eid, 0) if sp_eid
                              else torch.zeros((0,), dtype=torch.long))
    return data


def step_to_graph(step_path: str) -> Data:
    """One <guid>.step -> heterogeneous PyG Data. All solids kept as islands."""
    comp = Compound.load_from_step(str(step_path))
    if comp is None:
        raise StepGraphError("STEP read failed")
    solids = list(comp.solids())
    if len(solids) >= 1:
        graphs = []
        for s in solids:
            try:
                graphs.append(solid_to_graph(s))
            except StepGraphError:
                continue
        if not graphs:
            raise StepGraphError("no usable solids")
        return _merge_graphs(graphs)
    if sum(1 for _ in comp.faces()) == 0:
        raise StepGraphError("no faces in STEP file (empty transfer)")
    return solid_to_graph(comp)


if __name__ == "__main__":
    path = [a for a in sys.argv[1:] if not a.startswith("--")][0]
    g = step_to_graph(path)
    nfaces = int((g.node_type == 0).sum()); nedges = int((g.node_type == 1).sum())
    rels = torch.bincount(g.edge_type, minlength=NUM_RELATIONS).tolist()
    axf = int(g.entity_axis_valid[g.node_type == 0].sum())
    axe = int(g.entity_axis_valid[g.node_type == 1].sum())
    print(f"nodes           : {g.num_nodes}  ({nfaces} faces + {nedges} edges)")
    print(f"x               : {tuple(g.x.shape)}   (NODE_FEAT_DIM={NODE_FEAT_DIM})")
    print(f"edge_index      : {tuple(g.edge_index.shape)}")
    print(f"relations       : convex={rels[0]} concave={rels[1]} smooth={rels[2]} incidence={rels[3]}")
    print(f"analytic axes   : faces {axf}/{nfaces}   edges {axe}/{nedges}")
    # spot-check radius channels: edge radius slot = 2 + FACE_BLOCK + len(CURVE_TYPES) + 2
    er_slot = 2 + FACE_BLOCK + len(CURVE_TYPES) + 2
    fr_slot = 2 + len(SURF_TYPES) + 3
    circ = g.x[g.node_type == 1][:, er_slot]
    cyl = g.x[g.node_type == 0][:, fr_slot]
    print(f"edge radius >0  : {int((circ > 0).sum())}/{nedges}  (circle/ellipse rims)")
    print(f"cyl radius  >0  : {int((cyl > 0).sum())}/{nfaces}  (cylindrical faces)")
    print("\nSANITY: analytic axes should be HIGH; radius>0 counts should match the "
          "number of cylindrical faces / circular edges. If axes are ~0, the OCC "
          "adaptor import path differs on your pythonocc -> fix _face_geom/_edge_geom.")
