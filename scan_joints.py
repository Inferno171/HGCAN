"""
scan_joints.py                                                       [HGCAN]
JSON-only dataset audit (no occwl):
  1. how common ground vs part-to-part joints are (scoping)
  2. the part-to-part joint-TYPE histogram (class balance)
  3. joint NAMES survey + name-token -> type correlation
     (to test whether designer-given joint names carry signal worth fusing)

Writes reports/joint_scan.csv (per assembly) and reports/joint_names.csv
(every joint: name, type, category) plus a console summary.

Run (from HGCAN/ project root):
  python scan_joints.py --config configs/base.yaml
  python scan_joints.py --config configs/base.yaml --limit 500
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def classify(doc):
    """Return list of (name, joint_type, category) for every joint, where
    category in {part2part, ground, self}. Uses entity.body / entity.occurrence
    fallback so 'occurrence_two=None but geometry_or_origin_two.entity.body set'
    (a real part-to-part joint) is classified correctly, not as ground."""
    occs = doc.get("occurrences") or {}
    root = doc.get("root") or {}
    # body -> owning occurrence (for resolving the 2nd party by body)
    body2occ = {}
    for oid, o in occs.items():
        for b in ((o or {}).get("bodies") or {}):
            body2occ[b] = oid
    root_bodies = set((root.get("bodies") or {}).keys())

    def side_party(j, side):
        """A hashable identity for one joint side: occurrence uuid, or ('body', uuid)."""
        occ = j.get(f"occurrence_{side}")
        if occ:
            return occ
        ent = ((j.get(f"geometry_or_origin_{side}") or {}).get("entity_one") or {})
        if ent.get("occurrence"):
            return ent["occurrence"]
        b = ent.get("body")
        if b:
            return body2occ.get(b, ("body", b))   # map to occ if possible, else body node
        return None

    joints = {**(doc.get("joints") or {}), **(doc.get("as_built_joints") or {})}
    out = []
    for j in joints.values():
        j = j or {}
        name = j.get("name", "")
        jt = (j.get("joint_motion") or {}).get("joint_type", "UNKNOWN")
        p1, p2 = side_party(j, "one"), side_party(j, "two")
        if p1 is None or p2 is None:
            cat = "ground"          # genuinely one-sided
        elif p1 == p2:
            cat = "self"
        else:
            cat = "part2part"
        out.append((name, jt, cat))
    return out


# joint-type short label for readability
def short_type(jt):
    return jt.replace("JointType", "")


# tokenize a joint name into lowercased word tokens
_TOKEN = re.compile(r"[a-zA-Z]+")
def tokens(name):
    return [t.lower() for t in _TOKEN.findall(name or "")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    roots = cfg["paths"]["dataset_roots"]
    reports_dir = ROOT / cfg["paths"]["reports"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    asm_csv = reports_dir / "joint_scan.csv"
    name_csv = reports_dir / "joint_names.csv"

    # assembly-level tallies
    n_total = n_no_joint = n_has_p2p = n_ground_only = 0
    tot_p2p = tot_ground = tot_self = 0
    jt_p2p = Counter()

    # name survey
    name_counter = Counter()                 # full joint name -> count (part2part only)
    token_type = defaultdict(Counter)        # name-token -> Counter(joint_type)
    type_token = defaultdict(Counter)        # joint_type -> Counter(name-token)
    n_named = n_blank = 0

    af = open(asm_csv, "w", newline="", encoding="utf-8")
    aw = csv.writer(af); aw.writerow(["assembly_id", "n_part2part", "n_ground", "n_self", "category"])
    nf = open(name_csv, "w", newline="", encoding="utf-8")
    nw = csv.writer(nf); nw.writerow(["assembly_id", "joint_name", "joint_type", "category"])

    seen = 0
    for r in roots:
        rp = Path(r)
        if not rp.exists():
            print(f"[warn] missing root: {rp}", flush=True); continue
        for aj in rp.glob("*/assembly.json"):
            if args.limit and seen >= args.limit:
                break
            seen += 1; n_total += 1
            try:
                doc = json.loads(aj.read_text(encoding="utf-8"))
            except Exception:
                continue
            joints = classify(doc)
            p2p = sum(1 for _, _, c in joints if c == "part2part")
            ground = sum(1 for _, _, c in joints if c == "ground")
            self_ = sum(1 for _, _, c in joints if c == "self")
            tot_p2p += p2p; tot_ground += ground; tot_self += self_

            cat = ("no_joint" if not joints else
                   "has_part2part" if p2p > 0 else "ground_only")
            if cat == "no_joint": n_no_joint += 1
            elif cat == "has_part2part": n_has_p2p += 1
            else: n_ground_only += 1
            aw.writerow([aj.parent.name, p2p, ground, self_, cat])

            for name, jt, c in joints:
                nw.writerow([aj.parent.name, name, jt, c])
                if c != "part2part":
                    continue
                jt_p2p[jt] += 1
                nm = (name or "").strip()
                if nm:
                    n_named += 1
                    name_counter[nm] += 1
                    for tok in tokens(nm):
                        token_type[tok][jt] += 1
                        type_token[jt][tok] += 1
                else:
                    n_blank += 1
        if args.limit and seen >= args.limit:
            break
    af.close(); nf.close()

    print("\n" + "=" * 64)
    print(f"DATASET JOINT SCAN  ({n_total} assemblies)")
    print("=" * 64)
    print(f"no joints            : {n_no_joint:6d} ({100*n_no_joint/max(n_total,1):.1f}%)")
    print(f"has part-to-part     : {n_has_p2p:6d} ({100*n_has_p2p/max(n_total,1):.1f}%)  <- trainable")
    print(f"ground-only          : {n_ground_only:6d} ({100*n_ground_only/max(n_total,1):.1f}%)")
    print(f"\ntotal joints: part2part {tot_p2p} | ground {tot_ground} | self {tot_self}")

    print("\npart-to-part joint TYPE histogram:")
    for k, v in jt_p2p.most_common():
        print(f"    {short_type(k):16s} {v:6d} ({100*v/max(tot_p2p,1):.1f}%)")

    print(f"\njoint NAMES: {n_named} named, {n_blank} blank "
          f"({100*n_named/max(n_named+n_blank,1):.1f}% named)")
    print("\ntop 25 joint names (part-to-part):")
    for nm, v in name_counter.most_common(25):
        print(f"    {nm[:28]:28s} {v:5d}")

    # --- does a name-token predict a type? show tokens with strong skew ---
    print("\nname-token -> dominant type (tokens seen >=20x, >=70% one type):")
    rows = []
    for tok, cnt in token_type.items():
        total = sum(cnt.values())
        if total < 20:
            continue
        top_jt, top_n = cnt.most_common(1)[0]
        frac = top_n / total
        if frac >= 0.70:
            rows.append((total, tok, short_type(top_jt), frac))
    for total, tok, jt, frac in sorted(rows, reverse=True)[:25]:
        print(f"    {tok:16s} -> {jt:12s} {100*frac:4.0f}%  (n={total})")
    if not rows:
        print("    (no strongly-predictive tokens -> names likely NOT useful as a feature)")

    print(f"\n[scan] per-assembly -> {asm_csv}")
    print(f"[scan] every joint   -> {name_csv}")


if __name__ == "__main__":
    main()

