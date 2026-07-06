"""
build_cache.py                                                       [HGCAN]
Batch-build the existence+type cache and print the aggregate label-quality report
(dataset-wide recall@candidate + class balance). This gate locks the data layer.

  cache/bodies/    <body_uuid>.pt      occwl graphs, extracted ONCE (reused on rebuild)
  cache/assembly/  <assembly_id>.pt    AssemblyPairData, cheap to rebuild
  reports/         <assembly_id>.json  per-assembly report (+ aggregate_summary.json)

Run (from the HGCAN/ project root):
  python build_cache.py --config configs/base.yaml --limit 50   # smoke test
  python build_cache.py --config configs/base.yaml               # full build
  python build_cache.py --config configs/base.yaml --report-only # re-aggregate only
"""
import argparse
import csv
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def discover_assemblies(roots):
    for r in roots:
        rp = Path(r)
        if not rp.exists():
            print(f"[warn] missing root: {rp}", flush=True)
            continue
        yield from rp.glob("*/assembly.json")


def has_joints(p) -> bool:
    """True if the assembly has >=1 part-to-part joint, resolved the SAME way the
    body-level builder does: each joint side maps to a body via
    geometry_or_origin_*.entity_one.body (fallback: single-body occurrence).
    A joint counts if its two sides are distinct bodies."""
    try:
        doc = json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return False
    occs = doc.get("occurrences") or {}

    def side_body(j, side):
        ent = ((j.get(f"geometry_or_origin_{side}") or {}).get("entity_one") or {})
        if ent.get("body"):
            return ent["body"]
        occ = j.get(f"occurrence_{side}")
        if occ:
            bl = list(((occs.get(occ) or {}).get("bodies") or {}))
            if len(bl) == 1:
                return bl[0]
        return None

    joints = {**(doc.get("joints") or {}), **(doc.get("as_built_joints") or {})}
    for j in joints.values():
        j = j or {}
        b1, b2 = side_body(j, "one"), side_body(j, "two")
        if b1 and b2 and b1 != b2:
            return True
    return False


def _process_one(payload: dict) -> dict:
    from data.build_assembly_dataset import build_pair_dataset
    from data.step_graph import step_to_graph
    import torch

    jpath = Path(payload["json_path"])
    aid = jpath.parent.name
    bodies_dir = Path(payload["bodies_dir"])
    asm_dir = Path(payload["asm_dir"])
    reports_dir = Path(payload["reports_dir"])
    data_pt = asm_dir / f"{aid}.pt"
    report_js = reports_dir / f"{aid}.json"
    skip = payload["skip_existing"]

    if skip and data_pt.exists() and report_js.exists():
        try:
            rep = json.loads(report_js.read_text()); rep["_status"] = "cached"
            return rep
        except Exception:
            pass

    base = jpath.parent

    def loader(buid, fname):
        cache_p = bodies_dir / f"{buid}.pt"
        if skip and cache_p.exists():
            try:
                g = torch.load(cache_p, weights_only=False)
                # freshness check: cached body must match the CURRENT feature layout.
                # If step_graph's NODE_FEAT_DIM changed, the cached body is stale ->
                # fall through and re-extract instead of silently using old features.
                from data.step_graph import NODE_FEAT_DIM
                if g is not None and hasattr(g, "x") and g.x.size(-1) == NODE_FEAT_DIM:
                    return g
            except Exception:
                pass
        p = base / fname
        if not p.exists():
            p = base / f"{buid}.step"
        if not p.exists():
            return None
        try:
            g = step_to_graph(str(p))
        except Exception:
            return None          # bad body (occwl topology, empty transfer, ...) -> skip it
        if g is not None:
            cache_p.parent.mkdir(parents=True, exist_ok=True)
            try:
                torch.save(g, cache_p)
            except Exception:
                pass
        return g

    try:
        data, rep = build_pair_dataset(
            str(jpath), loader,
            knn_k=payload["knn_k"], easy_per_pos=payload["easy_per_pos"], seed=payload["seed"],
        )
    except Exception as e:
        # full traceback -> reports/traces/<assembly_id>.txt ; one-line msg -> CSV
        tb = traceback.format_exc()
        trace_dir = reports_dir / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        (trace_dir / f"{aid}.txt").write_text(tb, encoding="utf-8")
        msg = f"{type(e).__name__}: {e}".replace("\n", " ").replace("\r", " ")[:300]
        return {"assembly_id": aid, "_status": "error", "_error": msg}

    data_pt.parent.mkdir(parents=True, exist_ok=True)
    report_js.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, data_pt)
    rep_json = dict(rep)
    rep_json["residuals"] = [[float(d), str(f)] for d, f in rep.get("residuals", [])]
    report_js.write_text(json.dumps(rep_json))
    rep["_status"] = "built"
    return rep


def aggregate(reports_dir, out_summary):
    from data.build_assembly_dataset import label_quality_report_pairs, JOINT_TYPES_K1
    import numpy as np
    reports = []
    for js in Path(reports_dir).glob("*.json"):
        if js.name == Path(out_summary).name:
            continue
        try:
            r = json.loads(js.read_text())
            r["residuals"] = [(float(d), f) for d, f in r.get("residuals", [])]
            reports.append(r)
        except Exception:
            continue
    if not reports:
        print("[aggregate] no reports found."); return
    print("\n" + "=" * 64)
    print(f"AGGREGATE OVER {len(reports)} ASSEMBLIES")
    print("=" * 64)
    label_quality_report_pairs(reports)
    recs = [r["recall_at_candidate"] for r in reports
            if isinstance(r.get("recall_at_candidate"), (int, float))
            and not np.isnan(r["recall_at_candidate"])]
    cls = {k: 0 for k in JOINT_TYPES_K1}
    for r in reports:
        for k, v in (r.get("class_hist") or {}).items():
            cls[k] = cls.get(k, 0) + int(v)
    summary = {
        "n_assemblies": len(reports),
        "recall_mean": float(np.mean(recs)) if recs else None,
        "recall_min": float(np.min(recs)) if recs else None,
        "total_pairs": sum(r.get("n_pairs_kept", 0) for r in reports),
        "total_positives": sum(r.get("n_pos", 0) for r in reports),
        "class_histogram": cls,
    }
    Path(out_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(out_summary).write_text(json.dumps(summary, indent=1))
    print(f"\n[aggregate] summary -> {out_summary}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    roots = cfg["paths"]["dataset_roots"]
    bodies_dir = ROOT / cfg["paths"]["cache_bodies"]
    asm_dir = ROOT / cfg["paths"]["cache_assembly"]
    reports_dir = ROOT / cfg["paths"]["reports"]
    summary = reports_dir / "aggregate_summary.json"
    knn_k = cfg["candidates"]["knn_k"]
    easy = cfg["candidates"]["easy_per_positive"]
    seed = cfg["train"]["seed"]
    keep_jointless = cfg["scan"]["keep_jointless_as_negatives"]
    skip = cfg["build"]["skip_existing"]
    workers = args.workers or cfg["build"]["num_workers"] or (os.cpu_count() or 4)

    for d in (bodies_dir, asm_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        aggregate(reports_dir, summary); return

    t0 = time.time()
    all_aj = list(discover_assemblies(roots))
    selected = all_aj if keep_jointless else [aj for aj in all_aj if has_joints(aj)]
    if args.limit:
        selected = selected[:args.limit]
    print(f"[scan] {len(all_aj)} assemblies, {len(selected)} joint-bearing "
          f"({time.time()-t0:.1f}s)", flush=True)

    payloads = [{
        "json_path": str(aj), "bodies_dir": str(bodies_dir), "asm_dir": str(asm_dir),
        "reports_dir": str(reports_dir), "knn_k": knn_k, "easy_per_pos": easy,
        "seed": seed, "skip_existing": skip,
    } for aj in selected]

    built = cached = failed = 0
    fails = {}
    t1 = time.time()

    # per-assembly status CSV (written incrementally so a crash still leaves a record)
    status_csv = reports_dir / "build_status.csv"
    csv_f = open(status_csv, "w", newline="", encoding="utf-8")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["assembly_id", "status", "n_occ", "n_joints",
                    "n_pairs", "n_pos", "recall_at_candidate", "error"])

    # max_tasks_per_child=1: each worker handles ONE assembly then is replaced, so a
    # corrupted OCC/occwl state can't poison later assemblies. We also read each
    # future with a timeout so a hung/dead worker is skipped instead of hanging.
    try:
        ex = ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1)
    except TypeError:
        ex = ProcessPoolExecutor(max_workers=workers)   # older Python
    per_task_timeout = 300   # seconds; a single assembly should never take this long

    with ex:
        futs = {ex.submit(_process_one, p): (p["json_path"], Path(p["json_path"]).parent.name)
                for p in payloads}
        n = 0
        for fut in as_completed(futs):
            n += 1
            _, aid = futs[fut]
            try:
                rep = fut.result(timeout=per_task_timeout)
            except Exception as e:
                rep = {"assembly_id": aid, "_status": "error",
                       "_error": f"worker died/timeout: {type(e).__name__}"}
            st = rep.get("_status")
            if st == "built":
                built += 1
            elif st == "cached":
                cached += 1
            else:
                failed += 1
                k = rep.get("_error", "unknown").split(":")[0]
                fails[k] = fails.get(k, 0) + 1

            # live trace of the assembly just finished (id + outcome)
            tag = {"built": "ok", "cached": "cached", "error": "FAIL"}.get(st, st)
            extra = f"  {rep.get('_error','')}" if st == "error" else \
                    f"  pairs={rep.get('n_pairs_kept',0)} pos={rep.get('n_pos',0)} " \
                    f"recall={rep.get('recall_at_candidate','')}"
            print(f"[{n}/{len(payloads)}] {tag:6s} {aid}{extra}", flush=True)

            # CSV row
            csv_w.writerow([
                rep.get("assembly_id", aid), st,
                rep.get("n_occ", ""), rep.get("n_joints", ""),
                rep.get("n_pairs_kept", ""), rep.get("n_pos", ""),
                rep.get("recall_at_candidate", ""), rep.get("_error", ""),
            ])
            csv_f.flush()

    csv_f.close()
    print(f"\n[build] done {time.time()-t1:.1f}s  built={built} cached={cached} failed={failed}")
    print(f"[build] per-assembly status -> {status_csv}")
    if fails:
        print("[build] failure reasons:")
        for k, v in sorted(fails.items(), key=lambda x: -x[1]):
            print(f"    {v:5d}  {k}")
    aggregate(reports_dir, summary)


if __name__ == "__main__":
    main()
