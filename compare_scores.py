#!/usr/bin/env python3
"""
compare_scores.py

Compare per-paper rubric scores between two JSON or JSONL files produced by LLM_score.py.
- Supports JSONL (one JSON object per line) and JSON arrays (list of objects)
- Aligns records by a key (default: "id")
- Computes per-category summary stats (count, mean diff, MAE, std of diff, Pearson r)
- Prints top-N largest absolute disagreements (by a chosen category, or "all")
- Filter to only show rows with non-zero (or thresholded) differences
- Select a subset of categories to analyze/export; list available categories
- Optionally write a CSV of paired scores and diffs

Usage:
    python compare_scores.py file1.jsonl file2.jsonl \
        [--key id] [--csv out.csv] [--top 25] [--by overall|all] \
        [--threshold 0.0] [--only-diff] [--cats progress,overall] [--list-cats] [--csv-only-diff]

Notes:
- Expects each record to have at least {id, scores:{...}}; title/model are optional.
- Categories default to the union of both files' score keys; use --cats to restrict.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import orjson as _orjson
    def jloads(s: str | bytes) -> Any: return _orjson.loads(s)
except Exception:
    def jloads(s: str | bytes) -> Any: return json.loads(s)  # type: ignore

Record = Dict[str, Any]


def read_json_or_jsonl(path: str) -> List[Record]:
    """Load a list of record dicts from either JSONL or JSON array."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        # Peek first non-whitespace char
        head = f.read(2048)
        f.seek(0)
        first = next((ch for ch in head if not ch.isspace()), "")
        if first == "[":
            try:
                data = jloads(f.read())
                if isinstance(data, list):
                    return [x for x in data if isinstance(x, dict)]
            except Exception:
                pass
        # Fallback: JSONL
        out: List[Record] = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = jloads(line)
                if isinstance(obj, dict):
                    out.append(obj)
            except Exception:
                # skip invalid lines
                continue
        return out


def to_map(records: Iterable[Record], key: str) -> Dict[str, Record]:
    out: Dict[str, Record] = {}
    for r in records:
        k = r.get(key)
        if isinstance(k, str) and k:
            out[k] = r
    return out


def get_scores(rec: Record) -> Dict[str, Optional[float]]:
    raw = rec.get("scores")
    if isinstance(raw, dict):
        out: Dict[str, Optional[float]] = {}
        for k, v in raw.items():
            try:
                out[k] = float(v)
            except Exception:
                out[k] = None
        return out
    return {}


def union_categories(m1: Dict[str, Record], m2: Dict[str, Record]) -> List[str]:
    cats: set[str] = set()
    for m in (m1, m2):
        for r in m.values():
            s = r.get("scores")
            if isinstance(s, dict):
                cats.update(k for k in s.keys())
    # Stabilize ordering: put common rubric order first if present
    preferred = [
        "progress", "creativity", "novelty", "technical_rigor", "clarity", "potential_impact", "overall"
    ]
    rest = [c for c in sorted(cats) if c not in preferred]
    ordered = [c for c in preferred if c in cats] + rest
    return ordered


def pearson_r(x: List[float], y: List[float]) -> Optional[float]:
    n = len(x)
    if n < 2:
        return None
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    den_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    den_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
    den = den_x * den_y
    if den == 0:
        return None
    return num / den


def summarize_diffs(pairs: List[Tuple[str, float, float, float]]) -> Dict[str, Any]:
    """Compute summary stats for a list of (id, s1, s2, diff) where diff=s2-s1."""
    if not pairs:
        return {"n": 0}
    diffs = [d for (_, _, _, d) in pairs]
    n = len(diffs)
    mean = sum(diffs) / n
    mae = sum(abs(d) for d in diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / n
    sd = math.sqrt(var)
    # Pearson between s1 and s2
    s1 = [a for (_, a, _, _) in pairs]
    s2 = [b for (_, _, b, _) in pairs]
    r = pearson_r(s1, s2)
    return {"n": n, "mean_diff": mean, "mae": mae, "std_diff": sd, "pearson_r": r}


def build_pairs(m1: Dict[str, Record], m2: Dict[str, Record], cat: str, key: str) -> List[Tuple[str, float, float, float]]:
    out: List[Tuple[str, float, float, float]] = []
    common = set(m1.keys()) & set(m2.keys())
    for pid in common:
        s1 = get_scores(m1[pid]).get(cat)
        s2 = get_scores(m2[pid]).get(cat)
        if s1 is None or s2 is None:
            continue
        out.append((pid, float(s1), float(s2), float(s2) - float(s1)))
    return out


def write_csv(
    path: str,
    m1: Dict[str, Record],
    m2: Dict[str, Record],
    cats: List[str],
    key: str,
    sort_by: str,
    top: int,
    threshold: float = 0.0,
    only_diff: bool = False,
) -> None:
    rows: List[Dict[str, Any]] = []
    common = sorted(set(m1.keys()) & set(m2.keys()))
    for pid in common:
        r1, r2 = m1[pid], m2[pid]
        row: Dict[str, Any] = {
            key: pid,
            "title_1": r1.get("title"),
            "title_2": r2.get("title"),
            "model_1": r1.get("model"),
            "model_2": r2.get("model"),
        }
        s1 = get_scores(r1)
        s2 = get_scores(r2)
        for c in cats:
            a, b = s1.get(c), s2.get(c)
            row[f"{c}_1"] = a
            row[f"{c}_2"] = b
            row[f"{c}_diff"] = (b - a) if (a is not None and b is not None) else None
        rows.append(row)
    # Optionally filter to only differing rows (any selected category meeting |diff| >= threshold)
    if only_diff or threshold > 0:
        def row_has_diff(r: Dict[str, Any]) -> bool:
            for c in cats:
                d = r.get(f"{c}_diff")
                if isinstance(d, (int, float)) and abs(d) >= threshold:
                    return True
            return False
        rows = [r for r in rows if row_has_diff(r)]

    # Sort and limit
    sort_key = f"{sort_by}_diff" if sort_by else "overall_diff"
    rows.sort(key=lambda r: (abs(r.get(sort_key) or 0)), reverse=True)
    if top > 0:
        rows = rows[:top]
    # Write
    fieldnames: List[str] = [key, "title_1", "title_2", "model_1", "model_2"]
    for c in cats:
        fieldnames += [f"{c}_1", f"{c}_2", f"{c}_diff"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare per-paper scores across two JSON/JSONL files")
    ap.add_argument("file1", help="First scores file (JSONL or JSON array)")
    ap.add_argument("file2", help="Second scores file (JSONL or JSON array)")
    ap.add_argument("--key", default="id", help="Record key to align on (default: id)")
    ap.add_argument("--by", default="overall", help="Category to sort top diffs by (default: overall). Use 'all' to iterate all categories.")
    ap.add_argument("--top", type=int, default=25, help="Show top-N absolute disagreements (default: 25). Use 0 for no limit.")
    ap.add_argument("--csv", dest="csv_out", default="", help="Optional CSV output path for full paired scores")
    ap.add_argument("--threshold", type=float, default=0.0, help="Only show/report diffs with |diff| >= THRESHOLD in the Top list; applies to CSV when --csv-only-diff.")
    ap.add_argument("--only-diff", action="store_true", help="Print only the Top disagreements (suppresses per-category stats)")
    ap.add_argument("--csv-only-diff", action="store_true", help="When writing CSV, include only rows where any selected category meets the threshold")
    ap.add_argument("--list-cats", action="store_true", help="List available categories from the files and exit")
    ap.add_argument("--cats", default="", help="Comma-separated subset of categories to analyze/export (default: all)")
    args = ap.parse_args()

    r1 = read_json_or_jsonl(args.file1)
    r2 = read_json_or_jsonl(args.file2)
    m1 = to_map(r1, args.key)
    m2 = to_map(r2, args.key)

    ids1, ids2 = set(m1.keys()), set(m2.keys())
    common = ids1 & ids2
    only1 = ids1 - ids2
    only2 = ids2 - ids1

    cats = union_categories(m1, m2)
    if args.list_cats:
        print("Available categories:")
        for c in cats:
            print(f"- {c}")
        return

    # Restrict categories if requested
    if args.cats.strip():
        requested = [c.strip() for c in args.cats.split(",") if c.strip()]
        # Preserve requested order, include only those present
        selected = [c for c in requested if c in cats]
        if selected:
            cats = selected

    print(f"File1: {args.file1} ({len(r1)} records, {len(ids1)} keyed)")
    print(f"File2: {args.file2} ({len(r2)} records, {len(ids2)} keyed)")
    print(f"Common ids: {len(common)} | Only in file1: {len(only1)} | Only in file2: {len(only2)}")
    if only1:
        print(f"  e.g., only in file1 (first 5): {list(sorted(only1))[:5]}")
    if only2:
        print(f"  e.g., only in file2 (first 5): {list(sorted(only2))[:5]}")

    if not args.only_diff:
        print()
        print("Per-category stats (over common ids with both scores):")
        for c in cats:
            pairs = build_pairs(m1, m2, c, args.key)
            stats = summarize_diffs(pairs)
            n = stats.get("n", 0)
            if n == 0:
                print(f"- {c}: n=0")
                continue
            r = stats.get("pearson_r")
            r_s = f"{r:.3f}" if isinstance(r, float) else "NA"
            print(
                f"- {c}: n={n}, mean_diff={stats['mean_diff']:.3f}, MAE={stats['mae']:.3f}, "
                f"std_diff={stats['std_diff']:.3f}, r={r_s}"
            )

    # Top disagreements
    def show_top_for_category(by_cat: str) -> None:
        pairs = build_pairs(m1, m2, by_cat, args.key)
        # Filter by threshold
        if args.threshold > 0:
            pairs = [p for p in pairs if abs(p[3]) >= args.threshold]
        pairs.sort(key=lambda x: abs(x[3]), reverse=True)
        print()
        lim = args.top if args.top > 0 else len(pairs)
        print(f"Top {lim} disagreements by |{by_cat}_diff| (threshold={args.threshold}):")
        if not pairs:
            print("  (none)")
            return
        for pid, s1, s2, d in pairs[: lim]:
            t1 = m1[pid].get("title")
            t2 = m2[pid].get("title")
            print(f"  {pid}: {by_cat}_1={s1:g}, {by_cat}_2={s2:g}, diff={d:+g} | title1={t1!r} | title2={t2!r}")

    if args.by == "all":
        for c in cats:
            show_top_for_category(c)
    else:
        by = args.by if args.by in cats else ("overall" if "overall" in cats else cats[0] if cats else "")
        if by:
            show_top_for_category(by)

    if args.csv_out:
        try:
            # For CSV, if by==all, use 'overall' when present, else the first category
            csv_sort_by = ("overall" if (args.by == "all" and "overall" in cats) else (args.by if args.by in cats else (cats[0] if cats else "overall")))
            write_csv(
                args.csv_out,
                m1,
                m2,
                cats,
                args.key,
                csv_sort_by,
                args.top,
                threshold=args.threshold,
                only_diff=args.csv_only_diff,
            )
            print()
            print(f"Wrote CSV: {args.csv_out}")
        except Exception as e:
            print(f"[warn] failed to write CSV: {e}")


if __name__ == "__main__":
    main()
