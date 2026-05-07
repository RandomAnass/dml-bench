#!/usr/bin/env python3
"""
Relabel pre-2026-04-13 22:01 result JSONs that carry method=dml_relobralo
(simplified softmax+EMA variant) to method=dml_softmax_balance, which is the
post-rename canonical name for that algorithm.

Two-step procedure per file:
  1. Read the JSON, rewrite the `method` field (and `key` if present) to
     `dml_softmax_balance`.
  2. Save under the parallel filename `*_dml_softmax_balance.json` in the same
     directory (so aggregation scripts that filter by `method=dml_softmax_balance`
     now find this data).
  3. MOVE the original `*_dml_relobralo.json` to a `_legacy_relobralo_simplified/`
     sibling subdirectory for archival. The original file's content is preserved
     unchanged in the archive — only the active dir loses the mislabeled entry.

This is non-destructive: every byte of the original data survives, but the
active result-tree no longer carries simplified-variant data labeled as
`dml_relobralo`. Future faithful-Bischof-Kraus reruns will write fresh
`*_dml_relobralo.json` files into the same dirs without label collision.

Usage:
    python scripts/relabel_legacy_relobralo.py --dry-run         # report only
    python scripts/relabel_legacy_relobralo.py                    # apply
    python scripts/relabel_legacy_relobralo.py --dirs results/tier3_benchmark
        # narrow scope
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RENAME_CUTOFF_TS = datetime(2026, 4, 13, 22, 1, 46, tzinfo=timezone.utc).timestamp()

DEFAULT_DIRS = [
    "results/tier1_benchmark",
    "results/tier2_benchmark",
    "results/tier3_benchmark",
    "results/spy_options",
    "results/unified_comparison/multi_seed",
    "results/unified_comparison/multi_seed_v1_lrm_eval",
    "results/unified_comparison/single_seed",
    "results/unified_comparison/smoke_test",
]


def is_legacy(path: Path) -> bool:
    """A file is legacy if its mtime is before the rename cutoff."""
    return path.stat().st_mtime < RENAME_CUTOFF_TS


def relabel_one(src: Path, dry_run: bool) -> tuple[bool, Path | None, Path | None]:
    """
    Returns (success, new_active_path, archived_original_path).
    """
    try:
        with open(src) as f:
            data = json.load(f)
    except Exception as exc:
        print(f"  SKIP {src}: read error {exc}", file=sys.stderr)
        return (False, None, None)

    if data.get("method") != "dml_relobralo":
        # Not actually labeled relobralo — defensive skip
        return (False, None, None)

    # Build the new filename. Most files end with "_dml_relobralo.json".
    # Some have additional suffixes (e.g., "_gk", "_lrm") which we preserve
    # by replacing only the canonical method substring.
    name = src.name
    if "_dml_relobralo" not in name:
        print(f"  SKIP {src}: filename does not contain '_dml_relobralo'", file=sys.stderr)
        return (False, None, None)
    new_name = name.replace("_dml_relobralo", "_dml_softmax_balance")

    new_active_path = src.parent / new_name
    archive_dir = src.parent / "_legacy_relobralo_simplified"
    archived_original_path = archive_dir / src.name

    # Patch the JSON content
    new_data = dict(data)
    new_data["method"] = "dml_softmax_balance"
    if "key" in new_data and isinstance(new_data["key"], str):
        new_data["key"] = new_data["key"].replace("dml_relobralo", "dml_softmax_balance")
    new_data.setdefault("relabeled_from", "dml_relobralo")
    new_data.setdefault("relabeled_reason",
                        "Pre-2026-04-13-22:01 dml_relobralo data is the simplified "
                        "softmax+EMA variant (now SoftmaxBalanceDmlLoss). Relabeled "
                        "to dml_softmax_balance for label consistency. Original file "
                        "archived under _legacy_relobralo_simplified/.")
    new_data.setdefault("relabeled_at_utc",
                        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))

    if dry_run:
        return (True, new_active_path, archived_original_path)

    # Write new file
    archive_dir.mkdir(parents=True, exist_ok=True)
    new_active_path.write_text(json.dumps(new_data, indent=2))
    # Archive original
    shutil.move(str(src), str(archived_original_path))
    return (True, new_active_path, archived_original_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be done, do not write or move anything.")
    p.add_argument("--dirs", nargs="+", default=DEFAULT_DIRS,
                   help="Directories (relative to repo root) to scan.")
    args = p.parse_args()

    print(f"Cutoff: 2026-04-13 22:01:46 UTC (commit a4ee3339)")
    print(f"Mode:   {'DRY RUN (no changes)' if args.dry_run else 'APPLY'}")
    print(f"Dirs:   {len(args.dirs)}")
    print()

    total_seen = total_legacy = total_relabeled = total_skipped = 0
    per_dir = {}

    for d_rel in args.dirs:
        d = ROOT / d_rel
        if not d.exists():
            print(f"  [skip] {d_rel} (does not exist)")
            continue
        legacy_files = sorted(
            p for p in d.glob("*_dml_relobralo*.json")
            if p.is_file() and is_legacy(p)
        )
        per_dir[d_rel] = len(legacy_files)
        total_seen += len(list(d.glob("*_dml_relobralo*.json")))
        total_legacy += len(legacy_files)

        if not legacy_files:
            print(f"  [{d_rel}] no legacy files")
            continue

        print(f"  [{d_rel}] {len(legacy_files)} legacy file(s) to relabel")
        for src in legacy_files:
            ok, _, _ = relabel_one(src, args.dry_run)
            if ok:
                total_relabeled += 1
            else:
                total_skipped += 1

    print()
    print(f"Summary: legacy={total_legacy}  relabeled={total_relabeled}  skipped={total_skipped}")
    if args.dry_run:
        print("(dry-run: no files were actually modified)")
    else:
        print("Originals archived under <dir>/_legacy_relobralo_simplified/.")
        print("Run aggregation scripts now and confirm method axis splits.")
    print()
    print("Per-dir breakdown:")
    for d, n in sorted(per_dir.items()):
        print(f"  {d:55s} {n:5d}")


if __name__ == "__main__":
    main()
