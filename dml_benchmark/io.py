"""
Result-corpus I/O helpers.

The benchmark stores every training run as one JSON file under
`results/<tier>/<key>.json`. Aggregator scripts that produce paper
numbers iterate over those files and compute summary statistics.

Historical pattern (caught in the 2026-04-30 codebase audit, finding
B1-B3): every aggregator had its own inline

    for p in tdir.glob("*.json"):
        try:
            r = json.load(open(p))
        except Exception:
            continue

A truncated or partially-written JSON silently disappeared from the
paired-statistics computation, biasing `n_paired` downward without
notice. This module replaces that pattern with a single helper that
warns on every drop and lets the caller assert an expected count.

Usage:
    from dml_benchmark.io import load_result_json, iter_result_jsons

    # one-shot load
    r = load_result_json(path)
    if r is None:
        ...        # handled and warned

    # bulk iteration with skip count
    rows, n_skipped = [], 0
    for r, skipped in iter_result_jsons(tier_dir.glob("*.json")):
        if r is None:
            n_skipped += 1
            continue
        rows.append(r)
    assert len(rows) >= EXPECTED, f"loaded {len(rows)}, expected >= {EXPECTED}"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple


def load_result_json(path, *, on_error: str = "warn") -> Optional[dict]:
    """Load a result JSON. Returns the parsed dict or None.

    Args:
        path: file path (str or Path).
        on_error: 'warn' (default) — print a warning to stderr and return None.
                  'raise'  — re-raise the exception.
                  'silent' — return None without warning. Reserve for tests.

    Raises ValueError if `on_error` is not one of the three.
    """
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        if on_error == "warn":
            print(f"WARN: skip {path}: {type(e).__name__}: {e}",
                  file=sys.stderr)
            return None
        elif on_error == "raise":
            raise
        elif on_error == "silent":
            return None
        else:
            raise ValueError(
                f"on_error must be 'warn' | 'raise' | 'silent', got {on_error!r}"
            )


def iter_result_jsons(paths: Iterable, *, on_error: str = "warn") \
        -> Iterator[Tuple[Optional[dict], Optional[Path]]]:
    """Stream a sequence of result JSONs.

    Yields `(record, path)` for every iterated path, with `record = None`
    when the file failed to parse. Callers can decide whether to skip
    or stop on errors. Combine with `assert n_loaded >= EXPECTED` to
    catch silent drift.
    """
    for p in paths:
        rec = load_result_json(p, on_error=on_error)
        yield rec, Path(p)
