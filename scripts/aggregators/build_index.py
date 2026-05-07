"""
Build an enriched _index.md by parsing the rendered review files.

Adds: title, OpenReview URL, decision (when present), reply-type counts,
and a 1-line indicator of whether the file has full Official_Reviews
or only post-rebuttal comments.

Usage:
    python papers/neurips_DB/scripts/build_index.py
"""
from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REVIEWS = HERE.parent / "reviews"

LABEL_NOTE = {
    "_v1_backup": None,
    "_takeaways": None,
    "_index.md": None,
}


def parse_md(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    # Title is the first '# ' line
    title_match = re.search(r"^# (.+)$", text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "(unknown title)"

    # Forum URL
    url_match = re.search(r"\*\*OpenReview forum:\*\* (\S+)", text)
    forum_url = url_match.group(1) if url_match else ""

    # Thread counts (from "## Thread summary" block)
    counts = {}
    block = re.search(r"## Thread summary\s*\n((?:- .+\n)+)", text)
    if block:
        for line in block.group(1).strip().splitlines():
            m = re.match(r"- (\w+): (\d+)", line)
            if m:
                counts[m.group(1)] = int(m.group(2))

    # Decision text — pick the first "**decision:**" block
    dec_match = re.search(r"\*\*decision:\*\*\s*\n\s*\n([^\n]+)", text)
    decision = dec_match.group(1).strip() if dec_match else ""

    # Has full reviews?
    has_full = counts.get("official_review", 0) > 0
    only_postrebuttal = (counts.get("official_review", 0) == 0
                        and counts.get("official_comment", 0) > 0)

    return {
        "title": title,
        "forum_url": forum_url,
        "counts": counts,
        "decision": decision,
        "has_full_reviews": has_full,
        "only_postrebuttal": only_postrebuttal,
    }


def main():
    files = sorted([p for p in REVIEWS.glob("*.md") if p.name not in LABEL_NOTE])
    if not files:
        raise SystemExit("No review files found.")

    rows = []
    for f in files:
        meta = parse_md(f)
        label = f.stem
        rows.append((label, meta, f.name))

    # Group by full vs partial coverage
    accepted_or_unknown = []
    rejects = []
    partial = []

    for label, meta, fname in rows:
        if "reject" in label.lower() or "withdraw" in label.lower() or "reject" in meta["decision"].lower():
            rejects.append((label, meta, fname))
        elif not meta["has_full_reviews"] and meta["only_postrebuttal"]:
            partial.append((label, meta, fname))
        else:
            accepted_or_unknown.append((label, meta, fname))

    out = ["# Reviews index", ""]
    out += [
        f"Total papers: {len(rows)}.",
        f"Full review threads: {sum(1 for _, m, _ in rows if m['has_full_reviews'])}.",
        f"Post-rebuttal-only (full reviews not public): {sum(1 for _, m, _ in rows if m['only_postrebuttal'])}.",
        f"Rejected / withdrawn: {len(rejects)}.",
        "",
        "Source script: `papers/neurips_DB/scripts/fetch_reviews.py`",
        "Index regen: `python papers/neurips_DB/scripts/build_index.py`",
        "",
    ]

    def fmt_counts(c):
        if not c:
            return ""
        parts = []
        for k in ("official_review", "rebuttal", "official_comment", "decision",
                  "meta_review", "submission", "unknown"):
            if k in c:
                parts.append(f"{c[k]} {k.replace('_', ' ')}")
        return " · ".join(parts)

    def render_section(title, items):
        if not items:
            return []
        block = [f"## {title}", ""]
        for label, m, fname in items:
            cstr = fmt_counts(m["counts"])
            decision_str = f" — *{m['decision']}*" if m["decision"] else ""
            block.append(f"- **[{label}](./{fname})**{decision_str}")
            block.append(f"  - {m['title']}")
            block.append(f"  - {m['forum_url']}")
            if cstr:
                block.append(f"  - {cstr}")
            block.append("")
        return block

    out += render_section("Accepted / decision-known", accepted_or_unknown)
    out += render_section("Rejected / withdrawn (signal: what reviewers wouldn't accept)", rejects)
    out += render_section("Post-rebuttal only (full reviews restricted)", partial)

    (REVIEWS / "_index.md").write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {REVIEWS / '_index.md'}")
    for label, m, _ in rows:
        c = m["counts"]
        marker = "✓" if m["has_full_reviews"] else ("◇" if m["only_postrebuttal"] else "?")
        print(f"  {marker} {label:<32} reviews={c.get('official_review', 0)} "
              f"rebuttals={c.get('rebuttal', 0)} comments={c.get('official_comment', 0)} "
              f"decision={m['decision'][:40]!r}")


if __name__ == "__main__":
    main()
