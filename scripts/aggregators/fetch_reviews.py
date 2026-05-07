"""
Fetch full OpenReview review threads (Official_Review, Decision, Official_Comment,
Author_Response, Meta_Review) for a list of papers and dump each as a markdown file.

Source: dml_bench_review_inspiration_v2.md §12 (verbatim, with relative output dir
adjusted to live alongside this script).

Usage:
    python papers/neurips_DB/scripts/fetch_reviews.py

Outputs:
    papers/neurips_DB/reviews/<paper_id>.md  — one file per paper
    papers/neurips_DB/reviews/_index.md      — summary index
"""

import os
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# -------------------- CONFIGURE --------------------
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "reviews"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Papers to fetch. Format: (paper_label, openreview_forum_id, venue_hint)
# venue_hint helps the script pick the right API (v1 for older venues, v2 for 2023+).
PAPERS = [
    # Closest analogues (have direct relevance to DML-Bench):
    ("forces_are_not_enough",  "A8pqQipwkt",      "tmlr"),       # TMLR
    ("pinnacle_neurips24",     "aekfb95slj",      "v2"),         # NeurIPS 2024 D&B (accepted)
    ("pinnacle_iclr24_reject", "ApjY32f3Xr",      "v2"),         # ICLR 2024 (rejected — useful! shows what didn't work)
    ("pdebench_neurips22",     "dh_MkX0QfrK",     "v1"),         # NeurIPS 2022 D&B (Outstanding)
    ("the_well_neurips24",     "00Sx577BT3",      "v2"),         # NeurIPS 2024 D&B
    ("dc_bench_neurips22",     "Bs8iFQ7AM6",      "v1"),         # NeurIPS 2022 D&B

    # Additional D&B benchmarks from your table:
    ("lagrangebench_neurips23",   "8ZRAHNT7E9",   "v2"),    # corrected ID from openreview search
    ("airfrans_neurips22",        "Zp8YmiQ_bDC",  "v1"),
    ("lips_neurips22",            "ObD_o92z4p",   "v1"),    # corrected ID from openreview search
    ("climsim_neurips23",         "W5If9P1xqO",   "v2"),
    ("bubbleml_neurips23",        "0Wmglu8zak",   "v2"),    # corrected ID from openreview search
    ("benchstructinf_neurips24",  "kKtalvwqBZ",   "v2"),    # corrected ID from openreview search (was kKtalZeNUz)
    ("libmoon_neurips24",         "etdXLAMZoc",   "v2"),    # corrected ID from openreview search
    ("dattri_neurips24",          "IkA54A6KKe",   "v2"),    # corrected ID from openreview search

    # Rejected / withdrawn papers in derivative-supervision / PINN space
    # (added 2026-04-30: review threads of unsuccessful submissions are valuable
    #  for identifying common reviewer concerns we should pre-empt).
    ("l_pinn_iclr24_reject",      "EP09OGPRzk",   "v2"),    # ICLR 2024 — Reject. Langevin-dynamics PINN.
    ("dc_pinns_iclr25_withdraw",  "U2ZtvonVQz",   "v2"),    # ICLR 2025 — Withdrawn after reviews. Derivative-constrained PINN — directly relevant to DML.

    # NeurIPS 2025 D&B Track papers (added 2026-05-04: closest comparators
    # to our submission since they reflect current track standards).
    ("dataset_fingerprints_neurips25",  "iKwHwCaddB", "v2"),  # spotlight — foundation-model data analysis
    ("agentrecbench_neurips25",          "fm77rDf9JS", "v2"),  # spotlight — LLM agent benchmark
    ("physgym_neurips25",                "w8uII2qAmd", "v2"),  # poster — LLM physics discovery
    ("data_juicer2_neurips25",           "DHA9uoeMQx", "v2"),  # spotlight — data processing system
    ("intermt_neurips25",                "4SUtAp2cm0", "v2"),  # spotlight — multimodal preference dataset
    ("trauma_voices_neurips25",          "qrFvHgZa7l", "v2"),  # spotlight — synthetic clinical dataset
    ("patientsim_neurips25",             "1THAjdP4QJ", "v2"),  # spotlight — clinical simulator
    ("mtbbench_neurips25",               "anzoPBV4jI", "v2"),  # poster — clinical multimodal benchmark
    ("eurospeech_neurips25",             "26VLybEQ2h", "v2"),  # spotlight — multilingual speech corpus
    ("medsg_bench_neurips25",            "8CKhxBaWO5", "v2"),  # spotlight — medical image grounding
    ("medicalnarratives_neurips25",      "3rY182JOOZ", "v2"),  # poster — medical vision/lang dataset
    ("oceanbench_neurips25",             "wZGe1Kqs8G", "v2"),  # poster — global ocean forecasting
]
# NOTE: forum IDs above are the canonical IDs; if any 404s, search the OpenReview
# venue homepage for the paper title and use the id from the resulting forum URL.

API_V1 = "https://api.openreview.net/notes"
API_V2 = "https://api2.openreview.net/notes"
SLEEP_BETWEEN_PAPERS = 1.5   # be polite to the API


# -------------------- FETCH HELPERS --------------------
def fetch_notes(forum_id: str, api_url: str) -> list:
    """Fetch all notes (paper + replies) for a given forum id."""
    params = {"forum": forum_id, "details": "replies", "limit": 1000}
    r = requests.get(api_url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("notes", [])


def fetch_with_fallback(forum_id: str, hint: str) -> list:
    """Try v2 first then v1, since some venues only support one."""
    order = (API_V2, API_V1) if hint != "v1" else (API_V1, API_V2)
    last_exc = None
    for api in order:
        try:
            notes = fetch_notes(forum_id, api)
            if notes:
                return notes
        except Exception as e:
            last_exc = e
            continue
    if last_exc:
        raise last_exc
    return []


# -------------------- CLASSIFY NOTES --------------------
def note_kind(note: dict) -> str:
    """
    Classify a note as one of:
      'submission', 'official_review', 'decision', 'meta_review',
      'official_comment', 'rebuttal', 'unknown'
    Handles both API v1 (invitation) and v2 (invitations list).

    IMPORTANT: 2024+ NeurIPS D&B uses ".../Submission<N>/-/<TYPE>" so a
    naive "submission in inv" check incorrectly catches every reply. We
    check the specific note types FIRST and fall through to 'submission'
    only if none of them match.
    """
    invs = []
    if "invitation" in note:
        invs = [note["invitation"]]
    elif "invitations" in note:
        invs = note["invitations"]

    inv_str = " | ".join(invs).lower()

    # Specific reply types — check first.
    if "official_review" in inv_str or "/review" in inv_str:
        return "official_review"
    if "rebuttal" in inv_str or "author_response" in inv_str:
        return "rebuttal"
    if "decision" in inv_str:
        return "decision"
    if "meta_review" in inv_str or "metareview" in inv_str:
        return "meta_review"
    if "official_comment" in inv_str or "/comment" in inv_str:
        return "official_comment"

    # Submission detection: distinguish ".../Submission<N>/-/<X>" (a reply)
    # from "/-/Submission" (the paper itself). The paper is the only note
    # whose invitation ends in "/Submission" with no further suffix.
    # Concretely: paper invitations look like
    #   "venue/-/Submission"          (v1) or
    #   "venue/-/Submission" / "venue/-/Edit"  (v2 — at least one ends in Submission).
    import re
    if any(re.search(r"(?:^|/)submission(?:$|/-/edit$)", i.lower()) for i in invs):
        return "submission"
    # Fallback: a v2 paper note typically has parent==None or id==forum
    if note.get("id") == note.get("forum"):
        return "submission"
    return "unknown"


def get_content(note: dict) -> dict:
    """
    Get the content dict, normalizing across API v1 (flat) and v2 (nested 'value').
    """
    raw = note.get("content", {})
    flat = {}
    for k, v in raw.items():
        if isinstance(v, dict) and "value" in v:
            flat[k] = v["value"]
        else:
            flat[k] = v
    return flat


# -------------------- RENDER ONE PAPER --------------------
# Field-name handling. Schemas vary across venues:
#   TMLR              : summary_of_contributions, strengths_and_weaknesses, requested_changes, ...
#   NeurIPS D&B 2022  : "summary and contributions", "strengths", "weaknesses", "additional feedback"   (spaces, not underscores)
#   NeurIPS D&B 2024  : summary_and_contributions, opportunities_for_improvement, review, ...
#   ICLR              : summary, strengths, weaknesses, questions, soundness, presentation, ...
#
# Strategy: dump ALL non-empty fields, but order the high-value ones first so the
# critical reviewer feedback (weaknesses, requested changes, questions, ratings) is
# always at the top of the rendered note.
PRIORITIZED_FIELDS = [
    # Identity
    "title",
    # Summaries (different venue conventions)
    "summary",
    "summary_of_contributions",
    "summary_and_contributions",
    "summary and contributions",   # NeurIPS 2022 D&B (spaces)
    # The bit a paper team most wants to act on:
    "strengths",
    "weaknesses",
    "strengths_and_weaknesses",
    "opportunities_for_improvement",   # NeurIPS 2024 D&B's "weaknesses"
    "requested_changes",                # TMLR
    "questions",                        # ICLR
    # Reviewer's substantive prose
    "review",
    "main_review",
    # Conduct / scope flags
    "limitations",
    "broader_impact_concerns",
    "ethics",
    "flag_for_ethics_review",
    # Quality scores
    "soundness",
    "presentation",
    "contribution",
    "correctness",
    "clarity",
    "relation_to_prior_work",
    "relation to prior work",          # NeurIPS 2022 D&B (spaces)
    "documentation",
    "additional_feedback",
    "additional feedback",             # NeurIPS 2022 D&B (spaces)
    "claims_and_evidence",             # TMLR
    "audience",                         # TMLR
    "rating",
    "confidence",
    # Rebuttal/comment/decision specific
    "comment",
    "decision",
    "metareview",
    "code_of_conduct",
]

# Internal/admin fields not worth printing
SKIP_FIELDS = {
    "venue", "venueid", "_bibtex", "pdf", "supplementary_material",
    "html", "code", "TLDR", "tldr", "keywords", "primary_area",
    "authors", "authorids",
}


def render_note_content(c: dict) -> list[str]:
    """Render one note's content as a list of markdown lines.

    Prints prioritized fields first in PRIORITIZED_FIELDS order, then any other
    non-empty content fields (skipping SKIP_FIELDS) so we never silently drop a
    venue-specific field name we haven't seen before.
    """
    lines = []
    seen = set()
    for fld in PRIORITIZED_FIELDS:
        if fld in c:
            seen.add(fld)
            val = c[fld]
            if val in (None, "", [], {}):
                continue
            lines.append(f"**{fld}:**")
            lines.append("")
            if isinstance(val, list):
                val = "\n".join(f"- {x}" for x in val)
            lines.append(str(val))
            lines.append("")
    # Any remaining non-empty fields we didn't anticipate
    for fld, val in c.items():
        if fld in seen or fld in SKIP_FIELDS:
            continue
        if val in (None, "", [], {}):
            continue
        lines.append(f"**{fld}:**")
        lines.append("")
        if isinstance(val, list):
            val = "\n".join(f"- {x}" for x in val)
        lines.append(str(val))
        lines.append("")
    return lines


def render_paper_md(label: str, forum_id: str, notes: list) -> str:
    submission = next((n for n in notes if note_kind(n) == "submission"), None)
    title = "(unknown title)"
    authors = ""
    if submission:
        c = get_content(submission)
        title = c.get("title", title)
        authors = ", ".join(c.get("authors", []) or [])
    md = []
    md.append(f"# {title}")
    md.append("")
    md.append(f"- **Label:** `{label}`")
    md.append(f"- **OpenReview forum:** https://openreview.net/forum?id={forum_id}")
    md.append(f"- **PDF:** https://openreview.net/pdf?id={forum_id}")
    if authors:
        md.append(f"- **Authors:** {authors}")
    md.append("")
    md.append(f"> _Fetched {datetime.now(timezone.utc).isoformat()}_")
    md.append("")

    # Quick counts for the header so a reader knows the thread depth at a glance.
    by_kind_counts = {}
    for n in notes:
        if n is submission:
            continue
        by_kind_counts[note_kind(n)] = by_kind_counts.get(note_kind(n), 0) + 1
    if by_kind_counts:
        md.append("## Thread summary")
        md.append("")
        for k, v in sorted(by_kind_counts.items()):
            md.append(f"- {k}: {v}")
        md.append("")

    # Group notes by kind
    by_kind = {}
    for n in notes:
        if n is submission:
            continue
        k = note_kind(n)
        by_kind.setdefault(k, []).append(n)

    # Render each kind with sensible heading
    kind_order = [
        ("official_review", "Official Reviews"),
        ("rebuttal", "Author Rebuttals / Responses"),
        ("official_comment", "Official Comments (incl. multi-round discussion)"),
        ("decision", "Decision"),
        ("meta_review", "Meta Reviews"),
        ("unknown", "Other replies"),
    ]
    for kind_key, heading in kind_order:
        if kind_key not in by_kind:
            continue
        md.append(f"## {heading}")
        md.append("")
        # Sort by creation date so threads read in chronological order
        for n in sorted(by_kind[kind_key],
                        key=lambda x: x.get("cdate") or x.get("tcdate") or 0):
            c = get_content(n)
            signatures = n.get("signatures", []) or []
            sig = ", ".join(signatures) or "anonymous"
            ts = n.get("cdate") or n.get("tcdate")
            ts_str = (datetime.fromtimestamp(ts/1000, tz=timezone.utc).isoformat()
                      if ts else "unknown")
            md.append(f"### {sig} — {ts_str}")
            md.append("")
            md.extend(render_note_content(c))
            md.append("---")
            md.append("")
    return "\n".join(md)


# -------------------- MAIN --------------------
def main():
    index_lines = ["# Reviews index", ""]
    for label, forum_id, hint in PAPERS:
        out_path = OUT_DIR / f"{label}.md"
        if out_path.exists():
            print(f"[skip] {label} (already fetched, delete {out_path} to refetch)")
            index_lines.append(f"- [{label}](./{out_path.name}) (cached)")
            continue
        print(f"[fetch] {label} ({forum_id})")
        try:
            notes = fetch_with_fallback(forum_id, hint)
            if not notes:
                print(f"  (no notes returned; check forum id or login required)")
                index_lines.append(f"- {label}: NO DATA")
                continue
            md = render_paper_md(label, forum_id, notes)
            out_path.write_text(md, encoding="utf-8")
            n_replies = sum(1 for n in notes if note_kind(n) != "submission")
            print(f"  wrote {out_path} ({n_replies} replies)")
            index_lines.append(
                f"- [{label}](./{out_path.name}) — {n_replies} replies"
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            index_lines.append(f"- {label}: ERROR ({e})")
        time.sleep(SLEEP_BETWEEN_PAPERS)

    (OUT_DIR / "_index.md").write_text("\n".join(index_lines), encoding="utf-8")
    print(f"\nDone. See {OUT_DIR}/_index.md")


if __name__ == "__main__":
    main()
