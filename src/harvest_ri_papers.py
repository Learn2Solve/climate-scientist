#!/usr/bin/env python3
"""
Headless literature harvester for Rapid Intensification (RI) papers.

Uses OpenAlex for discovery and (optionally) downloads open-access PDFs
into a run-style artifacts folder. Writes a markdown summary under docs/papers/ri/.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.web_tools import ensure_dir, http_get, openalex_search_works, sha256_bytes, validate_http_url


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def slugify(text: str, *, max_len: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", text.strip()).strip("_")
    return text[:max_len] if text else "item"


def pick_pdf_url(item: dict[str, Any]) -> str:
    for key in ["pdf_url", "oa_url", "landing_page_url"]:
        v = item.get(key) or ""
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def download_pdf(url: str, out_dir: Path, *, max_bytes: int) -> tuple[Path | None, str]:
    ok, reason = validate_http_url(url)
    if not ok:
        return None, reason
    try:
        res = http_get(url, timeout_s=25.0, max_bytes=max_bytes)
    except Exception as exc:
        return None, f"fetch failed: {exc}"

    ctype = (res.headers.get("Content-Type") or "").lower()
    if "pdf" not in ctype and not url.lower().endswith(".pdf"):
        return None, f"not a pdf (content-type: {ctype})"

    sha = sha256_bytes(res.body)[:12]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"paper_{sha}.pdf"
    out_path.write_bytes(res.body)
    note = "truncated" if res.truncated else "ok"
    return out_path, note


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless RI literature harvester (OpenAlex).")
    parser.add_argument("--query", type=str, default="tropical cyclone rapid intensification")
    parser.add_argument("--from-date", type=str, default="2020-01-01")
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=Path("docs/papers/ri"))
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    parser.add_argument("--download-pdf", action="store_true")
    parser.add_argument("--max-bytes", type=int, default=10_000_000, help="Max bytes per PDF")
    parser.add_argument(
        "--include",
        type=str,
        default="",
        help="Comma-separated keywords; keep item if any match title/abstract/venue.",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default="",
        help="Comma-separated keywords; drop item if any match title/abstract/venue.",
    )
    args = parser.parse_args()

    ts = utc_compact()
    out_dir = args.out_dir
    ensure_dir(out_dir)

    if args.artifacts_dir is None:
        artifacts_dir = Path("runs") / f"harvest_{ts}" / "artifacts" / "web"
    else:
        artifacts_dir = args.artifacts_dir
    ensure_dir(artifacts_dir)

    search = openalex_search_works(
        args.query,
        max_results=int(args.max_results),
        from_publication_date=args.from_date,
        require_abstract=True,
        require_oa=False,
        require_pdf=False,
    )

    include_terms = [x.strip().lower() for x in args.include.split(",") if x.strip()]
    exclude_terms = [x.strip().lower() for x in args.exclude.split(",") if x.strip()]

    md_lines = []
    md_lines.append(f"# RI Literature Harvest ({ts})")
    md_lines.append("")
    md_lines.append(f"- query: `{args.query}`")
    md_lines.append(f"- from_date: `{args.from_date}`")
    md_lines.append(f"- results: {len(search.get('results') or [])}")
    md_lines.append(f"- artifacts: `{artifacts_dir}`")
    if include_terms:
        md_lines.append(f"- include: {include_terms}")
    if exclude_terms:
        md_lines.append(f"- exclude: {exclude_terms}")
    md_lines.append("")

    results = []
    filtered = []
    for item in (search.get("results") or []):
        title = item.get("title") or ""
        doi = item.get("doi") or ""
        venue = item.get("venue") or ""
        year = item.get("publication_year")
        date = item.get("publication_date") or ""
        cited = item.get("cited_by_count")
        abstract = item.get("abstract") or ""

        text_blob = " ".join([title, abstract, venue]).lower()
        if include_terms and not any(t in text_blob for t in include_terms):
            continue
        if exclude_terms and any(t in text_blob for t in exclude_terms):
            continue

        pdf_url = pick_pdf_url(item)
        pdf_path = None
        pdf_note = ""
        if args.download_pdf and pdf_url:
            pdf_path, pdf_note = download_pdf(pdf_url, artifacts_dir, max_bytes=int(args.max_bytes))

        entry = {
            "title": title,
            "doi": doi,
            "venue": venue,
            "publication_year": year,
            "publication_date": date,
            "cited_by_count": cited,
            "openalex_id": item.get("openalex_id") or "",
            "landing_page_url": item.get("landing_page_url") or "",
            "pdf_url": item.get("pdf_url") or "",
            "oa_url": item.get("oa_url") or "",
            "abstract": abstract,
            "pdf_path": str(pdf_path) if pdf_path else "",
            "pdf_note": pdf_note,
        }
        results.append(entry)

        slug = slugify(title)[:60]
        md_lines.append(f"## {title}")
        md_lines.append("")
        if year or date:
            md_lines.append(f"- date: `{date or year}`")
        if venue:
            md_lines.append(f"- venue: {venue}")
        if doi:
            md_lines.append(f"- doi: {doi}")
        if cited is not None:
            md_lines.append(f"- cited_by: {cited}")
        if entry["landing_page_url"]:
            md_lines.append(f"- landing_page: {entry['landing_page_url']}")
        if entry["pdf_url"]:
            md_lines.append(f"- pdf_url: {entry['pdf_url']}")
        if entry["oa_url"]:
            md_lines.append(f"- oa_url: {entry['oa_url']}")
        if pdf_path:
            md_lines.append(f"- downloaded_pdf: `{pdf_path}` ({pdf_note})")
        md_lines.append("")
        if abstract:
            md_lines.append("**Abstract (OpenAlex)**")
            md_lines.append(abstract.strip())
            md_lines.append("")
        md_lines.append(f"_slug: {slug}_")
        md_lines.append("")

    filtered = results

    out_md = out_dir / f"harvest_{ts}.md"
    out_json = out_dir / f"harvest_{ts}.json"
    out_md.write_text("\n".join(md_lines).strip() + "\n", encoding="utf-8")
    out_json.write_text(
        json.dumps(
            {
                "query": args.query,
                "from_date": args.from_date,
                "results": results,
                "include": include_terms,
                "exclude": exclude_terms,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if include_terms or exclude_terms:
        shortlist_md = out_dir / f"shortlist_{ts}.md"
        shortlist_json = out_dir / f"shortlist_{ts}.json"
        shortlist_lines = [f"# RI Shortlist ({ts})", "", f"- total: {len(filtered)}", ""]
        for entry in filtered:
            title = entry.get("title") or ""
            shortlist_lines.append(f"## {title}")
            if entry.get("publication_date"):
                shortlist_lines.append(f"- date: `{entry['publication_date']}`")
            if entry.get("venue"):
                shortlist_lines.append(f"- venue: {entry['venue']}")
            if entry.get("doi"):
                shortlist_lines.append(f"- doi: {entry['doi']}")
            if entry.get("landing_page_url"):
                shortlist_lines.append(f"- landing_page: {entry['landing_page_url']}")
            if entry.get("pdf_url"):
                shortlist_lines.append(f"- pdf_url: {entry['pdf_url']}")
            if entry.get("pdf_path"):
                shortlist_lines.append(f"- downloaded_pdf: `{entry['pdf_path']}`")
            shortlist_lines.append("")
        shortlist_md.write_text("\n".join(shortlist_lines).strip() + "\n", encoding="utf-8")
        shortlist_json.write_text(json.dumps({"results": filtered}, indent=2), encoding="utf-8")
        print(f"Wrote {shortlist_md}")
        print(f"Wrote {shortlist_json}")

    print(f"Wrote {out_md}")
    print(f"Wrote {out_json}")
    print(f"Artifacts dir: {artifacts_dir}")


if __name__ == "__main__":
    main()
