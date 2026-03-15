"""Full-text paper reading via arXiv, Semantic Scholar, and PDF extraction.

Provides multi-source literature search, PDF download + text extraction
(via pymupdf4llm), and literature context for novelty checking.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

log = logging.getLogger("daemon.literature")

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

_last_call: dict[str, float] = {}


def _rate_limit(api: str, interval: float) -> None:
    """Block until at least `interval` seconds since last call for `api`."""
    now = time.monotonic()
    last = _last_call.get(api, 0.0)
    wait = interval - (now - last)
    if wait > 0:
        time.sleep(wait)
    _last_call[api] = time.monotonic()


# ---------------------------------------------------------------------------
# LiteratureManager
# ---------------------------------------------------------------------------


class LiteratureManager:
    """Multi-source literature search, PDF download, and text extraction."""

    def __init__(self, config: Any, db: Any) -> None:
        self._config = config
        self._db = db
        self._papers_dir = Path(
            getattr(config, "literature_papers_dir", "data/papers")
        )
        self._max_pdf_bytes = int(
            getattr(config, "literature_max_pdf_bytes", 50_000_000)
        )
        self._max_fulltext_chars = int(
            getattr(config, "literature_max_fulltext_chars", 500_000)
        )

    # -- arXiv -----------------------------------------------------------

    def search_arxiv(
        self,
        query: str,
        max_results: int = 10,
        sort_by: str = "relevance",
    ) -> list[dict[str, Any]]:
        """Search arXiv and store results in the DB. Returns list of paper dicts."""
        try:
            import arxiv
        except ImportError:
            log.warning("arxiv package not installed")
            return []

        _rate_limit("arxiv", 0.15)

        sort_map = {
            "relevance": arxiv.SortCriterion.Relevance,
            "submitted": arxiv.SortCriterion.SubmittedDate,
            "updated": arxiv.SortCriterion.LastUpdatedDate,
        }
        sort_criterion = sort_map.get(sort_by, arxiv.SortCriterion.Relevance)

        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_criterion,
        )

        results: list[dict[str, Any]] = []
        try:
            for paper in client.results(search):
                paper_id = self._store_arxiv_paper(paper)
                results.append({
                    "paper_id": paper_id,
                    "title": paper.title,
                    "authors": ", ".join(a.name for a in paper.authors[:5]),
                    "year": paper.published.year if paper.published else None,
                    "arxiv_id": paper.get_short_id(),
                    "abstract": paper.summary[:500] if paper.summary else "",
                    "pdf_url": paper.pdf_url,
                    "categories": paper.categories[:5],
                })
        except Exception as exc:
            log.warning("arXiv search failed: %s", exc)

        return results

    def _store_arxiv_paper(self, paper: Any) -> str:
        """Dedup and store an arXiv paper. Returns paper_id."""
        arxiv_id = paper.get_short_id()

        # Check existing by arxiv_id
        existing = self._db.get_paper_by_arxiv_id(arxiv_id)
        if existing:
            # Merge pdf_url if missing
            if not existing.get("pdf_url") and paper.pdf_url:
                self._db.update_paper_ids(existing["id"], pdf_url=paper.pdf_url)
            return existing["id"]

        # Check by DOI
        doi = None
        if hasattr(paper, "doi") and paper.doi:
            doi = paper.doi
            existing = self._db.get_paper_by_doi(doi)
            if existing:
                self._db.update_paper_ids(
                    existing["id"], arxiv_id=arxiv_id, pdf_url=paper.pdf_url
                )
                return existing["id"]

        # Insert new
        from ulid import ULID
        paper_id = f"arxiv:{arxiv_id}"
        authors = "; ".join(a.name for a in paper.authors[:10])
        year = paper.published.year if paper.published else None

        self._db.insert_paper({
            "id": paper_id,
            "title": paper.title or "",
            "authors": authors,
            "year": year,
            "doi": doi,
            "abstract": paper.summary or "",
            "source": "arxiv",
            "url": paper.entry_id,
            "cited_by_count": 0,
        })
        self._db.update_paper_ids(
            paper_id, arxiv_id=arxiv_id, pdf_url=paper.pdf_url
        )
        return paper_id

    # -- Semantic Scholar ------------------------------------------------

    def search_semantic_scholar(
        self,
        query: str,
        max_results: int = 10,
        year: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search Semantic Scholar and store results. Returns list of paper dicts."""
        _rate_limit("s2", 1.1)

        params = {
            "query": query,
            "limit": min(max_results, 100),
            "fields": "title,authors,year,abstract,externalIds,citationCount,tldr,openAccessPdf",
        }
        if year:
            params["year"] = year

        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        query_string = "&".join(
            f"{k}={urllib.request.quote(str(v))}" for k, v in params.items()
        )
        full_url = f"{url}?{query_string}"

        results: list[dict[str, Any]] = []
        try:
            req = urllib.request.Request(full_url)
            req.add_header("User-Agent", "ClimateResearchAgent/1.0")
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())

            for item in data.get("data", []):
                paper_id = self._store_s2_paper(item)
                tldr = item.get("tldr")
                tldr_text = tldr.get("text", "") if isinstance(tldr, dict) else ""
                authors_list = item.get("authors", [])
                results.append({
                    "paper_id": paper_id,
                    "title": item.get("title", ""),
                    "authors": ", ".join(
                        a.get("name", "") for a in authors_list[:5]
                    ),
                    "year": item.get("year"),
                    "s2_id": item.get("paperId", ""),
                    "abstract": (item.get("abstract") or "")[:500],
                    "tldr": tldr_text,
                    "cited_by_count": item.get("citationCount", 0),
                })
        except Exception as exc:
            log.warning("Semantic Scholar search failed: %s", exc)

        return results

    def _store_s2_paper(self, item: dict[str, Any]) -> str:
        """Dedup and store a Semantic Scholar paper. Returns paper_id."""
        s2_id = item.get("paperId", "")
        ext_ids = item.get("externalIds") or {}
        doi = ext_ids.get("DOI")
        arxiv_id = ext_ids.get("ArXiv")

        # Dedup by s2_id
        if s2_id:
            existing = self._db.get_paper_by_s2_id(s2_id)
            if existing:
                return existing["id"]

        # Dedup by arxiv_id
        if arxiv_id:
            existing = self._db.get_paper_by_arxiv_id(arxiv_id)
            if existing:
                self._db.update_paper_ids(existing["id"], s2_id=s2_id)
                return existing["id"]

        # Dedup by DOI
        if doi:
            existing = self._db.get_paper_by_doi(doi)
            if existing:
                self._db.update_paper_ids(
                    existing["id"], s2_id=s2_id, arxiv_id=arxiv_id
                )
                return existing["id"]

        # Insert new
        paper_id = f"s2:{s2_id}" if s2_id else f"s2:{hashlib.sha256(item.get('title', '').encode()).hexdigest()[:16]}"
        authors_list = item.get("authors", [])
        authors_str = "; ".join(a.get("name", "") for a in authors_list[:10])

        # Extract PDF URL
        pdf_url = None
        oa_pdf = item.get("openAccessPdf")
        if isinstance(oa_pdf, dict):
            pdf_url = oa_pdf.get("url")

        self._db.insert_paper({
            "id": paper_id,
            "title": item.get("title", ""),
            "authors": authors_str,
            "year": item.get("year"),
            "doi": doi,
            "abstract": item.get("abstract") or "",
            "source": "semantic_scholar",
            "url": f"https://api.semanticscholar.org/CorpusID:{s2_id}" if s2_id else None,
            "cited_by_count": item.get("citationCount", 0),
        })
        self._db.update_paper_ids(
            paper_id, s2_id=s2_id, arxiv_id=arxiv_id, pdf_url=pdf_url
        )
        return paper_id

    # -- PDF download + extraction ---------------------------------------

    def fetch_paper_pdf(self, paper_id: str) -> dict[str, Any]:
        """Download PDF for a paper, extract text, store in DB. Returns status dict."""
        paper = self._db.get_paper_by_id(paper_id)
        if not paper:
            return {"ok": False, "error": f"Paper {paper_id} not found"}

        pdf_url = paper.get("pdf_url") or ""
        if not pdf_url:
            return {"ok": False, "error": "No PDF URL for this paper"}

        # Check if already fetched
        existing_file = self._db.get_paper_file(paper_id, "pdf")
        if existing_file and paper.get("full_text"):
            return {
                "ok": True,
                "status": "already_fetched",
                "chars": len(paper.get("full_text", "")),
            }

        try:
            pdf_path = self._download_pdf(paper_id, pdf_url)
            full_text = self._extract_pdf_text(pdf_path)

            # Store full text
            self._db.update_paper_full_text(paper_id, full_text)

            # Store file record
            file_size = pdf_path.stat().st_size
            sha256 = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
            self._db.insert_paper_file({
                "paper_id": paper_id,
                "file_type": "pdf",
                "local_path": str(pdf_path),
                "sha256": sha256,
                "size_bytes": file_size,
            })

            log.info(
                "Fetched PDF for %s: %d bytes, %d chars text",
                paper_id, file_size, len(full_text),
            )
            return {
                "ok": True,
                "status": "fetched",
                "pdf_path": str(pdf_path),
                "chars": len(full_text),
                "size_bytes": file_size,
            }
        except Exception as exc:
            log.warning("PDF fetch failed for %s: %s", paper_id, exc)
            return {"ok": False, "error": str(exc)}

    def _download_pdf(self, paper_id: str, pdf_url: str) -> Path:
        """Download a PDF to the local papers directory. Returns path."""
        safe_id = re.sub(r"[^a-zA-Z0-9._-]", "_", paper_id)
        paper_dir = self._papers_dir / safe_id
        paper_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = paper_dir / "paper.pdf"

        if pdf_path.exists():
            return pdf_path

        req = urllib.request.Request(pdf_url)
        req.add_header("User-Agent", "ClimateResearchAgent/1.0")

        with urllib.request.urlopen(req, timeout=60) as resp:
            content_length = resp.headers.get("Content-Length")
            if content_length and int(content_length) > self._max_pdf_bytes:
                raise ValueError(
                    f"PDF too large: {content_length} bytes "
                    f"(max {self._max_pdf_bytes})"
                )

            # Read in chunks with size limit
            data = bytearray()
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                data.extend(chunk)
                if len(data) > self._max_pdf_bytes:
                    raise ValueError(
                        f"PDF exceeded max size during download "
                        f"({len(data)} bytes)"
                    )

            pdf_path.write_bytes(bytes(data))

        return pdf_path

    def _extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text from PDF using pymupdf4llm. Returns markdown string."""
        try:
            import pymupdf4llm
        except ImportError:
            log.warning("pymupdf4llm not installed, falling back to basic extraction")
            return self._extract_pdf_text_fallback(pdf_path)

        try:
            text = pymupdf4llm.to_markdown(str(pdf_path))
        except Exception as exc:
            log.warning("pymupdf4llm extraction failed: %s, trying fallback", exc)
            return self._extract_pdf_text_fallback(pdf_path)

        # Truncate to max chars
        if len(text) > self._max_fulltext_chars:
            text = text[: self._max_fulltext_chars]

        return text

    def _extract_pdf_text_fallback(self, pdf_path: Path) -> str:
        """Fallback PDF text extraction using pymupdf directly."""
        try:
            import pymupdf
            doc = pymupdf.open(str(pdf_path))
            pages = []
            for page in doc:
                pages.append(page.get_text())
            text = "\n\n".join(pages)
            if len(text) > self._max_fulltext_chars:
                text = text[: self._max_fulltext_chars]
            return text
        except Exception as exc:
            log.warning("Fallback PDF extraction also failed: %s", exc)
            return ""

    # -- Reading ---------------------------------------------------------

    def read_paper(
        self, paper_id: str, max_chars: int = 50000
    ) -> dict[str, Any]:
        """Read a paper's full text. Auto-fetches PDF if needed."""
        paper = self._db.get_paper_by_id(paper_id)
        if not paper:
            return {"ok": False, "error": f"Paper {paper_id} not found"}

        full_text = paper.get("full_text") or ""

        # Auto-fetch if no full text but has PDF URL
        if not full_text and paper.get("pdf_url"):
            result = self.fetch_paper_pdf(paper_id)
            if result.get("ok"):
                paper = self._db.get_paper_by_id(paper_id)
                full_text = (paper or {}).get("full_text", "")

        if not full_text:
            # Return what we have (metadata + abstract)
            return {
                "ok": True,
                "paper_id": paper_id,
                "title": paper.get("title", ""),
                "authors": paper.get("authors", ""),
                "year": paper.get("year"),
                "abstract": paper.get("abstract", ""),
                "full_text": "",
                "note": "No full text available (no PDF URL)",
            }

        # Truncate to requested max
        truncated = full_text[:max_chars]
        return {
            "ok": True,
            "paper_id": paper_id,
            "title": paper.get("title", ""),
            "authors": paper.get("authors", ""),
            "year": paper.get("year"),
            "abstract": paper.get("abstract", ""),
            "full_text": truncated,
            "full_text_chars": len(full_text),
            "truncated": len(full_text) > max_chars,
        }

    # -- Combined search -------------------------------------------------

    def search_all(
        self,
        query: str,
        max_per_source: int = 5,
        sources: list[str] | None = None,
    ) -> dict[str, Any]:
        """Search arXiv + Semantic Scholar + OpenAlex. Returns combined results."""
        allowed = sources or ["arxiv", "semantic_scholar", "openalex"]
        results: dict[str, list[dict[str, Any]]] = {}

        if "arxiv" in allowed:
            try:
                results["arxiv"] = self.search_arxiv(query, max_results=max_per_source)
            except Exception as exc:
                log.warning("arXiv search error: %s", exc)
                results["arxiv"] = []

        if "semantic_scholar" in allowed:
            try:
                results["semantic_scholar"] = self.search_semantic_scholar(
                    query, max_results=max_per_source
                )
            except Exception as exc:
                log.warning("S2 search error: %s", exc)
                results["semantic_scholar"] = []

        if "openalex" in allowed:
            try:
                from agent.web_tools import openalex_search_works
                oa_results = openalex_search_works(query, max_results=max_per_source)
                oa_papers = []
                for p in oa_results.get("results", []):
                    oa_papers.append({
                        "paper_id": p.get("id", ""),
                        "title": p.get("title", ""),
                        "year": (p.get("publication_date") or "")[:4] or None,
                        "cited_by_count": p.get("cited_by_count", 0),
                    })
                results["openalex"] = oa_papers
            except Exception as exc:
                log.warning("OpenAlex search error: %s", exc)
                results["openalex"] = []

        total = sum(len(v) for v in results.values())
        return {"ok": True, "total": total, "results": results}

    # -- For novelty checker ---------------------------------------------

    def gather_literature_context(
        self,
        idea: str,
        max_external: int = 5,
        max_local: int = 10,
    ) -> str:
        """Gather literature context for novelty checking.

        Searches arXiv + Semantic Scholar for real-time results,
        plus local fulltext FTS.
        """
        parts: list[str] = []

        # Local fulltext search
        try:
            local_papers = self._db.search_papers_fulltext_fts(idea, limit=max_local)
            if local_papers:
                parts.append("=== Local paper corpus ===")
                for p in local_papers:
                    abstract = (p.get("abstract") or "")[:200]
                    parts.append(
                        f"- {p.get('title', '')} ({p.get('year', '?')}): {abstract}"
                    )
        except Exception:
            pass

        # arXiv search
        try:
            arxiv_results = self.search_arxiv(idea, max_results=max_external)
            if arxiv_results:
                parts.append("=== arXiv (recent) ===")
                for r in arxiv_results:
                    parts.append(
                        f"- {r.get('title', '')} ({r.get('year', '?')}): "
                        f"{r.get('abstract', '')[:200]}"
                    )
        except Exception:
            pass

        # Semantic Scholar search
        try:
            s2_results = self.search_semantic_scholar(idea, max_results=max_external)
            if s2_results:
                parts.append("=== Semantic Scholar ===")
                for r in s2_results:
                    tldr = r.get("tldr", "")
                    desc = tldr if tldr else (r.get("abstract") or "")[:200]
                    parts.append(
                        f"- {r.get('title', '')} ({r.get('year', '?')}, "
                        f"cited={r.get('cited_by_count', 0)}): {desc}"
                    )
        except Exception:
            pass

        return "\n".join(parts)
