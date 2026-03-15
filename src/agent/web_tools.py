from __future__ import annotations

import hashlib
import json
import re
import socket
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _host_is_private(host: str) -> bool:
    h = host.strip().lower()
    if not h:
        return True
    if h in {"localhost"} or h.endswith(".local"):
        return True
    try:
        ip = socket.gethostbyname(h)
    except Exception:
        # If we can't resolve, treat as unsafe for SSRF; caller can override by passing explicit IP.
        return True
    if ip.startswith("127.") or ip.startswith("10.") or ip.startswith("192.168.") or ip.startswith("169.254."):
        return True
    if ip.startswith("172."):
        try:
            second = int(ip.split(".")[1])
            if 16 <= second <= 31:
                return True
        except Exception:
            pass
    return False


def validate_http_url(url: str) -> tuple[bool, str]:
    try:
        u = urllib.parse.urlparse(url)
    except Exception as exc:
        return False, f"url parse error: {exc}"
    if u.scheme not in {"http", "https"}:
        return False, "only http/https urls allowed"
    if not u.netloc:
        return False, "missing host"
    host = u.hostname or ""
    if _host_is_private(host):
        return False, f"blocked host (private/localhost): {host}"
    return True, ""


class _HtmlText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip += 1
        if tag in {"p", "br", "li", "div", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip > 0:
            self._skip -= 1
        if tag in {"p", "li", "div"}:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if data and data.strip():
            self._chunks.append(data)

    def text(self) -> str:
        raw = " ".join(self._chunks)
        # normalize whitespace but keep newlines
        raw = re.sub(r"[ \t\f\v]+", " ", raw)
        raw = re.sub(r"\n\s+", "\n", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def html_to_text(html: str) -> str:
    p = _HtmlText()
    try:
        p.feed(html)
        p.close()
    except Exception:
        return html
    return p.text()


def decode_openalex_abstract(inverted_index: dict[str, Any] | None) -> str:
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""
    max_pos = -1
    for positions in inverted_index.values():
        if not isinstance(positions, list):
            continue
        for p in positions:
            try:
                max_pos = max(max_pos, int(p))
            except Exception:
                continue
    if max_pos < 0:
        return ""
    words = [""] * (max_pos + 1)
    for word, positions in inverted_index.items():
        if not isinstance(word, str) or not isinstance(positions, list):
            continue
        for p in positions:
            try:
                idx = int(p)
            except Exception:
                continue
            if 0 <= idx < len(words):
                words[idx] = word
    return " ".join(w for w in words if w).strip()


@dataclass(frozen=True)
class HttpResult:
    url: str
    status: int
    headers: dict[str, str]
    body: bytes
    truncated: bool


def http_get(url: str, *, timeout_s: float = 20.0, max_bytes: int = 2_000_000, user_agent: str = "climate-scientist-agent/0.1") -> HttpResult:
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
        status = int(getattr(resp, "status", 200))
        headers = {k: v for k, v in resp.headers.items()}
        data = resp.read(int(max_bytes) + 1)
        truncated = len(data) > int(max_bytes)
        if truncated:
            data = data[: int(max_bytes)]
        return HttpResult(url=url, status=status, headers=headers, body=data, truncated=truncated)


def openalex_search_works(
    query: str,
    *,
    max_results: int = 10,
    from_publication_date: str | None = None,
    sort: str = "publication_date:desc,cited_by_count:desc",
    require_abstract: bool = True,
    require_oa: bool = False,
    require_pdf: bool = False,
) -> dict[str, Any]:
    per_page = int(max(1, min(25, max_results)))

    filters = []
    if from_publication_date:
        filters.append(f"from_publication_date:{from_publication_date}")
    if require_abstract:
        filters.append("has_abstract:true")
    if require_oa:
        filters.append("open_access.is_oa:true")
    if require_pdf:
        filters.append("has_fulltext:true")

    params = {
        "search": query,
        "per-page": str(per_page),
        "sort": sort,
        "select": ",".join(
            [
                "id",
                "doi",
                "display_name",
                "publication_year",
                "publication_date",
                "cited_by_count",
                "primary_location",
                "best_oa_location",
                "open_access",
                "abstract_inverted_index",
            ]
        ),
    }
    if filters:
        params["filter"] = ",".join(filters)

    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params, safe=":,")
    res = http_get(url, timeout_s=20.0, max_bytes=2_000_000)
    data = json.loads(res.body.decode("utf-8", errors="replace"))

    out_results = []
    for w in (data.get("results") or [])[:per_page]:
        if not isinstance(w, dict):
            continue
        primary = w.get("primary_location") or {}
        best = w.get("best_oa_location") or {}
        oa = w.get("open_access") or {}

        abstract = decode_openalex_abstract(w.get("abstract_inverted_index"))
        out_results.append(
            {
                "title": w.get("display_name") or "",
                "openalex_id": w.get("id") or "",
                "doi": w.get("doi") or "",
                "publication_year": w.get("publication_year"),
                "publication_date": w.get("publication_date") or "",
                "cited_by_count": w.get("cited_by_count"),
                "venue": ((primary.get("source") or {}) if isinstance(primary, dict) else {}).get("display_name") if isinstance(primary, dict) else "",
                "landing_page_url": primary.get("landing_page_url") if isinstance(primary, dict) else "",
                "pdf_url": (best.get("pdf_url") if isinstance(best, dict) else "") or (primary.get("pdf_url") if isinstance(primary, dict) else ""),
                "is_oa": oa.get("is_oa") if isinstance(oa, dict) else None,
                "oa_url": oa.get("oa_url") if isinstance(oa, dict) else "",
                "abstract": abstract,
            }
        )

    return {"query": query, "url": url, "status": res.status, "truncated": res.truncated, "results": out_results}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

