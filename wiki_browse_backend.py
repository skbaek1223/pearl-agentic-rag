from __future__ import annotations

import json
import os
import pickle
import time
import unicodedata
from typing import Optional
from urllib.parse import quote, unquote, urlparse

import requests

WIKI_URL_PREFIX = "https://en.wikipedia.org/w/index.php?title="
DEFAULT_PAGES = os.environ.get("PEARL_ASEARCHER_PAGES_PATH")
DEFAULT_PAGE_INDEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "asearcher_page_index")


def parse_contents(doc: dict) -> tuple[str, str]:
    contents = doc.get("contents", doc.get("text", ""))
    title, _, text = contents.partition("\n")
    return title.strip().strip('"'), text.strip()


def title_to_url(title: str) -> str:
    return WIKI_URL_PREFIX + quote(title, safe="()'!,:-._*")


def url_to_title(url: str) -> str:
    u = url.strip().strip("<>").strip()
    for marker in ("index.php?title=", "index.php/", "/wiki/"):
        if marker in u:
            raw = u.split(marker, 1)[1].split("#")[0]
            raw = raw.split("&")[0] if marker == "index.php?title=" else raw.split("?")[0]
            return unquote(raw).split("#")[0].replace("_", " ").strip()
    path = urlparse(u).path if "://" in u else u
    seg = [s for s in path.split("/") if s]
    t = seg[-1] if seg else u
    return unquote(t).replace("_", " ").replace("-", " ").strip()


def _norm(title: str) -> str:
    return unicodedata.normalize("NFC", title).lower()


class ASearcherPageStore:

    def __init__(self, pages_path: str = DEFAULT_PAGES, index_dir: str = DEFAULT_PAGE_INDEX):
        with open(os.path.join(index_dir, "titles.pkl"), "rb") as f:
            self.titles: dict[str, tuple[int, int]] = pickle.load(f)
        self.lower: dict[str, str] = {}
        for t in self.titles:
            self.lower.setdefault(_norm(t), t)
        self.fd = os.open(pages_path, os.O_RDONLY)
        print(f"[asearcher-pages] {len(self.titles):,} pages", flush=True)

    def resolve(self, title: str) -> Optional[str]:
        if title in self.titles:
            return title
        cap = title[:1].upper() + title[1:]
        if cap in self.titles:
            return cap
        return self.lower.get(_norm(title))

    def page(self, title: str, max_chars: int) -> str:
        key = self.resolve(title)
        if key is None:
            return ""
        offset, n = self.titles[key]
        return json.loads(os.pread(self.fd, n, offset))["contents"][:max_chars]


class WikiBrowseBackend:
    def __init__(self, retriever_url: str, top_k: int = 5, page_max_chars: int = 10000,
                 chunk: int = 256, timeout: float = 600, pages: Optional[ASearcherPageStore] = None):
        self.pages = pages or ASearcherPageStore()
        self.url = retriever_url.rstrip("/")
        self.top_k = top_k
        self.page_max_chars = page_max_chars
        self.chunk = chunk
        self.timeout = timeout
        r = self._request(requests.get, f"{self.url}/health", timeout=10)
        print(f"[wiki-backend] connected to {self.url} ({r.json()})", flush=True)

    @staticmethod
    def _request(method, url, retries: int = 10, wait: float = 10.0, **kw):
        for attempt in range(retries):
            try:
                r = method(url, **kw)
                r.raise_for_status()
                return r
            except requests.RequestException as e:
                if attempt + 1 == retries:
                    raise
                print(f"[wiki-backend] {url} failed ({e.__class__.__name__}), retry {attempt + 1}/{retries - 1}", flush=True)
                time.sleep(wait)

    def _batch(self, queries: list[str], k: int) -> list[list[dict]]:
        out: list[list[dict]] = []
        for i in range(0, len(queries), self.chunk):
            part = queries[i:i + self.chunk]
            r = self._request(requests.post, f"{self.url}/batch_search",
                              json={"queries": part, "top_k": k}, timeout=self.timeout)
            out.extend(r.json()["results"])
        return out

    def batch_search(self, queries: list[str]) -> list[list[dict]]:
        if not queries:
            return []
        results = []
        for docs in self._batch(queries, self.top_k):
            hits = []
            for d in docs:
                title, text = parse_contents(d)
                hits.append({"title": title, "text": text, "url": title_to_url(title)})
            results.append(hits)
        return results

    def batch_access(self, urls: list[str], hints: Optional[list[str]] = None) -> list[str]:
        return [self.pages.page(url_to_title(u), self.page_max_chars) for u in urls]
