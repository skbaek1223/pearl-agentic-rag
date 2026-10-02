from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

DEFAULT_E5 = os.environ.get("PEARL_E5_MODEL_PATH")


def split_words(text: str, words: int = 100) -> list[str]:
    toks = text.split()
    return [" ".join(toks[i:i + words]) for i in range(0, len(toks), words)]


class PageChunkSelector:
    def __init__(self, model_path: str = DEFAULT_E5, k: int = 5, words: int = 100,
                 device: str = "cuda", batch_size: int = 128):
        self.k, self.words, self.batch_size = k, words, batch_size
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16).to(self.device).eval()

    @torch.no_grad()
    def _embed(self, texts: list[str]) -> torch.Tensor:
        out = []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tok(texts[i:i + self.batch_size], padding=True, truncation=True,
                           max_length=512, return_tensors="pt").to(self.device)
            h = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            out.append(F.normalize((h * mask).sum(1) / mask.sum(1), dim=-1).float())
        return torch.cat(out) if out else torch.empty(0)

    def select_chunks(self, pages: list[str], queries: list[str]) -> list[list[str]]:
        chunked = [split_words(p, self.words) if p else [] for p in pages]
        todo = [i for i, c in enumerate(chunked) if len(c) > self.k]
        if not todo:
            return chunked
        q_emb = self._embed([f"query: {queries[i]}" for i in todo])
        flat = [(i, j) for i in todo for j in range(len(chunked[i]))]
        p_emb = self._embed([f"passage: {chunked[i][j]}" for i, j in flat])
        pos = 0
        result = list(chunked)
        for qi, i in enumerate(todo):
            n = len(chunked[i])
            scores = p_emb[pos:pos + n] @ q_emb[qi]
            pos += n
            result[i] = [chunked[i][j] for j in sorted(scores.topk(self.k).indices.tolist())]
        return result

    def select(self, pages: list[str], queries: list[str]) -> list[str]:
        return ["\n\n...\n\n".join(cs) for cs in self.select_chunks(pages, queries)]


def _url_key(url: str) -> str:
    from wiki_browse_backend import url_to_title
    return url_to_title(url).lower()


def remember_search(process: dict, query: str, hits: list[dict]):
    seen = process.setdefault("_url_query", {})
    for h in hits:
        seen[_url_key(h["url"])] = query
    process["_last_query"] = query


def query_for(process: dict, url: str) -> str:
    return (process.get("_url_query", {}).get(_url_key(url)) or process.get("_last_query")
            or process["question"])
