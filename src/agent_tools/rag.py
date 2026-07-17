"""Index de recherche hybride (BM25 + embeddings) sur la doc LangGraph.

2026 stack :
  - RecursiveCharacterTextSplitter (langchain_text_splitters)
  - BM25S (bm25s) — sparse
  - SentenceTransformer (all-MiniLM-L6-v2) — dense
  - Fusion RRF
  - Pas de cross-encoder (overkill à cette échelle)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import bm25s
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = Path(__file__).resolve().parent.parent.parent / "docs"


@dataclass
class Chunk:
    source: str
    title: str
    text: str
    index: int


# ── Index hybride ────────────────────────────────────────────────────
class DocIndex:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None
        self.bm25: bm25s.BM25 | None = None
        self._build()

    # ── Métadonnées ───────────────────────────────────────────────
    def _load_titles(self) -> dict[str, str]:
        index_file = DOCS_DIR / "INDEX.txt"
        titles: dict[str, str] = {}
        if index_file.exists():
            for line in index_file.read_text().strip().split("\n"):
                parts = line.split("  |  ")
                if len(parts) == 2:
                    slug = parts[0].removesuffix(".txt").strip()
                    titles[slug] = parts[1].split("(")[0].strip()
        return titles

    # ── Chunking RecursiveCharacterTextSplitter ──────────────────
    def _chunk_text(self, text: str, size: int = 512, overlap: int = 64) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
            length_function=len,
        )
        return splitter.split_text(text)

    def _build(self):
        titles = self._load_titles()
        for fpath in sorted(DOCS_DIR.glob("*.txt")):
            if fpath.name == "INDEX.txt":
                continue
            slug = fpath.stem
            title = titles.get(slug, slug)
            text = fpath.read_text(encoding="utf-8")
            clean = re.sub(r"\s+", " ", text).strip()
            if not clean:
                continue
            for i, chunk_text in enumerate(self._chunk_text(clean)):
                self.chunks.append(
                    Chunk(source=slug, title=title, text=chunk_text, index=i)
                )

        if not self.chunks:
            raise RuntimeError(f"Aucun document trouvé dans {DOCS_DIR}")

        chunk_texts = [c.text for c in self.chunks]

        self.embeddings = self.model.encode(
            chunk_texts,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        self.bm25 = bm25s.BM25()
        corpus_tokens = bm25s.tokenize(chunk_texts)
        self.bm25.index(corpus_tokens)

    # ── Recherche hybride avec RRF ───────────────────────────────
    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        dense_results = self._search_dense(query, k * 3)
        sparse_results = self._search_sparse(query, k * 3)
        fused: dict[int, float] = {}

        for rank, (idx, _) in enumerate(dense_results):
            fused[idx] = fused.get(idx, 0) + 1.0 / (60 + rank + 1)
        for rank, (idx, _) in enumerate(sparse_results):
            fused[idx] = fused.get(idx, 0) + 1.0 / (60 + rank + 1)

        ranked = sorted(fused.items(), key=lambda x: -x[1])[:k]
        return [(self.chunks[idx], score) for idx, score in ranked]

    def _search_dense(self, query: str, k: int) -> list[tuple[int, float]]:
        q_emb = self.model.encode(query, normalize_embeddings=True)
        scores = np.dot(self.embeddings, q_emb)
        top_indices = np.argsort(scores)[-k:][::-1]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def _search_sparse(self, query: str, k: int) -> list[tuple[int, float]]:
        if self.bm25 is None:
            return []
        query_tokens = bm25s.tokenize(query)
        results, scores = self.bm25.retrieve(query_tokens, k=k)
        indices = results[0].tolist()
        flat_scores = scores[0].tolist()
        return list(zip(indices, flat_scores))


# ── Singleton réutilisable ───────────────────────────────────────────
_index: DocIndex | None = None


def get_index() -> DocIndex:
    global _index
    if _index is None:
        _index = DocIndex()
    return _index
