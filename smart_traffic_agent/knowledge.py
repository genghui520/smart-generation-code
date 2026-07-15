from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from .models import KnowledgeChunk, RetrievedChunk
from .utils import tokenize, write_json


# 与 build_rag_index.py 保持一致的 embedding 模型
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"


class _BGEEmbedding:
    """封装 sentence-transformers 用于查询时 embedding。"""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.name = model_name

    def __call__(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts, show_progress_bar=False, normalize_embeddings=True
        )
        return embeddings.tolist()


class KnowledgeBase:
    """FOCAS 知识库：支持 Chroma 向量检索 + 词项匹配降级。"""

    def __init__(
        self,
        chunks: list[KnowledgeChunk] | None = None,
        *,
        vector_db_dir: str | Path | None = None,
    ) -> None:
        self.chunks = chunks or []
        self._collection = None
        self._chroma_client = None
        self._embed_fn = None

        if vector_db_dir:
            self._load_vector_db(vector_db_dir)

    # ── 向量库初始化 ──────────────────────────────────────

    def _load_vector_db(self, path: str | Path) -> None:
        """加载 Chroma 持久化向量库。"""
        path = Path(path)
        if not path.exists():
            print(f"[KnowledgeBase] vector DB not found: {path}")
            return
        self._chroma_client = chromadb.PersistentClient(path=str(path))
        try:
            self._collection = self._chroma_client.get_collection("focas_knowledge")
            self._embed_fn = _BGEEmbedding(EMBEDDING_MODEL)
            count = self._collection.count()
            print(f"[KnowledgeBase] loaded: {path} ({count} docs)")
        except Exception as e:
            print(f"[KnowledgeBase] load failed: {e}")
            self._collection = None

    @property
    def vector_search_available(self) -> bool:
        return self._collection is not None

    # ── 序列化 ────────────────────────────────────────────

    @classmethod
    def from_json(cls, path: Path) -> "KnowledgeBase":
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        chunks = [KnowledgeChunk(**item) for item in raw.get("chunks", raw)]
        return cls(chunks)

    def to_json(self, path: Path) -> None:
        write_json(path, {"chunks": self.chunks})

    # ── 添加文本 ──────────────────────────────────────────

    def add_text(
        self,
        text: str,
        *,
        source: str,
        metadata: dict[str, str] | None = None,
        chunk_size: int = 900,
    ) -> None:
        metadata = metadata or {}
        paragraphs = [part.strip() for part in text.splitlines() if part.strip()]
        buffer: list[str] = []
        current_len = 0
        index = len(self.chunks) + 1

        for paragraph in paragraphs:
            if current_len + len(paragraph) > chunk_size and buffer:
                self._append_chunk(index, source, "\n".join(buffer), metadata)
                index += 1
                buffer = []
                current_len = 0
            buffer.append(paragraph)
            current_len += len(paragraph)

        if buffer:
            self._append_chunk(index, source, "\n".join(buffer), metadata)

    # ── 检索（混合：先向量，降级到词项） ──────────────────

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievedChunk]:
        """搜索最相关的知识片段。

        优先使用 Chroma 向量检索；若不可用则降级到词项匹配。
        """
        if self._collection is not None:
            return self._vector_search(query, top_k=top_k)
        return self._term_search(query, top_k=top_k)

    def search_by_scenario(
        self, query: str, scenario: str, *, top_k: int = 3
    ) -> list[RetrievedChunk]:
        """按场景过滤 + 向量检索。"""
        if self._collection is not None:
            return self._vector_search(
                query, top_k=top_k, where={"scenario": scenario}
            )
        return self._term_search(query, top_k=top_k)

    def search_by_type(
        self, query: str, rule_type: str, *, top_k: int = 3
    ) -> list[RetrievedChunk]:
        """按规则类型过滤 + 向量检索。"""
        return self.search_rules_by_type(query, rule_type, top_k=top_k)

    def search_rules_by_type(
        self, query: str, rule_type: str, *, top_k: int = 3
    ) -> list[RetrievedChunk]:
        """仅在指定规则类型中检索。"""
        if self._collection is not None:
            return self._vector_search(
                query, top_k=top_k, where={"rule_type": rule_type}
            )
        return [
            item
            for item in self._term_search(query, top_k=top_k * 4)
            if item.chunk.metadata.get("rule_type") == rule_type
            or item.chunk.metadata.get("type") == rule_type
        ][:top_k]

    def search_api(
        self, query: str, *, top_k: int = 5
    ) -> list[RetrievedChunk]:
        """仅在 API 函数中检索。"""
        if self._collection is not None:
            return self._vector_search(
                query, top_k=top_k, where={"source_type": "api"}
            )
        return self._term_search(query, top_k=top_k)

    def search_rules(
        self, query: str, *, top_k: int = 5
    ) -> list[RetrievedChunk]:
        """仅在规则中检索。"""
        if self._collection is not None:
            return self._vector_search(
                query, top_k=top_k, where={"source_type": "rule"}
            )
        return self._term_search(query, top_k=top_k)

    def search_scenario_organization(
        self, query: str, scenario: str, *, top_k: int = 2
    ) -> list[RetrievedChunk]:
        """检索场景中心知识组织结果。"""
        if self._collection is not None:
            return self._vector_search(
                query,
                top_k=top_k,
                where={"$and": [{"source_type": "scenario"}, {"scenario": scenario}]},
            )
        return self._term_search(query, top_k=top_k)

    # ── 向量检索核心 ──────────────────────────────────────

    def _vector_search(
        self,
        query: str,
        *,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        if self._collection is None or self._embed_fn is None:
            return []

        # 先用模型对 query 做 embedding
        query_emb = self._embed_fn([query])

        params: dict[str, Any] = {
            "query_embeddings": query_emb,
            "n_results": min(top_k, 50),
        }
        if where:
            params["where"] = where

        try:
            results = self._collection.query(**params)
        except Exception as e:
            print(f"[KnowledgeBase] 向量检索失败: {e}")
            return []

        retrieved: list[RetrievedChunk] = []
        if not results["documents"]:
            return retrieved

        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunk = KnowledgeChunk(
                chunk_id=meta.get("chunk_id", meta.get("source_chunk_id", "")),
                text=doc,
                metadata=dict(meta),  # 转为普通 dict
            )
            # Chroma 距离越小越相似，转为 0~1 分数
            score = max(0.0, 1.0 - dist / 2.0)
            retrieved.append(RetrievedChunk(chunk=chunk, score=round(score, 4)))

        return retrieved

    # ── 词项匹配降级 ──────────────────────────────────────

    def _term_search(self, query: str, *, top_k: int = 5) -> list[RetrievedChunk]:
        query_terms = set(expand_terms(tokenize(query)))
        scored: list[RetrievedChunk] = []

        for chunk in self.chunks:
            metadata_text = " ".join(str(value) for value in chunk.metadata.values())
            chunk_terms = set(
                expand_terms(tokenize(chunk.text + " " + metadata_text))
            )
            if not chunk_terms:
                continue
            overlap = query_terms.intersection(chunk_terms)
            score = len(overlap) / max(len(query_terms), 1)
            if score:
                scored.append(RetrievedChunk(chunk=chunk, score=score))

        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

    # ── 内部工具 ──────────────────────────────────────────

    def _append_chunk(
        self, index: int, source: str, text: str, metadata: dict[str, str]
    ) -> None:
        chunk_metadata = {"source": source, **metadata}
        self.chunks.append(
            KnowledgeChunk(
                chunk_id=f"{Path(source).stem}-{index:04d}",
                text=text,
                metadata=chunk_metadata,
            )
        )


def expand_terms(terms: list[str]) -> list[str]:
    aliases = {
        "coordinate": ["position", "axis", "坐标", "轴"],
        "position": ["coordinate", "坐标"],
        "spindle": ["主轴", "speed"],
        "program": ["nc", "gcode", "程序"],
        "parameter": ["参数", "config"],
        "alarm": ["报警", "fault"],
        "status": ["state", "状态"],
        "traffic": ["流量", "packet", "capture"],
        "capture": ["pcap", "抓包", "流量"],
        "坐标": ["coordinate", "position", "axis"],
        "主轴": ["spindle", "speed"],
        "程序": ["program", "nc", "gcode"],
        "参数": ["parameter", "config"],
        "报警": ["alarm", "fault"],
        "状态": ["status", "state"],
        "流量": ["traffic", "packet", "capture"],
    }
    expanded = list(terms)
    for term in terms:
        expanded.extend(aliases.get(term, []))
    return expanded


def sample_knowledge() -> KnowledgeBase:
    chunks = [
        KnowledgeChunk(
            chunk_id="api-coordinate-0001",
            text=(
                "ReadPosition reads current X/Y/Z coordinates during CNC motion. "
                "Parameters: axis list, coordinate system, sample interval. "
                "Use it while an NC program is running."
            ),
            metadata={"type": "api", "interface": "ReadPosition", "scene": "coordinate_motion"},
        ),
        KnowledgeChunk(
            chunk_id="api-status-0001",
            text=(
                "ReadRunStatus returns idle, running, paused, completed, or alarm. "
                "Call it before, during, and after program execution."
            ),
            metadata={"type": "api", "interface": "ReadRunStatus", "scene": "program_state"},
        ),
        KnowledgeChunk(
            chunk_id="api-program-0001",
            text=(
                "UploadProgram, SelectProgram, StartProgram, and StopProgram control "
                "NC program lifecycle for simulator execution."
            ),
            metadata={"type": "api", "interface": "ProgramLifecycle", "scene": "program_state"},
        ),
        KnowledgeChunk(
            chunk_id="api-feed-0001",
            text=(
                "ReadFeedSpeed reads feed speed during movement. It is useful for "
                "traffic generated by coordinate and machining state changes."
            ),
            metadata={"type": "api", "interface": "ReadFeedSpeed", "scene": "coordinate_motion"},
        ),
        KnowledgeChunk(
            chunk_id="nc-motion-0001",
            text=(
                "A straight interpolation NC program can use G90 G01 moves over "
                "X, Y, and Z, with M03 spindle start and M30 program end."
            ),
            metadata={"type": "nc_template", "scene": "coordinate_motion"},
        ),
    ]
    return KnowledgeBase(chunks)
