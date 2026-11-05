from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
import threading

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from .models import KnowledgeChunk, RetrievedChunk
from .utils import tokenize, write_json


# 与 build_rag_index.py 保持一致的 embedding 模型
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"


_EMBEDDING_MODEL_CACHE: dict[str, SentenceTransformer] = {}
_EMBEDDING_MODEL_LOCK = threading.Lock()


def _shared_sentence_transformer(model_name: str) -> SentenceTransformer:
    """Load one embedding model per process and share it across RAG stores."""
    model = _EMBEDDING_MODEL_CACHE.get(model_name)
    if model is not None:
        return model
    with _EMBEDDING_MODEL_LOCK:
        model = _EMBEDDING_MODEL_CACHE.get(model_name)
        if model is None:
            cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
            safe_name = "models--" + model_name.replace("/", "--")
            snapshots = cache_root / "hub" / safe_name / "snapshots"
            local_candidates = [
                item for item in snapshots.iterdir()
                if item.is_dir() and (item / "config.json").is_file()
            ] if snapshots.is_dir() else []
            resolved_model = str(max(local_candidates, key=lambda item: item.stat().st_mtime)) if local_candidates else model_name
            model = SentenceTransformer(
                resolved_model,
                trust_remote_code=True,
                local_files_only=resolved_model != model_name,
            )
            _EMBEDDING_MODEL_CACHE[model_name] = model
        return model


class _BGEEmbedding:
    """封装 sentence-transformers 用于查询时 embedding。"""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.model = _shared_sentence_transformer(model_name)
        self.name = model_name

    def __call__(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts, show_progress_bar=False, normalize_embeddings=True
        )
        return embeddings.tolist()


def _metadata_matches(metadata: dict[str, Any], where: dict[str, Any]) -> bool:
    if "$and" in where:
        return all(_metadata_matches(metadata, item) for item in where["$and"])
    return all(metadata.get(key) == value for key, value in where.items())


class KnowledgeBase:
    """Protocol knowledge base with Chroma/stable-vector and term fallback.

    ``collection_name`` keeps protocol stores isolated.  Existing callers
    retain the FOCAS default, while LSV-2 uses its own collection.
    """

    def __init__(
        self,
        chunks: list[KnowledgeChunk] | None = None,
        *,
        vector_db_dir: str | Path | None = None,
        collection_name: str = "focas_knowledge",
    ) -> None:
        self.chunks = chunks or []
        self.vector_db_path: str | None = str(vector_db_dir) if vector_db_dir else None
        self.collection_name = collection_name
        self.vector_db_status = "not_configured"
        self.vector_db_error = ""
        self._collection = None
        self._chroma_client = None
        self._embed_fn = None
        self._stable_embeddings: np.ndarray | None = None
        self._stable_records: list[dict[str, Any]] = []
        self._query_cache: dict[tuple[str, int, str], list[RetrievedChunk]] = {}

        if vector_db_dir:
            self._load_vector_db(vector_db_dir)

    @property
    def fallback_available(self) -> bool:
        """Whether deterministic local chunks can still answer lookups."""
        return bool(self.chunks)

    @property
    def status(self) -> dict[str, Any]:
        """Return an explicit, UI-safe knowledge source diagnosis."""
        vector_loaded = self.vector_search_available
        if vector_loaded:
            overall = "vector"
        elif self.fallback_available:
            overall = "fallback"
        elif self.vector_db_path and self.vector_db_status in {"missing", "error"}:
            overall = "unavailable"
        else:
            overall = "empty"
        return {
            "status": overall,
            "vector_db": {
                "status": self.vector_db_status,
                "path": self.vector_db_path,
                "collection": self.collection_name,
                "loaded": vector_loaded,
                "error": self.vector_db_error,
            },
            "fallback_available": self.fallback_available,
            "chunk_count": len(self.chunks),
        }

    # ── 向量库初始化 ──────────────────────────────────────

    def _load_vector_db(self, path: str | Path) -> None:
        """加载 Chroma 持久化向量库。"""
        path = Path(path)
        if not path.exists():
            self.vector_db_status = "missing"
            self.vector_db_error = f"vector DB not found: {path}"
            print(f"[KnowledgeBase] {self.vector_db_error}")
            return
        stable_embeddings = path / "stable_embeddings.npy"
        stable_records = path / "stable_records.jsonl"
        if stable_embeddings.exists() and stable_records.exists():
            try:
                self._stable_embeddings = np.load(stable_embeddings, mmap_mode="r")
                with stable_records.open("r", encoding="utf-8") as file:
                    self._stable_records = [json.loads(line) for line in file if line.strip()]
                self._embed_fn = _BGEEmbedding(EMBEDDING_MODEL)
                if len(self._stable_records) != len(self._stable_embeddings):
                    raise ValueError("stable vector/metadata count mismatch")
                self.vector_db_status = "loaded"
                self.vector_db_error = ""
                print(f"[KnowledgeBase] loaded stable vector index: {path} ({len(self._stable_records)} docs)")
                return
            except Exception as e:
                self.vector_db_status = "error"
                self.vector_db_error = f"stable vector index load failed: {e}"
                print(f"[KnowledgeBase] stable vector index load failed: {e}")
                self._stable_embeddings = None
                self._stable_records = []
        try:
            self._chroma_client = chromadb.PersistentClient(path=str(path))
            self._collection = self._chroma_client.get_collection(self.collection_name)
            self._embed_fn = _BGEEmbedding(EMBEDDING_MODEL)
            count = self._collection.count()
            self.vector_db_status = "loaded"
            self.vector_db_error = ""
            print(f"[KnowledgeBase] loaded: {path} ({count} docs)")
        except Exception as e:
            self.vector_db_status = "error"
            self.vector_db_error = str(e)
            print(f"[KnowledgeBase] load failed: {e}")
            self._collection = None

    @property
    def vector_search_available(self) -> bool:
        return self._collection is not None or self._stable_embeddings is not None

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
        if self.vector_search_available:
            return self._vector_search(query, top_k=top_k)
        return self._term_search(query, top_k=top_k)

    def search_by_scenario(
        self, query: str, scenario: str, *, top_k: int = 3
    ) -> list[RetrievedChunk]:
        """按场景过滤 + 向量检索。"""
        if self.vector_search_available:
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
        if self.vector_search_available:
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
        if self.vector_search_available:
            return self._vector_search(
                query, top_k=top_k, where={"source_type": "api"}
            )
        return self._term_search(query, top_k=top_k)

    def lookup_api_exact(
        self,
        functions: list[str] | tuple[str, ...] | set[str],
        *,
        chunk_types: set[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Return API documentation only for exact function metadata matches."""
        wanted = {str(name).strip().casefold() for name in functions if str(name).strip()}
        if not wanted:
            return []
        records: list[dict[str, Any]] = []
        records.extend({
            "id": chunk.chunk_id, "text": chunk.text, "metadata": chunk.metadata,
        } for chunk in self.chunks)
        records.extend(self._stable_records)
        if not records and self._collection is not None:
            for function in sorted(wanted):
                try:
                    result = self._collection.get(where={"function": function})
                except Exception:
                    continue
                ids = result.get("ids", [])
                documents = result.get("documents", [])
                metadatas = result.get("metadatas", [])
                records.extend({"id": ids[i], "text": documents[i], "metadata": metadatas[i] or {}}
                               for i in range(min(len(ids), len(documents), len(metadatas))))
        found: dict[str, RetrievedChunk] = {}
        for record in records:
            metadata = dict(record.get("metadata", {}))
            if str(metadata.get("function", "")).strip().casefold() not in wanted:
                continue
            chunk_type = str(metadata.get("chunk_type", "")).strip()
            if chunk_types is not None and chunk_type not in chunk_types:
                continue
            chunk_id = str(metadata.get("chunk_id", metadata.get("source_chunk_id", record.get("id", ""))))
            if not chunk_id:
                continue
            found.setdefault(chunk_id, RetrievedChunk(
                KnowledgeChunk(chunk_id=chunk_id, text=str(record.get("text", "")), metadata=metadata),
                score=1.0,
            ))
        return sorted(found.values(), key=lambda item: (
            str(item.chunk.metadata.get("function", "")).casefold(),
            str(item.chunk.metadata.get("chunk_type", "")), item.chunk.chunk_id,
        ))

    def lookup_api_prerequisites_exact(
        self, functions: list[str] | tuple[str, ...] | set[str],
    ) -> list[RetrievedChunk]:
        """Return prerequisite rules whose applies_to contains an exact API name."""
        wanted = {str(name).strip().casefold() for name in functions if str(name).strip()}
        if not wanted:
            return []
        records: list[dict[str, Any]] = []
        records.extend({"id": chunk.chunk_id, "text": chunk.text, "metadata": chunk.metadata}
                       for chunk in self.chunks)
        records.extend(self._stable_records)
        if not records and self._collection is not None:
            try:
                result = self._collection.get(where={"rule_type": "api_prerequisite"})
                ids = result.get("ids", [])
                documents = result.get("documents", [])
                metadatas = result.get("metadatas", [])
                records.extend({"id": ids[i], "text": documents[i], "metadata": metadatas[i] or {}}
                               for i in range(min(len(ids), len(documents), len(metadatas))))
            except Exception:
                pass
        found: dict[str, RetrievedChunk] = {}
        for record in records:
            metadata = dict(record.get("metadata", {}))
            if metadata.get("rule_type") != "api_prerequisite":
                continue
            applies = metadata.get("applies_to", [])
            if isinstance(applies, str):
                try:
                    applies = json.loads(applies)
                except json.JSONDecodeError:
                    applies = [part.strip() for part in applies.split(",") if part.strip()]
            if not isinstance(applies, list) or not wanted.intersection(str(name).strip().casefold() for name in applies):
                continue
            chunk_id = str(metadata.get("chunk_id", metadata.get("source_chunk_id", record.get("id", ""))))
            if chunk_id:
                found.setdefault(chunk_id, RetrievedChunk(
                    KnowledgeChunk(chunk_id=chunk_id, text=str(record.get("text", "")), metadata=metadata),
                    score=1.0,
                ))
        return sorted(found.values(), key=lambda item: item.chunk.chunk_id)

    def search_rules(
        self, query: str, *, top_k: int = 5
    ) -> list[RetrievedChunk]:
        """仅在规则中检索。"""
        if self.vector_search_available:
            return self._vector_search(
                query, top_k=top_k, where={"source_type": "rule"}
            )
        return self._term_search(query, top_k=top_k)

    def search_alarms(self, query: str, *, top_k: int = 5) -> list[RetrievedChunk]:
        """仅检索报警号、报警位和恢复条件知识。"""
        if self.vector_search_available:
            return self._vector_search(query, top_k=top_k, where={"source_type": "alarm"})
        return [
            item for item in self._term_search(query, top_k=top_k * 4)
            if item.chunk.metadata.get("source_type") == "alarm"
        ][:top_k]

    def search_scenario_organization(
        self, query: str, scenario: str, *, top_k: int = 2
    ) -> list[RetrievedChunk]:
        """检索场景中心知识组织结果。"""
        if self.vector_search_available:
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
        if self._embed_fn is None:
            return []

        cache_key = (query, min(top_k, 50), json.dumps(where or {}, sort_keys=True, ensure_ascii=False))
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        # 先用模型对 query 做 embedding
        query_emb = self._embed_fn([query])

        if self._stable_embeddings is not None:
            query_vector = np.asarray(query_emb[0], dtype=np.float32)
            matrix = np.asarray(self._stable_embeddings)
            mask = np.ones(len(self._stable_records), dtype=bool)
            for index, record in enumerate(self._stable_records):
                metadata = record.get("metadata", {})
                if where and not _metadata_matches(metadata, where):
                    mask[index] = False
            candidate_indices = np.flatnonzero(mask)
            if not len(candidate_indices):
                return []
            scores = matrix[candidate_indices] @ query_vector
            # Alarm identifiers are exact structured keys; semantic similarity
            # alone can rank a neighboring alarm higher than the requested one.
            if where and where.get("source_type") == "alarm":
                query_tokens = set(re.findall(r"[A-Za-z]+\d{4}|[A-Za-z]+", query))
                for position, candidate_index in enumerate(candidate_indices):
                    metadata = self._stable_records[int(candidate_index)].get("metadata", {})
                    code = str(metadata.get("alarm_code", "")).upper()
                    prefix = str(metadata.get("alarm_prefix", "")).upper()
                    if code and code in {token.upper() for token in query_tokens}:
                        scores[position] += 2.0
                    elif prefix and prefix in {token.upper() for token in query_tokens}:
                        scores[position] += 0.5
            # API names are stable identifiers.  Prefer the exact API entry
            # when a caller asks for one by name, while retaining semantic
            # ranking for natural-language queries that contain no identifier.
            query_api_tokens = {
                token.lower()
                for token in re.findall(r"[A-Za-z][A-Za-z0-9_]+", query)
            }
            for position, candidate_index in enumerate(candidate_indices):
                metadata = self._stable_records[int(candidate_index)].get("metadata", {})
                names = {
                    str(metadata.get("function", "")).strip().lower(),
                    str(metadata.get("library_method", "")).strip().lower(),
                }
                names.discard("")
                if names.intersection(query_api_tokens):
                    scores[position] += 2.0
            # Per-function prerequisite rules are indexed by exact FOCAS
            # function name; semantic similarity alone cannot discriminate
            # "cnc_rdgrpid" from its subword-neighbors, so boost rules whose
            # applies_to contains a function named verbatim in the query.
            if where and where.get("rule_type") == "api_prerequisite":
                query_tokens = {t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query)}
                for position, candidate_index in enumerate(candidate_indices):
                    metadata = self._stable_records[int(candidate_index)].get("metadata", {})
                    applies_raw = metadata.get("applies_to", "")
                    if isinstance(applies_raw, str):
                        try:
                            applies = json.loads(applies_raw)
                        except json.JSONDecodeError:
                            applies = []
                    else:
                        applies = applies_raw or []
                    if isinstance(applies, list) and any(
                        str(item).strip().lower() in query_tokens for item in applies
                    ):
                        scores[position] += 2.0
            order = np.argsort(-scores)[: min(top_k, 50)]
            retrieved = []
            for position in order:
                record = self._stable_records[int(candidate_indices[position])]
                metadata = dict(record.get("metadata", {}))
                chunk = KnowledgeChunk(
                    chunk_id=metadata.get("chunk_id", metadata.get("source_chunk_id", record.get("id", ""))),
                    text=str(record.get("text", "")),
                    metadata=metadata,
                )
                retrieved.append(RetrievedChunk(chunk=chunk, score=round(float(scores[position]), 4)))
            self._query_cache[cache_key] = list(retrieved)
            return retrieved

        if self._collection is None:
            return []

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

        if len(self._query_cache) >= 256:
            self._query_cache.pop(next(iter(self._query_cache)))
        self._query_cache[cache_key] = list(retrieved)
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
