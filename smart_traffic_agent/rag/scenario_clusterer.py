from __future__ import annotations

import json
import math
import csv
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..utils import tokenize, write_json
from .scenario_taxonomy import ProtocolTaxonomy, default_taxonomy


DEFAULT_SCENARIO_CLUSTERS_PATH = Path("rag_indexes/focas/scenario_clusters_auto.json")
DEFAULT_SCENARIO_21_CLUSTERS_PATH = DEFAULT_SCENARIO_CLUSTERS_PATH
DEFAULT_SCENARIO_CLUSTER_REVIEW_PATH = Path("rag_indexes/focas/scenario_cluster_review.csv")
DEFAULT_RULE_CHUNKS_PATH = Path("rag_indexes/focas/rule_chunks.jsonl")
DEFAULT_SCENARIO_KNOWLEDGE_PATH = Path("rag_indexes/focas/scenario_knowledge.json")
DEFAULT_MIN_CLUSTER_COUNT = 6
DEFAULT_MAX_CLUSTER_COUNT = 36


@dataclass(slots=True)
class KnowledgeUnit:
    unit_id: str
    source_type: str
    source_scenario: str
    rule_type: str
    text: str
    api_features: list[str]
    nc_or_operation_features: list[str]
    expected_signals: list[str]
    semantic_objects: list[str]
    trigger_type: str
    source: dict[str, Any]


@dataclass(slots=True)
class ClusterAssignment:
    unit_id: str
    cluster_id: str
    score: float
    labels: dict[str, Any]
    source: dict[str, Any]


def build_scenario_21_clusters(
    output_path: Path = DEFAULT_SCENARIO_CLUSTERS_PATH,
    taxonomy: ProtocolTaxonomy | None = None,
    rule_chunks_path: Path = DEFAULT_RULE_CHUNKS_PATH,
    scenario_knowledge_path: Path | None = None,
    cluster_count: int | None = None,
    min_clusters: int = DEFAULT_MIN_CLUSTER_COUNT,
    max_clusters: int = DEFAULT_MAX_CLUSTER_COUNT,
) -> dict[str, Any]:
    taxonomy = taxonomy or default_taxonomy()
    units = load_knowledge_units(
        taxonomy=taxonomy,
        rule_chunks_path=rule_chunks_path,
        scenario_knowledge_path=scenario_knowledge_path,
    )
    payload = scenario_21_cluster_payload(
        taxonomy,
        units,
        cluster_count=cluster_count,
        min_clusters=min_clusters,
        max_clusters=max_clusters,
    )
    write_json(output_path, payload)
    return payload


def write_cluster_review_csv(
    clusters_path: Path = DEFAULT_SCENARIO_CLUSTERS_PATH,
    output_path: Path = DEFAULT_SCENARIO_CLUSTER_REVIEW_PATH,
) -> list[dict[str, Any]]:
    payload = json.loads(clusters_path.read_text(encoding="utf-8"))
    rows = [cluster_review_row(cluster) for cluster in payload.get("clusters", [])]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "cluster_id",
                "member_count",
                "dominant_trigger",
                "dominant_objects",
                "dominant_apis",
                "dominant_nc_or_operation",
                "dominant_signals",
                "dominant_rule_types",
                "suggested_scene_name",
                "suggested_action",
                "review_note",
                "representative_units",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def cluster_review_row(cluster: dict[str, Any]) -> dict[str, Any]:
    profile = cluster.get("profile", {})
    objects = values(profile, "semantic_objects", 4)
    apis = values(profile, "api_features", 4)
    nc_features = values(profile, "nc_or_operation_features", 4)
    signals = values(profile, "expected_signals", 4)
    rule_types = values(profile, "rule_types", 4)
    trigger = first_profile_value(profile, "trigger_types")
    suggested_name = suggest_scene_name(trigger, objects, apis, nc_features, signals)
    return {
        "cluster_id": cluster.get("cluster_id", ""),
        "member_count": cluster.get("member_count", 0),
        "dominant_trigger": trigger,
        "dominant_objects": "; ".join(objects),
        "dominant_apis": "; ".join(apis),
        "dominant_nc_or_operation": "; ".join(nc_features),
        "dominant_signals": "; ".join(signals),
        "dominant_rule_types": "; ".join(rule_types),
        "suggested_scene_name": suggested_name,
        "suggested_action": suggest_review_action(cluster, suggested_name),
        "review_note": "",
        "representative_units": " | ".join(
            preview(item.get("text_preview", ""), limit=90) for item in cluster.get("top_members", [])[:3]
        ),
    }


def values(profile: dict[str, Any], key: str, limit: int = 3) -> list[str]:
    return [str(item.get("value", "")) for item in profile.get(key, [])[:limit] if item.get("value")]


def first_profile_value(profile: dict[str, Any], key: str) -> str:
    items = profile.get(key, [])
    return str(items[0].get("value", "")) if items else ""


def suggest_scene_name(
    trigger: str,
    objects: list[str],
    apis: list[str],
    nc_features: list[str],
    signals: list[str],
) -> str:
    primary_object = objects[0] if objects else ""
    primary_api = apis[0] if apis else ""
    if primary_object in {"alarm", "diagnosis"} or "alm" in primary_api or "diagnos" in primary_api:
        return "alarm_diagnosis"
    if primary_object in {"parameter", "macro_variable"} or primary_api in {"cnc_rdparam", "cnc_wrparam", "cnc_rdmacro", "cnc_wrmacro"}:
        return "parameter_macro_access"
    if primary_object == "tool_offset" or "tool" in primary_api or "tofs" in primary_api:
        return "tool_offset_management"
    if primary_object == "work_coordinate" or "zofs" in primary_api:
        return "work_coordinate_setting"
    if primary_object == "program" or primary_api in {"cnc_search", "cnc_dnc", "cnc_upload", "cnc_download"}:
        return "program_lifecycle"
    if primary_object == "spindle" or "sp" in primary_api:
        return "spindle_control"
    if primary_object in {"axis", "feed"} or primary_api in {"cnc_rdposition", "cnc_rdpos", "cnc_rdactf", "cnc_actf"}:
        return "coordinate_feed_motion"
    if primary_object == "connection" or "ether" in primary_api or "unsolic" in primary_api:
        return "ethernet_connection_exception"
    if primary_object == "pmc_signal" or primary_api.startswith("pmc_"):
        return "pmc_signal_monitoring"

    text = " ".join([trigger, *objects, *apis, *nc_features, *signals]).lower()
    rules = [
        ("coordinate_feed_motion", ["axis", "feed", "position", "rdposition", "actf"]),
        ("program_lifecycle", ["program", "upload", "download", "search", "dnc"]),
        ("tool_offset_management", ["tool_offset", "tool", "tofs", "rdtool"]),
        ("work_coordinate_setting", ["work_coordinate", "zofs", "g54", "g55", "g56"]),
        ("parameter_macro_access", ["parameter", "macro", "rdparam", "wrparam", "rdmacro"]),
        ("alarm_diagnosis", ["alarm", "alm", "diagnos"]),
        ("spindle_control", ["spindle", "s value", "rdsp", "acts"]),
        ("ethernet_connection_exception", ["connection", "ethernet", "unsolic", "socket"]),
        ("abnormal_request_exception", ["abnormal", "invalid", "error_code", "ew_"]),
        ("pmc_signal_monitoring", ["pmc", "di", "do", "io_signal"]),
        ("general_status_collection", ["machine_status", "statinfo", "status"]),
    ]
    for name, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return name
    return "manual_review_required"


def suggest_review_action(cluster: dict[str, Any], suggested_name: str) -> str:
    count = int(cluster.get("member_count", 0))
    if suggested_name == "manual_review_required":
        return "review"
    if count < 10:
        return "merge_or_review"
    if count > 220:
        return "split_or_keep_as_broad_scene"
    return "keep_or_merge_by_semantics"


def scenario_21_cluster_payload(
    taxonomy: ProtocolTaxonomy,
    knowledge_units: list[KnowledgeUnit] | None = None,
    cluster_count: int | None = None,
    min_clusters: int = DEFAULT_MIN_CLUSTER_COUNT,
    max_clusters: int = DEFAULT_MAX_CLUSTER_COUNT,
) -> dict[str, Any]:
    knowledge_units = knowledge_units or taxonomy_as_units(taxonomy)
    vectors = vectorize_units(knowledge_units)
    selected = select_cluster_count(vectors, cluster_count, min_clusters, max_clusters)
    cluster_count = selected["selected_k"]
    labels, scores = cluster_vectors(vectors, cluster_count)
    clusters = build_natural_cluster_rows(labels, scores, knowledge_units)
    assignments = build_assignments(labels, scores, knowledge_units, clusters)

    return {
        "protocol": taxonomy.protocol,
        "knowledge_type": "naturally_clustered_fine_grained_scenarios",
        "method": "feature_vector_kmeans_clustering",
        "cluster_count_selection": {
            **selected,
            "rationale": (
                "The system clusters document-derived knowledge units by their traffic-generation "
                "features. The number of clusters is selected by comparing candidate k values with "
                "a centroid-based separation score and cluster-size validity constraints. Therefore "
                "the final scenario count is produced by the clustering process rather than fixed in advance."
            ),
        },
        "feature_dimensions": {
            "T": "traffic trigger type",
            "O": "semantic object",
            "A": "API category or function",
            "N": "NC command or operation feature",
            "E": "expected traffic signal",
        },
        "clustering_rule": (
            "Knowledge units are converted into weighted feature vectors from T/O/A/N/E labels "
            "and document terms. K-means clustering groups units with similar trigger mechanisms, "
            "semantic objects, API functions, NC/operation features, and expected traffic signals. "
            "Cluster names are assigned after clustering from high-frequency features, not used as "
            "predefined cluster prototypes."
        ),
        "clusters": clusters,
        "assignments": [asdict(item) for item in assignments],
        "statistics": {
            "scenario_cluster_count": len(clusters),
            "unique_cluster_count": len({row["cluster_id"] for row in clusters}),
            "knowledge_unit_count": len(knowledge_units),
            "assigned_unit_count": len(assignments),
            "nonempty_cluster_count": sum(1 for row in clusters if row["member_count"] > 0),
        },
    }


def load_knowledge_units(
    *,
    taxonomy: ProtocolTaxonomy,
    rule_chunks_path: Path = DEFAULT_RULE_CHUNKS_PATH,
    scenario_knowledge_path: Path | None = None,
) -> list[KnowledgeUnit]:
    units: list[KnowledgeUnit] = []
    if rule_chunks_path.exists():
        units.extend(rule_chunk_to_unit(row) for row in load_jsonl(rule_chunks_path))
    if scenario_knowledge_path and scenario_knowledge_path.exists():
        units.extend(scenario_knowledge_to_units(scenario_knowledge_path))
    return units or taxonomy_as_units(taxonomy)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def scenario_knowledge_to_units(path: Path) -> list[KnowledgeUnit]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    units: list[KnowledgeUnit] = []
    for scenario in payload.get("scenarios", []):
        scenario_id = str(scenario.get("scenario_id", ""))
        rules = scenario.get("rules", {})
        for rule_type, items in rules.items():
            for index, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                row = {
                    **item,
                    "rule_id": item.get("rule_id") or f"{scenario_id}-{rule_type}-{index + 1}",
                    "rule_type": rule_type,
                    "scenario": scenario_id,
                    "distinguishing_signals": scenario.get("distinguishing_signals", []),
                }
                units.append(rule_chunk_to_unit(row, source_type="scenario_knowledge"))
    return units


def taxonomy_as_units(taxonomy: ProtocolTaxonomy) -> list[KnowledgeUnit]:
    units = []
    for scenario in taxonomy.scenarios:
        row = {
            "rule_id": f"taxonomy-{scenario.scenario_id}",
            "rule_type": "scenario_seed",
            "scenario": scenario.scenario_id,
            "rule_text": " ".join(
                [
                    scenario.scenario_id,
                    scenario.name,
                    scenario.goal,
                    " ".join(scenario.typical_nc_program),
                    " ".join(scenario.recommended_api_functions),
                    " ".join(scenario.distinguishing_signals),
                ]
            ),
            "recommended_api_functions": scenario.recommended_api_functions,
            "nc_program_requirements": scenario.typical_nc_program,
            "distinguishing_signals": scenario.distinguishing_signals,
        }
        units.append(rule_chunk_to_unit(row, source_type="taxonomy_seed"))
    return units


def rule_chunk_to_unit(row: dict[str, Any], source_type: str = "rule_chunk") -> KnowledgeUnit:
    text = row_text(row)
    semantic_objects = infer_semantic_objects(text.lower())
    trigger_type = infer_trigger_type(text.lower(), str(row.get("rule_type", "")))
    return KnowledgeUnit(
        unit_id=str(row.get("rule_id") or row.get("chunk_id") or row.get("source_chunk_id") or hash(text)),
        source_type=source_type,
        source_scenario=str(row.get("scenario", "")),
        rule_type=str(row.get("rule_type", "")),
        text=text,
        api_features=extract_api_features(row, text),
        nc_or_operation_features=extract_nc_operation_features(row, text),
        expected_signals=extract_expected_signals(row, text, semantic_objects),
        semantic_objects=semantic_objects,
        trigger_type=trigger_type,
        source={
            "source_file": row.get("source_file"),
            "source_chunk_id": row.get("source_chunk_id"),
            "page_start": row.get("page_start"),
            "page_end": row.get("page_end"),
            "section_title": row.get("section_title"),
            "source_scenario": row.get("scenario"),
            "rule_type": row.get("rule_type"),
        },
    )


def row_text(row: dict[str, Any]) -> str:
    fields = ["rule_text", "function", "category", "text", "section_title"]
    parts = [str(row.get(field, "")) for field in fields]
    for field in [
        "nc_program_requirements",
        "operation_sequence",
        "collection_timing",
        "recommended_api_functions",
        "allowed_operations",
        "restricted_operations",
        "abnormal_traffic_allowed",
        "distinguishing_signals",
        "quality_checks",
    ]:
        value = row.get(field)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value:
            parts.append(str(value))
    return " ".join(part for part in parts if part)


def extract_api_features(row: dict[str, Any], text: str) -> list[str]:
    features = []
    for value in row.get("recommended_api_functions", []) or []:
        if isinstance(value, str) and value:
            features.append(value)
    for token in text.replace("(", " ").replace(")", " ").replace(",", " ").split():
        if token.startswith(("cnc_", "pmc_")):
            features.append(token.strip(";:,.，。"))
    return sorted(set(features))


def extract_nc_operation_features(row: dict[str, Any], text: str) -> list[str]:
    features = []
    for field in ["nc_program_requirements", "operation_sequence", "allowed_operations", "restricted_operations"]:
        value = row.get(field)
        if isinstance(value, list):
            features.extend(str(item) for item in value[:4])
    lowered = text.lower()
    tokens = [
        "G00",
        "G01",
        "G02",
        "G03",
        "G28",
        "G30",
        "G40",
        "G41",
        "G42",
        "G43",
        "G44",
        "G49",
        "G54",
        "G55",
        "G56",
        "G57",
        "G58",
        "G59",
        "M03",
        "M04",
        "M05",
        "M30",
        "S value",
        "F value",
        "MDI",
        "JOG",
        "handwheel",
        "upload",
        "download",
        "pause",
        "resume",
        "timeout",
        "invalid",
    ]
    features.extend(token for token in tokens if token.lower() in lowered)
    return sorted(set(features))


def extract_expected_signals(row: dict[str, Any], text: str, semantic_objects: list[str]) -> list[str]:
    signals = [str(item) for item in row.get("distinguishing_signals", []) or [] if item]
    lowered = text.lower()
    signal_terms = [
        "position_change",
        "feed_speed_change",
        "spindle_speed_change",
        "spindle_state_change",
        "run_status_change",
        "alarm_bits_change",
        "diagnosis_value_change",
        "parameter_value_observed",
        "io_bit_change",
        "socket_error",
    ]
    signals.extend(term for term in signal_terms if term in lowered)
    if not signals:
        signals.extend(f"{item}_change" for item in semantic_objects[:2])
    return sorted(set(signals))


def infer_trigger_type(text: str, rule_type: str = "") -> str:
    if rule_type == "safety_rule" or contains_any(text, ["socket", "timeout", "invalid", "error", "illegal", "protect", "ew_"]):
        return "abnormal_trigger"
    if contains_any(text, ["alarm", "diagnos", "pmc", "history"]):
        return "diagnosis_trigger"
    if contains_any(text, ["param", "offset", "macro", "zofs", "tofs", "work coordinate"]):
        return "data_access_trigger"
    if contains_any(text, ["upload", "download", "select", "mdi", "jog", "pause", "resume", "start", "stop"]):
        return "operation_trigger"
    return "motion_trigger"


def infer_semantic_objects(text: str) -> list[str]:
    mapping = [
        ("axis", ["axis", "position", "coordinate", "g01", "g28", "g30"]),
        ("feed", ["feed", "actf", "f value"]),
        ("spindle", ["spindle", "s value", "m03", "m04", "m05", "acts"]),
        ("program", ["program", "upload", "download", "search"]),
        ("parameter", ["parameter", "param"]),
        ("tool_offset", ["tool", "offset", "tofs", "g43", "g44", "g49"]),
        ("work_coordinate", ["work coordinate", "zofs", "g54", "g55", "g56", "g57", "g58", "g59"]),
        ("macro_variable", ["macro"]),
        ("alarm", ["alarm"]),
        ("diagnosis", ["diagnos"]),
        ("pmc_signal", ["pmc", "di", "do"]),
        ("history", ["history"]),
        ("connection", ["socket", "connect", "ethernet", "cnc_allclibhndl3"]),
        ("error_code", ["error", "invalid", "ew_"]),
    ]
    objects = [name for name, terms in mapping if contains_any(text, terms)]
    return objects or ["machine_status"]


def contains_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def vectorize_units(units: list[KnowledgeUnit]) -> list[dict[str, float]]:
    raw_vectors = [unit_feature_counter(unit) for unit in units]
    document_frequency: Counter[str] = Counter()
    for vector in raw_vectors:
        document_frequency.update(vector.keys())

    total = len(raw_vectors)
    vectors = []
    for vector in raw_vectors:
        weighted = {}
        for term, count in vector.items():
            idf = math.log((1 + total) / (1 + document_frequency[term])) + 1.0
            weighted[term] = count * idf
        vectors.append(normalize_vector(weighted))
    return vectors


def select_cluster_count(
    vectors: list[dict[str, float]],
    requested_count: int | None,
    min_clusters: int,
    max_clusters: int,
) -> dict[str, Any]:
    if requested_count and requested_count > 0:
        return {
            "mode": "manual",
            "selected_k": min(requested_count, len(vectors)),
            "candidate_scores": [],
        }
    if not vectors:
        return {"mode": "auto", "selected_k": 0, "candidate_scores": []}

    upper = min(max_clusters, len(vectors))
    lower = min(max(2, min_clusters), upper)
    candidates = []
    for k in range(lower, upper + 1):
        labels, scores = cluster_vectors(vectors, k, max_iterations=20)
        quality = cluster_quality(vectors, labels, scores, k)
        candidates.append({"k": k, **quality})

    best = max(candidates, key=lambda item: item["selection_score"])
    acceptable_score = best["selection_score"] * 0.97
    stable_candidates = [
        item
        for item in candidates
        if item["selection_score"] >= acceptable_score and item["tiny_cluster_ratio"] <= 0.05
    ]
    selected = stable_candidates[0] if stable_candidates else best
    return {
        "mode": "auto",
        "selected_k": selected["k"],
        "min_k": lower,
        "max_k": upper,
        "criterion": "parsimonious centroid separation with small-cluster penalty",
        "best_k_by_score": best["k"],
        "best_score": best["selection_score"],
        "selected_score": selected["selection_score"],
        "parsimony_rule": "select the smallest k reaching at least 97% of the best score and with tiny-cluster ratio <= 0.05",
        "candidate_scores": candidates,
    }


def cluster_quality(
    vectors: list[dict[str, float]],
    labels: list[int],
    own_scores: list[float],
    cluster_count: int,
) -> dict[str, float]:
    centroids = recompute_centroids(vectors, labels, cluster_count)
    margins = []
    for vector, label, own in zip(vectors, labels, own_scores):
        other = max(
            (dot(vector, centroid) for index, centroid in enumerate(centroids) if index != label),
            default=0.0,
        )
        own_distance = max(0.0, 1.0 - own)
        other_distance = max(0.0, 1.0 - other)
        denominator = max(own_distance, other_distance, 1e-9)
        margins.append((other_distance - own_distance) / denominator)

    counts = Counter(labels)
    tiny_threshold = max(2, int(len(vectors) * 0.005))
    tiny_ratio = sum(1 for count in counts.values() if count < tiny_threshold) / max(1, cluster_count)
    nonempty_ratio = len(counts) / max(1, cluster_count)
    avg_silhouette = sum(margins) / max(1, len(margins))
    balance = normalized_entropy(list(counts.values()))
    selection_score = avg_silhouette + 0.08 * balance + 0.05 * nonempty_ratio - 0.18 * tiny_ratio
    return {
        "selection_score": round(selection_score, 6),
        "avg_silhouette": round(avg_silhouette, 6),
        "balance": round(balance, 6),
        "tiny_cluster_ratio": round(tiny_ratio, 6),
        "nonempty_ratio": round(nonempty_ratio, 6),
    }


def normalized_entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0 or len(counts) <= 1:
        return 0.0
    entropy = 0.0
    for count in counts:
        p = count / total
        entropy -= p * math.log(p)
    return entropy / math.log(len(counts))


def unit_feature_counter(unit: KnowledgeUnit) -> Counter[str]:
    features: Counter[str] = Counter()
    add_weighted(features, [f"T:{unit.trigger_type}"], 4)
    add_weighted(features, [f"O:{item}" for item in unit.semantic_objects], 4)
    add_weighted(features, [f"A:{item}" for item in unit.api_features], 5)
    add_weighted(features, [f"N:{item}" for item in unit.nc_or_operation_features], 3)
    add_weighted(features, [f"E:{item}" for item in unit.expected_signals], 3)
    add_weighted(features, [f"R:{unit.rule_type}"], 2)

    text_tokens = [
        token
        for token in tokenize(unit.text)
        if len(token) >= 2 and token not in {"the", "and", "with", "for", "from", "this", "that"}
    ]
    add_weighted(features, [f"W:{token}" for token in text_tokens[:160]], 1)
    return features


def add_weighted(counter: Counter[str], terms: list[str], weight: int) -> None:
    for term in terms:
        if term:
            counter[term.lower()] += weight


def normalize_vector(vector: dict[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm == 0:
        return vector
    return {key: value / norm for key, value in vector.items()}


def cluster_vectors(
    vectors: list[dict[str, float]],
    cluster_count: int,
    max_iterations: int = 30,
) -> tuple[list[int], list[float]]:
    if not vectors:
        return [], []
    cluster_count = min(cluster_count, len(vectors))
    centroids = initialize_centroids(vectors, cluster_count)
    labels = [-1] * len(vectors)

    for _ in range(max_iterations):
        changed = False
        for index, vector in enumerate(vectors):
            label = best_centroid(vector, centroids)
            if labels[index] != label:
                labels[index] = label
                changed = True
        centroids = recompute_centroids(vectors, labels, cluster_count)
        if not changed:
            break

    scores = [dot(vectors[index], centroids[labels[index]]) for index in range(len(vectors))]
    return labels, scores


def initialize_centroids(vectors: list[dict[str, float]], cluster_count: int) -> list[dict[str, float]]:
    first = max(range(len(vectors)), key=lambda index: len(vectors[index]))
    selected = [first]
    while len(selected) < cluster_count:
        candidate = max(
            (index for index in range(len(vectors)) if index not in selected),
            key=lambda index: 1.0 - max(dot(vectors[index], vectors[item]) for item in selected),
        )
        selected.append(candidate)
    return [vectors[index] for index in selected]


def best_centroid(vector: dict[str, float], centroids: list[dict[str, float]]) -> int:
    return max(range(len(centroids)), key=lambda index: dot(vector, centroids[index]))


def recompute_centroids(
    vectors: list[dict[str, float]],
    labels: list[int],
    cluster_count: int,
) -> list[dict[str, float]]:
    grouped: list[dict[str, float]] = [defaultdict(float) for _ in range(cluster_count)]
    counts = [0] * cluster_count
    for vector, label in zip(vectors, labels):
        counts[label] += 1
        for term, value in vector.items():
            grouped[label][term] += value

    centroids = []
    for index, centroid in enumerate(grouped):
        if counts[index] == 0:
            fallback = vectors[index % len(vectors)]
            centroids.append(fallback)
        else:
            centroids.append(normalize_vector(dict(centroid)))
    return centroids


def dot(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(term, 0.0) for term, value in left.items())


def build_natural_cluster_rows(
    labels: list[int],
    scores: list[float],
    units: list[KnowledgeUnit],
) -> list[dict[str, Any]]:
    grouped: dict[int, list[tuple[KnowledgeUnit, float]]] = defaultdict(list)
    for label, score, unit in zip(labels, scores, units):
        grouped[label].append((unit, score))

    raw_rows = []
    for label, members in sorted(grouped.items()):
        members.sort(key=lambda item: item[1], reverse=True)
        profile = summarize_cluster_profile([unit for unit, _ in members])
        raw_rows.append(
            {
                "raw_label": label,
                "name": cluster_name(profile),
                "profile": profile,
                "member_count": len(members),
                "top_members": [
                    {
                        "unit_id": unit.unit_id,
                        "score": round(score, 4),
                        "rule_type": unit.rule_type,
                        "source_scenario": unit.source_scenario,
                        "text_preview": preview(unit.text),
                        "source": unit.source,
                    }
                    for unit, score in members[:8]
                ],
            }
        )

    raw_rows.sort(key=lambda row: row["name"])
    used_ids: Counter[str] = Counter()
    rows = []
    for index, row in enumerate(raw_rows, start=1):
        base_id = slug(row["name"]) or f"cluster_{index:02d}"
        used_ids[base_id] += 1
        suffix = f"_{used_ids[base_id]}" if used_ids[base_id] > 1 else ""
        rows.append(
            {
                "cluster_id": f"cluster_{index:02d}_{base_id}{suffix}",
                "cluster_index": index,
                **row,
            }
        )
    return rows


def summarize_cluster_profile(units: list[KnowledgeUnit]) -> dict[str, Any]:
    return {
        "trigger_types": most_common(unit.trigger_type for unit in units),
        "semantic_objects": most_common(item for unit in units for item in unit.semantic_objects),
        "api_features": most_common(item for unit in units for item in unit.api_features),
        "nc_or_operation_features": most_common(item for unit in units for item in unit.nc_or_operation_features),
        "expected_signals": most_common(item for unit in units for item in unit.expected_signals),
        "rule_types": most_common(unit.rule_type for unit in units),
        "source_scenarios": most_common(unit.source_scenario for unit in units if unit.source_scenario),
    }


def most_common(values: Any, limit: int = 8) -> list[dict[str, Any]]:
    counter = Counter(value for value in values if value)
    return [{"value": value, "count": count} for value, count in counter.most_common(limit)]


def cluster_name(profile: dict[str, Any]) -> str:
    objects = [item["value"] for item in profile.get("semantic_objects", [])[:2]]
    apis = [item["value"] for item in profile.get("api_features", [])[:1]]
    triggers = [item["value"].replace("_trigger", "") for item in profile.get("trigger_types", [])[:1]]
    parts = objects + apis + triggers
    return "_".join(parts) if parts else "unlabeled_cluster"


def first_value(profile: dict[str, Any], key: str) -> str:
    values = profile.get(key, [])
    return str(values[0]["value"]) if values else ""


def build_assignments(
    labels: list[int],
    scores: list[float],
    units: list[KnowledgeUnit],
    clusters: list[dict[str, Any]],
) -> list[ClusterAssignment]:
    id_by_raw_label = {row["raw_label"]: row["cluster_id"] for row in clusters}
    return [
        ClusterAssignment(
            unit_id=unit.unit_id,
            cluster_id=id_by_raw_label[label],
            score=round(score, 4),
            labels={
                "T": unit.trigger_type,
                "O": unit.semantic_objects,
                "A": unit.api_features,
                "N": unit.nc_or_operation_features[:8],
                "E": unit.expected_signals,
            },
            source=unit.source,
        )
        for label, score, unit in zip(labels, scores, units)
    ]


def slug(text: str) -> str:
    safe = []
    for char in text.lower():
        if char.isalnum():
            safe.append(char)
        elif char in {"_", "-", " ", "/"}:
            safe.append("_")
    return "_".join(part for part in "".join(safe).split("_") if part)


def preview(text: str, limit: int = 180) -> str:
    text = " ".join(text.split())
    return text[:limit].rstrip()
