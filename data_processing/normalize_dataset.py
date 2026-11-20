"""Normalize HotpotQA and 2WikiMultiHopQA records to a common schema.

Gold answers and supporting facts are retained for evaluation. Do not pass
those fields to retrieval or answer generation.
"""

from __future__ import annotations

from typing import Any


def _get_first(record: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return default


def _normalize_context(raw_context: Any) -> list[dict[str, Any]]:
    """Accept official [title, sentences] pairs or dict-shaped contexts."""
    normalized: list[dict[str, Any]] = []
    if isinstance(raw_context, dict):
        # Some converted datasets represent a sequence as parallel arrays.
        titles = raw_context.get("title", [])
        sentence_groups = raw_context.get("sentences", [])
        raw_context = list(zip(titles, sentence_groups))

    if not isinstance(raw_context, list):
        raise ValueError("Record context must be a list or a context dictionary")

    for index, item in enumerate(raw_context):
        if isinstance(item, dict):
            title = item.get("title") or f"untitled_{index}"
            sentences = item.get("sentences", [])
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            title, sentences = item[0], item[1]
        else:
            raise ValueError(f"Unsupported context item at index {index}: {item!r}")

        if isinstance(sentences, str):
            sentences = [sentences]
        if not isinstance(sentences, list):
            raise ValueError(f"Context sentences for {title!r} must be a list")

        normalized.append({
            "title": str(title),
            "sentences": [str(sentence) for sentence in sentences],
        })
    return normalized


def _normalize_supporting_facts(raw_facts: Any) -> list[dict[str, Any]]:
    """Accept official [[title, sent_id], ...] or columnar dict format."""
    if raw_facts is None:
        return []

    if isinstance(raw_facts, dict):
        titles = raw_facts.get("title", [])
        sent_ids = raw_facts.get("sent_id", [])
        raw_facts = list(zip(titles, sent_ids))

    if not isinstance(raw_facts, list):
        raise ValueError("supporting_facts must be a list or dictionary")

    facts: list[dict[str, Any]] = []
    for item in raw_facts:
        if isinstance(item, dict):
            title = item.get("title")
            sent_id = item.get("sent_id")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            title, sent_id = item[0], item[1]
        else:
            raise ValueError(f"Unsupported supporting fact: {item!r}")

        facts.append({"title": str(title), "sent_id": int(sent_id)})
    return facts


def normalize_record(record: dict[str, Any], dataset_name: str, split: str) -> dict[str, Any]:
    """Convert one source record into the project's canonical QA schema."""
    question_id = _get_first(record, "_id", "id")
    question = record.get("question")
    answer = record.get("answer")

    if question_id is None or not isinstance(question, str) or not question.strip():
        raise ValueError("Each record must contain an ID and a non-empty question")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"Record {question_id!r} has no gold answer; use an answerable split")

    contexts = _normalize_context(record.get("context"))
    supporting_facts = _normalize_supporting_facts(record.get("supporting_facts"))

    # Enrich supporting facts with their source sentence when available.
    context_lookup = {item["title"]: item["sentences"] for item in contexts}
    for fact in supporting_facts:
        sentences = context_lookup.get(fact["title"], [])
        sent_id = fact["sent_id"]
        fact["sentence"] = sentences[sent_id] if 0 <= sent_id < len(sentences) else None

    # Preserve 2Wiki's evidence chains and other dataset-specific annotations,
    # without requiring them for the shared RAG interface.
    evidence_chain = _get_first(record, "evidences", "evidence", default=[])

    return {
        "id": f"{dataset_name}:{question_id}",
        "source_id": str(question_id),
        "dataset": dataset_name,
        "split": split,
        "question": question.strip(),
        "answer": answer.strip(),
        "question_type": _get_first(record, "type", "reasoning_type"),
        "difficulty": record.get("level"),
        "contexts": contexts,
        "supporting_facts": supporting_facts,
        "evidence_chain": evidence_chain,
    }
