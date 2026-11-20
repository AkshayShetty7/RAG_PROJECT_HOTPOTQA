"""Build deduplicated, sentence-preserving corpora from benchmark contexts."""
from __future__ import annotations
import argparse, json, re
from collections import defaultdict
from pathlib import Path

DATASETS = ("hotpotqa", "2wikimultihopqa")

def normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title).strip().casefold()

def prepare_documents(benchmark_path: str | Path, output_dir: str | Path):
    data = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
    questions = data.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("Benchmark must contain a non-empty 'questions' list")

    # Keep dataset identity and exact context version in the key. Some releases
    # contain the same page title with different sentence lists; merging by title
    # alone would corrupt sentence IDs and supporting-fact alignment. Exact duplicate
    # context versions are deduplicated, while distinct versions remain separate.
    import hashlib
    docs = {ds: {} for ds in DATASETS}
    for q in questions:
        ds = q.get("dataset")
        if ds not in docs:
            raise ValueError(f"Unsupported dataset label: {ds!r}")
        qid = q.get("id")
        contexts = q.get("contexts")
        if not isinstance(contexts, list) or not contexts:
            raise ValueError(f"Question {qid} has no contexts")
        for ctx in contexts:
            title = str(ctx.get("title", "")).strip()
            sentences = ctx.get("sentences")
            if not title or not isinstance(sentences, list):
                raise ValueError(f"Malformed context in question {qid}")
            clean_sentences = [str(x).strip() for x in sentences]
            clean_sentences = [(sid, text) for sid, text in enumerate(clean_sentences) if text]
            title_key = normalize_title(title)
            content_key = tuple(text for _, text in clean_sentences)
            key = (title_key, content_key)
            doc = docs[ds].get(key)
            if doc is None:
                digest = hashlib.sha1(repr(key).encode("utf-8")).hexdigest()[:12]
                doc = {
                    "document_id": f"{ds}:{digest}", "dataset": ds,
                    "title": title, "sentences": clean_sentences,
                    "question_ids": set()
                }
                docs[ds][key] = doc
            doc["question_ids"].add(qid)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for ds, by_title in docs.items():
        records = []
        for doc in by_title.values():
            records.append({
                "document_id": doc["document_id"], "dataset": ds,
                "title": doc["title"],
                "sentences": [
                    {"sent_id": sid, "text": text}
                    for sid, text in doc["sentences"]
                ],
                "question_ids": sorted(doc["question_ids"])
            })
        records.sort(key=lambda x: x["title"].casefold())
        path = out / ds / "documents.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        summary[ds] = {"documents": len(records), "sentences": sum(len(x["sentences"]) for x in records), "path": str(path)}
    (out / "document_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

def main():
    p = argparse.ArgumentParser(description="Prepare benchmark document corpora")
    p.add_argument("--benchmark", default="datasets/benchmark_70_30.json")
    p.add_argument("--output-dir", default="data/benchmark_corpus")
    args = p.parse_args()
    summary = prepare_documents(args.benchmark, args.output_dir)
    for ds, info in summary.items():
        print(f"{ds}: {info['documents']} documents, {info['sentences']} sentences -> {info['path']}")

if __name__ == "__main__": main()
