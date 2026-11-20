
"""Run HotpotQA retrieval comparisons and answer generation with resumable JSONL output."""

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from evaluation.dataset_loader import load_benchmark
from evaluation.retrieval_metrics import score_retrieval
from evaluation.answer_metrics import exact_match, token_f1
from rag.benchmark_pipeline import BenchmarkRetriever
from rag.groq_generator import GroqGenerator


METHODS = ("dense", "bm25", "hybrid", "hybrid_rerank")


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_done(path, record_type):
    """Return IDs of valid completed records only."""
    if not path.exists():
        return set()

    done = set()

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        try:
            row = json.loads(line)
            qid = row.get("id")

            if not qid:
                continue

            if record_type == "retrieval":
                retrieval = row.get("retrieval", {})
                if all(
                    method in retrieval
                    and isinstance(retrieval[method].get("chunks"), list)
                    and "metrics" in retrieval[method]
                    for method in METHODS
                ):
                    done.add(qid)

            elif record_type == "answer":
                if isinstance(row.get("response"), str) and row["response"].strip():
                    done.add(qid)

        except (json.JSONDecodeError, AttributeError, TypeError):
            continue

    return done


def main():
    load_dotenv()

    ap = argparse.ArgumentParser(
        description="Run the 100-question HotpotQA RAG benchmark"
    )

    ap.add_argument(
        "--benchmark",
        default="datasets/benchmark_hotpotqa_100.json"
    )
    ap.add_argument(
        "--index-dir",
        default="data/benchmark_indexes_hotpotqa_100"
    )
    ap.add_argument(
        "--output-dir",
        default="results/hotpotqa_100/raw"
    )
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument(
        "--generate",
        action="store_true",
        help="Generate answers using Hybrid RRF + BGE reranker"
    )
    ap.add_argument("--sleep", type=float, default=2.2)
    ap.add_argument("--limit", type=int, default=0)

    args = ap.parse_args()

    questions = load_benchmark(args.benchmark)

    # Enforce the HotpotQA-only experiment scope.
    wrong_datasets = {
        q.get("dataset") for q in questions
        if q.get("dataset") != "hotpotqa"
    }

    if wrong_datasets:
        raise ValueError(
            f"Expected HotpotQA-only benchmark. Found: {wrong_datasets}"
        )

    if args.limit:
        questions = questions[:args.limit]

    if not questions:
        raise ValueError("No benchmark questions found")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    metric_path = out / "retrieval_metrics.jsonl"
    answer_path = out / "answers_hybrid_rerank.jsonl"

    retrieval_done = read_done(metric_path, "retrieval")
    answer_done = read_done(answer_path, "answer")

    # Load only the HotpotQA index.
    retriever = BenchmarkRetriever(
        "hotpotqa",
        index_root=args.index_dir
    )

    generator = GroqGenerator() if args.generate else None

    print(f"Benchmark questions: {len(questions)}")
    print(f"Index directory: {args.index_dir}")
    print(f"Output directory: {out}")
    print(f"Indexed chunks: {len(retriever.chunks)}")

    for n, q in enumerate(questions, 1):
        qid = q["id"]
        chunks = None

        # Run all four retrieval methods.
        if qid not in retrieval_done:
            row = {
                "id": qid,
                "dataset": q["dataset"],
                "question": q["question"],
                "reference": q["answer"],
                "retrieval": {}
            }

            for method in METHODS:
                rerank = method == "hybrid_rerank"
                base_method = "hybrid" if rerank else method

                results = retriever.retrieve(
                    q["question"],
                    base_method,
                    args.top_k,
                    rerank=rerank
                )

                method_chunks = [x["chunk"] for x in results]

                row["retrieval"][method] = {
                    "metrics": score_retrieval(
                        q,
                        [c.__dict__ for c in method_chunks]
                    ),
                    "chunks": [c.__dict__ for c in method_chunks]
                }

                if rerank:
                    chunks = method_chunks

            append_jsonl(metric_path, row)

        # Generate answers using hybrid retrieval + reranking only.
        if generator and qid not in answer_done:
            if chunks is None:
                results = retriever.retrieve(
                    q["question"],
                    "hybrid",
                    args.top_k,
                    rerank=True
                )
                chunks = [x["chunk"] for x in results]

            evidence = [{"chunk": c} for c in chunks]

            generated = generator.generate(q["question"], evidence)
            prediction = generated["answer"]

            append_jsonl(
                answer_path,
                {
                    "id": qid,
                    "dataset": q["dataset"],
                    "user_input": q["question"],
                    "retrieved_contexts": [c.text for c in chunks],
                    "response": prediction,
                    "reference": q["answer"],
                    "exact_match": exact_match(prediction, q["answer"]),
                    "token_f1": token_f1(prediction, q["answer"]),
                    "generation": generated
                }
            )

            time.sleep(args.sleep)

        if n % 10 == 0 or n == len(questions):
            print(f"Processed {n}/{len(questions)}")

    print(f"Retrieval results: {metric_path}")

    if generator:
        print(f"Answer results: {answer_path}")


if __name__ == "__main__":
    main()
