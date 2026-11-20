
"""
Ragas evaluation for the 1,000-question RAG benchmark.

Compatible with:
    ragas==0.4.4
    openai==1.109.1
    httpx==0.28.1

Metrics:
    1. Faithfulness
    2. Factual Correctness (F1)
    3. Context Recall

Features:
    - Groq OpenAI-compatible API
    - Windows certificate trust via truststore
    - Sequential metric execution
    - Per-metric checkpointing
    - Resume incomplete evaluations
    - Optional question limit for testing
"""

import argparse
import csv
import json
import math
import os
import ssl
import time
from pathlib import Path

import httpx
import truststore

from dotenv import load_dotenv
from openai import OpenAI

from ragas.llms import llm_factory
from ragas.metrics.collections import (
    Faithfulness,
    FactualCorrectness,
    ContextRecall,
)


# Preserve the existing factual-correctness column name
# for compatibility with earlier result files.
FACTUAL_COL = "factual_correctness(mode=f1)"

SCORE_COLUMNS = [
    "faithfulness",
    FACTUAL_COL,
    "context_recall",
]

ERROR_COLUMNS = [
    "faithfulness_error",
    "factual_correctness_error",
    "context_recall_error",
]

OUTPUT_COLUMNS = [
    "id",
    "dataset",
    "user_input",
    "retrieved_contexts",
    "response",
    "reference",
    *SCORE_COLUMNS,
    *ERROR_COLUMNS,
]


def parse_score(value):
    """Return a valid score in [0, 1], or None if missing/invalid."""
    if value is None or str(value).strip() == "":
        return None

    try:
        score = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        return None

    return score


def load_existing_results(output_path):
    """Load previously saved results for resumable evaluation."""
    existing = {}

    if not output_path.exists():
        return existing

    with output_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            question_id = row.get("id")

            if question_id:
                existing[question_id] = row

    return existing


def save_results(output_path, results):
    """
    Save all results after each question.

    Write to a temporary file first and replace the destination
    to reduce the risk of losing the previous checkpoint.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_name(
        output_path.name + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=OUTPUT_COLUMNS,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in results.values():
            writer.writerow(row)

    os.replace(temporary_path, output_path)


def create_evaluator():
    """Create a Groq-backed Ragas LLM using Windows trusted certificates."""
    model = os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-120b",
    )

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY was not found. Check your project .env file."
        )

    ssl_context = truststore.SSLContext(
        ssl.PROTOCOL_TLS_CLIENT
    )

    http_client = httpx.Client(
        verify=ssl_context,
        timeout=120.0,
    )

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        http_client=http_client,
        max_retries=3,
    )

    evaluator_llm = llm_factory(
        model=model,
        provider="openai",
        client=client,
    )

    print(f"Evaluator model: {model}")
    print("Groq client initialized successfully.")

    return evaluator_llm, http_client


def create_metrics(evaluator_llm):
    """Initialize the three Ragas collections-based metrics."""
    return {
        "faithfulness": Faithfulness(
            llm=evaluator_llm
        ),
        FACTUAL_COL: FactualCorrectness(
            llm=evaluator_llm,
            mode="f1",
        ),
        "context_recall": ContextRecall(
            llm=evaluator_llm
        ),
    }


def score_metric(metric, metric_name, row):
    """Run one metric and return its validated numeric score."""
    if metric_name == "faithfulness":
        result = metric.score(
            user_input=row["user_input"],
            retrieved_contexts=row["retrieved_contexts"],
            response=row["response"],
        )

    elif metric_name == FACTUAL_COL:
        result = metric.score(
            response=row["response"],
            reference=row["reference"],
        )

    elif metric_name == "context_recall":
        result = metric.score(
            user_input=row["user_input"],
            retrieved_contexts=row["retrieved_contexts"],
            reference=row["reference"],
        )

    else:
        raise ValueError(
            f"Unknown metric: {metric_name}"
        )

    score = parse_score(result.value)

    if score is None:
        raise ValueError(
            f"{metric_name} returned an invalid score: "
            f"{result.value!r}"
        )

    return score


def prepare_result_row(source_row, previous=None):
    """Build a result row while preserving previous valid scores."""
    previous = previous or {}

    row = {
        "id": source_row["id"],
        "dataset": source_row["dataset"],
        "user_input": source_row["user_input"],
        "retrieved_contexts": json.dumps(
            source_row["retrieved_contexts"],
            ensure_ascii=False,
        ),
        "response": source_row["response"],
        "reference": source_row["reference"],
    }

    for column in SCORE_COLUMNS:
        old_score = parse_score(previous.get(column))
        row[column] = (
            old_score if old_score is not None else ""
        )

    for column in ERROR_COLUMNS:
        row[column] = previous.get(column, "")

    return row


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run resumable Ragas evaluation."
    )

    parser.add_argument(
        "--input",
        default="results/raw/answers_hybrid_rerank.jsonl",
        help="Input answer JSONL file.",
    )

    parser.add_argument(
        "--output",
        default="results/ragas/ragas_scores.csv",
        help="Output CSV file.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help=(
            "Number of questions between progress summaries. "
            "Evaluation is sequential and saved after every question."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of questions to evaluate.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    # Load input questions.
    with input_path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        source_rows = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    if args.limit is not None:
        source_rows = source_rows[:args.limit]

    if not source_rows:
        raise RuntimeError(
            "No evaluation rows were found in the input file."
        )

    required_fields = {
        "id",
        "dataset",
        "user_input",
        "retrieved_contexts",
        "response",
        "reference",
    }

    for index, row in enumerate(source_rows):
        missing = required_fields - row.keys()

        if missing:
            raise ValueError(
                f"Input row {index} is missing fields: {missing}"
            )

        if not isinstance(row["retrieved_contexts"], list):
            raise ValueError(
                f"retrieved_contexts must be a list: {row['id']}"
            )

    # Load previous checkpoints.
    results = load_existing_results(output_path)

    evaluator_llm, http_client = create_evaluator()

    try:
        metrics = create_metrics(evaluator_llm)

        total = len(source_rows)
        completed = 0
        failed = 0

        print(f"Questions to process: {total}")

        for index, source_row in enumerate(source_rows, start=1):
            question_id = source_row["id"]

            result_row = prepare_result_row(
                source_row,
                results.get(question_id),
            )

            print(
                f"\n[{index}/{total}] "
                f"{question_id}"
            )

            for metric_name, metric in metrics.items():

                # Reuse previously saved valid scores.
                if parse_score(result_row.get(metric_name)) is not None:
                    print(
                        f"  {metric_name}: already saved; skipping"
                    )
                    continue

                error_column = {
                    "faithfulness": "faithfulness_error",
                    FACTUAL_COL: "factual_correctness_error",
                    "context_recall": "context_recall_error",
                }[metric_name]

                try:
                    score = score_metric(
                        metric,
                        metric_name,
                        source_row,
                    )

                    result_row[metric_name] = score
                    result_row[error_column] = ""

                    print(
                        f"  {metric_name}: {score:.4f}"
                    )

                except Exception as exc:
                    failed += 1

                    result_row[metric_name] = ""

                    error_message = (
                        f"{type(exc).__name__}: {exc}"
                    )

                    result_row[error_column] = error_message

                    print(
                        f"  ERROR in {metric_name}: "
                        f"{error_message}"
                    )

            # Save every question immediately.
            results[question_id] = result_row
            save_results(output_path, results)

            if all(
                parse_score(result_row.get(column)) is not None
                for column in SCORE_COLUMNS
            ):
                completed += 1

            if index % args.batch_size == 0 or index == total:
                print(
                    f"\nProgress: {index}/{total} questions processed"
                )
                print(
                    f"Fully scored in this run: {completed}"
                )
                print(
                    f"Metric failures in this run: {failed}"
                )
                print(f"Checkpoint: {output_path}")

            # Small delay between questions to reduce request bursts.
            if index < total:
                time.sleep(1)

        fully_scored = sum(
            all(
                parse_score(row.get(column)) is not None
                for column in SCORE_COLUMNS
            )
            for row in results.values()
        )

        print("\nEvaluation finished.")
        print(f"Input questions processed: {total}")
        print(f"Fully scored questions in output: {fully_scored}")
        print(f"Metric failures in this run: {failed}")
        print(f"Saved results: {output_path}")

    finally:
        http_client.close()


if __name__ == "__main__":
    main()
