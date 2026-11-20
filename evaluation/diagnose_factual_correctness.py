
import os
import ssl
import csv
import re
import time
import unicodedata
import argparse
import truststore
import httpx

from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import FactualCorrectness

load_dotenv()


def normalize(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = text.casefold()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="results/hotpotqa_100/ragas/ragas_scores.csv",
    )

    parser.add_argument(
        "--output",
        default="results/hotpotqa_100/ragas/fc_diagnostic.csv",
    )

    parser.add_argument("--limit", type=int, default=41)
    parser.add_argument("--repeats", type=int, default=1)

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    with input_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    # Select exact/normalized matches that originally scored zero.
    candidates = [
        row for row in rows
        if normalize(row["response"]) == normalize(row["reference"])
        and row.get("factual_correctness(mode=f1)", "").strip() == "0.0"
    ]

    print("Original rows:", len(rows))
    print("Exact normalized matches with FC=0:", len(candidates))

    candidates = candidates[:args.limit]

    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found")

    ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

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

    metric = FactualCorrectness(
        llm=evaluator_llm,
        mode="f1",
    )

    output_rows = []

    try:
        for index, row in enumerate(candidates, start=1):
            print(f"\n[{index}/{len(candidates)}] {row['id']}")
            print("Response:", repr(row["response"]))
            print("Reference:", repr(row["reference"]))

            for repeat in range(1, args.repeats + 1):
                try:
                    result = metric.score(
                        response=row["response"],
                        reference=row["reference"],
                    )

                    score = result.value
                    error = ""

                    print(f"  Run {repeat}: {score}")

                except Exception as exc:
                    score = ""
                    error = f"{type(exc).__name__}: {exc}"
                    print(f"  Run {repeat}: ERROR {error}")

                output_rows.append({
                    "id": row["id"],
                    "repeat": repeat,
                    "response": row["response"],
                    "reference": row["reference"],
                    "original_score": row[
                        "factual_correctness(mode=f1)"
                    ],
                    "diagnostic_score": score,
                    "error": error,
                })

                time.sleep(1)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open(
            "w", encoding="utf-8", newline=""
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "id",
                    "repeat",
                    "response",
                    "reference",
                    "original_score",
                    "diagnostic_score",
                    "error",
                ],
            )
            writer.writeheader()
            writer.writerows(output_rows)

        print("\nSaved:", output_path)

    finally:
        http_client.close()


if __name__ == "__main__":
    main()
