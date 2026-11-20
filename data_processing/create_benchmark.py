"""Create a reproducible 

0:30 HotpotQA/2Wiki evaluation benchmark.

Example:
python -m data_processing.create_benchmark \
  --hotpot datasets/hotpotqa/dev_distractor.json \
  --wiki datasets/2wikimultihopqa/dev.json \
  --hotpot-count 700 --wiki-count 300 \
  --seed 42 --output datasets/benchmark_70_30.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from .loaders import load_hotpotqa, load_2wiki


def create_benchmark(
    hotpot_path: str,
    wiki_path: str,
    hotpot_count: int = 700,
    wiki_count: int = 300,
    seed: int = 42,
    hotpot_split: str = "validation",
    wiki_split: str = "validation",
) -> dict:
    if hotpot_count <= 0 or wiki_count < 0:
        raise ValueError("HotpotQA count must be positive and 2Wiki count cannot be negative")

    hotpot = load_hotpotqa(hotpot_path, hotpot_split)
    wiki = load_2wiki(wiki_path, wiki_split)

    if len(hotpot) < hotpot_count:
        raise ValueError(f"HotpotQA has {len(hotpot)} records; {hotpot_count} requested")
    if len(wiki) < wiki_count:
        raise ValueError(f"2Wiki has {len(wiki)} records; {wiki_count} requested")

    rng = random.Random(seed)
    selected_hotpot = rng.sample(hotpot, hotpot_count)
    selected_wiki = rng.sample(wiki, wiki_count)
    combined = selected_hotpot + selected_wiki
    rng.shuffle(combined)

    # Defensive validation: unique source IDs and usable context/evidence.
    ids = [item["id"] for item in combined]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate normalized question IDs detected")
    if any(not item["contexts"] for item in combined):
        raise ValueError("At least one selected question has no context passages")

    counts = Counter(item["dataset"] for item in combined)
    return {
        "benchmark_name": "hotpotqa_70_2wikimultihopqa_30",
        "version": 1,
        "seed": seed,
        "sampling": {
            "hotpotqa": {"split": hotpot_split, "selected": hotpot_count},
            "2wikimultihopqa": {"split": wiki_split, "selected": wiki_count},
        },
        "total_questions": len(combined),
        "dataset_counts": dict(counts),
        "questions": combined,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hotpot", required=True, help="HotpotQA answerable JSON split")
    parser.add_argument("--wiki", required=True, help="2Wiki answerable JSON split")
    parser.add_argument("--hotpot-count", type=int, default=700)
    parser.add_argument("--wiki-count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hotpot-split", default="validation")
    parser.add_argument("--wiki-split", default="validation")
    parser.add_argument("--output", default="datasets/benchmark_70_30.json")
    args = parser.parse_args()

    benchmark = create_benchmark(
        hotpot_path=args.hotpot,
        wiki_path=args.wiki,
        hotpot_count=args.hotpot_count,
        wiki_count=args.wiki_count,
        seed=args.seed,
        hotpot_split=args.hotpot_split,
        wiki_split=args.wiki_split,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved benchmark: {output_path}")
    print(f"Total questions: {benchmark['total_questions']}")
    print(f"Dataset counts: {benchmark['dataset_counts']}")
    print(f"Sampling seed: {benchmark['seed']}")
    print("Reminder: answer/supporting_facts/evidence_chain are evaluation-only labels.")


if __name__ == "__main__":
    main()
