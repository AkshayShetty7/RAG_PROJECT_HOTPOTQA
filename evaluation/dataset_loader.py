import json
from pathlib import Path

def load_benchmark(path="datasets/benchmark_70_30.json"):
    data=json.loads(Path(path).read_text(encoding="utf-8"))
    questions=data["questions"]
    if not questions:
        raise ValueError("Benchmark contains no questions.")
    return questions
