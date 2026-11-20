"""Load raw JSON benchmark files from the official dataset formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .normalize_dataset import normalize_record


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a top-level JSON list in {path}")
    return data


def load_hotpotqa(path: str | Path, split: str = "validation") -> list[dict[str, Any]]:
    records = load_json_records(path)
    return [normalize_record(row, "hotpotqa", split) for row in records]


def load_2wiki(path: str | Path, split: str = "validation") -> list[dict[str, Any]]:
    records = load_json_records(path)
    return [normalize_record(row, "2wikimultihopqa", split) for row in records]
