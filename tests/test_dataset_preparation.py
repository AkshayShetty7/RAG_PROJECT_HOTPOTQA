import json
import tempfile
import unittest
from pathlib import Path

from data_processing.create_benchmark import create_benchmark
from data_processing.normalize_dataset import normalize_record


class DatasetPreparationTests(unittest.TestCase):
    def test_normalize_hotpot_style_record(self):
        row = {
            "_id": "abc",
            "question": "Which city is mentioned?",
            "answer": "Paris",
            "context": [["Page A", ["Paris is a city."]]],
            "supporting_facts": [["Page A", 0]],
        }
        normalized = normalize_record(row, "hotpotqa", "validation")
        self.assertEqual(normalized["id"], "hotpotqa:abc")
        self.assertEqual(normalized["supporting_facts"][0]["sentence"], "Paris is a city.")

    def test_reproducible_70_30_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            hotpot_rows = []
            wiki_rows = []
            for i in range(4):
                hotpot_rows.append({
                    "_id": f"h{i}", "question": f"H question {i}", "answer": "H answer",
                    "context": [["H page", ["H sentence"]]],
                    "supporting_facts": [["H page", 0]],
                })
                wiki_rows.append({
                    "_id": f"w{i}", "question": f"W question {i}", "answer": "W answer",
                    "context": [["W page", ["W sentence"]]],
                    "supporting_facts": [["W page", 0]],
                    "evidences": [["subject", "relation", "object"]],
                })
            hotpot_path = tmp_path / "hotpot.json"
            wiki_path = tmp_path / "wiki.json"
            hotpot_path.write_text(json.dumps(hotpot_rows), encoding="utf-8")
            wiki_path.write_text(json.dumps(wiki_rows), encoding="utf-8")

            first = create_benchmark(str(hotpot_path), str(wiki_path), 3, 2, 123)
            second = create_benchmark(str(hotpot_path), str(wiki_path), 3, 2, 123)
            self.assertEqual(first, second)
            self.assertEqual(first["dataset_counts"], {"hotpotqa": 3, "2wikimultihopqa": 2})
            self.assertEqual(first["total_questions"], 5)


if __name__ == "__main__":
    unittest.main()
