
import json
from pathlib import Path
from dotenv import load_dotenv

from rag.groq_generator import GroqGenerator

load_dotenv()

source = Path("results/pilot_100/raw/answers_hybrid_rerank.jsonl")
output = Path("results/pilot_100/repairs/answers_repaired.jsonl")

target_id = "hotpotqa:5a74f8bd5542993748c8976a"

# Load the original answer record.
records = [
    json.loads(line)
    for line in source.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

record = next((x for x in records if x["id"] == target_id), None)

if record is None:
    raise ValueError(f"Question not found: {target_id}")

if record.get("response", "").strip():
    raise ValueError("This question already has a non-empty answer.")

# Reconstruct the evidence objects expected by GroqGenerator.
class Chunk:
    def __init__(self, text, index):
        self.text = text
        self.title = f"Retrieved context {index + 1}"

evidence = [
    {"chunk": Chunk(text, i)}
    for i, text in enumerate(record["retrieved_contexts"])
]

generator = GroqGenerator()

generated = generator.generate(
    record["user_input"],
    evidence
)

if not generated["answer"].strip():
    raise RuntimeError("Repair produced an empty answer.")

repaired = dict(record)
repaired["response"] = generated["answer"]
repaired["generation"] = generated
repaired["exact_match"] = None
repaired["token_f1"] = None
repaired["repair_status"] = "regenerated"

output.parent.mkdir(parents=True, exist_ok=True)

with output.open("a", encoding="utf-8") as f:
    f.write(json.dumps(repaired, ensure_ascii=False) + "\n")

print("Original answer:", repr(record["response"]))
print("Repaired answer:", repr(generated["answer"]))
print("Finish reason:", generated["finish_reason"])
print("Attempts:", generated["attempts"])
print("Saved:", output)
