
import csv
import re
import unicodedata
from pathlib import Path

INPUT = Path(
    "results/hotpotqa_100/ragas/ragas_scores.csv"
)

OUTPUT = Path(
    "results/hotpotqa_100/ragas/factual_correctness_analysis.csv"
)

FC_COL = "factual_correctness(mode=f1)"


def normalize(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = text.casefold()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


with INPUT.open(
    "r", encoding="utf-8-sig", newline=""
) as f:
    rows = list(csv.DictReader(f))

for row in rows:
    response = row.get("response", "")
    reference = row.get("reference", "")

    exact_match = (
        bool(normalize(response))
        and normalize(response) == normalize(reference)
    )

    raw_value = row.get(FC_COL, "").strip()

    try:
        raw_score = float(raw_value)
    except (ValueError, TypeError):
        raw_score = None

    adjusted_score = (
        1.0 if exact_match else raw_score
    )

    row["normalized_exact_match"] = int(exact_match)
    row["factual_correctness_ragas_raw"] = (
        raw_score if raw_score is not None else ""
    )
    row["factual_correctness_adjusted"] = (
        adjusted_score if adjusted_score is not None else ""
    )
    row["factual_correctness_override_applied"] = int(
        exact_match and raw_score != 1.0
    )

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

fieldnames = list(rows[0].keys())

with OUTPUT.open(
    "w", encoding="utf-8", newline=""
) as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("Questions:", len(rows))
print(
    "Normalized exact matches:",
    sum(r["normalized_exact_match"] for r in rows)
)
print(
    "Ragas FC mean:",
    round(
        sum(float(r[FC_COL]) for r in rows if r[FC_COL].strip())
        / sum(bool(r[FC_COL].strip()) for r in rows),
        4
    )
)
print(
    "Adjusted FC mean:",
    round(
        sum(float(r["factual_correctness_adjusted"])
            for r in rows if str(r["factual_correctness_adjusted"]).strip())
        / sum(bool(str(r["factual_correctness_adjusted"]).strip())
              for r in rows),
        4
    )
)
print("Saved:", OUTPUT)
