
import os
import ssl
import truststore
import httpx

from dotenv import load_dotenv
from openai import OpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import FactualCorrectness

load_dotenv()

model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError("GROQ_API_KEY not found in .env")

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

tests = [
    {
        "name": "A_exact_short_answer",
        "response": "Tim Cluess",
        "reference": "Tim Cluess",
    },
    {
        "name": "B_exact_full_sentence",
        "response": "Tim Cluess was the head coach.",
        "reference": "Tim Cluess was the head coach.",
    },
    {
        "name": "C_short_vs_full_sentence",
        "response": "Tim Cluess",
        "reference": "Tim Cluess was the head coach.",
    },
    {
        "name": "D_wrong_answer",
        "response": "Steve Prohm",
        "reference": "Tim Cluess",
    },
]

try:
    print("Evaluator:", model)
    print("-" * 60)

    for test in tests:
        print("\nTest:", test["name"])
        print("Response:", test["response"])
        print("Reference:", test["reference"])

        try:
            result = metric.score(
                response=test["response"],
                reference=test["reference"],
            )

            print("Factual Correctness:", result.value)

            # Inspect the detailed metric result if available.
            print("Details:", getattr(result, "details", None))

        except Exception as exc:
            print("ERROR:", type(exc).__name__, str(exc))

        print("-" * 60)

finally:
    http_client.close()
