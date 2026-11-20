
import os
import ssl
import asyncio
import truststore
import httpx

from dotenv import load_dotenv
from openai import OpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import FactualCorrectness

load_dotenv()

TESTS = [
    ("Dutch", "Dutch"),
    ("U2", "U2"),
    ("Personal History", "Personal History"),
    ("Royal Air Force", "Royal Air Force"),
    ("Tim Cluess", "Tim Cluess"),
]

async def inspect_case(metric, response, reference):
    print("\n" + "=" * 65)
    print("RESPONSE:", repr(response))
    print("REFERENCE:", repr(reference))

    # Direction 1: response claims checked against reference
    response_claims = await metric._decompose_claims(response)
    print("\n1. Claims extracted from response:")
    print(repr(response_claims))

    if response_claims:
        response_verdicts = await metric._verify_claims(
            response_claims, reference
        )
        print("Response claim verification:")
        print(repr(response_verdicts))
    else:
        print("Response produced NO claims.")

    # Direction 2: reference claims checked against response
    reference_claims = await metric._decompose_claims(reference)
    print("\n2. Claims extracted from reference:")
    print(repr(reference_claims))

    if reference_claims:
        reference_verdicts = await metric._verify_claims(
            reference_claims, response
        )
        print("Reference claim verification:")
        print(repr(reference_verdicts))
    else:
        print("Reference produced NO claims.")

    # Final score using the normal Ragas method
    result = await metric.ascore(
        response=response,
        reference=reference,
    )
    print("\n3. Final Ragas Factual Correctness:", result.value)


async def main():
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found in .env")

    ssl_context = truststore.SSLContext(
        ssl.PROTOCOL_TLS_CLIENT
    )

    http_client = httpx.AsyncClient(
        verify=ssl_context,
        timeout=120.0,
    )

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            http_client=httpx.Client(
                verify=ssl_context,
                timeout=120.0,
            ),
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

        print("Evaluator:", model)

        for response, reference in TESTS:
            await inspect_case(metric, response, reference)

        client.close()

    finally:
        await http_client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
