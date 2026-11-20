
import os
import time
import ssl
import httpx
import truststore

from groq import Groq


class GroqGenerator:

    def __init__(self, model=None):

        self.model = model or os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-120b"
        )

        # Use Windows trusted root certificates
        ssl_context = truststore.SSLContext(
            ssl.PROTOCOL_TLS_CLIENT
        )

        http_client = httpx.Client(
            verify=ssl_context,
            timeout=120.0
        )

        self.client = Groq(
            api_key=os.environ["GROQ_API_KEY"],
            http_client=http_client
        )

    def generate(self, question, evidence, max_retries=5):

        context = "\n\n".join(
            f"[{i+1}] {x['chunk'].title}\n{x['chunk'].text}"
            for i, x in enumerate(evidence)
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "Answer the question using only the supplied evidence. "
                    "Return only the shortest answer that directly answers the question. "
                    "Do not explain your reasoning. "
                    "Do not include citations, source references, sentence IDs, or labels. "
                    "Do not include phrases such as 'The answer is'. "
                    "Preserve names, dates, and numerical values accurately. "
                    "If the evidence is insufficient, respond with exactly: "
                    "'Insufficient information'. "
                    "Output plain text only. Do not invent facts."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Evidence:\n{context}\n\n"
                    "Provide only the answer."
                )
            }
        ]

        # Increase the completion budget when the model runs out of tokens.
        token_budgets = [512, 1024, 2048, 4096, 4096]

        last_finish_reason = None

        for attempt in range(max_retries):

            token_limit = token_budgets[
                min(attempt, len(token_budgets) - 1)
            ]

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    max_completion_tokens=token_limit
                )

                choice = response.choices[0]
                usage = response.usage

                answer = (choice.message.content or "").strip()
                finish_reason = choice.finish_reason
                last_finish_reason = finish_reason

                usage_data = {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens
                }

                # Retry if the completion budget was exhausted.
                if finish_reason == "length":
                    print(
                        f"Generation truncated at {token_limit} tokens. "
                        f"Retrying with a larger budget..."
                    )
                    continue

                # Never silently accept an empty answer.
                if not answer:
                    if attempt < max_retries - 1:
                        print(
                            f"Empty completion (finish_reason={finish_reason}). "
                            f"Retrying with {token_budgets[min(attempt + 1, len(token_budgets) - 1)]} tokens..."
                        )
                        continue

                    raise RuntimeError(
                        "Groq returned an empty answer after retries. "
                        f"Last finish_reason: {finish_reason}"
                    )

                return {
                    "answer": answer,
                    "model": self.model,
                    "finish_reason": finish_reason,
                    "attempts": attempt + 1,
                    "usage": usage_data
                }

            except RuntimeError:
                raise

            except Exception as exc:

                status = getattr(exc, "status_code", None)
                error_name = type(exc).__name__

                retryable = (
                    status == 429
                    or (status is not None and status >= 500)
                    or error_name in (
                        "APIConnectionError",
                        "APITimeoutError",
                        "ConnectError",
                        "ReadTimeout",
                        "ConnectTimeout"
                    )
                )

                if retryable and attempt < max_retries - 1:
                    wait = min(60, 5 * (2 ** attempt))
                    print(
                        f"Groq temporary error: {error_name}. "
                        f"Retrying in {wait} seconds..."
                    )
                    time.sleep(wait)
                    continue

                raise

        raise RuntimeError(
            "Groq generation failed after all retries. "
            f"Last finish_reason: {last_finish_reason}"
        )
