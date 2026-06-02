import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)

MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
            base_url="https://openrouter.ai/api/v1",
        )
    return _client


def target_direction(complexity_v1: float, mean_complexity: float) -> str:
    return (
        "simpler and easier to read"
        if complexity_v1 > mean_complexity
        else "more detailed and intellectually rich"
    )


def rewrite_paragraph(text: str, direction: str, reason: str) -> str:
    prompt = (
        f"Rewrite the following paragraph to be {direction}. "
        f"Context: {reason} "
        "Preserve the core meaning. Return only the rewritten paragraph.\n\n"
        f"Original:\n{text}"
    )
    try:
        response = _get_client().chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error("OpenRouter rewrite failed: %s", e)
        return "[Rewrite unavailable]"
