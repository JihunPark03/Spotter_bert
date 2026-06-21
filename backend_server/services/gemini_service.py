from google import genai
from google.genai import types
from utils.cache import make_cache_key, get_cache, set_cache
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

PROMPT_PATH = Path("gemini_prompt.txt")


def _load_system_prompt() -> str:
    try:
        return PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("gemini_prompt.txt not found - fallback prompt used")
        return (
            "You are an extractor that outputs exactly four '-키: 값' lines "
            "(must include location)."
        )


def extract_features(user_text: str, client: genai.Client):
    key = make_cache_key(user_text, namespace="gemini")

    cached = get_cache(key)
    if cached and "reply" in cached:
        return {**cached, "cached": True}

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=user_text,
        config=types.GenerateContentConfig(
            system_instruction=_load_system_prompt(),
            temperature=0.0,
        ),
    )

    result = {"reply": response.text}
    set_cache(key, result)
    return {**result, "cached": False}
