from ml_client import request_inference
from utils.cache import make_cache_key, get_cache, set_cache


def detect_ad(text: str):
    key = make_cache_key(text, namespace="detect")

    cached = get_cache(key)
    if cached and "prob_ad" in cached:
        return {**cached, "cached": True}

    prob = request_inference(text)
    result = {
        "prob_ad": round(prob, 4) * 100,
        "is_ad": prob >= 0.5,
    }
    set_cache(key, result)
    return {**result, "cached": False}
