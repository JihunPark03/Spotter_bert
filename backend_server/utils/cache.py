import os, json, hashlib, redis
from typing import Optional, Dict, Any

CACHE_TTL = 3600
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True,
    socket_connect_timeout=0.5,
    socket_timeout=0.5,
)


def make_cache_key(text: str, namespace: str = "summary") -> str:
    hash_key = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
    return f"{namespace}:{hash_key}"


def get_cache(key: str) -> Optional[Dict[str, Any]]:
    try:
        data = redis_client.get(key)
        if not data:
            return None
        return json.loads(data)
    except redis.exceptions.RedisError as e:
        print(f"[Redis] get_cache failed: {e}")
        return None

def set_cache(key: str, data: dict):
    try:
        redis_client.set(key, json.dumps(data), ex=CACHE_TTL)
    except redis.exceptions.RedisError as e:
        print(f"[Redis] set_cache failed: {e}")
