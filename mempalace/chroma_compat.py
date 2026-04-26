import hashlib
import math
import os
import re
from typing import Iterable

import chromadb

COLLECTION_NAME = "mempalace_drawers"
EMBED_DIM = 256
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class LocalHashEmbeddingFunction:
    @staticmethod
    def name() -> str:
        return "mempalace_local_hash"

    @staticmethod
    def build_from_config(config: dict[str, object]) -> "LocalHashEmbeddingFunction":
        LocalHashEmbeddingFunction.validate_config(config)
        return LocalHashEmbeddingFunction()

    def get_config(self) -> dict[str, object]:
        return {}

    @staticmethod
    def validate_config(config: dict[str, object]) -> None:
        return None

    def is_legacy(self) -> bool:
        return False

    def __call__(self, input: Iterable[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in input]

    def _embed_text(self, text: str) -> list[float]:
        vec = [0.0] * EMBED_DIM
        tokens = TOKEN_RE.findall((text or "").lower())
        if not tokens:
            return vec
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8"), usedforsecurity=False).digest()
            idx = int.from_bytes(digest[:2], "big") % EMBED_DIM
            sign = -1.0 if (digest[2] & 1) else 1.0
            weight = 1.0 + ((digest[3] % 7) / 10.0)
            vec[idx] += sign * weight
        norm = math.sqrt(sum(value * value for value in vec))
        if norm:
            vec = [value / norm for value in vec]
        return vec


def get_client(palace_path: str):
    os.makedirs(palace_path, exist_ok=True)
    return chromadb.PersistentClient(path=palace_path)


def get_collection(
    palace_path: str,
    collection_name: str = COLLECTION_NAME,
    create: bool = False,
):
    client = get_client(palace_path)
    embedding_function = LocalHashEmbeddingFunction()
    if create:
        return client.get_or_create_collection(
            collection_name,
            embedding_function=embedding_function,
        )
    try:
        return client.get_collection(
            collection_name,
            embedding_function=embedding_function,
        )
    except TypeError:
        # Older Chroma versions may not accept embedding_function on lookup.
        return client.get_collection(collection_name)
    except Exception:
        return client.create_collection(
            collection_name,
            embedding_function=embedding_function,
        )
