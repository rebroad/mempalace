"""MemPalace runtime tweaks loaded automatically via Python's sitecustomize hook.

On constrained hosts, the lite runtime uses local hash embeddings instead of
Chroma's default ONNX embedder. Recent chromadb builds decide whether to use the
thin-client path during import, so we seed that flag here before chromadb loads.
"""

from __future__ import annotations

import os
import sys
import types
from enum import Enum
from pathlib import Path
import importlib.util
from typing import Optional


def _lite_mode_from_env() -> bool:
    value = os.environ.get("MEMPALACE_LITE", "")
    return value.lower() not in {"", "0", "false", "no", "off"}


def _chromadb_utils_path() -> Optional[str]:
    spec = importlib.util.find_spec("chromadb")
    if spec is None or spec.origin is None:
        return None
    chromadb_root = Path(spec.origin).resolve().parent
    utils_path = chromadb_root / "utils"
    return str(utils_path) if utils_path.exists() else None


if _lite_mode_from_env():
    try:
        import pysqlite3.dbapi2 as _sqlite3

        sys.modules["sqlite3"] = _sqlite3
        _sqlite3.sqlite_version_info = (3, 45, 3)  # type: ignore[attr-defined]
        _sqlite3.sqlite_version = "3.45.3"  # type: ignore[attr-defined]
    except Exception:
        pass

    if "chromadb.utils" not in sys.modules:
        utils_pkg = types.ModuleType("chromadb.utils")
        utils_path = _chromadb_utils_path()
        utils_pkg.__path__ = [utils_path] if utils_path else []  # type: ignore[attr-defined]
        sys.modules["chromadb.utils"] = utils_pkg

    if "chromadb.auth.token_authn" not in sys.modules:
        token_authn = types.ModuleType("chromadb.auth.token_authn")

        class TokenTransportHeader(str, Enum):
            AUTHORIZATION = "Authorization"
            X_CHROMA_TOKEN = "X-Chroma-Token"

        token_authn.TokenTransportHeader = TokenTransportHeader  # type: ignore[attr-defined]
        sys.modules["chromadb.auth.token_authn"] = token_authn

    if "chromadb.telemetry.opentelemetry" not in sys.modules:
        opentelemetry = types.ModuleType("chromadb.telemetry.opentelemetry")

        class OpenTelemetryGranularity(str, Enum):
            NONE = "none"
            OPERATION = "operation"
            OPERATION_AND_SEGMENT = "operation_and_segment"
            ALL = "all"

        def trace_method(*_args, **_kwargs):
            def decorator(func):
                return func

            return decorator

        def add_attributes_to_current_span(*_args, **_kwargs):
            return None

        def otel_init(*_args, **_kwargs):
            return None

        class OpenTelemetryClient:  # noqa: D401
            def __init__(self, *args, **kwargs):
                self._dependencies = set()

            def dependencies(self):
                return self._dependencies

            def start(self):
                return None

            def stop(self):
                return None

            def reset_state(self):
                return None

        opentelemetry.OpenTelemetryGranularity = OpenTelemetryGranularity  # type: ignore[attr-defined]
        opentelemetry.trace_method = trace_method  # type: ignore[attr-defined]
        opentelemetry.add_attributes_to_current_span = add_attributes_to_current_span  # type: ignore[attr-defined]
        opentelemetry.otel_init = otel_init  # type: ignore[attr-defined]
        opentelemetry.OpenTelemetryClient = OpenTelemetryClient  # type: ignore[attr-defined]
        opentelemetry.tracer = None  # type: ignore[attr-defined]
        opentelemetry.granularity = OpenTelemetryGranularity.NONE  # type: ignore[attr-defined]
        sys.modules["chromadb.telemetry.opentelemetry"] = opentelemetry

    if "chromadb.utils.embedding_functions" not in sys.modules:
        embedding_functions = types.ModuleType("chromadb.utils.embedding_functions")
        embedding_functions.__package__ = "chromadb.utils"

        class ChromaLangchainEmbeddingFunction:  # noqa: D401
            pass

        class DefaultEmbeddingFunction:  # noqa: D401
            @staticmethod
            def name() -> str:
                return "default"

            @staticmethod
            def build_from_config(config):
                return DefaultEmbeddingFunction()

            def get_config(self):
                return {}

            def is_legacy(self) -> bool:
                return False

            def __call__(self, input):
                return []

        def get_builtins():
            return {"ChromaLangchainEmbeddingFunction", "DefaultEmbeddingFunction"}

        known_embedding_functions = {"default": DefaultEmbeddingFunction}

        def register_embedding_function(ef_class=None):
            def _register(cls):
                try:
                    known_embedding_functions[cls.name()] = cls
                except Exception:
                    pass
                return cls

            if ef_class is not None:
                return _register(ef_class)
            return _register

        def config_to_embedding_function(config):
            name = config.get("name", "default")
            cls = known_embedding_functions[name]
            return cls.build_from_config(config.get("config", {}))

        embedding_functions.ChromaLangchainEmbeddingFunction = ChromaLangchainEmbeddingFunction  # type: ignore[attr-defined]
        embedding_functions.DefaultEmbeddingFunction = DefaultEmbeddingFunction  # type: ignore[attr-defined]
        embedding_functions.get_builtins = get_builtins  # type: ignore[attr-defined]
        embedding_functions.known_embedding_functions = known_embedding_functions  # type: ignore[attr-defined]
        embedding_functions.register_embedding_function = register_embedding_function  # type: ignore[attr-defined]
        embedding_functions.config_to_embedding_function = config_to_embedding_function  # type: ignore[attr-defined]
        sys.modules["chromadb.utils.embedding_functions"] = embedding_functions
