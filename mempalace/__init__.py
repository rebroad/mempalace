"""MemPalace — Give your AI a memory. No API key required."""

import logging
import os
import sys
import types
from enum import Enum
from pathlib import Path
import importlib.util
import tempfile
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


def _sqlite_runtime_supports_chroma(sqlite3_module) -> tuple[bool, str]:
    """Probe the actual SQLite engine features Chroma lite needs.

    Version strings are not enough here. On pi3 we observed a runtime that
    reported a newer SQLite version while still lacking DROP COLUMN and the
    trigram FTS5 tokenizer that Chroma migrations require.
    """
    path = None
    conn = None
    try:
        fd, path = tempfile.mkstemp(suffix=".sqlite3", dir="/var/tmp")
        os.close(fd)
        conn = sqlite3_module.connect(path)
        cur = conn.cursor()
        cur.execute(
            "CREATE TABLE collections ("
            "id TEXT PRIMARY KEY, "
            "name TEXT NOT NULL, "
            "topic TEXT NOT NULL, "
            "dimension INTEGER, "
            "database_id TEXT NOT NULL, "
            "UNIQUE(name, database_id))"
        )
        conn.commit()
        cur.execute("ALTER TABLE collections DROP COLUMN topic")
        conn.commit()
        cur.execute(
            'CREATE VIRTUAL TABLE embedding_fulltext_search '
            'USING fts5(string_value, tokenize="trigram")'
        )
        conn.commit()
        return True, ""
    except Exception as exc:
        return False, str(exc)
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass
        if path:
            try:
                os.remove(path)
            except Exception:
                pass


def _require_sqlite_runtime_features(sqlite3_module) -> None:
    ok, detail = _sqlite_runtime_supports_chroma(sqlite3_module)
    if ok:
        return
    lib_hint = str(Path.home() / ".local" / "sqlite-3.45.3" / "lib")
    raise RuntimeError(
        "MemPalace lite mode requires a SQLite runtime with DROP COLUMN and "
        f'FTS5 trigram support. Current sqlite3 reports "{sqlite3_module.sqlite_version}" '
        f"but does not provide the required features ({detail}). "
        "Do not rely on a spoofed version string. Start Python with the correct "
        f'LD_LIBRARY_PATH, for example: LD_LIBRARY_PATH="{lib_hint}" ...'
    )


if _lite_mode_from_env():
    try:
        import pysqlite3.dbapi2 as _sqlite3

        sys.modules["sqlite3"] = _sqlite3
    except Exception:
        import sqlite3 as _sqlite3

    _require_sqlite_runtime_features(_sqlite3)

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

from .version import __version__  # noqa: E402

# chromadb telemetry: posthog capture() was broken in 0.6.x causing noisy stderr
# warnings ("capture() takes 1 positional argument but 3 were given"). In 1.x the
# posthog client is a no-op stub, so this is now harmless — kept as a guard in
# case future chromadb versions re-introduce real telemetry calls.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

# NOTE: the previous block set ``ORT_DISABLE_COREML=1`` on macOS arm64 as a
# supposed workaround for the #74 ARM64 segfault.  Two problems:
#
# 1. ONNX Runtime does not read that env var -- it has no global way to
#    disable a single execution provider, so the setdefault was a no-op.
# 2. #74 is a null-pointer crash in ``chromadb_rust_bindings.abi3.so``, not
#    an ONNX issue, so disabling CoreML would not have fixed it anyway.
#
# #521 has since traced the actual macOS arm64 crashes (both in mine and
# search paths) to the 0.x chromadb hnswlib binding.  Filtering
# CoreMLExecutionProvider at the ONNX layer leaves the hnswlib C++ crash
# intact, so the real fix is upgrading chromadb to 1.5.4+, which #581
# proposes.  See #397 for the history of this line.

def main(*args, **kwargs):
    """Compatibility shim for older console scripts.

    Some installed wrappers still do ``from mempalace import main``.
    Keep that import path working by delegating to the real CLI entry point.
    """

    from .cli import main as cli_main

    return cli_main(*args, **kwargs)


__all__ = ["__version__", "main"]
