# MemPalace on pi3 / armv7

This note records the steps needed to get MemPalace running on a Raspberry Pi 3
class machine (`armv7`, Debian Bullseye-era userspace, system glibc `2.28`).
The goal was to keep the full MemPalace toolchain working locally, without
depending on `onnxruntime` or other wheels that are not available on this
platform.

## What Broke

The stock install path did not work on `pi3` for a few separate reasons:

- The system Python was too old for the current MemPalace release line.
- Chroma's default embedding path pulled in `onnxruntime`, which is not
  available for this host in a usable form.
- Some prebuilt wheels for native packages assumed a newer glibc than the one
  shipped on the device.
- The bundled `sqlite3` version exposed by the system interpreter was too old
  for the Chroma code paths MemPalace uses.

## Working Runtime

The runtime that worked on `pi3` was:

1. Python `3.9.19` installed locally at `~/.local/python3.9.19`.
2. A dedicated virtual environment at `~/src/mempalace/.venv`.
3. A local SQLite build exposed through `pysqlite3` so Chroma could use a
   modern SQLite version.
4. Local BLAS/OpenBLAS libraries on `LD_LIBRARY_PATH` for the native numeric
   stack.
5. Source builds for the native pieces that did not have usable armv7 wheels.

In practice, the MemPalace runtime needed these environment variables when it
was launched from the Telegram bot or from a shell on the Pi:

```bash
export LD_LIBRARY_PATH="$HOME/.local/openblas-bullseye/lib:$HOME/.local/atlas-bullseye/lib:$LD_LIBRARY_PATH"
export XDG_CACHE_HOME="$HOME/.cache"
export MEMPALACE_PALACE_PATH="$HOME/.mempalace/palace"
```

## Repository Changes

The repository changes that made the armv7 setup work were:

- Added `mempalace/chroma_compat.py` to centralize Chroma client/collection
  creation.
- Replaced the default embedding path with a local hash-based embedding
  function so MemPalace can run without `onnxruntime` on `pi3`.
- Updated the ingest/search/repair/Layers/MCP call sites to use the new
  compatibility helper instead of calling `chromadb.PersistentClient(...)`
  directly.
- Added no-op MCP resource responses so the server does not fail when clients
  probe `resources/list`, `resources/templates/list`, or `resources/read`
  during startup.

The relevant code paths are:

- [`mempalace/chroma_compat.py`](../mempalace/chroma_compat.py)
- [`mempalace/miner.py`](../mempalace/miner.py)
- [`mempalace/convo_miner.py`](../mempalace/convo_miner.py)
- [`mempalace/layers.py`](../mempalace/layers.py)
- [`mempalace/searcher.py`](../mempalace/searcher.py)
- [`mempalace/palace_graph.py`](../mempalace/palace_graph.py)
- [`mempalace/cli.py`](../mempalace/cli.py)
- [`mempalace/mcp_server.py`](../mempalace/mcp_server.py)

## Verification

The armv7 build was checked with the main flows that MemPalace needs for the
Telegram bot:

- `mempalace init`
- `mempalace mine --mode convos`
- `mempalace search`
- MCP server startup and resource discovery

The important result is that MemPalace now runs locally on `pi3` without
requiring `onnxruntime`. The trade-off is that the local hash embedding is a
pragmatic fallback, not a drop-in replacement for the upstream semantic
embedding stack.
