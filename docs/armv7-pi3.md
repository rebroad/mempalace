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
2. A shared virtual environment at `~/.local/mempalace-venv`.
3. A local SQLite build installed at `~/.local/sqlite-3.45.3`.
4. `pysqlite3` built from source against that local SQLite install, so Chroma
   uses a real SQLite engine with `RETURNING` support instead of the older
   system `sqlite3` module.
5. Local BLAS/OpenBLAS libraries on `LD_LIBRARY_PATH` for the native numeric
   stack.
6. Source builds for the native pieces that did not have usable armv7 wheels.

In practice, the MemPalace runtime needed these environment variables when it
was launched from the Telegram bot or from a shell on the Pi:

```bash
export LD_LIBRARY_PATH="$HOME/.local/sqlite-3.45.3/lib:$HOME/.local/openblas-bullseye/usr/lib/arm-linux-gnueabihf/openblas-pthread:$HOME/.local/atlas-bullseye/usr/lib/arm-linux-gnueabihf:$LD_LIBRARY_PATH"
export XDG_CACHE_HOME="$HOME/.cache"
export MEMPALACE_PALACE_PATH="$HOME/.mempalace/palace"
export MEMPALACE_LITE=1
export PYTHONPATH="$HOME/src/mempalace${PYTHONPATH:+:$PYTHONPATH}"
```

## Repository Changes

The repository changes that made the armv7 setup work were:

- Added `mempalace/chroma_compat.py` to centralize Chroma client/collection
  creation.
- Replaced the default embedding path with a local hash-based embedding
  function so MemPalace can run without `onnxruntime` on `pi3`.
- Added lite-mode runtime shims in `mempalace/__init__.py` and `sitecustomize.py`
  so Chroma can start on armv7 without ONNX, telemetry/auth side paths, or the
  old stdlib SQLite module getting in the way.
- Updated the ingest/search/repair/Layers/MCP call sites to use the new
  compatibility helper instead of calling `chromadb.PersistentClient(...)`
  directly.
- Added no-op MCP resource responses so the server does not fail when clients
  probe `resources/list`, `resources/templates/list`, or `resources/read`
  during startup.

The relevant code paths are:

- [`mempalace/chroma_compat.py`](../mempalace/chroma_compat.py)
- [`mempalace/__init__.py`](../mempalace/__init__.py)
- [`sitecustomize.py`](../sitecustomize.py)
- [`mempalace/miner.py`](../mempalace/miner.py)
- [`mempalace/convo_miner.py`](../mempalace/convo_miner.py)
- [`mempalace/layers.py`](../mempalace/layers.py)
- [`mempalace/searcher.py`](../mempalace/searcher.py)
- [`mempalace/palace_graph.py`](../mempalace/palace_graph.py)
- [`mempalace/cli.py`](../mempalace/cli.py)
- [`mempalace/mcp_server.py`](../mempalace/mcp_server.py)

## Repair / Bootstrap

On `pi3`, the Telegram bot can now drive the runtime bootstrap itself with:

```bash
~/src/codex-telegram-bot/scripts/telegram_bot.py --repair-mempalace-runtime-and-exit
```

That path is intended to be the operational recovery entry point. It will:

1. Create `~/.local/mempalace-venv` if missing.
2. Ensure local SQLite `3.45.3` exists under `~/.local/sqlite-3.45.3`.
3. Build or rebuild `pysqlite3` against that local SQLite.
4. Reinstall the key Python packages needed by the armv7 MemPalace runtime.
5. Run a smoke test that proves:
   - SQLite `RETURNING` works
   - Chroma collection create/upsert works
   - `mempalace mine --mode convos` works

If the host is missing essential build tools, the repair script will try
`apt-get` (or `sudo -n apt-get`) for the basic toolchain first.

## Verification

The armv7 build was checked with the main flows that MemPalace needs for the
Telegram bot:

- `mempalace init`
- `mempalace mine --mode convos`
- `mempalace search`
- MCP server startup and resource discovery
- Telegram bot prune/import flow via `/conversation empty`

The important result is that MemPalace now runs locally on `pi3` without
requiring `onnxruntime`. The trade-off is that the local hash embedding is a
pragmatic fallback, not a drop-in replacement for the upstream semantic
embedding stack.
