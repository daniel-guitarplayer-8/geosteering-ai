# External Integrations

**Analysis Date:** 2026-05-24

## APIs & External Services

**Scientific Literature:**
- Semantic Scholar (S2) API — search and retrieve scientific papers
  - Client: `httpx` (async HTTP in `tools/consensus-mcp-server/server.py`)
  - Auth: `S2_API_KEY` env var (optional; public access works without key)
  - Endpoint: `https://api.semanticscholar.org/graph/v1/paper/search`
  - MCP tool: `search_papers`, `get_paper_details`
- ArXiv API — preprint search
  - Client: `httpx` (XML response parsing in `tools/consensus-mcp-server/server.py`)
  - Auth: none required
  - MCP tool: `search_arxiv`
  - Cache: `docs/reference/papers/` (local JSON cache)

## Data Storage

**Databases:**
- None (no database engine used)

**File Storage:**
- Local filesystem — all simulation output (`.dat`, `.out`, `.jac` files written by `tatu.x`)
- Hankel filter weights: `geosteering_ai/simulation/filters/*.npz` and `*.json` (committed, loaded by `filters/loader.py`)
- Numba JIT cache: `$TMPDIR/geosteering_numba_cache` (permissions 0o700, set in `cli/main.py`)
- JAX XLA compilation cache: `$JAX_COMPILATION_CACHE_DIR` (e.g., `/content/jax_cache` in Colab)
- Model checkpoints: filesystem path via `GEOSTEERING_MODEL_PATH` env var (Docker / inference)
- Paper cache: `docs/reference/papers/` (MCP consensus server local cache)

**Caching:**
- LRU plot cache: `geosteering_ai/simulation/tests/sm_plot_cache.py` — in-memory, configurable `max_bytes` (default: 10% RAM, floor 500 MB, ceiling 4 GB via psutil); persisted to QSettings (`cache/max_bytes_mb`, `cache/maxlen`)

## Authentication & Identity

**Auth Provider:**
- None (no user authentication system; API is open in MVP)
- `S2_API_KEY` — optional Semantic Scholar token (in `.mcp.json` env block)
- GitHub token: implicitly used by GitHub Actions (`actions/checkout@v4`) via `GITHUB_TOKEN`

## MCP Servers (Claude Code Integration)

The project registers MCP servers in `.mcp.json` and uses additional MCP plugins from the Claude ecosystem. These expose project capabilities directly to Claude Code as native tools.

**Registered in `.mcp.json`:**
- **consensus** — scientific paper search
  - Command: `python tools/consensus-mcp-server/server.py`
  - Deps: `mcp>=1.0.0`, `httpx>=0.25.0`
  - 4 MCP tools: `search_papers`, `get_paper_details`, `search_arxiv`, `list_cached_papers`
  - Source: `tools/consensus-mcp-server/server.py`
- **colab-mcp** — Google Colab browser automation
  - Command: `uvx git+https://github.com/googlecolab/colab-mcp`
  - Timeout: 60,000 ms
  - Used for: running GPU training/validation notebooks from Claude Code sessions
  - Access via Bash hook permission: `mcp__colab-mcp__open_colab_browser_connection`

**Local MCP servers (instantiated on demand, not in `.mcp.json`):**
- **physics-validator-mcp** — validates simulation physics correctness
  - Source: `tools/physics-validator-mcp/server.py`
  - Deps: `mcp>=1.25.0,<2.0`, `numpy>=1.24`
  - 6 MCP tools: `check_fortran_parity`, `check_maxwell_symmetry`, `check_decoupling_factors`, `check_errata_immutable`, `check_skin_depth`, `run_canonical_models`
  - Wraps: `geosteering_ai.simulation.validation.*` modules
- **numba-profiler-mcp** — benchmarks/profiles Numba simulation performance
  - Source: `tools/numba-profiler-mcp/server.py`
  - Deps: `mcp>=1.25.0,<2.0`, `numpy>=1.24`, `numba>=0.60`
  - 6 MCP tools: `run_scenario_benchmark`, `compare_branches`, `check_cpu_topology`, `check_oversubscription`, `profile_kernel`, `analyze_jit_cache`
  - Wraps: `geosteering_ai.simulation.multi_forward.simulate_multi` + `_workers.py` utilities

**External Claude AI MCP plugins (settings.json `enabledPlugins`):**
- **claude-md-management** — manages CLAUDE.md and project instructions
- **Figma** (via MCP server at runtime, `claude.ai Figma`) — design/UI integration; not actively used in this project but available
- **context7** — on-demand library documentation fetching (TF, Keras, NumPy, etc.)

## Fortran/Python Bridge (f2py)

**Integration type:** Compiled shared library
- Wrapper source: `Fortran_Gerador/tatu_f2py_wrapper.f08`
- Compiled artifacts (Python 3.11 Darwin): `Fortran_Gerador/tatu_f2py.cpython-311-darwin.so`
- Build command: `make f2py_wrapper` (in `Fortran_Gerador/`)
- Python import: `import tatu_f2py` — calls Fortran subroutines directly from Python
- Usage: `geosteering_ai/simulation/validation/compare_fortran.py` for parity checks
- Also used: `geosteering_ai/data/synthetic_generator.py` and simulation validation modules
- Status: compiled for Python 3.11 (Anaconda); Python 3.13 path uses subprocess to `tatu.x` binary

**Subprocess interface (production path):**
- `geosteering_ai/simulation/multi_forward.py` — spawns `tatu.x` as subprocess for Fortran backend
- Workers: `geosteering_ai/simulation/_workers.py` — `ProcessPoolExecutor` with ephemeral pool

## Monitoring & Observability

**Error Tracking:**
- Not integrated (no Sentry, Rollbar, etc.)

**Logs:**
- Python `logging` module throughout `geosteering_ai/` (never `print()`)
- Logger retrieved via `geosteering_ai/utils/logger.py`
- Structured log format; level controlled via `PipelineConfig`
- CI benchmark logs: `benchmark_ci.log` (written in GitHub Actions, not persisted)
- Performance baseline: `.claude/perf_baseline.json` — JSON with throughput metrics per scenario

**Performance baseline system:**
- File: `.claude/perf_baseline.json`
- Updated by: CI `geosteering-cli benchmark --scenario E --n 200` step
- Hook: `.claude/hooks/check-perf-regression.sh` — WARN-only regression gate (PostToolUse)
- Doc: `docs/PERFORMANCE_BASELINE.md`

## CI/CD & Deployment

**Source Control:**
- GitHub: `github.com/daniel-guitarplayer-8/geosteering-ai`
- Main branch: `main`; development: `develop`; features: `feat/sprint-*`

**CI Pipeline:**
- Service: GitHub Actions
- Config: `.github/workflows/ci.yml`
- Runners: `ubuntu-latest`
- Matrix: Python `["3.13", "3.12"]`
- Steps in order:
  1. `actions/checkout@v4`
  2. `actions/setup-python@v5`
  3. `pip install -e ".[dev,viz,train,hpo]"`
  4. `sudo apt-get install -y xvfb` (headless GUI)
  5. `python -m compileall geosteering_ai/ -q` (compile check)
  6. `geosteering-warmup --verbose` (warm Numba JIT cache, timeout 5 min)
  7. `pytest tests/ -v --tb=short --ignore=tests/test_gpu.py --ignore=tests/test_simulation_manager_gui.py`
  8. `xvfb-run -a pytest tests/test_simulation_manager_gui.py` (GUI tests, `continue-on-error: true`, timeout 10 min)
  9. `python -m geosteering_ai.cli benchmark --scenario E --n 200` (smoke benchmark, `continue-on-error: true`)
  10. `mypy geosteering_ai/ --ignore-missing-imports` (`continue-on-error: true`)

**Docker CI Pipeline:**
- Config: `.github/workflows/docker.yml`
- Trigger: pushes/PRs to `main`/`develop` that touch `Dockerfile.cpu`, `geosteering_ai/api/**`, `pyproject.toml`
- Steps: `docker/setup-buildx-action@v3`, `docker/build-push-action@v6` (GHA cache), smoke `/health` + `/predict` endpoints
- External actions: `actions/checkout@v4`, `docker/setup-buildx-action@v3`, `docker/build-push-action@v6`

**Hosting:**
- Google Colab Pro+ for GPU training and validation
- Docker CPU image (`Dockerfile.cpu`) for REST API deployment
- GitHub as primary source registry (no PyPI publishing currently)

## Google Colab Integration

**Access method:** colab-mcp browser automation + manual notebook upload
- MCP server: `googlecolab/colab-mcp` (via uvx in `.mcp.json`)
- Token refresh: `.claude/hooks/colab-token-refresh.sh` (runs on every Bash tool use)
- GPU runtime: T4 (standard) or A100 (Pro+ compute units)

**Notebooks:**
- `notebooks/colab_templates/train_v240_mp16.ipynb` — mixed-precision training (TF AMP)
- `notebooks/colab_templates/validate_sprint_o0_o1_gpu.ipynb` — Sprint O0/O1 JAX GPU validation
- `notebooks/colab_templates/validate_sprint_o1_gpu_tests.ipynb` — pytest GPU suite in Colab
- `notebooks/colab_templates/validate_jax_gpu_v240.ipynb` — JAX GPU parity validation
- `notebooks/colab_templates/benchmark_tfdata_mp16.ipynb` — TF data pipeline benchmarks
- `notebooks/validate_gpu_colab.ipynb` — 824-test GPU validation (Keras 3.x compat)

**Install pattern in Colab:**
```python
GIT_REPO_URL = "https://github.com/daniel-guitarplayer-8/geosteering-ai.git"
# pip install -e ".[all]" then explicitly reinstall jax[cuda12]
# Sets JAX_PLATFORMS=cuda,cpu, JAX_COMPILATION_CACHE_DIR=/content/jax_cache
```

## Pre-Commit Hooks

Registered in `.pre-commit-config.yaml`:
- `pre-commit/pre-commit-hooks v4.6.0` — trailing whitespace, EOF, JSON/YAML validity, large files, merge conflicts, private key detection
- `astral-sh/ruff-pre-commit v0.5.0` — ruff lint + format (files: `geosteering_ai/`, `tests/`)
- `pre-commit/mirrors-mypy v1.10.0` — type check `geosteering_ai/` (excludes `simulation/tests/`)
- Local: `check-anti-patterns` — blocks 13 anti-pattern TSV entries from `.claude/anti-patterns.txt`

## Claude Code Hooks (`.claude/settings.json`)

**PreToolUse (Edit|Write):**
- `backup-pre-edit.sh` — creates `.backups/` snapshot before every file edit
- `acquire-lock.sh` — multi-agent write lock (`.claude/locks/`)
- `check-anti-patterns.sh` — BLOCK on 13 anti-patterns (physics constants, globals(), print())
- `validate-physics.sh` — enforces errata values (FREQUENCY_HZ, SPACING_METERS, etc.)
- `validate-no-pytorch.sh` — blocks PyTorch imports in production paths
- `protect-critical-files.sh` — guards critical config files from accidental overwrites
- `check-version-references.sh` — enforces ADR-0001 version SSoT rule

**PreToolUse (Bash):**
- `colab-token-refresh.sh` — refreshes Colab MCP browser session token

**PostToolUse (Edit|Write):**
- `release-lock.sh` — releases multi-agent write lock
- `compile-check.sh` — `python -m compileall` on edited file
- `lint-v2-standards.sh` — ruff lint check on edited file
- `autoformat.sh` — ruff format on edited file
- `validate-scientific-refs.sh` — validates scientific references in code (passive hook)
- `run-fortran-parity.sh` — runs Fortran parity tests after simulation edits (timeout 30s)
- `check-ptbr-accentuation.sh` — validates PT-BR accented words in edited files

**Stop (session end):**
- `release-lock.sh --all` — force-release all locks
- `run-pytest.sh` — runs pytest suite (timeout 120s)

**SessionStart:**
- `setup-environment.sh` — configures environment variables on startup
- `reinject-errata.sh` — re-injects physics errata constants on context compact

## Webhooks & Callbacks

**Incoming:**
- None (no webhook endpoints in the API MVP; `/health` and `/predict` are polling endpoints)

**Outgoing:**
- Semantic Scholar API — outgoing GET requests from `tools/consensus-mcp-server/server.py`
- ArXiv API — outgoing GET requests (XML) from `tools/consensus-mcp-server/server.py`
- GitHub API — via `gh` CLI in development workflow (PRs, reviews); not automated

## Environment Configuration

**Required env vars (production API):**
- `GEOSTEERING_MODEL_PATH` — path to saved TF model directory

**Required env vars (simulation / Colab):**
- `NUMBA_CACHE_DIR` — set to `$TMPDIR/geosteering_numba_cache` by `cli/main.py`
- `NUMBA_NUM_THREADS` — set by `SimulationThread.run()` / `_run_numba_parallel_sync`
- `JAX_PLATFORMS` — `cuda,cpu` (Colab) or `cpu` (local dev)
- `JAX_COMPILATION_CACHE_DIR` — XLA cache path
- `JAX_ENABLE_X64` — `True` for double-precision Jacobian tests

**Optional env vars:**
- `S2_API_KEY` — Semantic Scholar API key (consensus MCP server)
- `QT_QPA_PLATFORM=offscreen` — headless Qt in CI (xvfb fallback)
- `OMP_NUM_THREADS` — OpenMP thread count for Fortran `tatu.x`
- `GEOSTEERING_API_DOCS_ENABLED` — `"0"` disables FastAPI /docs in production (Docker)

**Secrets location:**
- No secrets committed to repository
- `S2_API_KEY` passed via shell environment to `.mcp.json` `${S2_API_KEY}` interpolation
- GitHub Actions secrets: `GITHUB_TOKEN` (automatic), no custom secrets declared

---

*Integration audit: 2026-05-24*
