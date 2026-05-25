# Technology Stack

**Analysis Date:** 2026-05-24

## Languages

**Primary:**
- Python 3.13 — all package code (`geosteering_ai/`), tests, CLI, API, MCP servers
- Fortran 2008 (F08) — legacy EM simulator (`Fortran_Gerador/PerfilaAnisoOmp.f08`, `magneticdipoles.f08`, `filtersv2.f08`, `utils.f08`, `parameters.f08`, `RunAnisoOmp.f08`)

**Secondary:**
- Bash — CI scripts, hooks (`/.claude/hooks/*.sh`), Fortran bench scripts (`Fortran_Gerador/bench/run_bench.sh`)

## Runtime

**Environment:**
- CPython 3.13 (pinned in `.python-version`)
- Supported range: `>=3.12,<3.15` (`pyproject.toml:11`)
- FORBIDDEN: Python 3.14+ (no wheels for PyQt6, JAX, SciPy)

**Package Manager:**
- pip with setuptools 68+ and wheel
- Lockfile: not committed (managed via `pip install -e ".[dev,all]"`)
- Local venv: `~/Geosteering_AI_venv`

**Fortran Toolchain:**
- Compiler: `gfortran` (GNU Fortran), OpenMP via `-fopenmp`
- Build system: `Fortran_Gerador/Makefile` (GNU Make)
- Production flags: `-O3 -march=native -ffast-math -funroll-loops -std=f2008`
- Debug flags: `-O0 -fno-fast-math -fsignaling-nans` (for numeric parity validation)
- Output binary: `Fortran_Gerador/tatu.x`
- macOS workaround: ld-classic wrapper for Darwin 25+ (`Makefile:30-36`)

## Frameworks

**Deep Learning (Primary):**
- TensorFlow `>=2.13` — model training, inference, Keras 3.x layers
  - Declared in `[train]` and `[all]` extras (`pyproject.toml:37`)
  - Used across `geosteering_ai/models/`, `geosteering_ai/training/`, `geosteering_ai/losses/`, `geosteering_ai/inference/`
  - GPU: NVIDIA CUDA 12 (via `jax[cuda12]` in Colab; TF uses system CUDA)
  - PyTorch is **banned** in production paths — hook `validate-no-pytorch.sh` blocks it

**Simulation Backends:**
- JAX — GPU/XLA EM field computation (`geosteering_ai/simulation/_jax/`)
  - Files: `kernel.py`, `dipoles_unified.py`, `dipoles_native.py`, `forward_pure.py`, `propagation.py`, `rotation.py`, `geometry_jax.py`, `hankel.py`, `multi_forward.py`
  - Colab GPU: installed as `jax[cuda12]` with explicit reinstall after TF
  - Uses `jax.vmap`, `jax.lax.fori_loop`, `jax.pure_callback` for Numba interop
  - JIT cache: `/content/jax_cache` in Colab, `JAX_COMPILATION_CACHE_DIR` env var
- Numba `>=0.60` — CPU JIT-compiled EM simulator (`geosteering_ai/simulation/_numba/`)
  - Files: `kernel.py`, `dipoles.py`, `propagation.py`, `rotation.py`, `geometry.py`, `hankel.py`
  - Uses `@njit(cache=True, fastmath=True)` for Hankel digital filters
  - Cache: `$TMPDIR/geosteering_numba_cache` (tmpfs, set in `cli/main.py` before heavy imports)

**GUI:**
- PyQt6 (preferred) / PySide6 (fallback) — Simulation Manager desktop app
  - Compat layer: `geosteering_ai/simulation/tests/sm_qt_compat.py`
  - 20+ GUI modules under `geosteering_ai/simulation/tests/`
  - Main app: `simulation_manager.py` (~10,700 lines)
  - Plot backends (opt-in): matplotlib, PyQtGraph, Plotly (WebEngine), Vispy

**Web API:**
- FastAPI `>=0.110` — REST API for DL inference (`geosteering_ai/api/`)
  - Routes: `geosteering_ai/api/routes/health.py`, `geosteering_ai/api/routes/predict.py`
  - Pydantic `>=2.5` — request/response schemas (`geosteering_ai/api/schemas.py`)
  - Uvicorn `>=0.27` — ASGI server (`geosteering_ai/api/app.py`)
  - Entry point: `geosteering-api` → `geosteering_ai.api.cli:main`

**Testing:**
- pytest `>=7.0` — test runner
  - Config: `pyproject.toml` `[tool.pytest.ini_options]`
  - Markers: `slow`, `gui`, `gpu`
- pytest-cov `>=4.0` — coverage
- pytest-qt `>=4.4` — Qt GUI tests (`qtbot` fixture, Sprint v2.33)
- httpx `>=0.27` — FastAPI TestClient for API tests

**Build/Dev:**
- pre-commit `>=3.0` — Quality Mesh Layer 2 (`.pre-commit-config.yaml`)
- ruff `v0.5.0` — linting + formatting (replaces flake8 + isort)
- mypy `>=1.0` — static typing (Python 3.13 mode, `--ignore-missing-imports`)
- watchdog `>=3.0` — file watcher daemon (`tools/file_watcher_daemon.py`)
- psutil `>=5.9` — CPU topology detection + lock manager PID checks

## Key Dependencies

**Critical (required by all installs):**
- `numpy>=1.24` — all numerical operations; array manipulation throughout
- `scipy>=1.10` — signal processing, optimization primitives
- `scikit-learn>=1.2` — scalers (`StandardScaler`, `MinMaxScaler`), train/test split
- `pyyaml>=6.0` — `PipelineConfig.from_yaml()`, all preset configs

**DL pipeline (`[all]` extra):**
- `tensorflow>=2.13` — 48 model architectures, 26 losses, training loop
- `optuna>=3.0` — hyperparameter optimization (`geosteering_ai/training/optuna_hpo.py`)
- `joblib>=1.2` — parallel inference, serialization
- `pandas>=2.0` — data loading, evaluation reports
- `tqdm>=4.65` — progress bars in training loop

**Visualization (`[viz]` extra):**
- `matplotlib>=3.5` — primary plot backend; EDA, holdout, training curves
- PyQtGraph — opt-in SM plot backend (lazy import)
- Plotly — opt-in SM plot backend, requires `PyQt6-WebEngine`
- Vispy — opt-in SM plot backend (rejected for production: `docs/ARCHITECTURE_v2.md`)

**Simulation validation (optional, not in pyproject.toml):**
- `empymod` — reference EM solver for Python (used only in `validation/compare_empymod.py`, lazy import)

**API stack (`[api]` extra):**
- `fastapi>=0.110`
- `pydantic>=2.5`
- `uvicorn[standard]>=0.27`

**MCP servers (`tools/`):**
- `mcp>=1.0.0` — MCP Python SDK (consensus server)
- `mcp>=1.25.0,<2.0` — MCP Python SDK (physics-validator, numba-profiler)
- `httpx>=0.25.0` — HTTP client for Semantic Scholar + ArXiv APIs

## Configuration

**Environment:**
- Python version pinned: `.python-version` → `3.13`
- Simulation env vars:
  - `NUMBA_CACHE_DIR` / `NUMBA_NUM_THREADS` — set in `geosteering_ai/cli/main.py` before heavy imports
  - `JAX_PLATFORMS`, `JAX_COMPILATION_CACHE_DIR`, `JAX_ENABLE_X64` — set in Colab notebooks
  - `QT_QPA_PLATFORM=offscreen` — headless GUI in CI
  - `S2_API_KEY` — optional Semantic Scholar key (`.mcp.json`)
  - `GEOSTEERING_MODEL_PATH` — path to saved TF model for API inference
- Key config files:
  - `pyproject.toml` — project metadata, deps, entry points, pytest/mypy config
  - `configs/baseline.yaml`, `configs/robusto.yaml`, `configs/geosinais_p4.yaml`, `configs/nstage_n2.yaml`, `configs/nstage_n3.yaml`, `configs/realtime_causal.yaml` — `PipelineConfig` presets
  - `.mcp.json` — MCP server definitions (consensus + colab-mcp)
  - `.claude/settings.json` — Claude Code hooks (PreToolUse/PostToolUse/Stop/SessionStart)
  - `.claude/settings.local.json` — local permission allowlist
  - `.pre-commit-config.yaml` — static analysis hooks (ruff, mypy, local)
  - `.claude/perf_baseline.json` — throughput baseline metrics for CI regression gate

**Build:**
- Python package: `setuptools>=68.0` / `wheel` via `pyproject.toml`
- Fortran: `Fortran_Gerador/Makefile` with GNU Make + gfortran
- f2py bridge: `Fortran_Gerador/tatu_f2py_wrapper.f08` → compiled `.so` (Python 3.11 only: `tatu_f2py.cpython-311-darwin.so`)
- Docker (CPU): `Dockerfile.cpu` — python:3.13-slim-bookworm, multi-stage build, `geosteering-ai:cpu` image

## Platform Requirements

**Development:**
- macOS (Apple Silicon M-series or x86_64) with Xcode CLT (for gfortran, ld-classic)
- Python 3.13 venv at `~/Geosteering_AI_venv`
- Numba cache in tmpfs; xvfb for headless GUI tests on Linux

**Production / GPU Compute:**
- Google Colab Pro+ with T4 or A100 GPU
  - `jax[cuda12]` — reinstalled after TF to ensure CUDA wheel
  - CUDA 12.x driver required for JAX GPU backend
  - Git-cloned from `github.com/daniel-guitarplayer-8/geosteering-ai`
- Docker CPU deployment: `Dockerfile.cpu` serving FastAPI on port 8000
- CI: `ubuntu-latest` (GitHub Actions), Python 3.13 primary + 3.12 fallback

---

*Stack analysis: 2026-05-24*
