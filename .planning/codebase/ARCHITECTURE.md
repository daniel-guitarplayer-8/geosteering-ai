<!-- refreshed: 2026-05-24 -->
# Architecture

**Analysis Date:** 2026-05-24

## System Overview

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                         ENTRY POINTS                                         │
├────────────────┬────────────────┬─────────────────┬────────────────────────┤
│  geosteering-  │  geosteering-  │  geosteering-   │  Jupyter Notebooks     │
│  cli simulate/ │  warmup        │  api (FastAPI)   │  (Colab GPU Training)  │
│  benchmark     │                │  :8000           │  notebooks/*.ipynb     │
│  `cli/_main.py`│ `cli/warmup.py`│ `api/app.py`     │                        │
└───────┬────────┴────────┬───────┴────────┬─────────┴────────────────────────┘
        │                 │                │
        ▼                 ▼                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                   PIPELINE CORE  (`geosteering_ai/`)                         │
│                                                                              │
│  config.py ← PipelineConfig dataclass (246 fields, canonical truth)         │
│                                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │  data/   │→ │  noise/  │→ │ models/  │→ │training/ │→ │ inference/   │  │
│  │DataPipe- │  │(on-the-  │  │Registry  │  │TrainingL-│  │InferencePipe-│  │
│  │line      │  │fly only) │  │48 archs  │  │oop /     │  │line /        │  │
│  └──────────┘  └──────────┘  └──────────┘  │NStage    │  │realtime.py   │  │
│                                             └──────────┘  └──────────────┘  │
│                                                                              │
│  ┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌─────────────────────────┐ │
│  │ losses/  │  │ evaluation/  │  │visualiz- │  │     simulation/         │ │
│  │Catalog   │  │ metrics,     │  │ation/    │  │ multi_forward.py        │ │
│  │26 losses │  │ comparison   │  │ plots    │  │ ├── _numba/ (Numba JIT) │ │
│  │LossFactry│  │ dod, reports │  │ EDA      │  │ ├── _jax/ (JAX/GPU)    │ │
│  └──────────┘  └──────────────┘  └──────────┘  │ ├── filters/ (Hankel)  │ │
│                                                  │ └── tests/ (Qt GUI)   │ │
│  utils/ (logger, timer, validation, formatting, system, io)               │ │
│  multi_agent/ (LockManager, conflict_matrix — agent concurrency)          │ │
└──────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                   SIMULATION BACKENDS                                        │
├──────────────┬─────────────────┬───────────────────┬────────────────────────┤
│  Fortran     │  Numba JIT      │  JAX              │  Python pure (dev)     │
│  `tatu.x` /  │  `_numba/`      │  `_jax/`          │  `_jax/forward_pure.py`│
│  f2py        │  kernel.py      │  kernel.py        │                        │
│  `Fortran_   │  Paridade ref.  │  GPU-capable      │                        │
│  Gerador/`   │  <1e-12         │  vmap/pmap        │                        │
└──────────────┴─────────────────┴───────────────────┴────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| `PipelineConfig` | Single source of truth for all 246 pipeline flags; fail-fast validation in `__post_init__` | `geosteering_ai/config.py` |
| `SimulationConfig` | All 50+ simulator parameters (backend, parallelism, tiling, JAX strategy, compensation, tilted antennas) | `geosteering_ai/simulation/config.py` |
| `DataPipeline` | Orchestrates raw → decoupling → split → scaler-fit → val/test offline → train raw; two modes (offline / on-the-fly) | `geosteering_ai/data/pipeline.py` |
| `ModelRegistry` | Registry of 48 architectures across 9 families; validates causal compatibility | `geosteering_ai/models/registry.py` |
| `LossFactory` | Factory for 26 loss functions; priority chain Morales > Prob > base | `geosteering_ai/losses/factory.py` |
| `TrainingLoop` | Keras compile + fit with noise curriculum; wraps callbacks factory | `geosteering_ai/training/loop.py` |
| `NStageTrainer` | N-stage progressive training with per-stage mini-curriculum; mutually exclusive with curriculum | `geosteering_ai/training/nstage.py` |
| `InferencePipeline` | Serializable FV + GS + scalers for deployment; mirrors training chain without noise | `geosteering_ai/inference/pipeline.py` |
| `SlidingWindowInference` / `GeoSteeringSession` | Causal real-time inference with 1 sliding window; uncertainty bands | `geosteering_ai/inference/realtime.py` |
| `CurriculumSchedule` | 3-phase noise ramp: clean (epochs 0→N) → linear ramp → stable at max | `geosteering_ai/noise/curriculum.py` |
| `simulate_multi` | Top-level dispatcher: routes to Numba/JAX/Fortran backend; manages worker pool | `geosteering_ai/simulation/multi_forward.py` |
| `SimulationManager` (Qt GUI) | PyQt6 desktop app: generate synthetic training data via GUI; 10.7k LOC | `geosteering_ai/simulation/tests/simulation_manager.py` |
| `LockManager` | File-based concurrency control for multi-agent Claude sessions | `geosteering_ai/multi_agent/lock_manager.py` |
| FastAPI app | REST API: `/health`, `/predict`; MVP from Sprint v2.39 | `geosteering_ai/api/app.py` |

## Pattern Overview

**Overall:** Layered Modular Package with Registry/Factory pattern

**Key Characteristics:**
- `PipelineConfig` is the sole global state; every function receives it as a parameter — no `globals().get()` calls
- Factory pattern for all extensible components: `ModelRegistry.build()`, `LossFactory.get()`, `build_callbacks()`
- Single data path: `DataPipeline` enforces `raw → noise → FV → GS → scale` — no alternative offline noise paths
- Fail-fast validation: `PipelineConfig.__post_init__()` and `SimulationConfig.__post_init__()` assert physical errata before any computation
- PyTorch is blocked in production paths by `.claude/hooks/validate-no-pytorch.sh`; only permitted in `geosteering_ai/adapters/pytorch_adapter.py` (planned Sprint v2.30)

## Layers

**Data Layer:**
- Purpose: Load Fortran `.dat` binary files, split by geological model (never by sample), compute Feature Views and Geosignals, fit scalers on clean data
- Location: `geosteering_ai/data/`
- Contains: `loading.py`, `splitting.py`, `feature_views.py`, `geosignals.py`, `scaling.py`, `pipeline.py`, `boundaries.py`, `sampling.py`, `surrogate_data.py`, `second_order.py`, `synthetic_generator.py`, `inspection.py`
- Depends on: `config.py`, `utils/`
- Used by: `training/`, `inference/`, `evaluation/`

**Noise Layer:**
- Purpose: On-the-fly noise injection in raw EM domain (A/m) — the only physically valid noise path
- Location: `geosteering_ai/noise/`
- Contains: `functions.py` (34 noise types as TF ops), `curriculum.py` (3-phase schedule), `utils.py` (numpy helper for debug/visualization only)
- Depends on: `config.py`
- Used by: `training/callbacks.py` (injects `noise_level_var` TF variable), `data/pipeline.py` (builds `train_map_fn` closure)

**Models Layer:**
- Purpose: Build Keras models from config; 48 architectures across 9 families
- Location: `geosteering_ai/models/`
- Contains: `registry.py`, `blocks.py` (23 reusable Keras blocks), `cnn.py`, `rnn.py`, `hybrid.py`, `tcn.py`, `transformer.py`, `unet.py`, `decomposition.py`, `advanced.py`, `surrogate.py`, `geosteering.py`
- Depends on: `config.py`; TensorFlow/Keras is the mandatory framework
- Used by: `training/`, `inference/`

**Losses Layer:**
- Purpose: 26 loss functions with physics-aware geophysical losses
- Location: `geosteering_ai/losses/`
- Contains: `catalog.py` (26 individual losses), `factory.py` (LossFactory), `geophysical.py` (geo-aware #14-#17), `pinns.py` (8 PINN scenarios, 1975 LOC)
- Depends on: `config.py`
- Used by: `training/loop.py`

**Training Layer:**
- Purpose: Keras compile + fit orchestration; callbacks, N-Stage, HPO
- Location: `geosteering_ai/training/`
- Contains: `loop.py`, `callbacks.py` (3538 LOC — 26 callbacks, builds from factory), `nstage.py`, `optuna_hpo.py`, `metrics.py`, `adaptation.py`
- Depends on: `config.py`, `data/`, `models/`, `losses/`, `noise/`
- Used by: notebooks, CLI

**Inference Layer:**
- Purpose: Deployment-ready inference; serializable pipeline; real-time geosteering
- Location: `geosteering_ai/inference/`
- Contains: `pipeline.py`, `realtime.py`, `export.py`, `uncertainty.py` (MC Dropout, Ensemble, INN)
- Depends on: `config.py`, `data/` (for FV/GS/scalers), `models/`
- Used by: `evaluation/`, `visualization/`, FastAPI `api/`, notebooks

**Evaluation Layer:**
- Purpose: Metrics, model comparison, DOD Picasso analysis, reports
- Location: `geosteering_ai/evaluation/`
- Contains: `metrics.py`, `comparison.py`, `advanced.py`, `dod.py`, `geosteering_metrics.py`, `geosteering_report.py`, `manifest.py`, `predict.py`, `realtime_comparison.py`, `report.py`, `config_report.py`
- Depends on: `inference/`, `config.py`

**Visualization Layer:**
- Purpose: Training curves, holdout overlays, EDA, Picasso DOD plots, real-time trajectory panels
- Location: `geosteering_ai/visualization/`
- Contains: `eda.py` (1323 LOC), `holdout.py`, `training.py`, `picasso.py`, `geosteering.py`, `realtime.py`, `error_maps.py`, `uncertainty.py`, `export.py`, `optuna_viz.py` (1231 LOC)
- Depends on: `evaluation/`, `config.py`

**Simulation Layer:**
- Purpose: Forward EM simulation for synthetic training data generation; 4 backends; PyQt6 GUI
- Location: `geosteering_ai/simulation/`
- Contains: `multi_forward.py` (top-level dispatcher, 1344 LOC), `config.py` (SimulationConfig, 1247 LOC), `forward.py` (Numba JIT entry), `_workers.py` (ephemeral ProcessPoolExecutor), `_numba/` (Numba JIT backend), `_jax/` (JAX/GPU backend), `filters/` (Hankel filter coefficients: Werthmuller 201pt, Kong 61pt, Anderson 801pt), `io/` (binary .dat I/O, model.in I/O), `postprocess/` (compensation, tilted antennas), `validation/` (Fortran parity, analytical checks), `visualization/` (benchmark and physics plots), `benchmarks/`, `tests/` (PyQt6 GUI + sm_workers.py)
- Key parcel: `tests/simulation_manager.py` (10759 LOC, PyQt6 GUI — the primary user interface for data generation)
- Depends on: Fortran `tatu.x` (via f2py or subprocess), Numba, JAX, `config.py`

**Utilities Layer:**
- Purpose: Shared cross-cutting concerns; imported by all modules
- Location: `geosteering_ai/utils/`
- Contains: `logger.py` (setup_logger, ColoredFormatter — use this, NEVER print()), `timer.py`, `validation.py`, `formatting.py`, `system.py` (is_colab, has_gpu), `io.py`

**Multi-Agent Layer:**
- Purpose: File-based locking for concurrent Claude agent sessions
- Location: `geosteering_ai/multi_agent/`
- Contains: `lock_manager.py` (LockManager, AgentConflictError, LockInfo), `conflict_matrix.py`

## Data Flow

### Primary Training Path (On-the-Fly Mode)

1. `.dat` binary load + `.out` metadata parse (`geosteering_ai/data/loading.py`)
2. EM decoupling: subtract ACp/ACx constants (`geosteering_ai/data/pipeline.py`)
3. Split by geological model — train (70%) / val (15%) / test (15%) with zero model overlap (`geosteering_ai/data/splitting.py`)
4. Fit scalers on clean FV+GS (temporary; train data stays RAW) (`geosteering_ai/data/scaling.py`)
5. Val/test transformed offline: FV → GS → scale
6. Per batch per epoch: `train_map_fn` closure applies noise(A/m) → FV_tf(noisy) → GS_tf(noisy) → scale_tf (`geosteering_ai/data/pipeline.py:build_train_map_fn`)
7. Keras model forward pass → loss computation (`geosteering_ai/training/loop.py`)
8. Callbacks: curriculum noise scheduling, early stopping, checkpointing (`geosteering_ai/training/callbacks.py`)

### Inference Path

1. Raw EM input → `InferencePipeline`: FV → GS → scale (NO noise) (`geosteering_ai/inference/pipeline.py`)
2. `model.predict()` → inverse target scaling → rho_h, rho_v (Ω·m)
3. Optional uncertainty quantification: MC Dropout or Ensemble (`geosteering_ai/inference/uncertainty.py`)
4. Real-time: `SlidingWindowInference` with causal-padded model, 1-window sliding (`geosteering_ai/inference/realtime.py`)

### Simulation Path (Data Generation)

1. User configures geological model via PyQt6 GUI or CLI (`simulate_multi` call)
2. `SimulationConfig` validated → dispatched to backend in `simulate_multi` (`geosteering_ai/simulation/multi_forward.py:661`)
3. Numba path: `_workers.py` spawns ephemeral `ProcessPoolExecutor`; each worker calls `forward.py::simulate()` → JIT-compiled `_numba/kernel.py`
4. JAX path: `_jax/multi_forward.py` uses vmap over (iTR, iAng); `_jax/kernel.py` with XLA compilation
5. Fortran path: subprocess calls `tatu.x` or f2py wrapper `tatu_f2py`
6. Output: `MultiSimulationResult` tensor → written as 22-column binary `.dat` + `.out` metadata

**State Management:**
- No global mutable state in production pipeline; `PipelineConfig` and `SimulationConfig` are passed explicitly
- `noise_level_var` is a `tf.Variable` updated each epoch by curriculum callback, captured in `train_map_fn` closure
- Numba JIT cache: `$TMPDIR/geosteering_numba_cache` (set by `cli/_main.py` before imports)
- Qt GUI state: held by `SimulationManager` Qt object hierarchy (`simulation/tests/simulation_manager.py`)

## Key Abstractions

**PipelineConfig:**
- Purpose: Single source of truth for all DL pipeline parameters; replaces 574 `globals().get()` calls
- Examples: `geosteering_ai/config.py` (1615 LOC, 246 fields)
- Pattern: Dataclass with `__post_init__` validation, YAML serialization, class-method presets (`baseline()`, `robusto()`, `nstage(n)`, `geosinais_p4()`, `realtime()`)

**SimulationConfig:**
- Purpose: All simulator parameters: backend selection, parallelism, tiling strategy, JAX strategy, multi-frequency/TR/dip sweeps, Hankel filter, compensation, tilted antennas
- Examples: `geosteering_ai/simulation/config.py` (1247 LOC, ~50 fields)
- Pattern: Dataclass; `backend` field selects among `"fortran_f2py"`, `"numba"`, `"jax"`, `"python"`

**ModelRegistry:**
- Purpose: Named registry of 48 architectures; decouples architecture selection from instantiation
- Examples: `geosteering_ai/models/registry.py`
- Pattern: Registry — call `ModelRegistry().build(config)` — never construct model classes directly

**LossFactory:**
- Purpose: Maps `config.loss_type` string to a Keras-compatible loss function
- Examples: `geosteering_ai/losses/factory.py`
- Pattern: Factory with `@LossFactory.register("name")` decorator; `LossFactory.get(config)` at training time

**DataPipeline:**
- Purpose: Encapsulates the entire data preparation chain; two automatic modes triggered by `config.needs_onthefly_fv_gs`
- Examples: `geosteering_ai/data/pipeline.py` (1120 LOC)
- Pattern: Orchestrator — call `pipeline.prepare(path)` then `pipeline.build_train_map_fn(noise_var)`

## Entry Points

**CLI (geosteering-cli):**
- Location: `geosteering_ai/cli/_main.py` (entry: `geosteering_ai.cli:main`)
- Triggers: `pip install -e .` installs `geosteering-cli` binary
- Responsibilities: `simulate` subcommand (multi-dimensional sweeps), `benchmark` subcommand (scenarios A-H), `version` subcommand

**Warmup CLI (geosteering-warmup):**
- Location: `geosteering_ai/cli/warmup.py`
- Triggers: Called in CI before pytest to pre-compile Numba JIT cache
- Responsibilities: Warm Numba tier-2 JIT synchronously; sets `NUMBA_CACHE_DIR`

**REST API (geosteering-api):**
- Location: `geosteering_ai/api/app.py`, routes in `geosteering_ai/api/routes/`
- Triggers: `geosteering-api` entry point starts uvicorn on :8000
- Responsibilities: `/health` endpoint, `/predict` endpoint (inference via InferencePipeline)

**PyQt6 GUI (Simulation Manager):**
- Location: `geosteering_ai/simulation/tests/simulation_manager.py` (10759 LOC)
- Triggers: Launched as standalone desktop app
- Responsibilities: Geological model configuration, multi-TR/freq/dip simulation, real-time progress, LRU plot cache, export `.dat`/`.out`

**Notebooks (Colab Orchestrators):**
- Location: `notebooks/` and `notebooks/colab_templates/`
- Triggers: Google Colab Pro+ GPU session; `pip install git+...@vX.Y.Z`
- Responsibilities: Training, evaluation, EDA, benchmarking — thin orchestrators calling the pip package

**MCP Servers (Claude agent tools):**
- Location: `tools/physics-validator-mcp/server.py`, `tools/numba-profiler-mcp/server.py`, `tools/consensus-mcp-server/server.py`
- Triggers: Registered in `.claude/settings.json` for Claude Code sessions
- Responsibilities: Physics validation (6 tools), Numba profiling (6 tools), scientific paper search

## Architectural Constraints

- **Threading:** Numba simulation uses ephemeral `ProcessPoolExecutor` (spawn method); `NUMBA_NUM_THREADS` set in parent process (`SimulationThread.run`) so workers inherit it — setting it inside workers causes `RuntimeError` in spawn mode
- **Global state:** No module-level mutable singletons in production paths; `noise_level_var` (`tf.Variable`) is the only shared mutable state in training and is explicitly captured in closures
- **Circular imports:** `simulation/config.py` imports from `simulation/filters/loader.py` and `utils/`; `multi_forward.py` imports `config.py` and `_workers.py` — no circular dependency as each layer imports only from lower layers
- **PyTorch prohibition:** `validate-no-pytorch.sh` hook blocks any `import torch` in `geosteering_ai/{models,losses,training,inference,evaluation,data,simulation,visualization,utils}/`
- **Physical errata (immutable):** `FREQUENCY_HZ=20000.0`, `SPACING_METERS=1.0`, `SEQUENCE_LENGTH=600`, `TARGET_SCALING="log10"`, `INPUT_FEATURES=[1,4,5,20,21]`, `OUTPUT_TARGETS=[2,3]`, `eps_tf=1e-12` — enforced in `PipelineConfig.__post_init__()`
- **Fortran parity:** Numba and JAX backends must maintain <1e-12 absolute difference with Fortran reference — enforced by 10 canonical model tests in `tests/test_simulation_compare_fortran.py`

## Anti-Patterns

### Using globals().get() for configuration

**What happens:** Code reads flags from `globals()` or module-level variables instead of receiving a `PipelineConfig` parameter.
**Why it's wrong:** Creates 574+ divergent defaults (the bug that motivated v2.0); untestable; no static analysis.
**Do this instead:** Every function and class takes `config: PipelineConfig` as a parameter. See `geosteering_ai/training/loop.py` for the correct pattern.

### Setting NUMBA_NUM_THREADS inside a worker process

**What happens:** Worker function calls `os.environ["NUMBA_NUM_THREADS"] = ...` inside `run_numba_chunk`.
**Why it's wrong:** In spawn mode, Numba is imported during worker bootstrap before the function runs — `RuntimeError: Cannot set NUMBA_NUM_THREADS`. Fixed in Sprint v2.29.1.
**Do this instead:** Set `os.environ["NUMBA_NUM_THREADS"]` in `SimulationThread.run` (the parent) with `try/finally` to restore; workers inherit via spawn. See `geosteering_ai/simulation/tests/sm_workers.py`.

### Noise in the normalized/scaled domain

**What happens:** Noise is applied after scaling (or offline by copying data K times).
**Why it's wrong:** Violates LWD physics — Feature Views and Geosignals computed from pre-noise data produce values that diverge from instrument measurements (GS_clean=0.35 dB ± 0.08 vs GS_noisy_EM=2.26 dB).
**Do this instead:** Apply noise ONLY in raw EM domain (A/m) in `train_map_fn` Step 1, before FV and GS computation. See `geosteering_ai/data/pipeline.py:build_train_map_fn`.

### Split by sample instead of by geological model

**What happens:** `train_test_split` on individual measurement samples, mixing models across splits.
**Why it's wrong:** Data leakage — same geological model appears in train and test; inflated metrics.
**Do this instead:** Always use `split_by_geological_model()` from `geosteering_ai/data/splitting.py` which partitions on model IDs.

## Error Handling

**Strategy:** Fail-fast at configuration time; structured logging everywhere else.

**Patterns:**
- `PipelineConfig.__post_init__()` raises `AssertionError` immediately on invalid values — never defer validation to runtime
- `SimulationConfig.__post_init__()` same pattern for simulator parameters
- Worker-level exceptions in `ProcessPoolExecutor` are caught and re-raised with context in `_workers.py:run_batch()`
- `try/except (ImportError, TypeError, ValueError)` granularity in topology detection (`_workers.py:detect_cpu_topology`) — never bare `except`

## Cross-Cutting Concerns

**Logging:** `setup_logger()` from `geosteering_ai/utils/logger.py`; structured with colored formatter; NEVER use `print()` anywhere in production code
**Validation:** Centralized in `PipelineConfig.__post_init__()` and `SimulationConfig.__post_init__()`; `ValidationTracker` in `utils/validation.py` for runtime checks
**Authentication:** Not applicable (local dev + Colab + API is internal research tool)

---

*Architecture analysis: 2026-05-24*
