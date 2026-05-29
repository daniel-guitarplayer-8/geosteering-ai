# Testing Patterns

**Analysis Date:** 2026-05-24

## Test Framework

**Runner:** pytest ≥7.0
**Config:** `pyproject.toml` seção `[tool.pytest.ini_options]`

**Assertion Library:** pytest built-in + NumPy (`np.allclose`, `np.array_equal`)

**Key pytest settings:**
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
pythonpath = ["."]
addopts = "-v --tb=short"
```

**Run Commands:**
```bash
# Suite completa (CPU-only, ignora GPU e GUI)
pytest tests/ -v --tb=short --ignore=tests/test_gpu.py --ignore=tests/test_simulation_manager_gui.py

# Subset rápido (fast path — sem Fortran parity, sem benchmarks)
pytest tests/ -v --tb=short -m "not slow and not fortran and not gui and not gpu"

# Apenas testes GUI (requer xvfb no Linux)
xvfb-run -a pytest tests/test_simulation_manager_gui.py -v --tb=short

# Apenas testes de paridade Fortran
pytest tests/test_simulation_compare_fortran.py -k "fortran_python_numba" --tb=short

# Apenas testes GPU (skip automático sem GPU)
pytest tests/ -m gpu -v --tb=short

# Com cobertura
pytest tests/ --cov=geosteering_ai --cov-report=html

# Watch mode (instalar pytest-watch separado)
ptw tests/ -- -v --tb=short
```

## Test File Organization

**Location:** `tests/` no root do projeto (diretório único — NÃO co-localizado com o código)

**Naming:** `test_<module_or_feature>.py`

**Structure:**
```
tests/
├── __init__.py                          # Necessário para imports relativos no Colab
├── conftest.py                          # Root conftest — QT_API + NUMBA_CACHE_DIR + GPU skip
├── conftest_qt.py                       # Fixtures pytest-qt (qt_binding, mock_simulation_thread, mock_sim_request)
├── _fortran_helpers.py                  # Helpers privados para testes Fortran (prefixo _)
│
│── # Pipeline DL
├── test_config.py                       # PipelineConfig errata, mutual exclusivity, YAML roundtrip
├── test_data_pipeline.py                # shapes, split P1, scaler fit on clean
├── test_noise.py                        # curriculum 3-phase, noise preserva zobs
├── test_models.py                       # forward pass para cada arquitetura (48 total)
├── test_losses.py                       # forward pass + gradients para 26 losses
├── test_training.py                     # TrainingLoop, callbacks (17+)
├── test_inference.py                    # InferencePipeline, realtime, export
├── test_evaluation.py                   # métricas, comparação
├── test_surrogate.py                    # SurrogateNet TCN + ModernTCN
├── test_pinns.py                        # 8 cenários PINN
├── test_boundaries.py                   # DTB (P5)
├── test_visualization.py                # plots, EDA
├── test_utils.py                        # logger, timer, validation
│
│── # CLI
├── test_cli_mvp.py                      # geosteering-cli simulate/benchmark/version
├── test_cli_warmup_entry_point.py       # geosteering-warmup --verbose
├── test_cli_warmup_background.py        # warmup em background
│
│── # API REST
├── test_api_health.py                   # /health endpoint
├── test_api_predict.py                  # /predict endpoint
├── test_api_schemas.py                  # Pydantic v2 schemas
│
│── # Simulação Numba
├── test_simulation_config.py
├── test_simulation_forward.py
├── test_simulation_multi.py
├── test_simulation_workers.py           # pool efêmero (v2.29 back-to-basics)
├── test_simulation_workers_ephemeral.py
├── test_simulation_workers_threading.py
├── test_simulation_numba_kernel.py
├── test_simulation_numba_geometry.py
├── test_simulation_numba_propagation.py
├── test_simulation_numba_dipoles.py
├── test_simulation_numba_rotation.py
├── test_simulation_numba_hankel.py
├── test_simulation_numba_specializations.py  # 1 .nbc por função + NUMBA_CACHE_DIR
├── test_simulation_lru_cache.py
├── test_simulation_parameters_seed.py
├── test_simulation_random_seed.py
│
│── # Simulação JAX
├── test_simulation_jax_foundation.py
├── test_simulation_jax_fortran_parity.py  # <1e-12 JAX vs Fortran
├── test_simulation_jax_multi.py
├── test_simulation_jax_batched_api.py
├── test_simulation_jax_perf_baseline.py
├── test_simulation_jax_performance.py
├── test_simulation_jax_propagation.py
├── test_simulation_jax_dipoles_native.py
├── test_simulation_jax_native_e2e.py
├── test_simulation_jacfwd_native.py
├── test_simulation_jacobian.py
│
│── # Paridade Fortran (gate sagrado)
├── test_simulation_compare_fortran.py   # 7 modelos canônicos — max_abs <1e-6
│
│── # Benchmarks e performance
├── test_simulation_benchmark.py
├── test_simulation_analytical_validation.py
├── test_perf_baseline_h.py              # Cenário H stress-test 8×8×8=512 combos
│
│── # GUI (pytest-qt)
├── test_simulation_manager_gui.py       # 16 testes Qt — rodados separados com xvfb
│
│── # Quality Mesh
├── test_known_bugs.py                   # 11 testes: KB-001/002/013/018/019 — XFAIL/SKIP para bugs conhecidos
├── test_hooks_i25.py                    # Sprint v2.24 — hooks I2.5
├── test_multi_agent_locks.py            # LockManager multi-agente
├── test_worktree_config.py
└── test_sprint_v224.py
```

## Pytest Markers

Definidos em `pyproject.toml` seção `[tool.pytest.ini_options]`:

| Marker | Descrição | Comportamento |
|--------|-----------|---------------|
| `slow` | Testes lentos (>30s) | Skip em CI rápido via `-m 'not slow'` |
| `gui` | Testes pytest-qt (requerem Qt6) | Rodados separados com `xvfb-run -a` |
| `gpu` | Requerem GPU física (TF/JAX CUDA) | Skip automático via `conftest.py` sem GPU |
| `fortran` | Dependem de `tatu.x` compilado | Skip automático se binário ausente |

**Uso em testes:**
```python
@pytest.mark.slow
@pytest.mark.parametrize("model_name", CANONICAL_MODELS)
def test_compare_fortran_python_numba(model_name: str, tmp_path: Path) -> None:
    ...

@pytest.mark.gui
def test_simulation_thread_signals(qtbot, mock_simulation_thread):
    ...

@pytest.mark.gpu
def test_jax_gpu_throughput(self):
    ...
```

## Root `conftest.py` (`tests/conftest.py`)

Executado antes de qualquer plugin pytest. Três responsabilidades:

**1. NUMBA_CACHE_DIR em tmpfs (v2.36 D3):**
```python
if "NUMBA_CACHE_DIR" not in os.environ:
    _cache_dir = os.path.join(tempfile.gettempdir(), "geosteering_numba_cache_test")
    os.makedirs(_cache_dir, mode=0o700, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = _cache_dir
```
Deve rodar antes de qualquer `import numba`. Idempotente — preserva override do usuário (CI, debug).

**2. QT_API alignment (Sprint v2.33):**
```python
if "QT_API" not in os.environ:
    try:
        import PyQt6.QtCore  # noqa: F401
        os.environ["QT_API"] = "pyqt6"
    except ImportError:
        ...
```
Previne cross-binding TypeError entre pytest-qt (PySide6) e sm_qt_compat (PyQt6).

**3. GPU skip automático via hook `pytest_collection_modifyitems`:**
```python
def pytest_collection_modifyitems(config, items):
    if _detect_gpu_available():
        return
    skip_gpu = pytest.mark.skip(reason="GPU física não disponível (Sprint v2.40)")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
```
Detecta GPU via `tf.config.list_physical_devices("GPU")` e `jax.devices()` (excluindo JAX-Metal macOS — não é CUDA Tensor Cores).

## Qt Fixtures (`tests/conftest_qt.py`)

Fixtures específicas da GUI, carregadas pelos testes via `pytest_plugins`:

| Fixture | Scope | Descrição |
|---------|-------|-----------|
| `qt_binding` | session | Retorna `"PyQt6"` ou `"PySide6"` em uso; raise RuntimeError se ausente |
| `mock_sim_request` | function | `SimRequest` mínimo (10 posições, 20 kHz, 1 TR, dip=0°) sem rodar simulação real |
| `mock_simulation_thread` | function | `MagicMock` com sinais Qt-like rastreáveis (`progress_update`, `finished_all`, `error`, `paused`, `resumed`, `cancelled`, `log`) |

Uso típico:
```python
def test_thread_accepts_request(qtbot, mock_sim_request, mock_simulation_thread):
    mock_simulation_thread.progress_update.emit(50, 100, 1234.5)
    assert mock_simulation_thread.progress_update.emit.called
```

## Fortran Parity Testing — Gate Sagrado (<1e-12)

### Estratégia

A paridade Fortran é um requisito inviolável: `max_abs_error < 1e-6` (relativo ao Fortran), com goal interno `<1e-12` para o caminho JIT Numba compilado.

**7 modelos canônicos** testados em `tests/test_simulation_compare_fortran.py`:
- `oklahoma_3`, `oklahoma_5`, `devine_8`, `oklahoma_15`, `oklahoma_28`, `hou_7`, `viking_graben_10`

**Guard CI-safe:**
```python
from tests._fortran_helpers import _tatu_runnable
FORTRAN_AVAILABLE = _tatu_runnable(DEFAULT_FORTRAN_EXEC)
fortran_required = pytest.mark.skipif(
    not FORTRAN_AVAILABLE,
    reason="tatu.x não executável..."
)
```
Skip automático se `Fortran_Gerador/tatu.x` não existir (CI Linux pode não ter o binário compilado em macOS).

**Padrão de teste:**
```python
@fortran_required
@pytest.mark.parametrize("model_name", CANONICAL_MODELS)
def test_compare_fortran_python_numba(model_name: str, tmp_path: Path) -> None:
    """Paridade Numba vs Fortran: max_abs < 1e-6 nos 7 modelos canônicos."""
    results = compare_fortran_python(
        canonical_model_name=model_name,
        backends=["numba"],
        n_positions=80,
        workdir=tmp_path,
    )
    r = results[0]
    assert r.passed, f"[{model_name}/numba] max_abs={r.max_abs_error:.2e}"
```

### Hook de Parity (Pre-commit e PostToolUse)

**Modo QUICK** (PostToolUse — arquivo `*_numba/*` ou `forward.py` editado):
- Roda apenas `oklahoma_3` (~8s)
- Trigger: edição de arquivos no caminho crítico Numba JIT

**Modo FULL** (pre-commit):
- Roda 7 modelos canônicos (~146s)
- Trigger: `FORTRAN_PARITY_MODE=full bash .claude/hooks/run-fortran-parity.sh`

**Bypass emergencial:**
```bash
CLAUDE_BYPASS_FORTRAN_PARITY=1 ...
```

**Anti-pattern KB-013** (BLOCK via `check-anti-patterns.sh`): `@njit(parallel=True` em `*_numba/kernel.py` — causa nested prange overhead que destrói throughput.

## Performance Baseline e Anti-Regressão

### Baseline

Arquivo: `.claude/perf_baseline.json`
Documentação: `docs/PERFORMANCE_BASELINE.md`

Cenários registrados:
| Chave | Cenário | Throughput Baseline | Tipo |
|-------|---------|-------------------|------|
| `E_n200` | E, n=200, cold | 69.336 mod/h | cold |
| `E_n200_warm` | E, n=200, warm cache | 105.423 mod/h | warm |
| `H_n2_stress` | H 8×8×8=512 combos, n=2 | 772 mod/h | warm |
| `jax_gpu_t4.*_hot` | JAX GPU Tesla T4 | ver `.claude/perf_baseline.json` | hot |
| `jax_gpu_a100.*_hot` | JAX GPU A100 40 GB (OFICIAL) | ver `.claude/perf_baseline.json` | hot |

Threshold: **90% do baseline** — WARN (não bloqueia).

### Gate de Anti-Regressão

Hook `check-perf-regression.sh` (WARN-only, não bloqueia commit):
```bash
bash .claude/hooks/check-perf-regression.sh
# ou com override:
SCENARIO=E N_MODELS=200 THRESHOLD_PCT=90 bash .claude/hooks/check-perf-regression.sh
```

Atualizar baseline após otimização intencional (>+5%): editar `.claude/perf_baseline.json` manualmente.

**Testes de performance na suite:**
- `tests/test_simulation_benchmark.py` — benchmarks CPU Numba
- `tests/test_simulation_jax_perf_baseline.py` — baselines JAX
- `tests/test_simulation_jax_performance.py` — throughput JAX
- `tests/test_perf_baseline_h.py` — Cenário H stress-test (Sprint v2.35)

## GUI Testing — pytest-qt + xvfb

**Framework:** `pytest-qt>=4.4` (adicionado em Sprint v2.33)
**Plugin:** `pytestmark = pytest.mark.gui` no arquivo de testes GUI

**Arquivo principal:** `tests/test_simulation_manager_gui.py` (16 testes, ~10.7k linhas Qt cobertas)

**Run local (macOS):**
```bash
pytest tests/test_simulation_manager_gui.py -v --tb=short
```

**Run CI (Linux headless):**
```bash
xvfb-run -a pytest tests/test_simulation_manager_gui.py -v --tb=short
```
Env: `QT_QPA_PLATFORM=offscreen` (setado automaticamente em `conftest.py` se `DISPLAY` ausente)

**O que os testes GUI cobrem:**
- Compatibilidade Qt (PyQt6/PySide6 via `sm_qt_compat`)
- `SimRequest` dataclass
- `SimulationThread` sinais (`progress_update`, `finished_all`, `error`, `paused`, `resumed`, `cancelled`)
- Pause/resume cooperativo
- Widgets leves

## GPU Testing (Auto-Skip sem GPU)

**Marker:** `@pytest.mark.gpu`

**Skip automático:** implementado em `tests/conftest.py` via `pytest_collection_modifyitems` — testes marcados como SKIPPED (não omitidos da coleção — mantém visibilidade do escopo).

**Detecção GPU:**
```python
def _detect_gpu_available() -> bool:
    try:
        import tensorflow as tf
        if tf.config.list_physical_devices("GPU"):
            return True
    except Exception:
        pass
    try:
        import jax
        for d in jax.devices():
            if d.platform == "gpu" and "metal" not in str(d).lower():
                return True
    except (ImportError, AttributeError, RuntimeError):
        pass
    return False
```
JAX-Metal (macOS) **excluído** — não é CUDA Tensor Cores (testes mp16 assumem CUDA).

**Testes GPU:**
- `tests/test_simulation_jax_perf_baseline.py` — baseline GPU T4/A100
- `tests/test_simulation_jax_performance.py` — throughput GPU
- `tests/test_simulation_jacfwd_native.py` — Jacobian JAX GPU

**Validação em Colab:** notebooks em `notebooks/` (ex: `validate_sprint_o0_o1_gpu.ipynb`, `validate_sprint_o1_gpu_tests.ipynb`)

## Numba JIT Cache Tests

**Arquivo:** `tests/test_simulation_numba_specializations.py` (5 guard tests, Sprint v2.31)

Garante:
1. Exatamente 1 especialização por função Numba crítica
2. Exatamente 1 arquivo `.nbc` em disco por função
3. `NUMBA_CACHE_DIR` configurado antes do import Numba

## Test Structure Patterns

**Suite organization:**
```python
class TestErrata:
    """Valores físicos críticos — Errata v4.4.5 + v5.0.15."""

    def test_frequency_hz_valid_range(self):
        """frequency_hz aceita range [100, 1e6] Hz."""
        config = PipelineConfig(frequency_hz=20000.0)
        assert config.frequency_hz == 20000.0

    def test_frequency_hz_rejects_out_of_range(self):
        """frequency_hz rejeita valores fora de [100, 1e6]."""
        with pytest.raises(AssertionError, match="frequency_hz"):
            PipelineConfig(frequency_hz=50.0)
```

**Fixtures padrão:**
```python
@pytest.fixture
def default_config() -> PipelineConfig:
    """PipelineConfig com defaults válidos para testes."""
    return PipelineConfig()

@pytest.fixture
def tmp_workdir(tmp_path) -> Path:
    """Diretório temporário isolado por teste."""
    return tmp_path / "workdir"
```

**Async testing (não é padrão aqui):** a suite não usa `pytest-asyncio` — o simulador e pipeline são síncronos no caminho crítico.

**Error testing pattern:**
```python
def test_rejects_invalid_value(self):
    with pytest.raises(AssertionError, match="spacing_meters"):
        PipelineConfig(spacing_meters=0.01)
```

**Parametrize pattern:**
```python
@pytest.mark.parametrize("model_name", CANONICAL_MODELS)
def test_model_forward_pass(model_name: str) -> None:
    config = PipelineConfig(model_type=model_name)
    model = ModelRegistry().build(config)
    x = tf.random.normal([2, 600, 5])
    y = model(x, training=False)
    assert y.shape == (2, 600, 2)
```

## Mocking

**Framework:** `unittest.mock.MagicMock` (stdlib — sem `pytest-mock` adicional)

**Pattern para GUI:**
```python
from unittest.mock import MagicMock

@pytest.fixture
def mock_simulation_thread() -> MagicMock:
    thread = MagicMock()
    for signal_name in ("progress_update", "finished_all", "error", "paused", ...):
        signal_mock = MagicMock()
        signal_mock.emit = MagicMock()
        signal_mock.connect = MagicMock()
        signal_mock.disconnect = MagicMock()
        setattr(thread, signal_name, signal_mock)
    return thread
```

**O que mockar:**
- `SimulationThread` em testes GUI (evita rodar simulação real)
- Binário `tatu.x` quando não compilado (via `skipif` — não mock)
- GPU (via skip automático em `conftest.py` — não mock)

**O que NÃO mockar:**
- `PipelineConfig` — testar validação real
- Funções Numba JIT — paridade deve ser com código real
- `compare_fortran_python` — a função é o próprio gate

## Fixtures e Helpers

**Test data (modelos canônicos):**
```python
from geosteering_ai.simulation.validation.canonical_models import get_canonical_model
m = get_canonical_model("oklahoma_3")
# m.rho_h, m.rho_v, m.esp — parâmetros do modelo geológico
```

**Fortran helpers (`tests/_fortran_helpers.py`):**
```python
from tests._fortran_helpers import (
    write_model_in_multi, compute_n_pos_for_dip, compute_pz_for_dip, _tatu_runnable
)
n_pos = write_model_in_multi(workdir=tmp_path, model_name="oklahoma_3", ...)
pz = compute_pz_for_dip(30.0, p_med=0.2)
```

**Localização de fixtures compartilhadas:**
- Root: `tests/conftest.py` — NUMBA_CACHE_DIR, QT_API, GPU skip
- Qt: `tests/conftest_qt.py` — `qt_binding`, `mock_simulation_thread`, `mock_sim_request`
- Private: `tests/_fortran_helpers.py` — helpers Fortran (prefixo `_`, não exportado)

## Coverage

**Requirements:** Não há target % enforçado automaticamente (não há `--cov-fail-under`).

**View Coverage:**
```bash
pytest tests/ --cov=geosteering_ai --cov-report=html
open htmlcov/index.html
```

**Areas with coverage:**
- `geosteering_ai/config.py` — alta (errata, presets, YAML)
- `geosteering_ai/simulation/` — alta (1665+ testes total)
- `geosteering_ai/models/`, `losses/`, `noise/` — cobertos por testes de forward pass + gradients
- `geosteering_ai/simulation/tests/sm_workers.py` — 16 testes GUI (Sprint v2.33 cobriu ~10.7k linhas Qt antes sem cobertura)

**Known coverage gap:** GPU paths (`@pytest.mark.gpu`) — validados em Google Colab Pro+ T4/A100 (não no CI CPU).

## CI Workflow (`.github/workflows/ci.yml`)

**Trigger:** push/PR em `main` e `develop`
**Matrix:** Python 3.13 (primária) e 3.12 (fallback)
**Runner:** ubuntu-latest

**Passos em ordem:**
1. Checkout + Setup Python
2. `pip install -e ".[dev,viz,train,hpo]"`
3. `sudo apt-get install xvfb` (headless GUI)
4. `python -m compileall geosteering_ai/ -q` (compile check)
5. `geosteering-warmup --verbose` (Sprint v2.32 — isola cold-start JIT, timeout 5min)
6. `pytest tests/ -v --tb=short --ignore=tests/test_gpu.py --ignore=tests/test_simulation_manager_gui.py`
7. `xvfb-run -a pytest tests/test_simulation_manager_gui.py -v --tb=short` (GUI, `continue-on-error: true`)
8. `python -m geosteering_ai.cli benchmark --scenario E --n 200` (observabilidade, `continue-on-error: true`)
9. `mypy geosteering_ai/ --ignore-missing-imports` (`continue-on-error: true`)

**GPU tests:** não rodam em CI (skip automático, executados em Google Colab Pro+)

## Colab Validation Notebooks

Notebooks em `notebooks/` para validação com GPU real:
- `notebooks/validate_sprint_o0_o1_gpu.ipynb` — Sprint O0/O1 em Colab T4/A100
- `notebooks/validate_sprint_o1_gpu_tests.ipynb` — testes GPU Sprint O1
- Instala via: `pip install git+...@tag` e roda suite GPU

---

*Testing analysis: 2026-05-24*
