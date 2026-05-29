# Coding Conventions

**Analysis Date:** 2026-05-24

## Language and Identifiers

**Primary Language:** Python 3.13 (pinned — 3.14+ proibido, sem wheels para PyQt6/JAX/SciPy)

**Identifier language:** English (variables, functions, classes, modules, constants)
**Comment/docstring language:** PT-BR with mandatory accents (acentuação obrigatória)

```python
# CORRETO:
# Inicialização do pipeline de dados com configuração padrão.
def build_data_pipeline(config: PipelineConfig) -> DataPipeline:

# PROIBIDO — identificador em PT-BR:
def construir_pipeline(configuracao):

# PROIBIDO — comentário sem acento:
# Inicializacao do pipeline de dados com configuracao padrao.
```

## Naming Patterns

**Files:**
- `snake_case.py` — todos os módulos Python
- `test_<module>.py` — arquivos de teste co-localizados em `tests/`
- `_private_module.py` — prefixo `_` para helpers privados (ex: `tests/_fortran_helpers.py`, `geosteering_ai/simulation/_numba/`)

**Functions:**
- `snake_case` — todas as funções públicas e privadas
- Prefixo `_` para funções internas/privadas não exportadas

**Classes:**
- `PascalCase` — todas as classes (ex: `PipelineConfig`, `ModelRegistry`, `LossFactory`, `SimulationThread`)

**Constants:**
- `UPPER_SNAKE_CASE` — constantes e defaults físicos (ex: `FREQUENCY_HZ`, `SPACING_METERS`, `DTYPE_22COL`)

**Type aliases / markers:**
- `snake_case` para pytest markers definidos em `pyproject.toml`

## Absolute Prohibitions

### PyTorch em Production Paths (BLOCK — exit 2)

`import torch` ou `from torch ...` são **bloqueados** pelo hook `validate-no-pytorch.sh` em:
- `geosteering_ai/models/`, `geosteering_ai/losses/`, `geosteering_ai/training/`
- `geosteering_ai/inference/`, `geosteering_ai/evaluation/`, `geosteering_ai/data/`
- `geosteering_ai/simulation/`, `geosteering_ai/visualization/`, `geosteering_ai/utils/`

Uso legítimo via adapter:
```python
from geosteering_ai.adapters import get_adapter
torch_adapter = get_adapter("pytorch")  # UNICO caminho permitido
```
Pesquisa exploratória: `geosteering_ai/research/` (não bloqueado).

### `print()` (WARN via anti-patterns)

Proibido em `geosteering_ai/`. Usar sempre `logger` do módulo `geosteering_ai/utils/`:
```python
# PROIBIDO:
print(f"Modelo criado: {model_type}")

# CORRETO:
from geosteering_ai.utils import get_logger
logger = get_logger(__name__)
logger.info("Modelo criado: %s", model_type)
```

### `globals().get()` (BLOCK via anti-patterns)

Proibido em qualquer arquivo `geosteering_ai/`. Usar sempre `PipelineConfig`:
```python
# PROIBIDO (KB-GLB):
model_type = globals().get("MODEL_TYPE", "ResNet_18")

# CORRETO:
def build_model(config: PipelineConfig) -> tf.keras.Model:
    model_type = config.model_type  # sempre do config
```

### Valores Físicos Errados (BLOCK — imutáveis)

Os seguintes defaults são verificados pelo hook `validate-physics.sh` e pelo `PipelineConfig.__post_init__()`:

| Constante | Valor PROIBIDO | Valor CORRETO |
|-----------|---------------|---------------|
| `FREQUENCY_HZ` | `2.0` (KB-FRQ) | `20000.0` |
| `SPACING_METERS` | `1000.0` (KB-SPC) | `1.0` |
| `SEQUENCE_LENGTH` | `601` | `600` |
| `TARGET_SCALING` | `"log"` (KB-LOG) | `"log10"` |
| `INPUT_FEATURES` | `[0,3,4,7,8]` (KB-INF) | `[1,4,5,20,21]` |
| `OUTPUT_TARGETS` | `[1,2]` (KB-OTG) | `[2,3]` |
| `eps_tf` | `1e-30` (KB-EPS) | `1e-12` |

### Known Bug Anti-Patterns (BLOCK via `.claude/anti-patterns.txt`)

| ID | Padrão | Severidade | Arquivo |
|----|--------|-----------|---------|
| KB-013 | `@njit(...parallel=True` | BLOCK | `*_numba/kernel.py` |
| KB-018 | `rng_seed = 42` hardcoded | BLOCK | `*simulation_manager.py` |
| KB-019 | `threads_per_worker=4` com `workers=4` | BLOCK | `*sm_workers.py` |
| KB-002 | `epoch / total_epochs` sem `+` | WARN | `*noise/curriculum.py` |
| KB-PYT | `import torch` | BLOCK | `*.py` |
| KB-PRT | `print(` em linha | WARN | `*geosteering_ai/*.py` |
| KB-GLB | `globals().get(` | BLOCK | `*geosteering_ai/*.py` |

## Code Patterns Obrigatórios

### 1. PipelineConfig como Parâmetro (Nunca Globals)

```python
# CORRETO:
def build_model(config: PipelineConfig) -> tf.keras.Model:
    hidden = config.hidden_units
    ...

# PROIBIDO:
def build_model():
    hidden = globals().get("HIDDEN_UNITS", 128)  # KB-GLB — BLOQUEADO
```

### 2. Factory Pattern para Componentes

```python
# CORRETO:
model = ModelRegistry().build(config)           # geosteering_ai/models/registry.py
loss_fn = LossFactory.get(config)              # geosteering_ai/losses/factory.py
callbacks = build_callbacks(config, model, noise_var)  # geosteering_ai/training/

# PROIBIDO:
if config.loss_type == "rmse":
    loss_fn = rmse_loss
elif config.loss_type == "mae":
    loss_fn = mae_loss  # lógica imperativa — usar Factory
```

### 3. DataPipeline com Cadeia Explícita

```python
pipeline = DataPipeline(config)
data = pipeline.prepare(dataset_path)           # raw → split → fit_scaler
map_fn = pipeline.build_train_map_fn(noise_var) # noise → FV → GS → scale
```

Cadeia fisicamente correta (obrigatória):
```
train_raw → noise(A/m) → FV_tf(noisy) → GS_tf(noisy) → scale → modelo
```
- Scaler fitado em dados **limpos** (FV+GS clean, temporário)
- Val/test transformados offline
- Train permanece raw para on-the-fly

### 4. Presets YAML para Reprodutibilidade

```python
config = PipelineConfig.from_yaml("configs/robusto.yaml")
# OU presets de classe:
config = PipelineConfig.robusto()
config = PipelineConfig.nstage(n=3)
config = PipelineConfig.geosinais_p4()
config = PipelineConfig.realtime(model_type="WaveNet")
```

## Mandatory Documentation Patterns (D1–D14)

Todo arquivo em `geosteering_ai/` DEVE seguir os 14 padrões abaixo.

### D1 — Mega-header Unicode (topo de cada módulo .py)

```python
# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/models/registry.py                                        ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : Registry de modelos — fábrica centralizada                 ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Models                                                     ║
# ║  Versão      : v2.0                                                       ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : YYYY-MM-DD                                                 ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : TensorFlow 2.x / Keras 3.x                                 ║
# ║  Dependências: config, utils                                              ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    [Descrição em PT-BR com acentuação]                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
```

14 campos obrigatórios: nome do arquivo, módulo, projeto, subsistema, versão, autor, criação, status, framework, dependências, finalidade (mínimo).

### D2 — Cabeçalho de Seção com 4+ Linhas de Contexto

```python
# ═════════════════════════════════════════════════════════════════════════
# Seção: Inicialização do Stem Conv1D
# ─────────────────────────────────────────────────────────────────────────
# Convolution inicial de campo receptivo maior (7×1) para capturar
# features EM de baixa frequência. strides=1 preserva N_MEDIDAS.
# Kernel 7 cobre ~7m de profundidade no perfil de poço,
# ideal para detectar contrastes de camada espessa.
# ═════════════════════════════════════════════════════════════════════════
```

### D3 — Diagramas ASCII com Unicode Borders (quando ≥3 caminhos/categorias)

```python
#   ┌──────────────────────┬────────────┬────────────────────────┐
#   │  Modelo              │  Blocos    │  Tipo                  │
#   ├──────────────────────┼────────────┼────────────────────────┤
#   │  ResNet_18 (★)       │  8         │  ResidualBlock         │
#   │  ResNet_34           │  16        │  ResidualBlock         │
#   └──────────────────────┴────────────┴────────────────────────┘
```

### D4 — Atributos de Config com 4+ Linhas por Grupo

Cada grupo de atributos em `geosteering_ai/config.py` documenta propósito físico, range válido, default e referência.

### D5 — Docstrings Google-style com 5+ Campos

```python
def build_resnet18(input_shape, output_channels, use_causal, **kwargs):
    """Constrói modelo ResNet-18 para inversão 1D de resistividade.

    Estrutura do modelo:
      ┌──────────────────────────────────────────────────────────┐
      │  Input (batch, None, N_FEATURES)                        │
      │    ↓                                                    │
      │  Stem: Conv1D(64, 7) → BN → ReLU → Dropout            │
      │    ↓                                                    │
      │  Output: Dense(output_channels, 'linear')               │
      └──────────────────────────────────────────────────────────┘

    Args:
        input_shape: Shape (None, N_FEATURES). None permite sequência
            de comprimento variável (multi-ângulo). Default: (None, 5).
        output_channels: Número de saídas (resistividades). Default: 2
            para P1 baseline [ρ_h, ρ_v].
        use_causal: Se True, usa padding causal para inferência em tempo
            real (janela deslizante). Se False, padding "same" (offline).

    Returns:
        tf.keras.Model compilado, pronto para treinamento.

    Raises:
        ValueError: se input_shape incompatível com a arquitetura.

    Note:
        Ref: He et al. "Deep Residual Learning for Image Recognition"
        (CVPR 2016) — skip connections estabilizam gradientes em redes
        profundas, permitindo treinar 4 estágios sem degradação.

    Example:
        >>> model = build_resnet18((None, 5), 2, use_causal=False)
        >>> model.summary()
    """
```

Campos obrigatórios: descrição, Args (com significado físico), Returns, Raises, Note (com referência), Example.

### D6 — Docstrings de Classes com Attributes + Example

Toda classe pública documenta `Attributes:` (um por linha com tipo e descrição) e `Example:` com snippet executável.

### D7 — Comentários Inline Semânticos

```python
x = Conv1D(64, 7, padding=_padding, strides=1)(x)  # stem: RF=7m, captura gradientes longos
```

### D8 — Inventário de Exports com `__all__` Semântico

Todos os módulos públicos definem `__all__` com lista explícita e comentários descritivos.

### D9 — Logging Estruturado (Nunca `print`)

```python
logger = get_logger(__name__)
logger.info("Treinamento iniciado: epochs=%d, lr=%.2e", config.epochs, config.lr)
logger.warning("Scaler fitado em dados ruidosos — violar P1 (errata)")
logger.error("Falha na leitura de %s: %s", dataset_path, str(exc))
```

### D10 — Constantes com Documentação Física

```python
# Frequência padrão LWD (20 kHz) — range validado: [100, 1e6] Hz
# Derivado do .out Fortran (PerfilaAnisoOmp.f08). NUNCA alterar para 2.0.
FREQUENCY_HZ: float = 20000.0
```

### D11 — Tabelas de Fórmulas ASCII em Catálogos

Módulos de catálogos (`losses/catalog.py`, `models/`) incluem tabelas ASCII com fórmulas matemáticas.

### D12 — Cross-references `Note:` em Docstrings

Toda função pública referencia módulos relacionados, decisões arquiteturais ou documentação física.

### D13 — Branch Comments com Layout de Saída

```python
if view == "H1_logH2":
    # ── H1_logH2: H1 cru preserva SNR em alta atenuação,
    #    H2 log10-transformado comprime faixa dinâmica larga de Hzz.
    #    Saída: [Re(H1), Im(H1), log10|H2|, φ(H2)]
    #    Motivação física: Hzz varia 4+ ordens de magnitude.
    ...
```

### D14 — Diagrama noise × FV × GS

Obrigatório em `geosteering_ai/data/pipeline.py` — diagrama ASCII da cadeia completa de dados.

## Code Style

**Formatting:** `ruff-format` (substituiu Black)
- Config: `pyproject.toml` (sem seção `[tool.ruff]` extra — usa defaults ruff)
- Pre-commit hook: `ruff-format` em `^(geosteering_ai|tests)/.*\.py$`

**Linting:** `ruff` (substituiu flake8 + isort)
- Args: `["--fix", "--exit-non-zero-on-fix"]`
- Scope: `^(geosteering_ai|tests)/.*\.py$`

**Type checking:** `mypy`
- `python_version = "3.13"`
- `warn_return_any = true`
- `warn_unused_configs = true`
- `ignore_missing_imports = true`
- Exclude: `^geosteering_ai/simulation/tests/`
- Config: `pyproject.toml` seção `[tool.mypy]`

## Import Organization

**Order (enforced by ruff):**
1. `from __future__ import annotations` (primeiro, quando necessário)
2. Standard library
3. Third-party (`numpy`, `tensorflow`, `jax`, etc.)
4. Internal (`from geosteering_ai.config import PipelineConfig`)

**Type-checking imports:**
```python
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from geosteering_ai.simulation.multi_forward import MultiSimulationResultBatch
```

## Error Handling

**Strategy:** Fail-fast no `PipelineConfig.__post_init__()` — errata validada em tempo de construção, não em runtime tardio.

**Patterns:**
- `AssertionError` com mensagem descritiva para validações de errata (ex: `assert 100.0 <= FREQUENCY_HZ <= 1e6, "frequency_hz"`)
- `ValueError` para argumentos inválidos em funções públicas
- Try-except granular: capturar `ImportError`, `TypeError`, `ValueError` explicitamente — nunca `except Exception` para mascarar erros sérios (`SystemError`, `MemoryError`)
- `RuntimeError` para estados impossíveis em inicialização (ex: nenhum binding Qt6 disponível)

## Logging

**Framework:** `logging` stdlib via wrapper `get_logger` em `geosteering_ai/utils/`

**Pattern:**
```python
from geosteering_ai.utils import get_logger
logger = get_logger(__name__)
```

**Níveis:**
- `logger.debug(...)` — fluxo interno, detalhes de implementação
- `logger.info(...)` — marcos de execução (início/fim de treinamento, carga de dados)
- `logger.warning(...)` — situações suspeitas mas recuperáveis (KB-019 oversubscrição)
- `logger.error(...)` — falhas de operação (I/O, dados corrompidos)

**Nunca usar `print()`** — KB-PRT alerta no stderr ao detectar `print(` em `geosteering_ai/`.

## Module Design

**Exports:** Todo módulo público define `__all__` explícito (D8).
**Encoding:** `# -*- coding: utf-8 -*-` no topo (obrigatório para PT-BR).
**Docstring de módulo:** imediatamente após o encoding header — descreve propósito, cobertura, e uso típico.

## Pre-commit Setup

Instalar:
```bash
pip install pre-commit && pre-commit install
```

Rodar manualmente:
```bash
pre-commit run --all-files
```

Config: `.pre-commit-config.yaml`

## Hooks Enforcement Chain

| Hook | Evento | Arquivo | Severity |
|------|--------|---------|----------|
| `backup-pre-edit.sh` | PreToolUse Edit | `.claude/hooks/` | INFO (sempre) |
| `check-anti-patterns.sh` | PreToolUse Edit\|Write | `.claude/hooks/` | BLOCK/WARN |
| `validate-no-pytorch.sh` | PreToolUse Edit\|Write | `.claude/hooks/` | BLOCK (exit 2) |
| `validate-physics.sh` | PreToolUse Edit\|Write | `.claude/hooks/` | BLOCK |
| `check-ptbr-accentuation.sh` | PostToolUse Edit\|Write | `.claude/hooks/` | WARN |
| `run-fortran-parity.sh` | PostToolUse Edit (kernel) | `.claude/hooks/` | BLOCK (exit 1) |
| `check-perf-regression.sh` | PostToolUse (manual/CI) | `.claude/hooks/` | WARN |
| `check-version-references.sh` | PreToolUse | `.claude/hooks/` | WARN |
| `ruff` + `ruff-format` | pre-commit | `.pre-commit-config.yaml` | BLOCK |
| `mypy` | pre-commit | `.pre-commit-config.yaml` | BLOCK |
| `check-anti-patterns-precommit.sh` | pre-commit | `.pre-commit-config.yaml` | BLOCK |
| `validate-physics.sh` | pre-commit (config.py, configs/*.yaml) | `.pre-commit-config.yaml` | BLOCK |
| `fortran-parity-full` | pre-commit (simulation/_numba/ ou forward.py) | `.pre-commit-config.yaml` | BLOCK (7 modelos ~146s) |

**Bypass emergencial** (usar com cautela):
```bash
CLAUDE_BYPASS_ANTI_PATTERNS=1 ...  # desabilita check-anti-patterns
CLAUDE_BYPASS_FORTRAN_PARITY=1 ... # desabilita parity gate
CLAUDE_BYPASS_PTBR=1 ...           # desabilita accentuation check
```

## PT-BR Accents (Mandatory)

Nunca usar formas sem acento em comentários, docstrings, ou documentação:

| PROIBIDO | CORRETO |
|---------|---------|
| implementacao | implementação |
| configuracao | configuração |
| nao | não |
| funcao | função |
| informacao | informação |
| definicao | definição |

Catálogo completo: `.claude/ptbr-words.txt` (60+ pares TSV).
Hook `check-ptbr-accentuation.sh` alerta (WARN) ao detectar violações em `.md`, `.py`, `.sh`, `.f08`, `.f90`.

---

*Convention analysis: 2026-05-24*
