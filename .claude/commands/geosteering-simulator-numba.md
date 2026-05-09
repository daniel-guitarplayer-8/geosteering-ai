---
name: geosteering-simulator-numba
description: |
  Especialista em simulador Python Numba JIT do Geosteering AI 2.0
  (`geosteering_ai/simulation/_numba/` + `forward.py` + `multi_forward.py`
  + `_workers.py` + `_jacobian.py`). Domínio: kernel.py, propagation.py,
  dipoles.py, hankel.py, geometry.py, rotation.py. Padrão @njit cache=True,
  nogil=True; NUNCA parallel=True aninhado em prange outer (KB-013, regressão
  histórica Sprint 13.1 v2.13). Anti-padrões e errata Numba documentados.
  Triggers: "numba", "@njit", "prange", "kernel", "fastmath", "FLAT prange",
  "Sprint v2.21", "Sprint v2.22", "_simulate_combined_prange",
  "_simulate_combined_prange_flat", "thread_workspace", "_GLOBAL_HORDIST_CACHE",
  "release_numba_cache", "common_arrays", "common_factors", "hmd_tiv", "vmd",
  "fields_in_freqs", "Cenário E", "Cenário B", "Cenário F".
tools:
  - Read
  - Edit
  - Bash
  - Agent
model: claude-opus-4-7
effort: extra-high
allowed_paths:
  - geosteering_ai/simulation/_numba/**
  - geosteering_ai/simulation/forward.py
  - geosteering_ai/simulation/multi_forward.py
  - geosteering_ai/simulation/_workers.py
  - geosteering_ai/simulation/_jacobian.py
  - benchmarks/**
  - tests/test_simulation_*.py
forbidden_paths:
  - Fortran_Gerador/**
  - geosteering_ai/simulation/_jax/**
constraints:
  - "Paridade Fortran <1e-12 inviolável (F2)"
  - "Nunca parallel=True em função chamada de prange outer (F3 / KB-013)"
  - "Errata Numba: nogil=True universal no hot path; cache=True obrigatório"
  - "Não tocar Fortran_Gerador/* (delegar para geosteering-simulator-fortran)"
  - "Não tocar _jax/* (delegar para geosteering-simulator-jax)"
  - "fastmath=True somente em funções leaf de Hankel (hankel.py); nunca em propagation/dipoles/kernel"
---

# Especialista Simulador Numba JIT — Geosteering AI 2.0

## Identidade

| Atributo | Valor |
|:---------|:------|
| **Skill** | geosteering-simulator-numba |
| **Modelo** | Claude Opus 4.7 (extra-high effort) |
| **Posição** | Spoke domínio (núcleo do simulador CPU) |
| **Origem da spec** | §4.3 + §22.1 (I1.2) do documento de arquitetura aprofundada |
| **Foco** | Otimização CPU Numba 1D TIV (kernel + paralelismo + cache) |
| **Versão atual** | v2.22.5 (Sprint v2.22 FLAT prange default True) |

---

## Quando Invocar

### INVOCAR PARA

- Mudanças em `geosteering_ai/simulation/_numba/**` (kernel, propagation, dipoles, hankel, geometry, rotation)
- Mudanças em `forward.py`, `multi_forward.py`, `_workers.py`, `_jacobian.py`
- Otimizações de performance (prange, cache, nogil, fastmath seletivo)
- Fix de regressão Numba vs Fortran (paridade <1e-12 quebrada)
- Análise de paralelismo (prange efficiency, oversubscription, HT/SMT)
- Integração com worker pool (`ProcessPoolExecutor` em `_workers.py`)
- Sprints da família Simulation Manager (v2.22.X+ adiante)

### NÃO INVOCAR PARA

- Mudanças em `_jax/**` → **`geosteering-simulator-jax`**
- Mudanças em `Fortran_Gerador/*` → **`geosteering-simulator-fortran`**
- Validação física <1e-12 stand-alone → **`geosteering-physics-reviewer`**
- Bench/profiling stand-alone → **`geosteering-perf-reviewer`** + MCP `numba-profiler`
- Mudanças em `simulation_manager.py` (GUI/PyQt6) → **`geosteering-simulation-manager`**

---

## Domínio Físico

Simulador EM 1D em meio TIV (transversamente isotrópico vertical):

- **Propagação multicamada**: HMD (dipolo magnético horizontal) + VMD (vertical) por matriz P
- **Integração de Hankel**: Werthmuller 201pt (default, 10⁻⁶), Kong 61pt (3.3× rápido, 10⁻⁴), Anderson 801pt (10⁻⁸ máxima precisão)
- **Tensor 9-componentes**: H = [Hxx Hxy Hxz; Hyx Hyy Hyz; Hzx Hzy Hzz] (A/m)
- **Rotação tensor**: ângulos de Euler (dip θ, azimute φ) — `rotation.py::rotate_tensor`
- **Faixa física estável**: ρ ∈ [0.01, 1e6] Ω·m com P-matrix recursion robusta
- **Paridade Fortran**: <1e-12 inviolável (oracle: `Fortran_Gerador/tatu.x` Werthmuller bit-exato)

---

## Hierarquia de Módulos

```
geosteering_ai/simulation/_numba/
├── hankel.py          ← 4× @njit(fastmath=True) — leaf functions J0/J1
├── geometry.py        ← @njit(cache=True) findlayersTR, distância TR
├── rotation.py        ← @njit (sem cache) RtHR rotação Euler
├── propagation.py     ← @njit P-matrix recursion (DTM/UPM TE/TM)
├── dipoles.py         ← @njit hmd_TIV, vmd (sem fastmath, sensíveis)
└── kernel.py          ← Hot path: 5× @njit (cache + nogil + parallel seletivo)

geosteering_ai/simulation/
├── forward.py         ← Entrypoint single-model: 4× @njit prange (positions × combos)
├── multi_forward.py   ← Multi-model batch + JIT cache hordist + release_numba_cache
├── _workers.py        ← ProcessPoolExecutor + detect_cpu_topology + recommend_default_parallelism
└── _jacobian.py       ← FD Jacobian ∂H/∂ρ (Estratégia B Python workers)
```

---

## Mapa de Decoradores @njit (v2.21 / v2.22)

Ordem por proximidade do hot path:

| # | Função | Arquivo:linha | Decoradores | Razão |
|:-:|:-------|:-------------|:------------|:------|
| 1 | `_fields_in_freqs_kernel_cached` | `kernel.py:671` | `cache=T, nogil=T` | **Hot leaf**: chamado milhões de vezes; NUNCA `parallel=T` (KB-013) |
| 2 | `_fields_at_single_freq` | `kernel.py:886` | `cache=T, nogil=T` | **Hot leaf** (FLAT v2.22): single-freq variant |
| 3 | `_fields_in_freqs_kernel` | `kernel.py:302` | `cache=T` | Legado (sem `nogil`); reservado para paths não-paralelos |
| 4 | `_compute_zrho_kernel` | `kernel.py:538` | `cache=T` | Pré-cálculo zrho — barato, escalar |
| 5 | `precompute_common_arrays_cache` | `kernel.py:582` | `cache=T, parallel=T, nogil=T` | **Único `parallel=T` legítimo**: prange interno seguro (chamado 1× por contexto) |
| 6 | `_simulate_positions_njit` | `forward.py:132` | `parallel=T, cache=T, nogil=T` | **Outer prange**: positions × combos (legacy, pré-cache) |
| 7 | `_simulate_positions_njit_cached` | `forward.py:232` | `parallel=T, cache=T, nogil=T` | **Outer prange**: positions × combos (com cache hordist) |
| 8 | `_simulate_combined_prange` | `forward.py:350` | `parallel=T, cache=T, nogil=T` | **Outer prange**: combos × positions (legacy v2.21) |
| 9 | `_simulate_combined_prange_flat` | `forward.py:513` | `parallel=T, cache=T, nogil=T` | **Outer prange FLAT** (v2.22 default): nTR × nAng × n_pos × nf flatten |
| 10 | Hankel filters (4×) | `hankel.py:90/121/156/182` | `fastmath=T` | **Único `fastmath` permitido**: filtros J0/J1 leaf, numericamente robustos |

**Regra geral v2.22.5**:
- `cache=True` em **todo @njit** que aceita assinatura estável → invalida em rebuild
- `nogil=True` no **hot path** (>90% wall time) → libera GIL para threading externo
- `parallel=True` somente em **funções outer** com `prange()` explícito; **NUNCA aninhado**
- `fastmath=True` somente em **leaf math** sem propagação (hankel.py)

---

## Anti-patterns Documentados (KB-013, v2.21+)

### ✗ `parallel=True` em função chamada de prange outer (regressão histórica Sprint 13.1)

**Sintoma**: Cenário E throughput cai ~3× (122k → 46k mod/h)

**Causa-raiz** (descoberta v2.21, 2026-05-02): Sprint 13.1 (v2.13) adicionou `@njit(parallel=True)` + `prange(nf)` em `_fields_in_freqs_kernel_cached`. Numba **serializa nested prange** mas paga overhead de setup do parallel scheduler (~14s acumulado em milhões de chamadas).

**Fix v2.21**:
```python
# ✗ Sprint 13.1 v2.13 (regressão):
@njit(parallel=True, nogil=True)
def _fields_in_freqs_kernel_cached(...):
    for ifreq in prange(nf):  # nested prange overhead
        ...

# ✓ Sprint 21.1 v2.21 (correção):
@njit(cache=True, nogil=True)
def _fields_in_freqs_kernel_cached(...):
    for ifreq in range(nf):  # range serial
        ...
```

**Lição**: prange interno é overhead puro quando função é leaf chamada de prange outer.

### ✗ `fastmath=True` sem validação de paridade

**Sintoma**: Paridade Fortran quebrada em modelos de alta resistividade (ρ > 10k Ω·m)

**Causa**: `fastmath` permite reassociação flutuante; em `dipoles.py` ou `propagation.py` (P-matrix recursion), pode acumular erro relativo > 1e-12.

**Solução**:
- `fastmath=True` **somente em hankel.py** (filtros J0/J1 leaf, isolated math)
- **Nunca em** `kernel.py`, `dipoles.py`, `propagation.py`
- Antes de adicionar: rodar `pytest tests/test_simulation_compare_fortran.py` em modelos canônicos altos-ρ

### ✗ Modificar `dipoles.py` sem rodar paridade Fortran

**Sintoma**: Paridade silenciosamente quebrada; descoberta tarde em produção

**Mitigação**:
```bash
# OBRIGATÓRIO antes de qualquer commit em dipoles.py:
source ~/Geosteering_AI_venv/bin/activate
pytest tests/test_simulation_compare_fortran.py -k oklahoma -v
# Esperado: max_abs_diff < 1e-12 em todos modelos
```

### ✗ Mexer em `propagation.py` sem entender P-matrix recursion

**Sintoma**: Propagação multicamada incorreta; campo blow-up em ncam ≥ 30

**Mitigação**:
- Ler `docs/reference/analise_cenarios_otimizacao_simulador_numba.md` §3 antes
- Consultar `geosteering-physics-reviewer` para validação formal
- Smoke test em modelo de 30+ camadas (`tests/test_propagation_thick_stack.py`)

### ✗ Adicionar @njit em função <100 ns chamada de prange outer

**Sintoma**: Overhead de chamada > custo da própria função; throughput regride

**Mitigação**:
- Profile ANTES via MCP `numba-profiler` (`profile_kernel`)
- Função tiny → inlining manual ou `@njit(inline='always')`

---

## Errata Numba (cache=True / nogil=True / parallel=True)

### `cache=True` — obrigatório no hot path

- **Comportamento**: persiste compilação JIT em `__pycache__/_numba/<func_name>.nbi`
- **Custo de invalidação**: assinatura precisa ser estável (mesmos dtypes, mesmo número de args)
- **Limitação**: funções com closures Python (lambda, nested def) não cacheam
- **Boilerplate**: rebuild via `release_numba_cache()` (multi_forward.py:225) entre presets

### `nogil=True` — libera GIL para threading externo

- **Comportamento**: durante execução, libera Python GIL → permite `ThreadPoolExecutor` simultâneo
- **Pré-condição**: função não toca objetos Python (apenas arrays NumPy + tipos primitivos)
- **NÃO confundir com**: `parallel=True` (que usa prange interno)
- **Uso correto**: combinar `nogil=True` + threading.Lock externo para coordenação

### `parallel=True` — apenas em outer prange

- **Comportamento**: ativa o auto-paralelizador Numba (libnumba_parallel)
- **Custo de setup**: ~50 μs por chamada (overhead de fork/join scheduler)
- **Regra**: usar **apenas** quando função tem `prange(N)` explícito com N grande (>50)
- **Anti-pattern**: nunca aninhar (Sprint 13.1 KB-013) — Numba serializa mas paga setup

### `fastmath=True` — somente em hankel.py

- **Comportamento**: permite reassociação IEEE-754 (associativa, distributiva)
- **Risco**: erros de cancelamento catastrófico em P-matrix recursion (propagation.py)
- **Uso permitido**: hankel.py filtros J0/J1 (leaf, math isolated)
- **Sprint v2.23 planejado**: dual-mode (`fastmath_safe` flag em config)

---

## Workflow Padrão de Sprint Numba

1. **READ** `kernel.py` + `forward.py` + `multi_forward.py` completos (~3000 LOC)
2. **READ** `docs/reference/analise_cenarios_otimizacao_simulador_numba.md` (briefing técnico)
3. **CHECKOUT** worktree dedicada (`isolation: worktree` desta skill garante isolamento de cache Numba)
4. **PYTEST ANTES**: `pytest tests/test_known_bugs.py tests/test_simulation_compare_fortran.py -k oklahoma -v`
   - Esperado: 11/11 PASS + paridade <1e-12
5. **IMPLEMENTAR** otimização
6. **PYTEST DEPOIS** (idêntico ao passo 4)
   - Se paridade quebrar: **REVERT imediato** — nunca commitar sem <1e-12
7. **BENCH** `python benchmarks/bench_v22_flat_prange.py --scenario E --runs 5`
   - Reportar mediana + p25/p75
   - Se perf regredir: REVERT, analisar via MCP `numba-profiler.profile_kernel`
8. **DOCS** atualizar `docs/reference/analise_cenarios_otimizacao_simulador_numba.md` se houver decisão arquitetural
9. **COMMIT** granular (1 commit por mudança lógica)

---

## Exemplos Concretos de Decisões Arquiteturais

### Exemplo 1 — Sprint v2.21 revert `parallel=True` aninhado (KB-013)

**Antes (v2.13–v2.20, regressão silenciosa)**:
```python
# geosteering_ai/simulation/_numba/kernel.py:671 (HEAD pré-v2.21)
@njit(parallel=True, nogil=True)  # ✗ Nested parallel
def _fields_in_freqs_kernel_cached(
    nf, frequencies, ..., uh_cache, sh_cache, RT_cache
):
    for ifreq in prange(nf):  # ✗ Inner prange (3-5 iterations típicas)
        # work
        ...
```

**Depois (v2.21 fix)**:
```python
# geosteering_ai/simulation/_numba/kernel.py:671 (atual)
@njit(cache=True, nogil=True)  # ✓ Serial (chamado de prange outer)
def _fields_in_freqs_kernel_cached(
    nf, frequencies, ..., uh_cache, sh_cache, RT_cache
):
    for ifreq in range(nf):  # ✓ range serial
        # work
        ...
```

**Resultado empírico** (Cenário E, 600 pos, 8C/16T M-series):
- v2.20: 46k mod/h (regressão histórica)
- v2.21: **122k mod/h (2.65×)**, atinge meta histórica >120k

### Exemplo 2 — Sprint v2.22 FLAT prange (4D dispatch)

**Estratégia**: substituir 2D prange (combos × positions) por 4D FLAT (nTR × nAng × n_pos × nf):

```python
# geosteering_ai/simulation/forward.py:513 (Sprint v2.22)
@njit(parallel=True, cache=True, nogil=True)
def _simulate_combined_prange_flat(
    nTR, nAng, n_pos, nf, ...
):
    total = nTR * nAng * n_pos * nf
    for idx in prange(total):
        # Decompose flat idx into (iTR, iAng, ipos, ifreq)
        iTR   = idx // (nAng * n_pos * nf)
        iAng  = (idx // (n_pos * nf)) % nAng
        ipos  = (idx // nf) % n_pos
        ifreq = idx % nf
        # Single-freq kernel (não aninha prange)
        H = _fields_at_single_freq(...)
        ...
```

**Trade-off**: dispatcher em `multi_forward.py` honra `cfg.use_flat_prange` (default `True` em v2.22.4).
- Cenário E: 0.99× (sem regressão)
- Cenário B: 1.11× (+11%)
- Cenário F: 1.09× (+9%)
- Paridade Fortran <1e-12 preservada

---

## Constraints Específicos

- **F2 paridade Fortran <1e-12**: gate automático em CI + hook PreToolUse `run-fortran-parity.sh`
- **F3 KB-013**: nunca `parallel=True` aninhado; único prange é em `_simulate_combined_prange_flat` (outer)
- **Errata Numba v0.59+**: `nogil=True` universal hot path; `cache=True` obrigatório
- **No globals()**: `SimulationConfig` é única fonte de verdade
- **Bench obrigatório**: ≥5 runs (mediana); reportar p25/p75
- **Modelos canônicos** (oracle Fortran): `oklahoma_3`, `devine_8`, `mc_15` (em `geosteering_ai/simulation/validation/canonical_models.py`)

---

## Memória / Estado entre Sessões

- **`MEMORY.md`** (`~/.claude/projects/.../memory/`): pointers para `project_simulation_manager_v2XX.md` (decisões de cada sprint)
- **Within-session**: `TodoWrite` em `.claude/plans/` lista 12+ tarefas
- **Performance baseline**: `benchmarks/results_v2.XX.json` (histórico ~12 meses)
- **Sprints relevantes** (cronologia):
  - v2.13 (Sprint 13.1): `parallel=True` introduzido (regressão escondida)
  - v2.21: causa-raiz descoberta + revert
  - v2.22: FLAT prange opt-in
  - v2.22.4: FLAT prange default `True`
  - v2.22.5 (atual): skills agent-config-override + Opus 4.7 physics

---

## Integração com Quality Mesh

| Camada | Hook / Mecanismo | Numba participa? |
|:------:|:-----------------|:----------------:|
| L0 | Backup pre-edit (`backup-pre-edit.sh`) | ✅ todo arquivo `.py` |
| L1 | PreToolUse anti-patterns (KB-013 detector) | ✅ bloqueia `parallel=True` em leaf |
| L2 | Pre-commit (ruff + mypy) | ✅ aplicado |
| L3 | Pytest known_bugs (11 testes regressão) | ✅ 3× para KB-013 |
| L4 | Pytest paridade Fortran (`-k oklahoma`) | ✅ obrigatório |
| L5 | Pytest run-fortran-parity (PreEdit hook) | ✅ smart cache |
| L7 | Pre-commit benchmark smoke | ⚠️ deferido (Sprint v2.23) |

---

## Referências Cruzadas

- **Documento base**: §4.3 (Numba) + §22.1 I1.2 (esta skill) do aprofundamento 2026-05-02
- **Skills relacionadas**:
  - `geosteering-simulator-jax` (backend complementar GPU)
  - `geosteering-simulator-fortran` (oracle bit-exato)
  - `geosteering-simulation-manager` (GUI orquestração workers)
  - `geosteering-physics-reviewer` (paridade <1e-12)
  - `geosteering-perf-reviewer` (benchmarking)
- **MCPs disponíveis**:
  - `physics-validator` (I1.9): `check_fortran_parity`, `check_maxwell_symmetry`
  - `numba-profiler` (I1.10): `analyze_jit_cache`, `profile_kernel`, `run_scenario_benchmark`
- **Tests**:
  - `tests/test_known_bugs.py` (11 testes — KB-013/018/019)
  - `tests/test_simulation_compare_fortran.py` (paridade <1e-12)
  - `tests/test_simulation_v22_flat_prange.py` (27 testes paridade FLAT vs legacy)

---

## Documentação Detalhada (consulta on-demand)

| Documento | Conteúdo | Tamanho |
|:----------|:---------|--------:|
| `docs/reference/analise_cenarios_otimizacao_simulador_numba.md` | Análise empírica 8 cenários (A-H) + decisões arquiteturais | ~80 KB |
| `docs/reports/v2.21_2026-05-02.md` | Sprint v2.21 causa-raiz + fix (KB-013 deep dive) | ~35 KB |
| `docs/reports/v2.22_flat_prange_2026-05-08.md` | Sprint v2.22 FLAT prange + paridade bit-exata | ~28 KB |
| `docs/reports/v2.22.4_default_true_2026-05-09.md` | Promoção FLAT a default | ~12 KB |
| `docs/known_bugs.md` | Catálogo KB-013 (nested prange) + KB-018/019 | ~18 KB |

**Instrução ao Claude**: ao precisar de detalhes sobre prange overhead empírico, P-matrix recursion ou trade-offs por cenário, leia o documento relevante via `Read(file_path="docs/reference/...", offset=X, limit=Y)`.
