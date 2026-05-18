# Sprint v2.36 — Plano Executivo Geosteering AI Numba JIT

## Contexto

A Sprint v2.35 (2026-05-15) entregou Cenário H no CI/CD e fechou o ciclo de
estabilização/observabilidade (v2.33–v2.35). Restaram 5 débitos identificados
no relatório `docs/reports/v2.35_numba_jit_status_2026-05-15.md`:

| ID  | Débito                                                | Impacto                    | Risco |
|:---:|:------------------------------------------------------|:---------------------------|:-----:|
| D1  | `E_n200_warm` baseline = `null`                       | CI sem proteção hot-path   | ALTO  |
| D2  | Cenário H sem baseline formal em `perf_baseline.json` | Stress-test sem regressão  | MÉDIO |
| D3  | `NUMBA_CACHE_DIR` ausente do `tests/conftest.py`      | LLVM recompile na suite    | BAIXO |
| D4  | Flag `cfg.use_fastmath` inócua (não lida)             | Confusão de API            | BAIXO |
| D5  | Cenário B regrediu −19% (376k → 303k) em v2.21        | Multi-freq LWD afetado     | ALTO  |

Esta sprint:
- **Fecha 4 débitos** (D1+D2+D3+D4).
- **Investiga D5** (path FLAT vs não-FLAT em multi-freq+multi-TR).
- **Tenta otimização O2** (tile/block) com paridade Fortran <1e-12 preservada.

Resultado esperado: `1665+ PASS / 0 FAIL`, hook anti-regressão protegendo
warm path, regressão Cenário B explicada (recuperada ou roadmap v2.37+).

---

## Validações da exploração (estado atual confirmado)

### Códigos-fonte verificados

| Caminho                                          | Achado                                                                                            |
|:-------------------------------------------------|:--------------------------------------------------------------------------------------------------|
| [forward.py:132](geosteering_ai/simulation/forward.py#L132) `_simulate_positions_njit` | `@njit(parallel=True, cache=True, nogil=True)` + `prange(n_positions)`        |
| [forward.py:232](geosteering_ai/simulation/forward.py#L232) `_simulate_positions_njit_cached` | Idem + 9 caches pré-computados (path single-TR legado v2.5)            |
| [forward.py:513](geosteering_ai/simulation/forward.py#L513) `_simulate_combined_prange_flat` | `prange(nTR × nAng × n_pos × nf)` — **PATH QUENTE EM PRODUÇÃO desde v2.22.4** |
| [multi_forward.py:1075](geosteering_ai/simulation/multi_forward.py#L1075) | Dispatcher `if cfg.use_flat_prange: flat else: combined` (default True) |
| [config.py:376](geosteering_ai/simulation/config.py#L376)         | `use_flat_prange: bool = True` (default desde v2.22.4)                              |
| [config.py:379](geosteering_ai/simulation/config.py#L379)         | `use_fastmath: bool = False` — **0 leituras dinâmicas** (puramente documental)      |
| [cli/main.py:46-75](geosteering_ai/cli/main.py#L46-L75)           | `NUMBA_CACHE_DIR` em tmpfs (Sprint v2.31)                                            |
| [tests/conftest.py](tests/conftest.py)                            | 40 LOC, **só QT_API**, sem `NUMBA_CACHE_DIR`                                         |
| [cli/benchmark.py](geosteering_ai/cli/benchmark.py)               | Cenários A–H suportados (H = 8×8×8 v2.35); `--workers`, `--threads` OK              |
| `.claude/perf_baseline.json`                                      | `E_n200_warm.throughput_mod_h = null` (placeholder v2.34); sem Cenário H              |
| `docs/PERFORMANCE_BASELINE.md`                                    | Sem seção Cenário H; sem Cenário B regression note                                   |

### ⚠ Gaps detectados nas instruções da Sprint v2.36 vs estado atual

1. **`--runs` não existe no CLI**. O comando `geosteering-cli benchmark` aceita
   `--scenario`, `--n`, `--workers`, `--threads`, mas **NÃO `--runs`**. O bench
   deve ser feito via loop `for i in 1..5; do geosteering-cli benchmark ...; done`
   OU adicionar flag `--runs` em `cli/benchmark.py`. **Decisão**: loop bash
   (sem alteração de API, MENOR escopo).

2. **Alvo de O2 (tile/block) — `_simulate_positions_njit_cached` vs `_simulate_combined_prange_flat`**:
   o briefing menciona o **primeiro**, mas o **segundo** é o path quente em
   produção (default desde v2.22.4). Tile/block em `_simulate_positions_njit_cached`
   afeta apenas o path legado single-TR (`simulate()` em `forward.py`).
   **Decisão a confirmar com o usuário antes de implementar** (ver AskUserQuestion).

3. **Cenário B tem 2 definições**:
   - `cli/benchmark.py` Cenário B = `(n_pos=100, nf=1, nTR=1, nAng=1)` — caminho leve
   - `benchmarks/bench_v22_flat_prange.py` Cenário B = `(n_pos=200, nf=1, nTR=3, nAng=4)` — multi-array LWD
   - Regressão −19% reportada no relatório v2.35 vem da **segunda definição**
     (bench_v22). D5 deve investigar usando `benchmarks/bench_v22_flat_prange.py`.

---

## FRENTE 1 — Débitos imediatos D1+D2+D3 (não tocam caminho crítico Numba)

### D1 — Medir `E_n200_warm` (estimativa 30 min)

**Arquivos**:
- `.claude/perf_baseline.json` (update)
- `docs/PERFORMANCE_BASELINE.md` (seção 2.1 atualizada)

**Procedimento**:
```bash
source ~/Geosteering_AI_venv/bin/activate
geosteering-warmup --verbose
# 5 runs Cenário E (warm), extrair "mod/h"
for i in 1 2 3 4 5; do
  geosteering-cli benchmark --scenario E --n 200 2>&1 | grep -oE '[0-9,]+ mod/h' | head -1
done | sort -n
# Anotar mediana (3º valor após sort)
```

**Aceite**: `perf_baseline.json` campo `E_n200_warm.throughput_mod_h != null`;
mediana esperada ≥76k mod/h (~110% de cold 69.336).

### D2 — Baseline Cenário H formal (estimativa 1h, 5 runs × ~120s)

**Arquivos**:
- `.claude/perf_baseline.json` (nova entrada `H_n2_stress`)
- `docs/PERFORMANCE_BASELINE.md` (tabela seção 1 + nova nota Cenário H)
- `tests/test_perf_baseline_h.py` (criar guard)

**Procedimento**:
```bash
for i in 1 2 3 4 5; do
  geosteering-cli benchmark --scenario H --n 2 --workers 2 --threads 2 2>&1 | tee /tmp/h_run_$i.log
  grep -oE '[0-9,]+ mod/h' /tmp/h_run_$i.log | head -1
done | sort -n
# Mediana → throughput_mod_h
```

**Aceite**: entrada `H_n2_stress` adicionada com `type: "warm"`, `version: "v2.36"`;
guard test passa.

### D3 — `NUMBA_CACHE_DIR` em `conftest.py` (estimativa 15 min)

**Arquivo**: `tests/conftest.py` (raiz)

**Padrão**: bloco no **topo** (antes de imports `from geosteering_ai...`):
```python
import os
import tempfile

_cache_dir = os.path.join(
    tempfile.gettempdir(), "geosteering_numba_cache_test"
)
try:
    os.makedirs(_cache_dir, mode=0o700, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", _cache_dir)
except OSError:
    pass
```

**Aceite**:
- `pytest tests/test_simulation_numba_specializations.py::test_numba_cache_dir_set` PASS.
- `pytest tests/ -v` sem aumento de tempo (cache compartilhado entre testes).

---

## FRENTE 2 — D4: Remover flag `use_fastmath` inócua (estimativa 45 min)

**Justificativa**: Confirmado por exploração que:
- `use_fastmath` declarado em `config.py:379` com docstring `puramente documental`.
- **0 leituras dinâmicas** no código (`grep -rn cfg.use_fastmath` retorna apenas docstring).
- Hard-coded `fastmath=True` em 8 funções (geometry, rotation, hankel, filters) NÃO depende de `cfg`.

**Estratégia**: **Remoção direta** (sem `DeprecationWarning`), pois:
- Versão `v2.36` é minor (API estável `geosteering_ai.simulation.config`).
- Nenhum YAML preset em `configs/*.yaml` referencia `use_fastmath`.
- 2 testes em `tests/test_sprint_v224.py:174-198` validam o default — remover esses testes.
- `tests/test_simulation_v223_fastmath_threads.py` valida `fastmath=True` **hard-coded** — manter.

**Arquivos**:
- `geosteering_ai/simulation/config.py:379-404` — remover campo + docstring 25 LOC
- `tests/test_sprint_v224.py:174-198` — remover testes obsoletos (frente 1.4)
- `docs/CHANGELOG.md` — entrada de breaking change documentada

**Aceite**:
- `grep -rn use_fastmath geosteering_ai/ tests/` retorna 0 matches.
- Suite total ≥ 1663 PASS (perda de 2 testes obsoletos).

---

## FRENTE 3 — O2: Tile/Block Processing (estimativa 4-6h)

### Decisão (confirmada pelo usuário)

Aplicar tile/block **apenas** em `_simulate_positions_njit_cached`
(`forward.py:232`) — path single-TR legado. **NÃO** tocar
`_simulate_combined_prange_flat` (hot path desde v2.22.4) nesta sprint.

Justificativa:
- Prova de conceito isolada: menor risco de regressão Cenário E.
- Se paridade <1e-12 OK + ganho >5%, extensão ao hot path fica para v2.37
  com benchmarking dedicado.
- Cenário E (path quente em `simulate_multi`) **não muda** diretamente;
  benchmark protege contra regressão indireta via testes do `simulate()`.

### Implementação proposta

**Arquivos**:
- `geosteering_ai/simulation/forward.py` — adicionar `_simulate_positions_njit_cached_tiled`
- `geosteering_ai/simulation/config.py` — novo campo `tile_size: int = 4`, `use_tiled_positions: bool = False`
- `geosteering_ai/simulation/__init__.py` — exportar nova função (se necessário)
- `tests/test_simulation_v236_tile_block.py` (CRIAR — TDD primeiro)
- `benchmarks/bench_v236_tile_block.py` (CRIAR — 5 runs × 3 cenários)

**Estrutura alvo**:

```python
@njit(parallel=True, cache=True, nogil=True)
def _simulate_positions_njit_cached_tiled(
    positions_z, dz_half, r_half, dip_rad,
    n, rho_h, rho_v, esp, h_arr, prof_arr, eta,
    freqs_hz, krJ0J1, wJ0, wJ1,
    u_cache, s_cache, uh_cache, sh_cache,
    RTEdw_cache, RTEup_cache, RTMdw_cache, RTMup_cache, AdmInt_cache,
    H_tensor, z_obs, rho_h_at_obs, rho_v_at_obs,
    tile_size,
):
    n_positions = positions_z.shape[0]
    n_tiles = (n_positions + tile_size - 1) // tile_size
    for t in _prange(n_tiles):                    # prange OUTER apenas (KB-013)
        jstart = t * tile_size
        jend = min(jstart + tile_size, n_positions)
        for j in range(jstart, jend):              # serial INNER no tile
            # corpo idêntico a _simulate_positions_njit_cached
            ...
```

**Regras obrigatórias**:
- `parallel=True` apenas no dispatcher externo (consistente com v2.21 KB-013).
- `prange` apenas no loop de tiles (não no inner).
- `cache=True` + `nogil=True` obrigatórios.
- Paridade bit-exata vs `_simulate_positions_njit_cached` ANTES de qualquer benchmark
  (`np.array_equal(H_legacy, H_tiled) == True` em 10 modelos canônicos).

**Dispatcher**:
- Wrapper em `forward.py::simulate()` chama tiled se `cfg.use_tiled_positions and cfg.tile_size > 1`.
- Default OFF (`use_tiled_positions: bool = False`), promover True só após validação.

**Critérios de aceite**:
- 10+ testes paridade TILE vs legacy: `np.array_equal == True`.
- Paridade Fortran <1e-12 preservada (10/10 canônicos via `pytest -m fortran_parity`).
- Cenário E benchmark mediana ≥ 62.402 mod/h (threshold 90% de 69.336).
- Cenário A NÃO regride: ≥ 131.767 mod/h (threshold 90% de 146.408).
- Ganho relativo de tile_size=4 vs flat ≥ 5% em algum cenário (caso contrário, REVERT — não vale a complexidade).

**Pseudocódigo de teste** (`tests/test_simulation_v236_tile_block.py`):

```python
@pytest.mark.parametrize("tile_size", [1, 2, 4, 8])
def test_tile_block_parity_vs_legacy(tile_size, model_canonical):
    cfg_legacy = SimulationConfig(use_tiled_positions=False)
    cfg_tiled = SimulationConfig(use_tiled_positions=True, tile_size=tile_size)
    H_legacy = simulate(model_canonical, cfg_legacy).H_tensor
    H_tiled = simulate(model_canonical, cfg_tiled).H_tensor
    assert np.array_equal(H_legacy, H_tiled)
```

---

## FRENTE 4 — D5: Investigar regressão Cenário B (estimativa 2-3h)

**Hipótese central**: `cfg.use_flat_prange=True` (default desde v2.22.4)
recupera a perda do Cenário B (376k → 303k mod/h após KB-013 fix em v2.21).

**Procedimento**:
1. Benchmark side-by-side em `benchmarks/bench_v22_flat_prange.py` (já existe, scenario B 200 pos × 3 TR × 4 ang × 1 freq):
   ```bash
   python benchmarks/bench_v22_flat_prange.py --scenario B --runs 5
   ```
   (Confirmar interface: bench já aceita `--runs`? Senão, wrap em bash loop.)
2. Capturar mediana com `use_flat_prange=True` e `False`.
3. Validar 2 hipóteses:
   - **H1**: FLAT 4D recupera ≥ 376k mod/h em B → D5 fechado, documentar.
   - **H2**: FLAT 4D NÃO recupera → regressão é arquitetural; planejar v2.37.

**Saídas**:
- `docs/reports/v2.36_cenario_b_analysis_YYYY-MM-DD.md` com tabela comparativa.
- Atualizar `docs/PERFORMANCE_BASELINE.md` com nota sobre Cenário B (com ou sem regressão).
- Se H2 (perda persistente): item em `docs/ROADMAP.md` para Sprint v2.37 (especialização multi-freq+multi-TR).

**Não-objetivo desta sprint**: implementar especialização. Apenas medir e documentar.

---

## Arquivos a criar (lista mínima)

| Caminho                                                            | Tipo    | Tamanho estimado |
|:-------------------------------------------------------------------|:--------|:-----------------|
| `tests/test_simulation_v236_tile_block.py`                         | Teste   | ~150 LOC         |
| `tests/test_perf_baseline_h.py`                                    | Guard   | ~30 LOC          |
| `benchmarks/bench_v236_tile_block.py`                              | Bench   | ~120 LOC         |
| `docs/reports/v2.36_<YYYY-MM-DD>.md`                               | Relat.  | seguir template  |
| `docs/reports/v2.36_cenario_b_analysis_<YYYY-MM-DD>.md`            | Relat.  | ~200 LOC         |

## Arquivos a modificar

| Caminho                                                            | Frente | Tipo de mudança                                   |
|:-------------------------------------------------------------------|:-------|:--------------------------------------------------|
| `.claude/perf_baseline.json`                                       | D1, D2 | Preencher `E_n200_warm`; adicionar `H_n2_stress`   |
| `docs/PERFORMANCE_BASELINE.md`                                     | D1, D2 | Atualizar tabela + seção 2.1                      |
| `tests/conftest.py` (raiz)                                         | D3     | Adicionar bloco NUMBA_CACHE_DIR no topo            |
| `geosteering_ai/simulation/config.py`                              | D4, O2 | Remover `use_fastmath`; adicionar `tile_size`+`use_tiled_positions` |
| `tests/test_sprint_v224.py`                                        | D4     | Remover 2 testes obsoletos (linhas 174-198)        |
| `geosteering_ai/simulation/forward.py`                             | O2     | Adicionar `_simulate_positions_njit_cached_tiled` + dispatcher em `simulate()` |
| `docs/CHANGELOG.md`                                                | TODOS  | Entrada v2.36                                      |
| `CLAUDE.md`                                                        | TODOS  | Bump v2.35 → v2.36 cabeçalho                       |

---

## Fluxo obrigatório por frente (gates de qualidade)

```
1. pytest tests/ -v --tb=short -m "fortran_parity or not slow"   # ANTES
2. Implementar (TDD primeiro nas Frentes 3 e 4)
3. pytest tests/test_simulation_v236_tile_block.py -v            # NOVOS
4. pytest tests/ -v -m "fortran_parity"                          # PARIDADE
5. Benchmark 5 runs — reportar mediana + p25/p75 + std
6. Se paridade quebrar → REVERT imediato (git restore .)
7. /coderabbit:code-review review pós-sprint
8. Gerar docs/reports/v2.36_YYYY-MM-DD.md
9. Atualizar docs/CHANGELOG.md + CLAUDE.md
```

**Skills a invocar (na ordem)**:
1. `/geosteering-simulator-numba` — domínio Numba JIT
2. `/geosteering-perf-baseline` — anti-regressão throughput
3. `/geosteering-perf-reviewer` — benchmark pós-implementação (Haiku rápido)
4. `/geosteering-documentation` — CHANGELOG + relatório final
5. `/coderabbit:code-review` — revisão pós-sprint (obrigatório se O2 mexer no caminho crítico)

**Skills opcionais**:
- `/geosteering-orchestrator` — só se >5 arquivos simultâneos ou fan-out necessário
- `/geosteering-physics-reviewer` — se houver dúvida sobre paridade Fortran

---

## Commits granulares (5 commits, 1 por frente lógica)

```
feat(perf): v2.36 — medir E_n200_warm + baseline Cenário H [D1+D2]
fix(tests): v2.36 — NUMBA_CACHE_DIR em conftest.py raiz [D3]
refactor(config): v2.36 — remover use_fastmath inócuo [D4]
feat(sim): v2.36 — tile/block processing _simulate_positions [O2]
fix(sim): v2.36 — investigação Cenário B regression [D5]
docs(sm): v2.36 — relatório + CHANGELOG + version bump
```

---

## Critério global de conclusão da sprint

| ✓   | Critério                                                                         |
|:---:|:---------------------------------------------------------------------------------|
| [ ] | Paridade Fortran <1e-12 em 10/10 modelos canônicos                              |
| [ ] | `E_n200_warm.throughput_mod_h != null` em `perf_baseline.json`                  |
| [ ] | `H_n2_stress` adicionado em `perf_baseline.json`                                |
| [ ] | Suite total ≥ 1665 PASS / 0 FAIL                                                |
| [ ] | Cenário E benchmark ≥ 62.402 mod/h (threshold 90% de 69.336 cold)                |
| [ ] | Cenário A benchmark ≥ 131.767 mod/h (threshold 90% de 146.408)                   |
| [ ] | 0 referências a `use_fastmath` fora de testes de hard-coded fastmath             |
| [ ] | 0 CodeRabbit findings bloqueantes                                                |
| [ ] | `docs/reports/v2.36_YYYY-MM-DD.md` criado                                        |
| [ ] | `docs/CHANGELOG.md` atualizado com entrada v2.36                                 |
| [ ] | `CLAUDE.md` cabeçalho atualizado (v2.35 → v2.36)                                 |

---

## Verificação (run end-to-end)

```bash
# 0. Setup
source ~/Geosteering_AI_venv/bin/activate
cd /Users/daniel/Geosteering_AI

# 1. Sanity check anterior
pytest tests/ -v -m "not slow and not gui" --tb=short  # baseline pre-sprint

# 2. Smoke teste novos modules
pytest tests/test_simulation_v236_tile_block.py -v
pytest tests/test_perf_baseline_h.py -v

# 3. Paridade Fortran
pytest tests/ -v -m "fortran_parity" --tb=short
# Esperado: 10/10 PASS

# 4. Benchmark anti-regressão
geosteering-cli benchmark --scenario E --n 200    # ≥62.402 mod/h
geosteering-cli benchmark --scenario A --n 1      # ≥131.767 mod/h

# 5. Hook anti-regressão
SCENARIO=E N_MODELS=200 bash .claude/hooks/check-perf-regression.sh
# Esperado: PASS sem WARN

# 6. CodeRabbit (se O2 mexeu em forward.py)
/coderabbit:code-review
```

---

## Riscos & Mitigação

| Risco                                                              | Mitigação                                      |
|:-------------------------------------------------------------------|:-----------------------------------------------|
| Tile/block quebra paridade Fortran                                 | TDD: testes paridade ANTES do benchmark. Revert imediato. |
| Cenário B regressão é arquitetural (FLAT não recupera)             | D5 só MEDE — implementação fica em v2.37. Sem pressão para hack. |
| Remoção de `use_fastmath` quebra preset YAML externo               | Grep preventivo em `configs/*.yaml`; rollback simples. |
| `E_n200_warm` baseline varia >10% entre runs                       | 5 runs + mediana; documentar p25/p75 em PERFORMANCE_BASELINE.md |
| O2 não gera ganho ≥5%                                              | REVERT a flag; documentar em report (tile_size não recompensa overhead em n_pos pequeno) |
