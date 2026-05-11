# Planejamento Sprints v2.25 → v2.27 — Alinhamento com `analise_cenarios_otimizacao_simulador_numba.md`

**Documento de planejamento técnico**
**Versão**: 1.0 (2026-05-10, pós-auditoria multi-agente da Sprint v2.24)
**Autor**: Daniel Leal · **Coordenação**: Claude (Opus 4.7 multi-agente)
**Status**: PLANEJAMENTO (Sprints não executadas ainda)
**Referência fonte**: `docs/reference/analise_cenarios_otimizacao_simulador_numba.md`
(produzido em 2026-05-02 com base no código v2.21)

---

## 1. Contexto e Motivação

Após a Sprint v2.24, foi realizada uma auditoria multi-agente do estado do
simulador Python Numba JIT contra o documento `analise_cenarios_otimizacao_simulador_numba.md`.
A análise identificou:

1. **5 das 6 otimizações** propostas no documento ainda não estão totalmente
   implementadas (O1 parcial, O3 parcial, O4-O6 pendentes; só O2 está completa
   desde v2.22).
2. **Risco numérico de cancelamento catastrófico** em alta resistividade
   (ρ > 1000 Ω·m) foi profundamente investigado e uma **solução superior**
   à proposta original do documento foi identificada (ver §2 abaixo).
3. **Inconsistências documentais menores** da Sprint v2.24 foram corrigidas
   (13 vs 12 decoradores; 7 modelos canônicos / 10 testes vs "13/13").

Este documento define o escopo, sequência e critérios de aceitação para
as Sprints v2.25, v2.26 e v2.27, garantindo cobertura completa das
otimizações O1-O6 com segurança numérica máxima.

---

## 2. Estratégia para Cancelamento Catastrófico em Alta-ρ — Decisão Técnica

### 2.1 Proposta original do documento (§7.4, §7.6, §8.3)

```
1. Adicionar 3 modelos canônicos de alta ρ ao gate Fortran <1e-12:
   - carbonato_seco_5c (ρ_max = 5000 Ω·m)
   - evaporita_3c     (ρ_max = 100000 Ω·m)
   - gas_seco_8c      (ρ_max = 10000 Ω·m)
2. Implementar dual-mode `hmd_tiv_fast` / `hmd_tiv_safe` controlado
   por `cfg.use_fastmath` (opt-in com risco em alta ρ).
3. Se gate falhar: manter fastmath=False em hmd_tiv/vmd e fastmath=True
   apenas em hankel.py (já implementado).
```

### 2.2 Estratégia adotada (auditoria multi-agente 2026-05-10)

**Substituir o dual-mode por `fastmath` SELETIVO + pairwise summation manual.**

```python
# Novo decorador em dipoles.py — hmd_tiv e vmd:
_FASTMATH_SAFE = {'nnan', 'ninf', 'contract', 'arcp', 'afn'}
# Excluídos deliberadamente:
#   'reassoc' — quebra ordem de soma (cancelamento catastrófico em alta ρ)
#   'nsz'     — pode alterar sinal de Im(H) próximo de zero
@njit(cache=True, fastmath=_FASTMATH_SAFE)
def hmd_tiv(...):
    sum_Ktedz_J1 = _pairwise_sum_complex(Ktedz_J1)
    sum_Ktm_J1   = _pairwise_sum_complex(Ktm_J1)
    # ... idem nas 7 reduções (linhas 665-671)
```

### 2.3 Justificativa técnica

| Aspecto | Dual-mode (proposta original) | Selective fastmath + pairwise (adotado) |
|:--------|:------------------------------|:----------------------------------------|
| **Numba support** | Sim (decorador estático) | **Sim, nativo desde Numba 0.49** (PR #3847) |
| **Performance** | +8-15% por função (fastmath full) | **+6-10% por função** (perde apenas 2pp) |
| **Robustez alta ρ** | **REATIVO** — só detecta falha | **PROATIVO** — elimina causa-raiz |
| **Manutenção** | 2 versões de cada função (16 arquivos) | 1 versão única + helper pairwise |
| **Erro de soma** | O(n·ε) com `reassoc` ativo | **O(ε·log n)** = O(ε·log 201) ≈ O(ε·7.65) |
| **Risco residual** | Falha silenciosa em ρ não-coberto | Praticamente eliminado |

**Pesquisa bibliográfica que fundamenta a decisão**:

- [Numba issue #2923](https://github.com/numba/numba/issues/2923) +
  [Numba PR #3847](https://github.com/numba/numba/pull/3847) — fine-grained
  fastmath flags suportados nativamente
- [Numba Performance Tips](https://numba.readthedocs.io/en/stable/user/performance-tips.html)
- [Pairwise summation — Wikipedia](https://en.wikipedia.org/wiki/Pairwise_summation)
- [Numba issue #10412](https://github.com/numba/numba/issues/10412) — Numba
  NÃO usa pairwise em `np.sum` (precisamos implementar à mão)
- [USGS — Hybrid fast Hankel transform](https://www.usgs.gov/publications/hybrid-fast-hankel-transform-algorithm-electromagnetic-modeling)
- [werthmuller.org/research](https://werthmuller.org/research/)

### 2.4 Defesa em profundidade

Os **3 modelos canônicos de alta ρ** propostos no documento original são
**mantidos** como camada extra de segurança (`test_simulation_high_resistivity_parity.py`),
mas agora atingem <1e-12 **sem precisar** de modo dual — porque o flag
seletivo já preserva a ordem de soma.

---

## 3. Sprint v2.25 — Escopo Detalhado

### 3.1 Frente A — O3 Seguro (Selective Fastmath + Pairwise Summation)

**Arquivos a modificar**:

- `geosteering_ai/simulation/_numba/dipoles.py` — substituir
  `@njit(cache=True)` em `hmd_tiv` (L180) e `vmd` (L708) por
  `@njit(cache=True, fastmath=_FASTMATH_SAFE)` + inserir 9 chamadas
  `_pairwise_sum_complex` (7 em `hmd_tiv` L665-L671 + 2 em `vmd` L961-L962)
- `geosteering_ai/simulation/_numba/__init__.py` — adicionar helper
  `_pairwise_sum_complex` e constante `_FASTMATH_SAFE`
- `geosteering_ai/simulation/config.py` — `use_fastmath: bool = False`
  recebe `DeprecationWarning` (remoção planejada v2.27)
- `docs/known_bugs.md` — adicionar **KB-020**: "Proibição de `reassoc`
  e `nsz` em fastmath de `dipoles.py` para evitar cancelamento
  catastrófico em alta ρ"
- `.claude/anti-patterns.txt` — adicionar regex bloqueante para
  `fastmath=True` (booleano) em `dipoles.py` (forçar uso de dict seletivo)

### 3.2 Frente B — 3 Modelos Canônicos Alta-ρ

**Arquivo novo**: `tests/test_simulation_high_resistivity_parity.py`

```python
HIGH_RHO_MODELS = {
    "carbonato_seco_5c": {
        "rho_h": np.array([2.0, 50.0, 5000.0, 50.0, 2.0]),
        "rho_v": rho_h * 2.0,
        "esp":   np.array([10.0, 5.0, 20.0, 5.0]),
    },
    "evaporita_3c": {
        "rho_h": np.array([1.5, 1e5, 1.5]),
        "rho_v": rho_h.copy(),
        "esp":   np.array([5.0, 50.0]),
    },
    "gas_seco_8c": {
        "rho_h": np.array([2.0, 10.0, 200.0, 1500.0, 10000.0, 1500.0, 10.0, 2.0]),
        "rho_v": rho_h * 1.5,
        "esp":   np.array([5.0, 3.0, 2.0, 1.0, 2.0, 3.0, 5.0]),
    },
}
```

**Gate**: paridade Fortran <1e-12 nos 3 modelos × 4 cenários
(single-freq, multi-freq, multi-TR, multi-angle).

### 3.3 Frente C — Kong 61pt CLI (O5)

**Arquivos a modificar**:

- `geosteering_ai/cli/simulate.py` — adicionar `--filter {werthmuller,kong,anderson}`
  (default: `werthmuller`)
- `geosteering_ai/cli/benchmark.py` — idem + também adicionar `--seed`
  (corrige débito menor identificado na auditoria)
- `geosteering_ai/simulation/config.py` — exposed (já existe `hankel_filter`,
  só passar do CLI para `SimulationConfig`)
- Help text deve indicar ganho esperado: "Kong 61pt: 3.3× mais rápido,
  precisão ε~1e-10 (adequado para geração de dados de treinamento)"

**5 novos testes em `tests/test_cli_mvp.py`**:

- `test_cli_simulate_filter_werthmuller_default`
- `test_cli_simulate_filter_kong`
- `test_cli_simulate_filter_anderson`
- `test_cli_benchmark_filter_propagation`
- `test_cli_simulate_invalid_filter_errors`

### 3.4 Frente D — Limpeza Cache Numba Automatizada

**Ressalva do physics-reviewer v2.24**: caches Numba residuais da v2.23
(sem `cache=True` explícito) podem coexistir com a v2.24 (com `cache=True`),
gerando colisão de hash silenciosa.

**Solução**: `.claude/hooks/setup-environment.sh` agora detecta mismatch
entre versão instalada e versão do cache, executando:

```bash
find geosteering_ai/simulation/_numba -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null
find geosteering_ai/simulation/_numba -name '*.nbi' -delete 2>/dev/null
find geosteering_ai/simulation/_numba -name '*.nbc' -delete 2>/dev/null
```

Detecção via stash de versão em `~/.geosteering-ai-cache-version`.

### 3.5 Multi-Agent Workflow Sprint v2.25

```
Fase 1 — Implementação paralela (3 agentes):
  ├─ Agent A (Opus 4.7 extra-high): O3 Seguro em dipoles.py + helper pairwise
  ├─ Agent B (Sonnet 4.6 high): 3 modelos alta-ρ + 12 testes
  └─ Agent C (Sonnet 4.6 high): Kong CLI + cache cleanup hook

Fase 2 — Reviews paralelos:
  ├─ physics-reviewer (Opus 4.7): valida gate alta-ρ <1e-12
  ├─ perf-reviewer (Haiku 4.5): valida +6-10% (sem regressão Cenário E)
  └─ code-reviewer (Sonnet 4.6): valida pairwise correctness + DRY

Fase 3 — coderabbit:code-review (CLI plain) + fixes finais
```

### 3.6 Critérios de Aceitação

- [ ] Paridade Fortran <1e-12 nos 7 modelos atuais + 3 novos alta-ρ
- [ ] Performance +6-10% em Cenário E (mediana 5 runs, stdev <2%)
- [ ] Performance ≥3× em Cenário E com `--filter kong` vs `--filter werthmuller`
- [ ] Suite total: 1665 + ~30 novos = ~1695+ PASS / 295 SKIP / 0 FAIL
- [ ] KB-020 documentado e bloqueado por anti-patterns
- [ ] `use_fastmath` flag emite `DeprecationWarning` com referência a v2.27
- [ ] `setup-environment.sh` automatiza limpeza de cache entre versões
- [ ] 0 CodeRabbit findings críticos

---

## 4. Sprint v2.26 — Escopo Detalhado

### 4.1 Frente A — O1 Adaptive Thread Count Dinâmico

Detalhamento do design baseado na análise do agente performance:

**`geosteering_ai/simulation/config.py`** — novos campos:

```python
@dataclass(frozen=True)
class SimulationConfig:
    # ...
    adaptive_threads: bool = False  # opt-in, default OFF
    adaptive_min_tasks_per_thread: int = 4  # configurável (doc O1 sugere 4)
```

**`geosteering_ai/simulation/multi_forward.py:1010`** — helper privado:

```python
def _apply_adaptive_thread_mask(
    cfg: SimulationConfig, n_combos: int, n_pos: int, nf: int
) -> Tuple[int, int]:
    """Computa máscara dinâmica de threads e retorna (target, previous)."""
    if not cfg.adaptive_threads or n_pos <= 1:
        return numba.get_num_threads(), numba.get_num_threads()  # bypass
    if cfg.use_flat_prange:
        effective_tasks = n_combos * n_pos * nf
    else:
        effective_tasks = n_combos * n_pos
    max_threads = numba.get_num_threads()
    optimal = min(max_threads, max(1, effective_tasks // cfg.adaptive_min_tasks_per_thread))
    if optimal != max_threads:
        numba.set_num_threads(optimal)
        logger.debug("Adaptive: %d → %d threads (tasks=%d)", max_threads, optimal, effective_tasks)
    return optimal, max_threads
```

**Uso em `simulate_multi`** (logo antes do despacho prange):

```python
target, prev = _apply_adaptive_thread_mask(cfg, n_combos, n_pos, nf)
try:
    result = _simulate_combined_prange(...)
finally:
    if target != prev:
        numba.set_num_threads(prev)  # restaura mesmo em exceção
```

### 4.2 Ganho Esperado

| Cenário | n_pos | n_combos | nf | Tasks (FLAT) | Threads ótimas | Ganho |
|:--------|:-----:|:--------:|:--:|:------------:|:--------------:|:------|
| Cenário 1 | 10 | 1 | 1 | 10 | 2 (8C CPU) | **+30-60%** |
| Cenário 1 | 30 | 1 | 1 | 30 | 7 | +10-20% |
| Cenário 1 | 100 | 1 | 1 | 100 | 8 (cap) | ~0% |
| Cenário 2 multi-freq | 30 | 1 | 16 | 480 | 8 (cap) | ~0% |
| Cenário E | 600 | 1 | 1 | 600 | 8 (cap) | ~0% |

### 4.3 5 Novos Testes em `tests/test_adaptive_threads.py`

1. `test_adaptive_threads_disabled_by_default` — paridade bit-exata vs v2.25
2. `test_adaptive_threads_reduces_for_small_npos` — mock `set_num_threads`,
   valida que `n_pos=10` resulta em `set_num_threads(2)`
3. `test_adaptive_threads_caps_at_cfg_num_threads` — não ultrapassa cfg
4. `test_adaptive_threads_restore_on_exception` — try/finally restaura estado
5. `test_adaptive_threads_paridade_fortran` — paridade <1e-12 com `adaptive=True`

### 4.4 Esforço Estimado

- **Implementação**: 0.5 dia
- **Testes**: 0.5 dia
- **Benchmark + revisão multi-agente**: 0.5-1 dia
- **Total**: 1.5-2 dias
- **Risco**: BAIXO (opt-in, bypass robusto, mesmo kernel — paridade por construção)

### 4.5 Frente B (opcional) — Mais Modelos Canônicos Alta-ρ

Se a Sprint v2.25 demonstrar robustez do fastmath seletivo, considerar
expandir o conjunto de teste com mais variações (modelos com transição
abrupta de baixa→alta ρ, casos extremos de anisotropia, etc.).

---

## 5. Sprint v2.27 — Escopo Detalhado

### 5.1 Frente A — O4 Cache de Contexto (§8.4)

**Motivação**: Em geosteering em tempo real (inversão Newton-Raphson com
mesmo modelo geológico, atualizando apenas posições), o `precompute_common_arrays_cache`
é recomputado a cada chamada — overhead de 0.5ms × N iterações.

**Implementação**: cache LRU em `geosteering_ai/simulation/_numba/__init__.py`:

```python
_PRECOMPUTE_CACHE: OrderedDict[CacheKey, CachedArrays] = OrderedDict()
_CACHE_MAX_SIZE = 64  # configurável via SimulationConfig.precompute_cache_size

CacheKey = Tuple[float, Tuple[float, ...], bytes, bytes, bytes, str]
# (hordist, freqs_tuple, rho_h.tobytes(), rho_v.tobytes(), esp.tobytes(), filter_name)
```

**Ganho**: +30% em inversão iterativa / geosteering em tempo real.

### 5.2 Frente B — O6 F7 einsum (§8.6)

**Substituir loop Python em `apply_tilted_antennas`**:

```python
# ATUAL (loop):
for k, (beta, phi) in enumerate(tilted_configs):
    H_tilted[k] = cos(b)*H[...,8] + sin(b)*(cos(p)*H[...,6] + sin(p)*H[...,7])

# PROPOSTO (einsum):
W = np.stack([np.cos(betas), np.sin(betas)*np.cos(phis), np.sin(betas)*np.sin(phis)], axis=-1)
components = np.stack([H[..., 8], H[..., 6], H[..., 7]], axis=-1)
H_tilted = np.einsum('...j,kj->k...', components, W)
```

**Ganho**: +3× em F7 para `n_tilted > 5`. Impacto global <2% (F7 não é
gargalo), mas elimina código frágil.

### 5.3 Frente C — GUI Testing (pytest-qt)

**Motivação**: bugs de regressão na GUI do Simulation Manager são frequentes
("alteração em widget X quebra widget Y"). Não existe cobertura automatizada.

**Adições**:

- `pip install pytest-qt` em `[dev]` extras de `pyproject.toml`
- `tests/test_simulation_manager_gui.py` — 5 testes:
  1. `test_gui_opens_without_error`
  2. `test_scenario_e_button_does_not_change_default_workers`
  3. `test_pool_warmup_thread_completes_before_first_run`
  4. `test_pause_cancel_button_state_machine`
  5. `test_golden_path_scenario_e_throughput_above_50k`

**Considerar separação MVC** (extrair `SimulationManagerController`) se
escopo permitir — caso contrário, deferir para v2.28.

### 5.4 Frente D — Remoção `use_fastmath` Flag

Após 2 sprints com `DeprecationWarning` (v2.25 e v2.26), remover
definitivamente o campo `use_fastmath` de `SimulationConfig` e
qualquer referência documental.

### 5.5 Frente E — Consolidação Hook PT-BR

**Ressalva do perf-reviewer v2.24**: `check-ptbr-accentuation.sh` executa
~105 chamadas `grep` por edição. Consolidar em um único `grep -f patterns.txt`:

```bash
# patterns.txt (gerado de ptbr-words.txt):
\bconfiguracao\b
\bnao\b
\bfuncao\b
# ... 60 palavras

grep -wif patterns.txt "$FILE_PATH"
```

Redução esperada: 105 forks → 1 fork. Tempo de hook: ~100-300ms → ~10-30ms.

---

## 6. Cobertura das 6 Otimizações do Documento de Análise

| Otim. | Descrição | Status v2.24 | Sprint v2.25 | Sprint v2.26 | Sprint v2.27 |
|:-----:|:----------|:-------------|:------------|:------------|:------------|
| **O1** | Adaptive thread count | ⚠️ Parcial (estático no `__post_init__`) | — | ✅ Dinâmico runtime | — |
| **O2** | FLAT prange | ✅ Default desde v2.22.4 | — | — | — |
| **O3** | fastmath em hmd_tiv+vmd | ⚠️ Parcial (só geometry/rotation/hankel) | ✅ Seletivo + pairwise (superior a dual-mode) | — | — |
| **O4** | Cache de contexto | ❌ Não implementado | — | — | ✅ LRU por modelo |
| **O5** | Kong 61pt CLI/GUI | ❌ Não implementado | ✅ CLI flag `--filter` | — | (GUI em v2.27 se MVC pronto) |
| **O6** | F7 einsum | ❌ Não implementado | — | — | ✅ einsum vetorizado |

**Cobertura total ao final de v2.27**: 6/6 otimizações entregues.

---

## 7. Riscos e Mitigações

| Risco | Probabilidade | Impacto | Mitigação |
|:------|:-------------:|:-------:|:----------|
| Pairwise summation introduz erro >1e-12 | Baixa | Crítico | Gate Fortran <1e-12 obrigatório antes do merge; testes unitários do helper |
| Selective fastmath não dá +6% | Média | Médio | Aceitar +4-5% se >0; documentar no relatório |
| Cache cleanup quebra desenvolvimento local | Baixa | Médio | Hook idempotente, apenas roda na 1ª invocação após mudança de versão |
| Adaptive threads regride performance em cenários grandes | Baixa | Médio | Default OFF; promover apenas se ganho >0 em ≥1 cenário e regressão <2% em todos |
| GUI pytest-qt incompatível com headless CI | Média | Baixo | Marca testes como `@pytest.mark.gui`; CI usa `--ignore` ou `xvfb-run` |
| Kong 61pt produz erro físico >1e-10 em alta ρ | Baixa | Médio | Já documentado §7.5 do doc análise; warning na CLI quando ρ_max > 10kΩ·m + Kong |

---

## 8. Decisões Pendentes (Não Bloqueantes)

1. **Quando promover `use_flat_prange` para sempre ON (sem flag)?**
   — proposta: v2.28+, após 4 sprints (v2.22.4 → v2.27) sem regressão
2. **`adaptive_threads` default ON em v2.27?**
   — depende de validação em v2.26 + 1 semana de uso opt-in
3. **MVC do Simulation Manager**
   — escopo de extração do `SimulationManagerController` ainda não definido;
   pode ser entregue como v2.27 Frente C completa ou como pré-requisito de v2.28
4. **`fastmath` global `=True` (deprecated)**
   — manter `DeprecationWarning` por 2 sprints (v2.25, v2.26) antes de
   remover em v2.27 (sequência aprovada por physics-reviewer)

---

## 9. Documentação a Atualizar Pós-Execução de Cada Sprint

Para cada Sprint v2.25, v2.26, v2.27 que for executada:

- [ ] `CLAUDE.md:16` — linha do Simulation Manager
- [ ] `docs/ROADMAP.md` — tabela de marcos
- [ ] `docs/CHANGELOG.md` — nova entrada
- [ ] `docs/reports/v2.{N}_{YYYY-MM-DD}.md` — relatório técnico detalhado
- [ ] `MEMORY.md` — entrada de Sprint
- [ ] `docs/reference/analise_cenarios_otimizacao_simulador_numba.md` —
  marcar otimização como "✅ IMPLEMENTADA em vX.Y" no roadmap §9.1
- [ ] `docs/known_bugs.md` — novos KBs se aplicável
- [ ] `.claude/anti-patterns.txt` — novos padrões se aplicável

---

## 10. Sequência Recomendada de Execução

```
1. Sprint v2.25 (O3 Seguro + Alta-ρ + Kong CLI)        ~3-4 dias
   ↓ Critério de gate: paridade Fortran <1e-12 em 10 modelos
2. Sprint v2.26 (O1 Adaptive Dinâmico)                  ~1.5-2 dias
   ↓ Critério de gate: paridade preservada + ganho ≥20% em Cenário 1 com n_pos=10
3. Sprint v2.27 (O4 Cache + O6 einsum + GUI + cleanup)  ~3-4 dias
   ↓ Critério de gate: 6/6 otimizações entregues
```

**Tempo total**: ~8-10 dias úteis para fechar cobertura completa do
documento de análise.

---

*Documento gerado em 2026-05-10 como artefato da auditoria multi-agente
pós-Sprint v2.24. Reorganiza o roadmap das Sprints v2.25-v2.27 para
garantir 100% de cobertura das 6 otimizações (O1-O6) do documento
`analise_cenarios_otimizacao_simulador_numba.md` com segurança numérica
superior à proposta original.*
