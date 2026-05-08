# Sprint v2.22 FLAT prange — Análise de Impacto e Plano de Ação

| Campo | Valor |
|:------|:------|
| **Versão alvo** | v2.22 (Sprint 22.1) |
| **Data** | 2026-05-08 |
| **Branch base** | `main` (após Etapa 1.5) |
| **Branch a criar** | `feat/simulation-manager-v2.22-flat-prange` |
| **Documento de referência** | `docs/reference/analise_cenarios_otimizacao_simulador_numba.md` |
| **Performance baseline** | v2.21 — Cenário E: 122k mod/h, Cenário B: 303k mod/h |
| **Meta v2.22** | Cenário B 303k → ≥600k mod/h, Cenário F (nf=4) +30%, sem regressão em A/E |

---

## Sumário Executivo

A "Sprint v2.22 FLAT prange" é uma refatoração do **kernel paralelo Numba** (`_simulate_combined_prange` em `geosteering_ai/simulation/forward.py`) que **colapsa quatro dimensões de paralelismo** (`nTR × nAng × n_pos × nf`) em **um único `prange`** plano, eliminando o aninhamento residual `range(nf)` serial dentro de `_fields_in_freqs_kernel_cached`.

A operação **NÃO afeta o backend JAX nem o simulador Fortran** — é uma mudança cirúrgica e isolada no caminho `@njit` Numba. As três perguntas do usuário são respondidas conforme:

1. **Sim**, a operação é exclusivamente no simulador Python Numba JIT (kernel Numba CPU).
2. **Nenhum impacto direto** no JAX GPU — o backend JAX já paraleliza naturalmente via `vmap` aninhado, mas há **lições conceituais transferíveis** para alinhar a estratégia entre os dois backends.
3. As **9 otimizações catalogadas** (O1–O6 + 3 deferred) têm cronograma definido nas versões v2.22 a v2.25, com a O2 (FLAT prange) sendo a primeira a ser executada por ser a **mais impactante e de menor risco**.

---

## 1. Pergunta 1 — A operação será feita no simulador Python Numba JIT?

### 1.1 Resposta direta

**Sim, exclusivamente no simulador Python Numba JIT.** A modificação é **isolada** ao caminho `@njit` em `geosteering_ai/simulation/forward.py` e ao kernel auxiliar em `geosteering_ai/simulation/_numba/kernel.py`.

### 1.2 Escopo cirúrgico — exatamente o que muda

```
geosteering_ai/simulation/
├── forward.py                  ← MODIFICA: novo kernel _simulate_combined_prange_flat
├── _numba/
│   ├── kernel.py               ← TOQUE MÍNIMO: opção de receber (ci, i_f) já decompostos
│   ├── propagation.py          ← INTOCADO
│   ├── dipoles.py              ← INTOCADO
│   ├── hankel.py               ← INTOCADO
│   ├── geometry.py             ← INTOCADO
│   └── rotation.py             ← INTOCADO
├── _jax/                       ← TOTALMENTE INTOCADO (todos os 9 arquivos)
│   ├── forward_pure.py         ← INTOCADO
│   ├── multi_forward.py        ← INTOCADO
│   ├── kernel.py               ← INTOCADO
│   ├── dipoles_native.py       ← INTOCADO
│   ├── dipoles_unified.py      ← INTOCADO
│   ├── propagation.py          ← INTOCADO
│   ├── geometry_jax.py         ← INTOCADO
│   ├── hankel.py               ← INTOCADO
│   └── rotation.py             ← INTOCADO
└── multi_forward.py            ← TOQUE MÍNIMO: dispatcher escolhe FLAT vs legado
```

### 1.3 Anatomia da mudança (estrutural)

**Estado atual (v2.21):**

```
@njit(parallel=True, cache=True, nogil=True)
def _simulate_combined_prange(...):
    n_total = n_combos * n_pos        # nTR × nAng × n_pos
    for k in prange(n_total):         # PARALELO (3 dimensões)
        ...
        cH = _fields_in_freqs_kernel_cached(...)   # ← dentro: range(nf) SERIAL!
        H_tensor[i_tr, i_ang, j, :, :] = cH
```

Problema: `_fields_in_freqs_kernel_cached` itera `for i_f in range(nf)` **serialmente** dentro de cada tarefa paralela, deixando potencial de paralelismo `nf` não explorado e gerando desbalanceamento quando `nf > 1` e `n_pos` é baixo.

**Estado proposto (v2.22 FLAT):**

```
@njit(parallel=True, cache=True, nogil=True)
def _simulate_combined_prange_flat(...):
    n_total = n_combos * n_pos * nf   # ← nf incluído no FLAT
    for k in prange(n_total):         # PARALELO (4 dimensões)
        # Decomposição de índice:
        i_combo = k // (n_pos * nf)
        rem     = k %  (n_pos * nf)
        j       = rem // nf
        i_f     = rem %  nf

        # Trabalha em UMA frequência por iteração (ao invés de nf):
        camad_t, camad_r = find_layers_tr(...)
        common_factors(...)               # apenas para i_f
        Hx, Hy, Hz = hmd_tiv(...)         # apenas para i_f
        Hx_v, Hy_v, Hz_v = vmd(...)       # apenas para i_f
        tH = rotate_tensor(...)
        H_tensor[i_tr, i_ang, j, i_f, :] = flatten_9(tH)
```

### 1.4 Por que é seguro

| Risco | Mitigação |
|:------|:----------|
| Quebra de paridade Fortran | Mesma física: cada `(i_combo, j, i_f)` produz **bit-exatamente** o mesmo `tH` que o caminho atual; apenas a ordem de despacho muda |
| Race condition | Cada tarefa `k` escreve em `H_tensor[i_tr, i_ang, j, i_f, :]` — slice **disjunto** entre tarefas → sem conflito |
| Redundância em `find_layers_tr` | `O(log n)` ≈ 50ns vs `hmd_tiv+vmd` ≈ 200μs → overhead 0,025% (desprezível) |
| Cache miss | Cache `u_unique[ci, i_f]` permanece imutável; cada tarefa lê uma slice 2D `(npt, n)` — padrão de acesso semelhante ao atual |
| Regressão Cenário E (nf=1) | Quando `nf=1`, FLAT reduz a `prange(n_combos × n_pos × 1)` ≡ `prange(n_combos × n_pos)` (idêntico ao atual) → sem regressão estrutural |

### 1.5 Critérios de aceite (do roadmap §9.2)

```
1. Cenário E (nf=1, n_pos=600):  sem regressão vs v2.21 (≥120 000 mod/h)
2. Cenário F (nf=4, n_pos=600):  ≥1.3× vs v2.21
3. Cenário G2 (nf=4, n_pos=30):  ≥2.5× vs v2.21
4. Cenário B (nf=1, multi-TR/Ang): ≥2× vs v2.21 (303k → ≥600k)
5. Paridade Fortran <1e-12 em 7 modelos canônicos
6. Suite pytest: 0 regressão (152+ testes simulation)
```

---

## 2. Pergunta 2 — Quais serão os impactos futuros no simulador JAX GPU?

### 2.1 Resposta direta

**Impacto direto: nenhum.** O backend JAX GPU é completamente independente do caminho `@njit` Numba CPU. Os dois compartilham apenas a definição da física (em arquivos Python puros) e os filtros de Hankel `.npz`.

**Impacto indireto: aprendizado arquitetural transferível.** A lição "FLAT é melhor que aninhado" se aplica também ao JAX, mas com nomenclatura diferente: `vmap` aninhado vs `vmap` plano.

### 2.2 Por que o JAX é independente

```
Pipeline JAX (caminho atual):
─────────────────────────────────────────────────────────────────────────
simulate_multi(cfg) com cfg.use_jax=True
  ↓
geosteering_ai/simulation/_jax/multi_forward.py::simulate_multi_jax()
  ↓
[2 estratégias coexistem desde v1.5.0+]
  │
  ├── jax_strategy="bucketed" (default)  ← Sprint 7.x
  │   └── _forward_bucket_jit  → vmap sobre n_pos (1 XLA program por bucket)
  │
  ├── jax_strategy="unified" (opt-in)   ← Sprint 10 Phase 2 (v1.5.0)
  │   └── _forward_pure_jax_unified_impl → 1 XLA program total (44× menos compilações)
  │
  └── jax_vmap_real=True (opt-in)       ← Sprint 12 (v1.6.0)
      └── _simulate_multi_jax_vmap_real  → vmap aninhado real (iTR, iAng) flat
```

### 2.3 Mapeamento conceitual entre Numba FLAT e JAX

| Conceito | Numba (CPU) | JAX (GPU) |
|:---------|:------------|:----------|
| Paralelismo plano | `prange(n_combos × n_pos × nf)` | `vmap` aplicado a array flat de shape `(n_total,)` |
| Custo de aninhamento | Overhead fork/join scheduler (~150μs/invocação) | Compilação XLA fragmentada (1 program por configuração) |
| Estado v2.21 | 3 dimensões FLAT, `nf` serial | `vmap` × `vmap` aninhado (já paraleliza `nf`) |
| Estado v2.22 (proposto) | 4 dimensões FLAT, `nf` paralelo | `jax_vmap_real=True` já implementa equivalente |

**Conclusão:** O JAX **já tem o equivalente do FLAT prange implementado** via `_simulate_multi_jax_vmap_real` (Sprint 12, v1.6.0). A Sprint v2.22 traz o backend Numba CPU ao **mesmo patamar arquitetural** do JAX GPU.

### 2.4 Impactos futuros indiretos (positivos)

#### 2.4.1 Convergência de design entre backends

Após v2.22, ambos os backends terão **paralelismo plano sobre 4 dimensões**:
- Numba CPU: `prange(nTR × nAng × n_pos × nf)` em CPU multi-core
- JAX GPU: `vmap` aninhado real sobre `(iTR, iAng, n_pos, nf)` em GPU SIMT

Isso simplifica:
- **Paridade entre backends** (`tests/test_simulation_jax_*.py`)
- **Documentação unificada** (uma única descrição de paralelismo)
- **Manutenção** (mudanças futuras seguem padrão similar)

#### 2.4.2 Habilita comparação direta de desempenho

Hoje: comparar Numba CPU vs JAX GPU é difícil porque os modelos de paralelismo são diferentes (Numba paraleliza só 3 dim, JAX 4 dim).

Pós-v2.22: ambos paralelizam 4 dimensões → **benchmark cabeça-a-cabeça** justo.

#### 2.4.3 Reuso de validação cruzada

A Sprint 12 (`jax_vmap_real`) provou que paralelizar `(iTR, iAng)` flat tem paridade `0.000e+00` com loop Python externo. A Sprint v2.22 pode **reaproveitar essa validação** como referência: se Numba FLAT bate com Numba não-FLAT bit-a-bit, e JAX já bate com Numba <1e-10, então JAX bit-a-bit com Numba FLAT está garantido.

### 2.5 Riscos JAX a monitorar

```
Riscos para o JAX (NÃO causados por v2.22, mas a observar):

1. SE futuramente unificarmos as 4 dimensões em UM SÓ vmap (1D flat),
   isso aumenta o tamanho do programa XLA → potencial impacto em
   compilação inicial (~30s vs ~5s atual).
   Mitigação: manter ambas as estratégias (bucketed + flat-1D) com flag.

2. SE a estratégia Numba FLAT inspirar uma estratégia "FLAT JAX"
   (1 vmap sobre n_total elementos), isso potencialmente reduz overhead
   de XLA dispatch em GPU pequena, MAS aumenta uso de memória (cópia
   completa do cache para cada elemento).
   Mitigação: medir antes de implementar.

3. NÃO há plano atual de mudar o JAX por causa da v2.22.
   O JAX continua sua trilha independente: Sprint 13 (XLA fusion check),
   Sprint 14 (mixed precision GPU), etc.
```

### 2.6 Compatibilidade com os 8 cenários JAX C1–C8 (do documento de arquitetura)

O documento `arquitetura_multiagente_geosteering_ai_aprofundamento_2026-05-02.md` (§Parte IV, decisão "JAX 8 cenários") catalogou 8 cenários de aplicação do backend JAX:

| Cenário JAX | Descrição | Afetado pela v2.22 (Numba) ? |
|:------------|:----------|:----------------------------:|
| C1: Sample-level forward | Inversão DL on-the-fly | Não (caminho JAX puro) |
| C2: Differentiable forward | Loss físico em treino | Não (`jax.grad` em JAX) |
| C3: Multi-position vmap | Sweep z em 1 modelo | Não (vmap nativo) |
| C4: Multi-frequency vmap | Sweep f em 1 modelo | Não (vmap nativo) |
| C5: Multi-model batch | 100k modelos via pmap | Não (pmap distribui em GPU) |
| C6: Hybrid CPU+GPU | Numba treino + JAX inferência | **Indiretamente:** treino mais rápido após v2.24 (Kong 61pt) |
| C7: Real-time inversion | LWD streaming GPU | Não (caminho JAX puro) |
| C8: Optuna search GPU | Hyperparameter tune | Não (caminho JAX puro) |

**Conclusão:** Os 8 cenários JAX evoluem de forma **paralela e independente** ao roadmap Numba.

---

## 3. Pergunta 3 — Status das otimizações em `analise_cenarios_otimizacao_simulador_numba.md`

### 3.1 Resposta direta

O documento `docs/reference/analise_cenarios_otimizacao_simulador_numba.md` (1684 linhas, gerado em 2026-05-02) cataloga **6 otimizações principais (O1–O6)** + **3 ações estruturais** (benchmarks, modelos canônicos alta-ρ, pré-cômputo Hankel avançado).

**Todas estão previstas para implementação na nova arquitetura**, distribuídas pelas versões **v2.22 a v2.25** do simulador, **sequencialmente** conforme o §9.1 do documento. A FLAT prange (O2) é a **primeira da fila** porque tem **maior impacto e menor risco**.

### 3.2 Catálogo completo das 9 otimizações

#### 3.2.1 O1 — Adaptive Thread Count para `n_pos` baixo

| Item | Detalhe |
|:-----|:--------|
| Versão alvo | **v2.23 Sprint 23.2** |
| Onde aplica | `simulate_multi()` em `multi_forward.py` (despacho) |
| Arquivos modificados | `multi_forward.py` (~20 LOC) |
| Cenários beneficiados | 1, 2, 3, 4 (com `n_pos < 50`) |
| Ganho esperado | +10–40% (eliminação de threads ociosas) |
| Risco | Baixo (overhead de `set_num_threads` ~1μs) |
| Tempo estimado | 1 dia |
| Status | **Programada — NÃO iniciada** |

```python
# Pseudocódigo da implementação:
effective_tasks = n_combos * n_pos * nf
optimal_threads = min(num_threads, max(1, effective_tasks // 4))
if optimal_threads != num_threads:
    numba.set_num_threads(optimal_threads)
    try:
        result = _simulate_combined_prange_flat(...)
    finally:
        numba.set_num_threads(num_threads)
```

#### 3.2.2 O2 — FLAT prange (`nf × nTR × nAng × n_pos`) ⭐ **SPRINT v2.22**

| Item | Detalhe |
|:-----|:--------|
| Versão alvo | **v2.22 Sprint 22.1** |
| Onde aplica | `_simulate_combined_prange` em `forward.py` |
| Arquivos modificados | `forward.py` (~150 LOC novos), `multi_forward.py` (toque), `kernel.py` (toque) |
| Cenários beneficiados | 2, 6, 7, 8 (todos com `nf > 1`) + B (multi-TR/Ang) |
| Ganho esperado | +1.3× a +2.9× (depende de `nf` e `n_pos`) |
| Risco | Médio (mudança em hot path; mitigado por testes de paridade) |
| Tempo estimado | 2–3 dias |
| Status | **Próxima a executar** |

#### 3.2.3 O3 — `fastmath=True` em `hmd_tiv` e `vmd`

| Item | Detalhe |
|:-----|:--------|
| Versão alvo | **v2.23 Sprint 23.1** |
| Onde aplica | `_numba/dipoles.py` (decoradores) |
| Arquivos modificados | `dipoles.py` (~10 LOC, dual-mode) |
| Cenários beneficiados | Todos |
| Ganho esperado | +5–10% global (8–15% nos kernels) |
| Risco | Médio (validação obrigatória em alta resistividade) |
| Tempo estimado | 1 dia (+ benchmark gate) |
| Status | **Programada — depende de v2.22** |

```python
# dipoles.py — proposta dual-mode:
@njit(fastmath=False)   # default (paridade <1e-12)
def hmd_tiv_precise(...): ...

@njit(fastmath=True)    # opt-in via cfg.use_fastmath
def hmd_tiv_fast(...):  ...

# em SimulationConfig:
use_fastmath: bool = False  # True = +8-15%, ε~1e-10
```

#### 3.2.4 O4 — Cache de contexto para tempo-real

| Item | Detalhe |
|:-----|:--------|
| Versão alvo | **v2.25 Sprint 25.1** |
| Onde aplica | Novo módulo `geosteering_ai/simulation/_cache_store.py` |
| Arquivos modificados | Novo (~150 LOC) + integração em `multi_forward.py` |
| Cenários beneficiados | Inversão iterativa (mesmo modelo, posições atualizadas) |
| Ganho esperado | +30% em geosteering Look-Ahead em tempo real |
| Risco | Alto (gerenciamento de memória LRU, hashing de modelo) |
| Tempo estimado | 2 dias |
| Status | **Programada — depende de v2.23** |

#### 3.2.5 O5 — Exposição clara de Kong 61pt para treinamento

| Item | Detalhe |
|:-----|:--------|
| Versão alvo | **v2.24 Sprint 24.1** |
| Onde aplica | GUI Simulation Manager + CLI `bench_v214_numba.py` |
| Arquivos modificados | `gui/widgets/parameters_panel.py`, `cli/parameters.py` |
| Cenários beneficiados | Geração de datasets de treinamento |
| Ganho esperado | +230% throughput de treinamento (3.3× rápido) |
| Risco | Baixo (já implementado; falta UI/discoverability) |
| Tempo estimado | 4h |
| Status | **Programada — UI work pendente** |

#### 3.2.6 O6 — Vetorização avançada de F7 com `np.einsum`

| Item | Detalhe |
|:-----|:--------|
| Versão alvo | **v2.25 Sprint 25.2** |
| Onde aplica | `multi_forward.py::apply_tilted_antennas()` |
| Arquivos modificados | `multi_forward.py` (~30 LOC) |
| Cenários beneficiados | F7 com `n_tilted > 5` |
| Ganho esperado | +3× em F7 (impacto global <2%) |
| Risco | Baixo |
| Tempo estimado | 2h |
| Status | **Programada — baixa prioridade** |

#### 3.2.7 Ações estruturais complementares

| Ação | Versão | Tempo |
|:-----|:------:|:-----:|
| Benchmarks F, G, G2, H, I, J, K_carb, K_evap, X | **v2.22 Sprint 22.2** | 4h |
| Modelos canônicos alta-ρ (carbonato, evaporita) | **v2.22 Sprint 22.3** | 3h |
| Pré-cômputo Hankel TE/TM avançado | **v2.24 Sprint 24.2** | 3–5 dias |

### 3.3 Tabela consolidada — Roadmap Numba CPU 2026

```
╔════════╦═══════════╦═══════════════════════════════════════╦══════════════╗
║ Versão ║ Sprint    ║ Otimização                            ║ Tempo        ║
╠════════╬═══════════╬═══════════════════════════════════════╬══════════════╣
║ v2.22  ║ 22.1      ║ O2 — FLAT prange (nf flat)           ║ 2-3 dias     ║
║ v2.22  ║ 22.2      ║ Benchmarks Cenários F/G/H/I/J/K      ║ 4h           ║
║ v2.22  ║ 22.3      ║ Modelos canônicos alta-ρ             ║ 3h           ║
╠════════╬═══════════╬═══════════════════════════════════════╬══════════════╣
║ v2.23  ║ 23.1      ║ O3 — fastmath em dipoles             ║ 1 dia        ║
║ v2.23  ║ 23.2      ║ O1 — adaptive thread count            ║ 1 dia        ║
╠════════╬═══════════╬═══════════════════════════════════════╬══════════════╣
║ v2.24  ║ 24.1      ║ O5 — Kong 61pt UI/CLI exposure        ║ 4h           ║
║ v2.24  ║ 24.2      ║ Pré-cômputo Hankel avançado           ║ 3-5 dias     ║
╠════════╬═══════════╬═══════════════════════════════════════╬══════════════╣
║ v2.25  ║ 25.1      ║ O4 — cache de contexto inv. iterativa ║ 2 dias       ║
║ v2.25  ║ 25.2      ║ O6 — F7 einsum vetorizado             ║ 2h           ║
╚════════╩═══════════╩═══════════════════════════════════════╩══════════════╝

Total acumulado: ~3 semanas de trabalho efetivo
Projeção throughput Cenário E: 122k → 165k mod/h (+35%)
Projeção throughput Cenário F: 100k → 220k mod/h (+120%, multi-freq)
Projeção throughput Cenário B: 303k → 600k+ mod/h (+98%, FLAT)
```

### 3.4 Como cada otimização se conecta com a nova arquitetura multiagente

O documento `arquitetura_multiagente_geosteering_ai_aprofundamento_2026-05-02.md` (§55, §66) prevê **dois agentes especializados** que orquestram este roadmap:

#### 3.4.1 Agente `numba-jit-engineer` (futuro, Etapa 2.1.3)

Responsabilidades específicas para cada otimização:
- O2 (FLAT prange): refatoração + benchmark + commit message
- O3 (fastmath): aplicação dual-mode + gate de paridade
- O1 (adaptive threads): heurística + tunning empírico
- Coordena com `numba-validator` (paridade Fortran)

#### 3.4.2 Agente `numba-validator` (futuro, Etapa 2.1.3)

Responsabilidades:
- Roda paridade Fortran <1e-12 após cada PR
- Roda 152+ testes simulation antes de aprovar merge
- Valida 3 modelos alta-ρ (carbonato, evaporita)
- Bloqueia merge se qualquer regressão detectada

#### 3.4.3 Hook futuro `numba-perf-baseline.sh` (Etapa 1.5+)

Já mencionado no roadmap pós-Etapa 1.5, mas **deferred**:
- PreToolUse: captura throughput baseline antes de PR
- PostToolUse: compara com baseline e bloqueia se regressão >5%
- Funciona junto com o `run-fortran-parity.sh` smart já em produção

#### 3.4.4 MCP Server `numba-profiler` (futuro, Etapa 4.2)

Auxiliar a longo prazo:
- Stdio MCP server local
- Wrapper de `numba.profiler` + `cProfile` + `linux perf`
- Expõe ferramenta `profile_kernel(scenario, n_runs)` ao agente
- Ajuda a identificar otimizações futuras (O7+)

### 3.5 Por que iniciar com FLAT prange (justificativa técnica)

```
Critério               │ O1   │ O2   │ O3   │ O4   │ O5   │ O6  │
═══════════════════════╪══════╪══════╪══════╪══════╪══════╪═════╡
Impacto na meta v2.22  │  +   │ ███  │  -   │  -   │  -   │  -  │
Risco de paridade      │  -   │  +   │ ███  │  +   │  -   │  -  │
Esforço (dias)         │ 1.0  │ 2-3  │ 1.0  │ 2.0  │ 0.5  │ 0.3 │
Bloqueio para outras   │  -   │ ✓✓✓  │  -   │  -   │  -   │  -  │
Validação reusável     │ baixa│ alta │ alta │ baixa│ baixa│ baixa│
═══════════════════════╪══════╪══════╪══════╪══════╪══════╪═════╡
Decisão                │ 2ª   │ 1ª   │ 3ª   │ 6ª   │ 4ª   │ 5ª  │

O2 vence porque:
1. Maior impacto na meta v2.22 (Cenário B 303k → 600k)
2. Habilita melhor mensuração das demais (benchmarks ficam consistentes)
3. Padroniza arquitetura paralela (igual ao JAX vmap_real)
4. Validação Fortran <1e-12 é direta (mesma física, ordem diferente)
5. Risco é confinado a 1 função (rollback trivial via flag)
```

---

## 4. Plano de Execução — Sprint v2.22 FLAT prange

### 4.1 Estrutura de commits (proposta)

```
Branch: feat/simulation-manager-v2.22-flat-prange (a partir de main)

Commit 1: feat(sim): novo kernel _simulate_combined_prange_flat
  - forward.py (+150 LOC)
  - kernel.py (~10 LOC) opção de receber cache slice por (ci, i_f)
  - Testes paridade Numba: novo == antigo bit-exato

Commit 2: feat(sim): dispatcher escolhe FLAT vs legado
  - multi_forward.py (~20 LOC)
  - SimulationConfig.use_flat_prange: bool = True (default novo)
  - Fallback: cfg.use_flat_prange=False reverte ao caminho v2.21

Commit 3: test(sim): paridade Fortran + 7 modelos canônicos
  - tests/test_simulation_v22_flat_prange.py (+200 LOC)
  - Inclui Cenário B, F, G2, J alta-ρ
  - Gate <1e-12 em todos

Commit 4: bench(sim): Cenários F, G, H, I, J, K_carb, K_evap, X
  - benchmarks/bench_v22_flat_prange.py (+150 LOC)
  - Tabela markdown gerada com t_atual, t_flat, speedup

Commit 5: docs(sim): relatório técnico v2.22
  - docs/reports/v2_22_flat_prange_2026-05-XX.md
  - CHANGELOG.md entrada
  - MEMORY.md pointer
  - Atualiza analise_cenarios_otimizacao_simulador_numba.md (status v2.22 → DONE)
```

### 4.2 Critérios de aceite (do roadmap §9.2)

```
GATE OBRIGATÓRIO antes de merge → main:

1. ✅ Cenário E (nf=1, n_pos=600):
   throughput >= 120 000 mod/h  (sem regressão)

2. ✅ Cenário F (nf=4, n_pos=600):
   throughput >= 1.30 × baseline (atual ~100k mod/h)
   → meta >= 130k mod/h

3. ✅ Cenário G2 (nf=4, n_pos=30):
   throughput >= 2.50 × baseline (atual ~30k mod/h)
   → meta >= 75k mod/h

4. ✅ Cenário B (nf=1, multi-TR/Ang):
   throughput >= 600 000 mod/h
   (atual v2.21: 303k → meta v2.22: 600k+)

5. ✅ Paridade Fortran <1e-12:
   - 7 modelos canônicos
   - 3 modelos alta-ρ (carbonato, evaporita, mista)

6. ✅ Suite pytest:
   - 152+ testes simulation: 0 regressão
   - 11 testes test_known_bugs.py: 11/11 PASS, 0 XFAIL

7. ✅ /code-review:
   - 0 findings ALTA/CRÍTICA
   - WARN aceitos com justificativa
```

### 4.3 Estimativa de tempo

| Fase | Atividade | Tempo |
|:----:|:----------|:-----:|
| 1 | Branch setup + análise detalhada do código atual | 1h |
| 2 | Implementação `_simulate_combined_prange_flat` | 3-4h |
| 3 | Dispatcher + flag `use_flat_prange` em config | 1h |
| 4 | Testes paridade Numba (FLAT vs legacy bit-exato) | 2h |
| 5 | Benchmarks F, G2, B + gate Cenário E | 2-3h |
| 6 | Validação Fortran <1e-12 (7 modelos canônicos) | 1-2h |
| 7 | Modelos alta-ρ (3 cenários novos) | 2-3h |
| 8 | /code-review + correções | 1-2h |
| 9 | Relatório técnico + CHANGELOG + MEMORY | 1-2h |
| 10 | Merge para main + tag v2.22 | 30min |
| | **TOTAL** | **2-3 dias** |

### 4.4 Riscos e mitigações

| Risco | Probabilidade | Impacto | Mitigação |
|:------|:-------------:|:-------:|:----------|
| Paridade Fortran quebra | Baixa | Alto | Reuso de testes existentes; commit reversível |
| Regressão Cenário E (nf=1) | Média | Médio | `use_flat_prange=False` mantém caminho legado |
| Numba não otimiza FLAT bem | Baixa | Médio | Profile + comparação com prange aninhado original |
| Memória cresce inesperadamente | Baixa | Baixo | Cache `u_unique` indexado por slice `[ci, i_f]` (sem copy) |
| Regressão em Cenário B | Baixa | Alto | É justamente o caso onde FLAT mais ganha; benchmark é gate |

---

## 5. Conexão com a Arquitetura Multi-Agente

### 5.1 Posicionamento no roadmap §22 do documento de arquitetura

O documento `arquitetura_multiagente_geosteering_ai_aprofundamento_2026-05-02.md` divide a evolução da nova arquitetura em **4 fases (~237h total)**. A Sprint v2.22 FLAT prange está em:

```
Fase 1 — Fundações & Tooling     [✅ Etapa 0 + 1 + 1.5 concluídas]
Fase 2 — Performance Core        [⏳ Sprint v2.22 = Item 2.1.1 (próximo)]
Fase 3 — Skills & Agentes        [⏳ Após Fase 2]
Fase 4 — MCP & Integração        [⏳ Após Fase 3]
```

### 5.2 Item §22.2.1.1 — Sprint v2.22 FLAT prange

```
Item §22.2.1.1 (do documento de arquitetura):
─────────────────────────────────────────────
  Sprint v2.22 — FLAT prange Numba CPU
  Branch: feat/simulation-manager-v2.22-flat-prange
  Objetivo: paralelizar nf no nível externo, eliminando aninhamento
  Pré-requisito: Etapa 1.5 concluída ✅
  Bloqueia: Sprint v2.23 (fastmath), v2.24 (Hankel), v2.25 (cache)
  Tempo: 2-3 dias
  Responsável: Daniel Leal (humano) + futuro numba-jit-engineer
```

### 5.3 Posicionamento das demais otimizações

```
Item §22.2.1.2 — Sprint v2.23 fastmath + adaptive threads (1 dia × 2)
Item §22.2.1.3 — Sprint v2.24 Kong 61pt UI + Hankel pre-computed (4h + 3-5 dias)
Item §22.2.1.4 — Sprint v2.25 cache contexto + F7 einsum (2 dias + 2h)
Item §22.2.1.5 — Sprint v2.26+ otimizações descobertas (TBD)
```

---

## 6. Próximas Decisões do Usuário

### 6.1 Decisão imediata

```
Pergunta: Iniciar Sprint v2.22 FLAT prange agora?

Sim → Plano de execução em §4.1:
  1. Criar branch feat/simulation-manager-v2.22-flat-prange
  2. Análise detalhada do código atual (forward.py, kernel.py)
  3. Implementação iterativa com TodoWrite tracking
  4. Validação contínua (paridade + benchmarks)
  5. Merge → main + tag v2.22

Não → Alternativas:
  • Path B: skill geosteering-orchestrator.md primeiro (Etapa 2.1.1)
  • Path C: revisão profunda do documento de arquitetura
  • Path D: outra direção definida pelo usuário
```

### 6.2 Decisões deferidas (futuras)

| Decisão | Quando | Como decidir |
|:--------|:-------|:-------------|
| Aplicar `fastmath=True` em produção? | Após v2.23.1 | Gate de paridade <1e-12 em alta-ρ |
| Ativar `use_flat_prange=True` por default? | Após v2.22 estável 1 semana | Telemetria de regressões em produção |
| Implementar cache de contexto (O4)? | v2.25 | Demanda de inversão iterativa em tempo real |
| Reformular JAX para 1-vmap FLAT? | v3.1+ | Após benchmarks GPU mostrarem necessidade |

---

## 7. Anexos

### 7.1 Tabela de impacto resumido (relê 11.2 do documento de análise)

```
         O1      O2      O3      O4      O5      O6
         Adapt   FLAT    fmath   ctx$    Kong    F7-einsum
Cen. 1   ●●○○   ○○○○   ●●○○   ●●○○   ●●●●   ○○○○
Cen. 2   ●●●○   ●●●●   ●●●○   ●●●○   ●●●●   ○○○○
Cen. 6   ●○○○   ●●●●   ●●●○   ●●●○   ●●●●   ○○○○
Cen. 7   ●○○○   ●●●●   ●●●○   ●●●○   ●●●●   ○○○○
Cen. 8   ○○○○   ●●●●   ●●●○   ●●●○   ●●●●   ○○○○

●●●● alto impacto  ●●●○ médio-alto  ●●○○ médio  ●○○○ baixo  ○○○○ sem
```

### 7.2 Projeção de throughput acumulado

```
Versão │ Cenário E (nf=1) │ Cenário F (nf=4) │ Cenário B (multi-TR) │
═══════╪══════════════════╪══════════════════╪══════════════════════╡
v2.21  │  122 000 mod/h   │  100 000 mod/h   │  303 000 mod/h       │
v2.22  │  122 000 mod/h   │  130 000 mod/h   │  600 000 mod/h       │ +FLAT
v2.23  │  132 000 mod/h   │  142 000 mod/h   │  650 000 mod/h       │ +fastmath
v2.24  │  148 000 mod/h   │  160 000 mod/h   │  680 000 mod/h       │ +Hankel
v2.25  │  165 000 mod/h   │  180 000 mod/h   │  750 000 mod/h       │ +context$
═══════╪══════════════════╪══════════════════╪══════════════════════╡
Meta   │  150 000+        │  150 000+        │  600 000+            │
Status │  ✅ atingida v2.24│ ✅ atingida v2.25│  ✅ atingida v2.22    │
```

### 7.3 Referências cruzadas

- `docs/reference/analise_cenarios_otimizacao_simulador_numba.md` (1684 linhas, 2026-05-02) — análise completa
- `docs/reports/arquitetura_multiagente_geosteering_ai_aprofundamento_2026-05-02.md` (§22, §55, §66) — roadmap multiagente
- `docs/reports/etapa_1_5_quality_mesh_polishing_2026-05-08.md` — estado atual Quality Mesh
- `geosteering_ai/simulation/forward.py:351-467` — `_simulate_combined_prange` atual
- `geosteering_ai/simulation/_numba/kernel.py` — `_fields_in_freqs_kernel_cached` (target da v2.22)
- `geosteering_ai/simulation/_jax/multi_forward.py:497+` — `_simulate_multi_jax_vmap_real` (referência arquitetural)

---

## 8. Conclusão

A **Sprint v2.22 FLAT prange** é uma operação **cirúrgica e isolada no backend Numba CPU**, sem impacto no JAX GPU ou Fortran. É a **otimização mais impactante e de menor risco** entre as 9 catalogadas no documento de análise (`analise_cenarios_otimizacao_simulador_numba.md`), e deve ser executada **antes** das demais por ser pré-requisito metodológico (estabelece padrão de paralelismo que as outras otimizações refinam).

Todas as 9 otimizações estão **previstas e mapeadas** no roadmap multi-agente (§22 do documento de arquitetura), com cronograma definido entre **v2.22 e v2.25** (aprox. 3 semanas de trabalho efetivo). A primeira é v2.22, programada para iniciar **assim que o usuário confirmar**.

**Aguardando instruções do usuário para iniciar a Sprint v2.22.**
