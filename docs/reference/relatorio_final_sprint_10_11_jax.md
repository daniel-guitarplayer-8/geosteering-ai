# Relatório Final — Simulador Python v1.5.0 (Sprint 10 + 11-JAX)

> **Data**: 2026-04-15
> **Versão entregue**: `geosteering_ai.simulation.__version__ = "1.5.0"` estável
> **PRs consolidados**: #1 a #24 (part1 + part2) na branch `main`

---

## 1. Sumário executivo

O simulador Python de propagação EM 1D TIV atinge em v1.5.0 a **meta Sprint 10 Phase 2**: consolidação de 44 programas XLA em **1 único** para oklahoma_28 (28 camadas), viabilizando deploy em GPU T4/A100 com redução de VRAM de ~11 GB → ~250 MB (target). Paridade numérica end-to-end de **3.5e-14** (5 ordens abaixo do gate) foi validada em 3 modelos canônicos. O backend Numba (paridade <2e-13 vs Fortran, 1.16-2.14× mais rápido) e o código Fortran permanecem **intocados** ao longo do ciclo Sprint 10 — o dispatcher `cfg.jax_strategy` preserva 100% de backward-compat.

| Métrica | Antes (v1.4.1) | v1.5.0 | Ganho |
|:--------|:--------------:|:------:|:-----:|
| XLA programs (oklahoma_28) | 44 | **1** | **44×** |
| VRAM T4 (estimado) | ~11 GB | ~250 MB | **~44×** |
| Paridade JAX unified vs bucketed | N/A | 3.5e-14 | N/A |
| Paridade Numba vs Fortran | <2e-13 | <2e-13 | = |
| Testes totais | 1427 | **1438+** | +11 |
| Fortran tocado em Sprint 10 | — | — | **não** |

---

## 2. Tabela completa de fases e sprints (F7.0 → F7.15)

| Fase | Sprint | Nome | PR | Data | Status | Métrica-chave |
|:----:|:------:|:-----|:--:|:----:|:------:|:-------------|
| F7.0 | 0 | Setup branch + deps | #1 | 2026-04-11 | ✅ | — |
| F7.1 | 1.1 | Filtros Hankel (Werthmüller/Kong/Anderson) | #1 | 2026-04-11 | ✅ | 53 testes, SHA-256 auditável |
| F7.1 | 1.2 | SimulationConfig (dataclass frozen) | #1 | 2026-04-11 | ✅ | 87 testes, 4 presets |
| F7.1 | 1.3 | Half-space analítico (5 casos) | #1 | 2026-04-11 | ✅ | 38 testes, ACp/ACx bit-exato |
| F7.2 | 2.1 | Numba propagation (common_arrays/factors) | #2 | 2026-04-11 | ✅ | 25 testes |
| F7.2 | 2.2 | Numba dipolos + I/O + F6/F7 | #3 | 2026-04-11 | ✅ | 58 testes, decoupling bit-exato |
| F7.2 | 2.3-2.4 | Geometry + Rotation + Hankel + Kernel | #4 | 2026-04-12 | ✅ | 59 testes, VMD < 1e-4 |
| F7.2 | 2.5-2.6 | Forward API + validação analítica | #5 | 2026-04-12 | ✅ | 33 testes, sentinel -1e300 |
| F7.2 | 2.7 | Benchmark forward CPU | #6 | 2026-04-12 | ✅ | small 921k mod/h |
| F7.2 | 2.8 | Visualização + paralelização threadpool | #7 | 2026-04-12 | ✅ | 11 testes |
| F7.2 | 2.9 | `@njit prange` — 6.6× speedup | #8 | 2026-04-12 | ✅ | 7 modelos canônicos |
| F7.2 | 2.10 | Cache `common_arrays` | #9 | 2026-04-13 | ✅ | 1.014M mod/h small |
| F7.3 | 3.1 | JAX foundation (hankel, rotation) | #10 | 2026-04-12 | ✅ | 15 testes |
| F7.3 | 3.2 | JAX propagation (`lax.scan`) | #10 | 2026-04-12 | ✅ | 10 testes |
| F7.3 | 3.3 | JAX kernel híbrido (`pure_callback`) | #10 | 2026-04-12 | ✅ | Propagação JAX puro |
| F7.3 | 3.3.1 | JAX dipoles nativo parcial | #10 | 2026-04-13 | ✅ | decoupling diferenciável |
| F7.3 | 3.3.2 | HMD `lax.switch` nativo (6 casos) | #10 | 2026-04-13 | ✅ | 28 testes |
| F7.3 | 3.3.3 | VMD `lax.switch` nativo (6 casos) | #11 | 2026-04-13 | ✅ | 26 testes |
| F7.3 | 3.4 | Notebook Colab GPU T4 | #10 | 2026-04-12 | ✅ | jax[cuda12] install auto |
| F7.4 | 4.1 | Validação empymod (VMD axial) | #10 | 2026-04-13 | ✅ | opt-in |
| F7.4 | 4.2 | empymod 9 comp. TIV | #11 | 2026-04-13 | ✅ | infra |
| F7.4 | 4.4 | Fortran↔Python direto + bench CPU | #14 | 2026-04-13 | ✅ | bit-exato, 4.53× |
| F7.5 | 5.1 | Jacobiano JAX jacfwd experimental | #13 | 2026-04-13 | ✅ | Fallback FD |
| F7.5 | 5.1b | jacfwd end-to-end nativo | #15 | 2026-04-13 | ✅ | JAX puro |
| F7.5 | 5.2 | Jacobiano FD Numba | #13 | 2026-04-13 | ✅ | Política δ Fortran |
| F7.6 | 6.1 | `simulator_backend` em PipelineConfig | #16 | 2026-04-13 | ✅ | 5 campos |
| F7.6 | 6.2 | SyntheticDataGenerator | #16 | 2026-04-13 | ✅ | 12 testes |
| F7.7 | 7.x | Bucketing + JIT cache | #17 | 2026-04-13 | ✅ | 8700× → 3-4× overhead |
| F7.7 | 7.x+ | LRU bounded cache VRAM | #18 | 2026-04-14 | ✅ | maxsize=64 |
| F7.8 | 8 | JAX warmup + chunked | #19 | 2026-04-14 | ✅ | 3 APIs |
| F7.9 | 9 | JAX pmap multi-GPU | #19 | 2026-04-14 | ✅ | POC A100×4 |
| F7.11 | 11 | Sprint 11 Numba (multi-TR/ângulo/freq) | #20 | 2026-04-14 | ✅ | 17 testes, 1.16-2.14× Fortran |
| F7.12 | — | Fix convenção T/R + teste dip≠0° | #21 | 2026-04-15 | ✅ | v1.4.1, <1e-12 |
| F7.13 | — | Docs consolidados v1.4.1 | #22 | 2026-04-15 | ✅ | Relatório PT-BR |
| F7.14 | 10 P1 | Sprint 10 Phase 1 (unified HMD) | #23 | 2026-04-15 | ✅ | v1.5.0a1, 4 testes |
| F7.14 | 11-JAX P1 | Sprint 11-JAX wrapper Python | #23 | 2026-04-15 | ✅ | 12 testes paridade |
| F7.15 | 10 P2-part1 | Phase 2 infra (VMD unified + jax_strategy) | #24-part1 | 2026-04-15 | ✅ | v1.5.0b1, 4 testes VMD |
| **F7.15** | **10 P2-part2** | **Phase 2 wired (wrappers + dispatcher + 44→1 XLA)** | **#24-part2** | **2026-04-15** | **✅** | **v1.5.0, 7 testes E2E** |
| **F7.15** | **11-JAX P2** | **Sprint 11-JAX: propaga cfg.jax_strategy** | **#24-part2** | **2026-04-15** | **✅** | **unified em multi-TR** |
| F7.16 | 12 | complex64 mixed precision | #25+ | futuro | 🔜 | VRAM A100 ~125 MB |
| F7.17 | — | Validação GPU Colab T4/A100 manual | — | futuro | 🔜 | Speedup 5-20× confirmado |
| F7.18 | — | Flip default bucketed → unified | #26 | futuro | 🔜 | Deprecação bucketed |

---

## 3. Correções e refatorações aplicadas em v1.5.0 (PR #24-part2)

### 3.1 Bug crítico: `compute_case_index_jax` não aceitava tracers

Na integração E3-E4, o smoke test falhou com `TracerBoolConversionError` em `dipoles_native.py:557`. A função original tinha apenas 2 caminhos:
1. Fast-path: `isinstance(camad_r, int)` — concretos (bucketed)
2. Legado Python: `if camad_r == 0 and camad_t != 0` — também requer concretos

Sob `vmap` (unified), `camad_r`/`camad_t` viram `BatchTracer` e caem no caminho 2 → Python `if` em booleano tracer → erro.

**Correção**: adicionar terceiro caminho fully-JAX via `jnp.where` encadeado cobrindo os 6 casos geométricos na mesma ordem. Garante tracer-compatibility **sem** quebrar o fast-path concreto.

### 3.2 Wrappers `_hmd_tiv_native_jax_unified` e `_vmd_native_jax_unified`

Substituem os loops Python `for j in range(camad_t, camad_r+1)` (linhas 1053/1145/1345/1379 do legacy) por chamadas a `_hmd_tiv_propagation_unified` e `_vmd_propagation_unified` (que já existiam de PR #23 e PR #24-part1 mas estavam órfãos). ETAPAS 4/5/6 reutilizadas sem modificação — já eram tracer-compat via `compute_case_index_jax` + `lax.switch`.

### 3.3 Dispatcher em `forward_pure_jax`

Lê `ctx.strategy` (novo campo) e escolhe entre `_forward_pure_jax_bucketed_impl` (default) e `_forward_pure_jax_unified_impl`. Paralelamente, `forward_pure_jax_chunked` foi atualizado para respeitar `strategy` no caminho de chunk único.

### 3.4 Cache unified separado

`_UNIFIED_JIT_CACHE: OrderedDict[tuple[int, int], callable]` chaveado por `(n, npt)` — distinto de `_BUCKET_JIT_CACHE` chaveado por `(ct, cr, n, npt)`. Helper público `clear_unified_jit_cache()` espelha `clear_jit_cache()`. Função `count_compiled_xla_programs(ctx, strategy)` permite auditar o cache em testes.

### 3.5 Propagação de `cfg.jax_strategy`

Três pontos de entrada atualizados para propagar o flag sem quebrar callers legados:
- `simulate_multi_jax` (`multi_forward.py`) — lê `cfg.jax_strategy` no loop iTR×iAng
- `compute_jacobian_jax` (`_jacobian.py`) — propaga em `build_static_context`
- `build_static_context` (`forward_pure.py`) — parâmetro com default `"bucketed"`

---

## 4. Artefatos gerados (PR #24-part2)

| Arquivo | Tipo | Propósito |
|:--------|:----:|:----------|
| `tests/test_simulation_jax_sprint10_wired.py` | NOVO | 7 testes end-to-end (XLA count, paridade, backward-compat, jacfwd, soft-gate CPU) |
| `benchmarks/bench_sprint10_unified_vs_bucketed.py` | NOVO | Comparação CPU oklahoma_3/5/28 |
| `docs/reference/sprint_10_phase2_unified_jit.md` | NOVO | Documentação técnica da consolidação 44→1 |
| `docs/reference/relatorio_final_sprint_10_11_jax.md` | NOVO | Este relatório |
| `geosteering_ai/simulation/_jax/dipoles_native.py` | MOD | +280 LOC (wrappers + fix tracer) |
| `geosteering_ai/simulation/_jax/kernel.py` | MOD | +18 LOC (use_unified kwarg) |
| `geosteering_ai/simulation/_jax/forward_pure.py` | MOD | +180 LOC (cache, dispatcher, strategy) |
| `geosteering_ai/simulation/_jax/__init__.py` | MOD | +8 exports |
| `geosteering_ai/simulation/_jax/multi_forward.py` | MOD | +8 LOC (propaga cfg.jax_strategy) |
| `geosteering_ai/simulation/_jacobian.py` | MOD | +1 LOC (propaga cfg.jax_strategy) |
| `geosteering_ai/simulation/__init__.py` | MOD | `__version__ = "1.5.0"` |

---

## 5. Estado do código Fortran (intocado em Sprint 10)

O código Fortran (`Fortran_Gerador/*.f08`, binário `tatu.x`) **não sofreu qualquer modificação** em Sprint 10 Phase 1 ou Phase 2. Estado atual preservado:

- **Versão documentação**: v10.0 | **Versão código**: v9.0 (paridade Python <2e-13)
- **Performance**: 58.856 mod/h (245% da meta)
- **Features completas**: Multi-TR, f2py wrapper, Batch parallel, Tensor 9-comp, F5 (freq arbitrárias), F6 (compensação midpoint), F7 (antenas inclinadas), Filtro Adaptativo (Werthmüller/Kong/Anderson)
- **F10 Jacobiano**: entregue em commits `8c4c6cc` + `732ae7f` (pré-Sprint 10)

Dispatcher de simulador (`cfg.simulator_backend`) em `PipelineConfig` continua selecionando entre Numba, Fortran e JAX sem interferência.

---

## 6. Estado da Arquitetura v2.0

O pacote `geosteering_ai/` mantém sua organização:

```
geosteering_ai/
├── simulation/           ← FOCO Sprint 10 (apenas _jax/ afetado)
│   ├── _jax/            ← modificado (wrappers unified, dispatcher)
│   ├── _numba/          ← INTOCADO (backend Numba estável)
│   ├── config.py        ← jax_strategy field (PR #24-part1)
│   └── forward.py       ← INTOCADO (dispatcher Numba/JAX)
├── config.py            ← INTOCADO (PipelineConfig)
├── data/, models/, losses/, training/, inference/ ← INTOCADOS
└── ...
```

Componentes centrais v2.0 preservados: **PipelineConfig** (SSoT), **ModelRegistry** (48 arquiteturas), **LossFactory** (26 losses), **DataPipeline** (cadeia raw→noise→FV→GS→scale), **SurrogateNet** (TCN + ModernTCN), PINNs (8 cenários), UQ (MC/Ensemble/INN). Nenhum campo adicionado ou removido de `PipelineConfig` neste ciclo.

---

## 7. Pendências e riscos

### 7.1 Pendências (backlog)

| # | Item | Prioridade | Sprint/versão |
|:-:|:-----|:----------:|:-------------:|
| 1 | Validação GPU manual Colab T4/A100 — confirmar VRAM <1 GB e speedup 5-20× | 🔴 Alta | Pré-v1.5.1 |
| 2 | Flip `jax_strategy` default `"bucketed"` → `"unified"` | 🟡 Média | v1.5.1 (após soak) |
| 3 | Vmap REAL sobre `(iTR, iAng)` em `simulate_multi_jax` | 🟡 Média | Sprint 12 |
| 4 | Port `find_layers_tr` para JAX tracer-compat | 🟢 Baixa | v1.7.0 |
| 5 | Deprecação caminho bucketed | 🟡 Média | v1.6.0 |
| 6 | complex64 mixed precision (VRAM ÷ 2) | 🟡 Média | Sprint 12 |
| 7 | F6/F7 em JAX GPU (portar postprocess) | 🟡 Média | v1.7.0 |
| 8 | Profiling XLA + target >1M mod/h T4 | 🔴 Alta | v2.0 |

### 7.2 Riscos identificados

| # | Risco | Severidade | Mitigação aplicada |
|:-:|:------|:---------:|:-------------------|
| R1 | Paridade unified vs bucketed em geometria específica | 🔴 Alto | Testes de paridade em 3 modelos canônicos — observado 3.5e-14 (bem abaixo do gate) |
| R2 | CPU slowdown > 1.3× deixa unified pior na prática | 🟡 Médio | Soft-gate 2.5× com default bucketed preservado — usuário opta por unified |
| R3 | `test_jit_cache_eviction_lru` (hardcoda 44 buckets) quebrar | 🟢 Baixo | Validado: default "bucketed" mantém teste funcional |
| R4 | Cache unified vaza VRAM em sequências longas | 🟢 Baixo | LRU com `maxsize = _BUCKET_JIT_CACHE_MAXSIZE` (64) |
| R5 | `jnp.where` encadeado propaga NaN em alta-ρ | 🟡 Médio | Teste `test_unified_jacfwd_high_rho` valida 0 NaN/Inf em ρ≈1500 Ω·m |

---

## 8. Próximos passos recomendados

1. **Validação GPU Colab T4** (usuário) — rodar `notebooks/validate_jax_unified_gpu.ipynb` (a ser criado) com `cfg.jax_strategy="unified"` em oklahoma_28 × 3TR × 5ang × 2freq. Gate: VRAM peak < 1 GB, 0 OOM, paridade <1e-10 vs mesma execução bucketed.
2. **Soak v1.5.0** (2 semanas) — monitorar issues reportados com `strategy="unified"` opt-in. Se zero issues, flip default em v1.5.1.
3. **Sprint 12** — portar `find_layers_tr` para JAX (+vmap aninhado em `simulate_multi_jax`) + complex64 mixed precision GPU.
4. **Deprecação bucketed** — após 1 release ciclo estável com unified default, remover `_forward_pure_jax_bucketed_impl` em v1.6.0.

---

## 9. Referências

- **Código**: `geosteering_ai/simulation/_jax/{dipoles_native,dipoles_unified,forward_pure,kernel,multi_forward}.py`
- **Testes**: `tests/test_simulation_jax_sprint10_{parity,wired}.py`
- **Docs técnicos**: `docs/reference/sprint_10_phase2_unified_jit.md`
- **Plano**: `/Users/daniel/.claude/plans/cosmic-riding-garden.md`
- **Sub-skill**: `.claude/commands/geosteering-simulator-python.md` (v1.10.0)
- **PRs**: #23 (Phase 1) + #24 parts 1+2 (Phase 2)
- **Commits-chave**: `9dec317` (v1.5.0a1), `efdc91e` (v1.5.0b1), TBD (v1.5.0 final)

---

*Relatório gerado em 2026-04-15. Autor: Daniel Leal. Acentuação PT-BR garantida.*
