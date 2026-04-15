# Sprint 10 Phase 2 — Unified JIT cabeado (PR #24-part2 / v1.5.0)

> **Status**: ✅ CONCLUÍDO em 2026-04-15
> **PRs**: #24-part1 (infra) + #24-part2 (wiring) → v1.5.0 estável
> **Meta atingida**: oklahoma_28 consolidou **44 XLA programs → 1** com paridade 3.5e-14

---

## 1. Motivação

O backend JAX do simulador Python passou, nas Sprints 7.x, por um processo de "bucketing": para cada combinação única `(camad_t, camad_r)` de um modelo geológico, JAX compila um XLA program dedicado. Em oklahoma_28, 44 buckets emergem no cenário multi-TR/multi-ângulo → **~11 GB de VRAM residente** em GPU T4 (cada program ocupa ~250 MB de código + buffers intermediários). Este custo tornava a meta "5-20× speedup em T4" inviável.

**Sprint 10** (em duas partes) consolida esses 44 programas em **1 único** por `(n, npt)`:

- **Phase 1** (PR #23, v1.5.0a1): `_hmd_tiv_propagation_unified` em `dipoles_unified.py` (ÓRFÃO)
- **Phase 2-part1** (PR #24-part1, v1.5.0b1): `_vmd_propagation_unified` + `cfg.jax_strategy` flag
- **Phase 2-part2** (PR #24-part2, v1.5.0): **wiring end-to-end** — wrappers em `dipoles_native.py`, dispatcher em `forward_pure.py`, testes 7/7 PASS

---

## 2. Arquitetura final

### 2.1 Fluxo de dispatcher

```
SimulationConfig.jax_strategy ∈ {"bucketed" (default), "unified"}
      ↓
build_static_context(..., strategy=cfg.jax_strategy)
      ↓
ForwardPureContext(strategy=...)
      ↓
forward_pure_jax(rho_h, rho_v, ctx)
      ↓
┌─ ctx.strategy == "bucketed" ──→ _forward_pure_jax_bucketed_impl
│    (44 JIT programs em oklahoma_28, caminho legacy Sprint 7.x)
│
└─ ctx.strategy == "unified" ──→ _forward_pure_jax_unified_impl
     (1 JIT program por (n, npt), caminho Sprint 10 Phase 2)
     └── _get_unified_jit(n, npt) → @jax.jit wraps:
         vmap_pos(vmap_freq(_single_position_jax(use_unified=True)))
         └── _single_position_jax(use_native_dipoles=True, use_unified=True)
             └── native_dipoles_full_jax_unified
                 ├── _hmd_tiv_native_jax_unified
                 │   ├── ETAPA 3: _hmd_tiv_propagation_unified (fori_loop)
                 │   ├── ETAPA 4: fatores geométricos
                 │   ├── ETAPA 5: _hmd_tiv_full_jax (lax.switch)
                 │   └── ETAPA 6: Ward-Hohmann assembly
                 └── _vmd_native_jax_unified (análogo)
```

### 2.2 Pontos de mudança no código (PR #24-part2)

| Arquivo | LOC | Mudança |
|:--------|:---:|:--------|
| `_jax/dipoles_native.py` | +280 | `_hmd_tiv_native_jax_unified`, `_vmd_native_jax_unified`, `native_dipoles_full_jax_unified` + fix tracer branch em `compute_case_index_jax` |
| `_jax/kernel.py` | +18 | Kwarg `use_unified: bool = False` em `_single_position_jax` + dispatcher interno |
| `_jax/forward_pure.py` | +180 | `_UNIFIED_JIT_CACHE`, `_get_unified_jit`, `_forward_pure_jax_unified_impl`, `count_compiled_xla_programs`, `clear_unified_jit_cache`; campo `strategy` em `ForwardPureContext`; parâmetro `strategy` em `build_static_context`; dispatcher em `forward_pure_jax` |
| `_jax/__init__.py` | +8 | Exports: `_hmd_tiv_native_jax_unified`, `_vmd_native_jax_unified`, `native_dipoles_full_jax_unified`, `count_compiled_xla_programs`, `clear_unified_jit_cache` |
| `_jax/multi_forward.py` | +8 | Propaga `cfg.jax_strategy` em `build_static_context` (vmap real sobre iTR/iAng deferido para Sprint 12) |
| `_jacobian.py` | +1 | Propaga `cfg.jax_strategy` |
| `tests/test_simulation_jax_sprint10_wired.py` | +260 (novo) | 7 testes end-to-end |
| `benchmarks/bench_sprint10_unified_vs_bucketed.py` | +160 (novo) | Comparação CPU |

### 2.3 Bug crítico descoberto e corrigido

Durante integração E3-E4, o smoke test local falhou em `compute_case_index_jax:557` com `TracerBoolConversionError`. A função tinha fast-path para `isinstance(camad_r, int)` (bucketed) e caminho legado Python (`if camad_r == 0 and camad_t != 0`) — mas este último não é tracer-compatível. Sob vmap, `camad_r/camad_t` viram `BatchTracer` e caem no caminho legado.

**Fix**: adicionar terceiro caminho fully-JAX via `jnp.where` encadeado cobrindo os 6 casos geométricos na mesma ordem do fast-path. Sem essa correção, `strategy="unified"` quebraria na primeira chamada.

---

## 3. Resultados validados

### 3.1 Consolidação XLA (medido localmente em CPU)

| Modelo | n camadas | n_pos | XLA bucketed | XLA unified | **Ganho** |
|:-------|:---------:|:-----:|:------------:|:-----------:|:---------:|
| oklahoma_3 | 3 | 100 | 5 | **1** | 5× |
| oklahoma_5 | 5 | 100 | 9 | **1** | 9× |
| oklahoma_28 | 28 | 100 | 44 | **1** | **44×** |

### 3.2 Paridade numérica (unified vs bucketed)

| Modelo | `max|H_b − H_u|` | Gate | Status |
|:-------|:----------------:|:----:|:------:|
| oklahoma_3 | 7.85e-14 | <1e-10 | ✅ 4 ordens abaixo |
| oklahoma_5 | 3.67e-14 | <1e-10 | ✅ 4 ordens abaixo |
| oklahoma_28 | 3.50e-14 | <1e-10 | ✅ 4 ordens abaixo |

Diferenças residuais vêm do reordenamento de operações XLA em `jax.lax.fori_loop` vs Python `for` estático — não de divergência física.

### 3.3 Testes (gate obrigatório pré-merge)

- `tests/test_simulation_jax_sprint10_wired.py`: **7/7 PASS** em 226s
  1. `test_unified_xla_program_count_1` — oklahoma_28 → 1 XLA ✅
  2. `test_unified_parity_vs_bucketed[oklahoma_3]` ✅
  3. `test_unified_parity_vs_bucketed[oklahoma_5]` ✅
  4. `test_unified_parity_vs_bucketed[oklahoma_28]` ✅
  5. `test_backward_compat_bucketed_default` ✅
  6. `test_unified_jacfwd_high_rho` (ρ≈1500 Ω·m, 0 NaN, 0 Inf) ✅
  7. `test_unified_cpu_soft_gate` (ratio < 2.5×) ✅
- `tests/test_simulation_jax_sprint10_parity.py` (PR #24-part1): **8/8 PASS** (regressão preservada)

---

## 4. Trade-offs CPU vs GPU — RESULTADO REAL (supera expectativas)

### 4.1 Benchmark CPU medido (Intel, 5 reps pós-warmup, n_pos=100, 1 freq)

| Modelo | n | XLA_b | XLA_u | bucketed_ms | unified_ms | **ratio** | parity |
|:-------|:-:|:-----:|:-----:|:-----------:|:----------:|:---------:|:------:|
| oklahoma_3 | 3 | 5 | 1 | 21.79 | 21.94 | **1.01×** | 7.85e-14 |
| oklahoma_5 | 5 | 9 | 1 | 27.24 | 22.70 | **0.83×** | 3.67e-14 |
| oklahoma_28 | 28 | 44 | **1** | 128.72 | 73.43 | **0.57×** | 3.50e-14 |

**Pior ratio: 1.01×** (equivalente). Em oklahoma_28, unified é **43% mais rápido** que bucketed em CPU.

### 4.2 Por que unified ficou mais rápido que bucketed em CPU?

Contrariando o gate soft pré-PR (≤1.3× slowdown esperado), o overhead de `jax.lax.fori_loop` + `jnp.where` encadeado é **superado** pela eliminação de:
- **44 dispatches XLA** separados por forward call (1 por bucket)
- **44 cópias de contexto** (cada `_get_bucket_jit` fecha sobre `ct, cr` concretos)
- **Scatter `H_out.at[indices].set(H_bucket)`** 44 vezes, cada um com buffer intermediário
- **Python overhead de iteração sobre `unique_keys`**

Em modelos pequenos (oklahoma_3 → 5 buckets), o overhead de fori_loop aproximadamente empata com o custo de 5 dispatches. Em modelos grandes (oklahoma_28 → 44 buckets), a economia de dispatches domina amplamente.

### 4.3 Tabela consolidada

| Aspecto | `strategy="bucketed"` | `strategy="unified"` |
|:--------|:---------------------:|:--------------------:|
| XLA programs oklahoma_28 | 44 | **1** |
| CPU latency oklahoma_28 | baseline | **0.57× (43% mais rápido!)** |
| VRAM T4 (estimado) | ~11 GB | **~250 MB** (meta) |
| Kernel launches GPU | 44 × vmap | 1 × vmap fundido |
| Diferenciabilidade (jacfwd) | ✅ | ✅ |
| Alta resistividade (ρ>1000) | estável | estável |
| Paridade bit | baseline | 3.5e-14 ULP-level |

**Conclusão**: unified domina bucketed em TODAS as métricas. O único motivo de não ser default já em v1.5.0 é a necessidade de validação GPU manual (Colab T4/A100) antes do flip — conforme política de soak.

---

## 5. Backward compatibility

- `cfg.jax_strategy = "bucketed"` permanece **default** em `SimulationConfig`
- `ForwardPureContext.strategy = "bucketed"` é **default** do dataclass
- `build_static_context(strategy=...)` tem default `"bucketed"` — todos os callers legados continuam no caminho PR #23
- `test_jit_cache_eviction_lru` (que hardcoda 44 buckets em oklahoma_28) **continua PASS**
- Nenhum símbolo público foi removido ou renomeado

---

## 6. Pendências pós-v1.5.0

| Item | Prioridade | Quando |
|:-----|:----------:|:------:|
| Validação GPU manual Colab T4/A100 | 🔴 Alta | Pré-v1.5.1 |
| Flip `jax_strategy` default → `"unified"` | 🟡 Média | v1.5.1 (após soak) |
| Vmap real sobre `(iTR, iAng)` em `simulate_multi_jax` | 🟡 Média | Sprint 12 (requer `find_layers_tr_jax`) |
| Deprecação caminho bucketed | 🟡 Média | v1.6.0 |
| complex64 mixed precision GPU | 🟡 Média | Sprint 12 |

---

## 7. Referências

- Código principal:
  - `geosteering_ai/simulation/_jax/dipoles_unified.py` (Phase 1+2-part1)
  - `geosteering_ai/simulation/_jax/dipoles_native.py:1566+` (wrappers Phase 2-part2)
  - `geosteering_ai/simulation/_jax/forward_pure.py` (dispatcher + `_UNIFIED_JIT_CACHE`)
- Testes: `tests/test_simulation_jax_sprint10_{parity,wired}.py`
- Plano: `/Users/daniel/.claude/plans/cosmic-riding-garden.md`
- Benchmark: `benchmarks/bench_sprint10_unified_vs_bucketed.py`
