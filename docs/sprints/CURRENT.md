# Sprint em execução — A1.6 (`A-jax-gpu-benchmark-redesign`)

> Este arquivo contém o plano detalhado da sprint **em execução**.
> Após o merge, deve ser renomeado para snapshot imutável (convenção:
> `v2.X.md`) e este arquivo fica vazio.

---

| Campo | Valor |
|:--|:--|
| **Versão (planejada)** | v2.43 (Sprint A1.6) |
| **Código backlog** | `A-jax-gpu-benchmark-redesign` |
| **Data início** | 2026-05-21 |
| **Branch** | `feat/a16-jax-gpu-benchmark-redesign` |
| **Trilha** | A (Simulador) |
| **Esforço estimado** | 4-6h implementação + 1h multi-agent review |
| **Modelo Claude** | Sonnet 4.6 |
| **Status** | EM EXECUÇÃO |

---

## Contexto

A Sprint A1 (`A-jax-gpu-validate`) fechou **DONE-PARTIAL**: paridade Fortran
<1e-12 confirmada em Colab T4 (163/163 PASS) mas gate de performance
reprovado em A/B/E (0.38×, 0.37×, 0.61× Numba). A auditoria
`v2.40.4_auditoria_resultados_sprint_a1` identificou 8 bugs metodológicos
no notebook (C1+C2 críticos, H1-H3 altos, M1-M3 médios) e duas
causas-raiz arquiteturais (loop Python serial + bucket cache explosion).

A Sprint A1.5 (`v2.42`, mergeada em main) implementou
`simulate_multi_jax_batched()` que resolve as causas-raiz: 1 trace XLA,
1 sync GPU→CPU, `_UNIFIED_JIT_CACHE` invariante a modelos.

Esta sprint (A1.6) reescreve o notebook para:

1. Substituir o loop Python `for m in models: simulate_multi_jax(...)` por
   uma única chamada `simulate_multi_jax_batched(rho_h_batch, ...)`.
2. Corrigir os 8 bugs metodológicos identificados.
3. Medir baseline Numba **localmente no T4** via `geosteering-cli benchmark`
   (não mais hardcoded Intel i9-9980HK Mac Intel).
4. Reportar Run 1 (cold-start) separado dos Runs 2-5 (hot-cache, com
   `statistics.median` apenas sobre hot).

---

## Escopo da Sprint

### Mudanças (1 arquivo de notebook + 3 docs)

| Arquivo | Mudança |
|:--|:--|
| `notebooks/colab_templates/validate_jax_gpu_v240.ipynb` | Rewrite completo (~20 células) |
| `docs/sprints/CURRENT.md` | Plan ativo (este arquivo) |
| `docs/sprints/v2.43.md` | Snapshot imutável pós-merge |
| `docs/CHANGELOG.md` | Append `[v2.43]` no topo |
| `docs/ROADMAP.md` | `A-jax-gpu-benchmark-redesign` CANDIDATE → DONE |

### Zero mudança em código de produção

A API `simulate_multi_jax_batched` já existe (v2.42, intocada nesta sprint).
A CLI `geosteering-cli benchmark` é consumida via subprocess. Sem testes
Python novos — notebook é deliverable para o usuário executar em Colab T4.

---

## Mapeamento bug → fix

| Bug | Sev | Fix nesta sprint |
|:--|:-:|:--|
| **C1** Warmup global shape fixa | CRIT | Warmup **por-cenário** com batched API (1 chamada por shape) |
| **C2** `statistics.median([T1..T5])` mascara cold-start | CRIT | Run 1 (cold) **separado** de `statistics.median([T2..T5])` (hot) |
| **H1** `d.platform == "gpu"` deprecada | HIGH | Usar `jax.default_backend() in ("gpu", "cuda")` |
| **H2** Sem `block_until_ready()` explícito | HIGH | Batched API já chama internamente; defensivo: ler `result.H_tensor.shape` |
| **H3** Warmup só `models[0]` | HIGH | Resolvido por design — `_UNIFIED_JIT_CACHE` chaveado `(n, npt)` |
| **M1** H dip 87.5° vs Numba 90° | MED | Documentar (validador JAX cap em 89°); gate só em A/B/E |
| **M2** `esp` alta variância | MED | Resolvido por design — batched cache não chaveia em esp |
| **M3** Numba baseline hardcoded i9-9980HK | MED | Medição local T4 via subprocess `geosteering-cli benchmark` |

---

## Critérios de Aceitação

- [ ] Notebook tem ≤20 células e parsa como JSON válido
- [ ] Célula warmup chama `simulate_multi_jax_batched` (não `simulate_multi_jax`)
- [ ] Célula benchmark chama `simulate_multi_jax_batched` com batches `(n_models, n)`
- [ ] `run_benchmark_batched_scenario` reporta `run_1_cold_mod_h` separado de `median_hot_mod_h`
- [ ] `measure_numba_baseline_t4` invoca CLI via subprocess + parse robusto
- [ ] H1 fix: GPU detection usa `jax.default_backend()`
- [ ] H2 fix: leitura de `result.H_tensor.shape` após cada call batched
- [ ] M3 fix: gate compara JAX hot vs Numba T4 LOCAL (não Intel hardcoded)
- [ ] Multi-agent reviews aplicados: 0 findings CRIT/ALTO sem fix
- [ ] CHANGELOG.md `[v2.43]` no topo após merge
- [ ] `docs/sprints/v2.43.md` snapshot imutável após merge
- [ ] `docs/sprints/CURRENT.md` esvaziado após merge

---

## Itens Fora de Escopo

- NÃO modificar `simulate_multi_jax_batched` (v2.42 intocado)
- NÃO modificar `geosteering-cli benchmark` (consumir como está)
- NÃO criar testes Python novos (notebook = deliverable user)
- NÃO `pmap` multi-GPU (Sprint A2)
- NÃO mixed precision complex64 (Sprint 13)
- NÃO chunking sobre modelos (Sprint A3)
- NÃO executar notebook em GPU local (Colab é deliverable user)

---

*Template alinhado com ADR-0001. Versão atribuída no primeiro commit da sprint.*
