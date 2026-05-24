# Sprint em execução

> Este arquivo contém o plano detalhado da sprint **em execução**.
> Após o merge, deve ser renomeado para snapshot imutável (convenção:
> `v2.X.md`) e este arquivo fica vazio.

---

## Sprint O0 — Pre-Flight (Pre-requisito de toda a iniciativa de otimização JAX GPU)

| Campo | Valor |
|:------|:------|
| **Code** | `O0-jax-gpu-preflight` |
| **Trilha** | A (JAX GPU) |
| **Iniciada** | 2026-05-24 |
| **Duração estimada** | 2 dias |
| **Branch** | `feat/sprint-o0-pre-flight-tests` |
| **Plano detalhado** | [docs/reports/v2.43_jax_gpu_optimization_plan.md](../reports/v2.43_jax_gpu_optimization_plan.md) — Parte III |
| **Backup pré-otimização** | `.backups/jax_simulator_pre_optimization_20260523_223707/` (30 arquivos) |

### Contexto

A baseline JAX GPU A100 está estabelecida (v2.43, 164/164 paridade Fortran
`<1e-12`, gate aprovado A: 3.55× / B: 3.10× / E: 2.19× Numba T4 local). Mas
investigação multi-agente revelou que a A100 opera a apenas **~7% do pico FP64
teórico** e perde para o histórico i9 per-device em E (0.39×) e G (0.37×).

A roadmap de otimização foi aprovada (Sprints O0–O4, ~10× ganho cumulativo
esperado). Sprint O0 é **BLOQUEANTE** — antes de qualquer modificação no código
de produção, precisamos de testes anti-regressão fechando 3 lacunas críticas:

1. Sem teste DIRETO `simulate_multi_jax_batched` vs Numba (apenas transitivo)
2. Combinação `unified + chunked + vmap_real` nunca testada juntas
3. Sem gate automático de throughput

### Critérios de Aceitação

- [ ] 6 testes Tier 1 criados em `tests/`:
  - [ ] T1.1 `test_forward_pure_bucketed_fortran_parity_canonical` (NEW)
  - [ ] T1.2 `test_batched_vs_numba_parity_direct` (extend batched_api)
  - [ ] T1.3 `test_pmap_parity_vs_forward_pure` (extend performance)
  - [ ] T1.4 `test_common_factors_tx_in_bottom_layer` (extend propagation)
  - [ ] T1.5 `test_unified_triple_flag_combination` (extend sprint12)
  - [ ] T1.6 `test_throughput_gpu_regression_gate` (NEW)
- [ ] `pytest tests/ --collect-only` coleta 6 novos testes sem erro
- [ ] Em ambiente sem CUDA (macOS dev), testes SKIPam gracedosamente
- [ ] Célula de profiling `jax.profiler.trace` adicionada em `validate_jax_gpu_v240.ipynb`
- [ ] Snapshot final em `docs/sprints/v2.44.md` após merge

### Estrutura de Execução (Multi-Agente)

3 agentes paralelos, cada um cobrindo 2 testes:

| Agente | Testes | Arquivos |
|:------:|:------:|:---------|
| 1 | T1.1 + T1.6 | `test_simulation_jax_fortran_parity.py` (NEW) + `test_simulation_jax_perf_baseline.py` (NEW) |
| 2 | T1.2 + T1.4 | extend `test_simulation_jax_batched_api.py` + extend `test_simulation_jax_propagation.py` |
| 3 | T1.3 + T1.5 | extend `test_simulation_jax_performance.py` + extend `test_simulation_jax_sprint12.py` |

### Próximas Sprints (após O0)

| Code | Trilha | Item | Status |
|:--|:-:|:--|:-:|
| `O1-jax-gpu-quick-wins` | A | 9 commits low-risk (rho_h_at_obs vetorizado, XLA cache persistente, donate_argnums, scan unroll, ...) | BACKLOG (dep: O0 ✓) |
| `O2-jax-gpu-chunking-async` | A | `lax.map(batch_size=K)` + async dispatch prefetch | BACKLOG (dep: O1) |
| `O3-jax-gpu-profile-driven` | A | Memory layout + eliminar overcompute switch (condicional ao profile) | BACKLOG (dep: O2) |
| `O4-jax-gpu-complex64-opt-in` | A | ADR + dtype dual + flag opt-in `cfg.dtype = "complex64"` | BACKLOG (dep: O3) |

---

*Template alinhado com ADR-0001. Versão `v2.44` será atribuída no primeiro commit
do Sprint O1 (Quick Wins). Sprint O0 é "pre-flight" — não consume versão.*
