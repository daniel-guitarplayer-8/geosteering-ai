# Sprint em execução

> Este arquivo contém o plano detalhado da sprint **em execução**.
> Após o merge, deve ser renomeado para snapshot imutável (convenção:
> `v2.X.md`) e este arquivo fica vazio.

---

**Nenhuma sprint em execução no momento.**

Sprint A1.6 (`A-jax-gpu-benchmark-redesign`) concluída em 2026-05-22 — ver
snapshot em [v2.43.md](v2.43.md).

Próximo candidato no backlog (ver
[docs/ROADMAP.md §0](../ROADMAP.md#0-backlog-priorizado-ssot)):

| Code | Trilha | Item | Status |
|:--|:-:|:--|:-:|
| `A-jax-gpu-dispatcher` | A | `simulation.simulate(cfg, backend=...)` com backend ∈ {jax_gpu, numba_cpu, auto}; auto-detect via `jax.devices()` | BACKLOG (dep: A1.6 ✓) |

**Bloqueio**: Sprint A2 (`A-jax-gpu-dispatcher`) só faz sentido após
usuário executar `validate_jax_gpu_v240.ipynb` em Colab Pro+ T4 e
confirmar o gate ≥1.5× Numba T4 LOCAL em A/B/E. Se o gate falhar,
investigar antes de A2.

---

*Template alinhado com ADR-0001. Versão atribuída no primeiro commit da sprint.*
