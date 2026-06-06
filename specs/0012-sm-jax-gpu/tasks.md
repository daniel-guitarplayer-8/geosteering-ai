---
Spec: 0012-sm-jax-gpu
Tarefas-de: spec.md
Status: planejado
Data: 2026-06-06
---

# Tarefas 0012 — JAX GPU no SM MVVM (subprocesso TLS-safe + seletor de backend)

## Constituição (gate)
Física só `simulate_batch` (<1e-12 JAX/Numba/Fortran). Processo da GUI NÃO importa JAX (subprocesso spawn).
`_run_simulation`/`SimRequest` picklable (já são). Caminho numba in-thread INALTERADO (sem regressão). Monólito
intocado. Sem ADR (subprocesso = padrão v2.29 já no projeto).

## Contratos (assinaturas)
```python
# gui/services/base.py
def _pool_run(fn, args, kwargs):            # roda fn num ProcessPool spawn (1 worker) — BLOQUEANTE
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
        return pool.submit(fn, *args, **kwargs).result()
class BaseService:
    def _run_in_subprocess(self, fn, *args, **kwargs): self._run_async(_pool_run, fn, args, kwargs)

# gui/services/simulation_service.py
def run(self, request):
    if request.backend in ("jax", "auto"): self._run_in_subprocess(_run_simulation, request)
    else: self._run_async(_run_simulation, request)

# apps/.../viewmodel.py: property backend (∈ {numba,jax,auto}) + _BACKENDS; validate() checa; run() passa.
# apps/.../view.py: combo "Backend:" [numba, jax, auto].
```

## Lista
| # | Tarefa | Cobre | Arquivos | Dep |
|:--|:--|:--|:--|:--:|
| **T01** | `_pool_run` + `BaseService._run_in_subprocess` | RF-1/AC-1 | `gui/services/base.py` | — |
| **T02** | `SimulationService.run` dispatch numba(in-thread)/jax-auto(subprocesso) | RF-2/AC-2 | `gui/services/simulation_service.py` | T01 |
| **T03** | VM property `backend` + validação; View combo "Backend:" | RF-3/AC-5 | `apps/.../viewmodel.py`, `view.py` | T02 |
| **T04** | Testes (paridade jax-gpu vs numba <1e-12 gated; e2e subprocess marshaling; VM backend; AC-1 GUI sem jax em sys.modules; numba in-thread inalterado) + GATE-V + revisão + commit | RF-4/RF-5/AC-1..6 | `tests/test_sm_jax_gpu.py` + regressão | T03 |

## Commit
Sem commitar até revisão; pré-aplicar auto-fixers (ruff-format two-step); footer Co-Authored-By.

## GATE-T
- [x] Toda tarefa → ≥1 AC; T04 = paridade <1e-12 (gated GPU) + AC-1 TLS-safe + regressão numba; última = verify+review.
