# Sprint 6.1 + 6.2 — Integração PipelineConfig + SyntheticDataGenerator

**PR**: #14c
**Branch**: `feature/pr14c-pipelineconfig-datapipeline`
**Data**: 2026-04-13
**Autor**: Daniel Leal

## Entregas

### Sprint 6.1 — `simulator_backend` em `PipelineConfig`

Novos campos (Seção 15.5 de `geosteering_ai/config.py`):

| Campo | Tipo | Default | Opções |
|:------|:-----|:--------|:-------|
| `simulator_backend` | str | `"fortran_f2py"` | `fortran_f2py`, `numba`, `jax` |
| `simulator_precision` | str | `"complex128"` | `complex64`, `complex128` |
| `simulator_device` | str | `"cpu"` | `cpu`, `gpu` |
| `simulator_jax_mode` | str | `"native"` | `native`, `hybrid` |
| `simulator_cache_common_arrays` | bool | `True` | — |

Validação `__post_init__` garante:
- backend/precision/device/jax_mode em listas válidas
- `device='gpu'` requer `backend='jax'` (Numba e Fortran são CPU-only)

**Default conservador**: `fortran_f2py` não altera pipelines legacy.

### Sprint 6.2 — `SyntheticDataGenerator`

Módulo novo `geosteering_ai/data/synthetic_generator.py` (~320 LOC):

- `GeneratedBatch` frozen dataclass com `H_tensor`, `rho_h`, `rho_v`, `esp`, `dat_22col` (compatível com `DTYPE_22COL`), `metadata`.
- `SyntheticDataGenerator(config)` — gera batches in-process usando o backend Python pedido (Numba ou JAX).
- Substitui `Fortran_Gerador/batch_runner.py` (que usa `ProcessPoolExecutor` + subprocess `tatu.x`).
- Compatibilidade 22-col preservada: `.dat` gerados são lidos por `loading.py` como se viessem do Fortran.

Amostragem: uniforme em log(ρ) por camada + espessuras uniformes (extensível em Sprint 6.3 para Sobol).

## Testes (12 novos, 12/12 PASS em 2,52s)

`tests/test_pipeline_simulator_backend.py`:
- 6 testes Sprint 6.1 (defaults, validações, mutual exclusivity).
- 6 testes Sprint 6.2 (shape, seed determinístico, round-trip .dat, erro fortran, alta ρ, throughput).

## Regressão

Suíte completa (7 arquivos-chave): **230/230 PASS em 49,69s**. Zero regressão.

## Não entregue (próximos sprints)

- **Sprint 6.3** — estratégias Sobol/mixed em `SyntheticDataGenerator` (port de `fifthBuildTIVModels.py`).
- **Sprint 6.4** — integração `SyntheticDataGenerator` com `DataPipeline.prepare()` via dispatcher automático.
- **Sprint F7.7** — pmap multi-GPU, XLA tuning.
