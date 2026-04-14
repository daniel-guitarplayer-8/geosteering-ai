# Sprint 8 + Sprint 9 — warmup/chunked + pmap multi-GPU (PR #14f)

**PR**: #14f · **Branch**: `feature/pr14f-jax-warmup-chunked-pmap` · **Data**: 2026-04-14
**Versão**: 1.3.2 → 1.3.3

## Motivação

Execução do notebook PR #14e em T4 ainda mostrou **~11 GB VRAM** e oklahoma_28
com **44 buckets** (maxsize auto-detect não funcionou em todas as T4s do Colab).
Diagnóstico: consolidar buckets via tracers requer reescrita extensa dos loops
Python em `_hmd_tiv_full_jax`/`_vmd_full_jax` (cases 5+6), inviável neste PR.

**Estratégia adotada**: amortizar a compilação dos 44 buckets via warmup
coletivo + reduzir pico de VRAM via chunking de posições + expor pmap para
multi-GPU (A100 × 4 em Colab Pro+).

## Entregas

### Sprint 8a — `warmup_all_buckets(ctx, rho_h_ref, rho_v_ref)`

Pré-compila **todos** os buckets de um `ForwardPureContext` antes da primeira
chamada cronometrada. Usa o shape exato das posições por bucket (crítico: JAX
compila por shape, e warmup com shape diferente não é reaproveitado).

Valor de retorno: número de buckets compilados.

### Sprint 8b — `forward_pure_jax_chunked(rho_h, rho_v, ctx, chunk_size=32)`

Processa posições em lotes menores para reduzir **pico** de VRAM durante vmap.
Para oklahoma_28 com 100 posições e `chunk_size=16`, espera-se redução de
pico em ~3-6× (validação pendente no notebook Colab).

Paridade bit-a-bit com `forward_pure_jax`: `max_abs = 2,84 × 10⁻¹⁵`.

### Sprint 9 — `forward_pure_jax_pmap(rho_h_batch, rho_v_batch, ctx)`

Distribui ``n_devices`` modelos simultaneamente via `jax.pmap`. Input:
`rho_h_batch.shape == (n_devices, n_layers)`. Funciona em mono-GPU (equivale
a vmap sobre batch) e escala para A100 × 4 no Colab Pro+.

## Validação CPU Intel i9

| Modelo | warmup (s) | forward pós-warmup (ms) | n_buckets |
|:-------|-----------:|------------------------:|----------:|
| oklahoma_28 | 63,7 | **334** | 44 |

Paridade mantida: `max_abs = 5,66 × 10⁻¹⁴` vs Numba.

## Testes novos (6/6 PASS; 15 total jax_performance)

- `test_warmup_all_buckets_returns_bucket_count`
- `test_forward_pure_jax_chunked_parity` — `max_abs < 1e-13` vs default
- `test_forward_pure_jax_chunked_small_passes_through` (n_pos ≤ chunk)
- `test_forward_pure_jax_chunked_validates_chunk_size` (raise se < 1)
- `test_forward_pure_jax_pmap_single_device`
- `test_forward_pure_jax_pmap_mismatch_raises`

Regressão total: **29/29 PASS em 191,9s**.

## Arquivos alterados

| Arquivo | Tipo | Δ LOC |
|:--------|:----:|------:|
| `geosteering_ai/simulation/_jax/forward_pure.py` | MOD | +180 |
| `geosteering_ai/simulation/__init__.py` | MOD | v1.3.3 |
| `tests/test_simulation_jax_performance.py` | MOD | +140 (6 testes) |
| `notebooks/bench_jax_gpu_colab_pr14f.ipynb` | NOVO | warmup+chunked+pmap |
| `docs/reference/sprint_8_9_warmup_chunked_pmap.md` | NOVO | este arquivo |
| `docs/ROADMAP.md` | MOD | F7.7.5 + F7.8.1 + F7.9.1 ✅ |
| `.claude/commands/geosteering-simulator-python.md` | MOD | v1.7.2 + seção 26 |

## Uso recomendado

```python
from geosteering_ai.simulation._jax.forward_pure import (
    build_static_context, forward_pure_jax,
    forward_pure_jax_chunked, warmup_all_buckets,
    forward_pure_jax_pmap,
    set_jit_cache_maxsize, clear_jit_cache,
)

# 1. Configura cache para o hardware
set_jit_cache_maxsize(64)  # T4=16, A100 40GB=32, A100 80GB=64

# 2. Pré-compila (amortiza tempo inicial)
ctx = build_static_context(...)
warmup_all_buckets(ctx)

# 3. Forward: use chunked para T4 com modelos grandes
H = forward_pure_jax_chunked(rho_h, rho_v, ctx, chunk_size=16)

# 4. Multi-GPU em A100 × 4
rho_h_batch = jnp.stack([rho_h_var[i] for i in range(4)])  # shape (4, n_layers)
rho_v_batch = jnp.stack([rho_v_var[i] for i in range(4)])
H_batch = forward_pure_jax_pmap(rho_h_batch, rho_v_batch, ctx)
```

## Pendências (futuros Sprints)

- **Sprint 10**: unificar 44 buckets em 1 JIT único via reescrita de
  `_hmd_tiv_full_jax`/`_vmd_full_jax` cases 5/6 com `lax.fori_loop`.
  Requer refatoração não-trivial da recorrência de propagação TE/TM.
- **Sprint 11**: `simulator_precision='complex64'` em GPU — reduz VRAM pela
  metade com perda aceitável de precisão para modelos ρ < 1000 Ω·m.
- Validação real do notebook `bench_jax_gpu_colab_pr14f.ipynb` em T4/A100.
