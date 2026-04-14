# Sprint 7.x+ — Cache LRU bounded para VRAM GPU (PR #14e)

**PR**: #14e
**Branch**: `feature/pr14e-jax-vram-optim`
**Data**: 2026-04-14
**Versão do subpacote**: 1.3.2
**Autor**: Daniel Leal

## Motivação

Execução do notebook `bench_jax_gpu_colab_pr14d.ipynb` em Colab Pro+ revelou:

| Hardware | oklahoma_3 ms | VRAM observado |
|:---------|--------------:|---------------:|
| T4 | 23,1 | **~12 GB** (75% de 16 GB) |
| A100 | 19,0 | **~60 GB** |

Embora os tempos por modelo sejam agora excelentes (speedup 2.500× vs PR #14b),
o consumo de VRAM é desproporcional ao tamanho do problema (3 camadas × 100 posições).

## Diagnóstico

Contagem real de buckets `(camad_t, camad_r)` por modelo canônico:

| Modelo | N_camadas | N_buckets |
|:-------|----------:|----------:|
| oklahoma_3 | 3 | **5** |
| oklahoma_5 | 5 | 9 |
| hou_7 | 7 | 13 |
| viking_graben_10 | 10 | 13 |
| oklahoma_28 | 28 | **44** |

Cada bucket gera **uma compilação XLA separada** (~10–100 MB cada em VRAM).
O `_BUCKET_JIT_CACHE` era um `dict` **sem limite** — em execuções sucessivas
ou modelos grandes (28 camadas), as entradas acumulavam até esgotar VRAM.

## Correções aplicadas

### 1. LRU bounded cache com eviction

```python
from collections import OrderedDict
_BUCKET_JIT_CACHE: OrderedDict = OrderedDict()
_BUCKET_JIT_CACHE_MAXSIZE: int = 64  # configurável
```

Ao inserir novo bucket:

```python
if len(_BUCKET_JIT_CACHE) >= _BUCKET_JIT_CACHE_MAXSIZE:
    _BUCKET_JIT_CACHE.popitem(last=False)  # evict oldest (LRU)
```

Ao reutilizar entrada existente:

```python
_BUCKET_JIT_CACHE.move_to_end(key)  # promove a MRU
```

### 2. API pública de controle

Três funções novas em `geosteering_ai/simulation/_jax/forward_pure.py`:

| Função | Uso |
|:-------|:----|
| `clear_jit_cache()` | Limpa todo o cache; chamar entre batches de modelos |
| `set_jit_cache_maxsize(N)` | Ajusta limite (recomendado: 16 p/ T4, 32 p/ A100 40GB, 64 p/ A100 80GB) |
| `get_jit_cache_info()` | Diagnóstico: `{n_entries, maxsize, keys}` |

### 3. Comentários D1-D14 nos novos símbolos

Documentação Google-style completa em PT-BR com: quando usar, impacto em VRAM,
configuração recomendada por hardware, exemplo.

## Validação (CPU local)

`tests/test_simulation_jax_performance.py` — **5 testes novos** (9 total):

| # | Teste | Gate |
|:-:|:------|:-----|
| 5 | `test_jit_cache_clear_and_info` | API pública funciona |
| 6 | `test_jit_cache_populates_after_forward` | Cache popula após forward |
| 7 | `test_jit_cache_eviction_lru` | oklahoma_28 com `maxsize=3` → exatamente 3 entradas |
| 8 | `test_jit_cache_parity_after_eviction` | Paridade Numba mantida após eviction |
| 9 | `test_set_jit_cache_maxsize_validation` | `maxsize < 1` levanta `ValueError` |

Regressão completa: **35/35 PASS em 160,89s** (config + simulation + pipeline + performance + LRU).

## Validação de paridade (CPU Intel i9)

| Modelo | maxsize | n_entries | Paridade vs Numba |
|:-------|--------:|----------:|------------------:|
| oklahoma_3 | 64 | 5 | `1,22 × 10⁻¹³` |
| oklahoma_28 | 10 | 10 (evict) | `5,66 × 10⁻¹⁴` |

## Uso recomendado em Colab Pro+

```python
from geosteering_ai.simulation._jax.forward_pure import (
    set_jit_cache_maxsize, clear_jit_cache,
)

# T4 (16 GB VRAM)
set_jit_cache_maxsize(16)

# A100 40 GB
set_jit_cache_maxsize(32)

# A100 80 GB ou memory-constrained
set_jit_cache_maxsize(64)  # default

# Entre batches de modelos
clear_jit_cache()
```

## Arquivos alterados

| Arquivo | Tipo | Δ LOC |
|:--------|:----:|------:|
| `geosteering_ai/simulation/_jax/forward_pure.py` | MOD | +80/−5 |
| `geosteering_ai/simulation/__init__.py` | MOD | versão 1.3.1 → 1.3.2 |
| `tests/test_simulation_jax_performance.py` | MOD | +90 (5 testes novos) |
| `notebooks/bench_jax_gpu_colab_pr14e.ipynb` | NOVO | monitora VRAM + maxsize auto |
| `docs/reference/sprint_7_x_plus_jax_vram.md` | NOVO | este arquivo |
| `docs/ROADMAP.md` | MOD | F7.7.4 ✅ |
| `.claude/commands/geosteering-simulator-python.md` | MOD | v1.7.0 → v1.7.1 + seção 25 |

## Próximos passos

- Execução real do notebook `bench_jax_gpu_colab_pr14e.ipynb` pelo usuário em
  Colab Pro+ (T4 ou A100) para confirmar redução efetiva de VRAM.
- **Sprint 8**: consolidação de buckets em 1 JIT único (via `jnp.take_along_axis`
  para `camad_t`/`camad_r` traceable) — pode eliminar totalmente o fanout XLA.
- **Sprint 9**: `jax.pmap` multi-GPU A100 para treino distribuído.
