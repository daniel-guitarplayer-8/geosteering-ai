# Sprint 7.x — Otimização de Performance JAX (CPU + GPU)

**PR**: #14d
**Branch**: `feature/pr14d-jax-performance`
**Data**: 2026-04-13
**Autor**: Daniel Leal
**Versão do subpacote**: 1.3.1

---

## 1. Motivação

Após PR #14b (Sprint 5.1b) introduzir `forward_pure_jax` como caminho JAX
diferenciável end-to-end, execução real revelou performance inaceitável:

| Hardware | Antes (ms/modelo oklahoma_3 100 pos) | Throughput | VRAM |
|:---------|-------------------------------------:|-----------:|-----:|
| macOS Intel i9 CPU | ~34.000 ms (medido, pior que reportado) | ~100 mod/h | — |
| Colab GPU T4 | 58.794 ms (reportado pelo usuário) | 61 mod/h | 12 GB |

Como referência, **Numba CPU faz 3,9 ms/modelo**. O JAX nativo era
**~8.700× mais lento** em CPU e **~15.000× mais lento** em GPU T4, com
consumo desproporcional de VRAM — indicando **dezenas de especializações
XLA** cacheadas simultaneamente.

---

## 2. Diagnóstico do root cause

A investigação estruturada (via Explore agent) identificou **5 causas** no
orquestrador `forward_pure_jax`:

| # | Causa | Arquivo:linha | Impacto |
|:-:|:------|:--------------|:-------:|
| H1 | **Ausência de `@jax.jit`** — cada chamada faz trace+compile+execute | `forward_pure.py:291` | **CRÍTICA** |
| H2 | **Loops Python duplos** sobre posições × frequências — sem `vmap`, sem fusion | `forward_pure.py:356-366` | **CRÍTICA** |
| H3 | `camad_t`/`camad_r` como **Python int** → cada combinação força recompile | `forward_pure.py:336, 358-359` | Alta |
| H4 | Loops `for j in range(camad_t, camad_r+1)` em branches HMD/VMD | `dipoles_native.py:1034, 1126, 1326, 1360` | Alta (se H3 não mitigada) |
| H5 | `compute_case_index_jax` usa `if z <= h0` com `z` tracer → `TracerBoolConversionError` | `dipoles_native.py:538-548` | Bloqueia JIT |

**Boa notícia**: os kernels HMD/VMD já usam `lax.switch` (6 branches), só a
orquestração tinha os bugs.

---

## 3. Solução implementada (3 camadas)

### 3.1 Bucketing por (camad_t, camad_r) — implementa Camadas 1+2+3 simultaneamente

A solução adotada agrupa posições por **bucket geométrico único** e
compila uma versão JIT por bucket. Para modelos com 3–22 camadas e
100–600 posições, tipicamente há **3–10 buckets distintos**.

Dentro de cada bucket:
- `camad_t`, `camad_r` são **estáticos** (fechados na closure do `jax.jit`),
  eliminando retrace por geometria.
- `jax.vmap(jax.vmap(_single_pos_one_freq, in_axes=(None, 0)), in_axes=(0, None))`
  fusa todas as posições × frequências num único kernel XLA.
- Cache em `_BUCKET_JIT_CACHE: dict` por `(ct, cr, n, npt)` garante
  reutilização entre chamadas sucessivas (ex.: treino on-the-fly).

### 3.2 Correção `compute_case_index_jax`

Quando `camad_r == camad_t` e `z`/`h0` são tracers (vmap sobre posições),
a expressão `if z <= h0` lança `TracerBoolConversionError`. Correção:

```python
if camad_r == camad_t:
    return jnp.where(z <= h0, 2, 3)  # tracer-safe branch
```

Os outros branches (que dependem só de `camad_r`, `camad_t`, `n` — todos
estáticos no bucket) continuam como Python `if`.

### 3.3 Preservação de compatibilidade

- **API pública inalterada**: `forward_pure_jax(rho_h, rho_v, ctx)` continua
  a mesma assinatura.
- **`jax.jacfwd` continua funcional**: validado em `test_jacfwd_native_*`.
- **Caminho JAX híbrido (`use_native_dipoles=False`) INTACTO**: nenhuma
  modificação em `fields_in_freqs_jax_batch`.
- **Paridade bit-a-bit** com Numba preservada: `max_abs = 3,63 × 10⁻¹⁴`.

---

## 4. Benchmarks medidos (CPU macOS Intel Core i9, 16 threads)

### 4.1 Resultado 4-way oficial (3 canônicos, 100 posições, 3 iter pós-warmup)

| Modelo | Fortran (ms) | **Numba (ms)** | JAX híbrido (ms) | **JAX nativo (ms)** | JAX nat vs Fortran |
|:-------|-------------:|---------------:|-----------------:|--------------------:|-------------------:|
| oklahoma_3 | 28,0 | **3,91** | 31.708,79 | **15,26** | **1,84× mais rápido** |
| oklahoma_5 | 231,7 | **3,46** | 31.863,37 | **23,86** | **9,71× mais rápido** |
| oklahoma_28 | 419,6 | **8,23** | 32.364,17 | **113,19** | **3,71× mais rápido** |

### 4.2 Comparação antes × depois (JAX nativo)

| Modelo | Antes (ms) | **Depois (ms)** | **Speedup** |
|:-------|-----------:|----------------:|------------:|
| oklahoma_3 | 17.135 | **15,26** | **1.123×** |
| oklahoma_5 | 17.111 | **23,86** | **717×** |
| oklahoma_28 | 17.359 | **113,19** | **153×** |

### 4.3 Gates atingidos

- ✅ CPU oklahoma_3 < 500 ms/modelo (gate Sprint 7.4): **15,3 ms**
- ✅ JIT cache reutilizado (5 chamadas < 300 ms cada): **~20 ms cada**
- ✅ Paridade vs Numba `max_abs < 1e-10`: **3,63 × 10⁻¹⁴**
- ✅ Alta ρ (1500 Ω·m) finito: testado
- ✅ JAX híbrido intacto: `test_jax_hybrid_path_preserved` PASS
- ✅ `jax.jacfwd` end-to-end funcional: 5/5 testes jacfwd PASS

---

## 5. Análise de trade-offs

### Por que JAX híbrido (31.700 ms) não foi otimizado?

- `fields_in_freqs_jax_batch` (hybrid via `pure_callback`) tem o mesmo
  padrão de loops Python. A otimização equivalente requereria
  movimentar o `pure_callback` para dentro de `vmap`, o que é
  tecnicamente possível mas **não é crítico**: híbrido foi mantido
  apenas como fallback opt-in (restrição 3 do usuário — não remover).
- Prioridade foi o caminho **nativo** que suporta `jax.jacfwd` e GPU.
- Otimização do híbrido fica como Sprint F7.7.x futuro (baixa prioridade).

### Por que não superamos Numba em CPU?

- Numba produz bare-metal code via LLVM; JAX tem overhead fundamental
  de tracing + dispatch XLA.
- Para **batches grandes** (`jax.vmap` sobre 1000+ modelos) ou **GPU**, o
  overhead amortiza e JAX supera.
- Para **jacobiano** (`jax.jacfwd`), JAX nativo é a **única** opção rápida —
  FD Numba tem custo `4 × n_layers × t_forward`, enquanto jacfwd nativo
  tem custo `~n_layers × t_forward` (forward mode diferenciação vetorizada).

### Consumo de VRAM (GPU T4)

- **Antes**: ~12 GB (cache XLA vazando com N compilações distintas).
- **Depois esperado**: < 4 GB (1 compilação por bucket × 3-10 buckets).
- Validação manual no notebook `bench_jax_gpu_colab_pr14d.ipynb`
  (execução pelo usuário).

---

## 6. Arquivos alterados

| Arquivo | Tipo | LOC | Mudança |
|:--------|:----:|----:|:--------|
| `geosteering_ai/simulation/_jax/forward_pure.py` | MOD | +130 / −25 | Substitui loops Python por bucketing + jit + vmap duplo; cache `_BUCKET_JIT_CACHE` |
| `geosteering_ai/simulation/_jax/dipoles_native.py` | MOD | +18 | `compute_case_index_jax` JAX-friendly (`jnp.where`) |
| `tests/test_simulation_jax_performance.py` | NOVO | ~150 | 4 testes gate: cache JIT, paridade, < 500ms, alta ρ |
| `notebooks/bench_jax_gpu_colab_pr14d.ipynb` | NOVO | 1 nb | Colab T4 atualizado com medição VRAM |
| `docs/reference/sprint_7_performance_jax.md` | NOVO | este arquivo |
| `docs/ROADMAP.md` | MOD | F7.7.1/2/3 ✅ |
| `.claude/commands/geosteering-simulator-python.md` | MOD | v1.6.0 → v1.7.0 + seção 24 |

Total: ~330 LOC novos + 180 LOC testes/docs.

---

## 7. Próximos passos

- **Sprint 7.7** (futuro, baixa prioridade): otimizar JAX híbrido
  (`fields_in_freqs_jax_batch`) com o mesmo padrão de bucketing. Hoje o
  caminho é preservado apenas como opt-in.
- **Sprint F7.8**: `jax.pmap` para multi-GPU (A100 × 4 em Colab Pro+).
- **Sprint F7.9**: integração de `SyntheticDataGenerator` (Sprint 6.2)
  com `forward_pure_jax` para geração de datasets em GPU.
- **Medição real GPU T4**: usuário executa `bench_jax_gpu_colab_pr14d.ipynb`
  e preenche tabela final do §4.
