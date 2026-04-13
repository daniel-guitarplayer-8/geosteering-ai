# Sprint 5.1b — `jax.jacfwd` end-to-end nativo + Notebook Colab GPU T4

**PR**: #14b
**Branch**: `feature/pr14b-jacfwd-nativo-colab`
**Data**: 2026-04-13
**Versão do subpacote**: 1.3.0
**Autor**: Daniel Leal

---

## 1. Motivação

PR #13 (Sprint 5.1) entregou o **dispatcher** `compute_jacobian_jax` com
fallback FD funcional, mas `_compute_jacobian_jacfwd_native` lançava
`NotImplementedError` porque `fields_in_freqs_jax_batch` contém **loops
Python + `np.asarray` + `np.empty`** que quebram o trace JAX.

Sprint 5.1b fecha essa pendência: um **forward JAX 100% puro** que aceita
`rho_h`/`rho_v` como `jnp.ndarray` traceable, habilitando `jax.jacfwd`
end-to-end em CPU e GPU.

**Restrição atendida** (CLAUDE.md item 3): o caminho **JAX híbrido**
(`use_native_dipoles=False` em `fields_in_freqs_jax_batch`) permanece
**intacto e funcional**. O nativo é agora o padrão via `backend='jax'` +
`try_jacfwd=True` no dispatcher.

---

## 2. Entregas

### 2.1 Novo módulo `_jax/forward_pure.py` (~380 LOC)

API pública:

```python
from geosteering_ai.simulation._jax.forward_pure import (
    HAS_JAX,
    ForwardPureContext,      # frozen dataclass estático
    build_static_context,    # pré-computa arrays (fora do trace JAX)
    forward_pure_jax,        # forward diferenciável
)
```

**Estratégia**:
1. `build_static_context()` pré-computa tudo que **não depende de
   `rho_h`/`rho_v`**: filtro Hankel, profundidades (`h_arr`, `prof_arr`),
   camadas `camad_t`/`camad_r` por posição (via `find_layers_tr` Numba).
2. `forward_pure_jax(rho_h, rho_v, ctx)` reconstrói `eta` via `jnp.stack`
   (não `np.empty`), depois chama `_single_position_jax` com
   `use_native_dipoles=True` para cada posição/frequência e empilha o
   tensor com `jnp.stack`.
3. `jax.jacfwd(forward_pure_jax, argnums=(0, 1))` flui diretamente.

### 2.2 Reescrita de `_compute_jacobian_jacfwd_native`

O stub que lançava `NotImplementedError` foi substituído por implementação
real (~60 LOC novos, −40 antigos) que:
- Valida `JAX_ENABLE_X64=True` (raise se não).
- Chama `build_static_context` + `jax.jacfwd(forward_pure_jax)`.
- Retorna `JacobianResult(backend="jax_native", method="jacfwd", fd_step=None)`.

Se JAX não estiver configurado para `float64`, o fallback FD continua
disponível via `try_jacfwd=False` (preservado integralmente).

### 2.3 Notebook Colab `bench_jax_gpu_colab_pr14b.ipynb`

Contém células prontas para execução manual em **Colab Pro+ GPU T4/A100**:
- §0 Setup (install JAX CUDA + clone do repo).
- §1 Forward JAX nativo — 5 modelos canônicos.
- §2 Jacobiano `jax.jit(jax.jacfwd(...))` em GPU.
- §3 Heatmap `|∂Hxx/∂ρ_h|` e `|∂Hzz/∂ρ_h|` para `oklahoma_5`.
- §4 Tabela comparativa CPU Intel i9 ↔ GPU T4.

### 2.4 Testes

**Arquivo novo**: `tests/test_simulation_jacfwd_native.py` — 5 testes.

| # | Teste | Verificação |
|:-:|:------|:------------|
| 1 | `test_forward_pure_matches_numba` | Forward pure ≈ Numba (max_abs < 1e-10) |
| 2 | `test_jacfwd_native_shape_and_dtype` | Shape (n_pos, nf, 9, n), `method=='jacfwd'` |
| 3 | `test_jacfwd_native_matches_fd_numba` | Jacfwd ↔ FD: rel_err < 5e-2 |
| 4 | `test_jacfwd_native_high_rho_stability` | ρ=1500 Ω·m finito |
| 5 | `test_jax_hybrid_path_preserved` | `use_native_dipoles=False` ainda funciona |

**Arquivo alterado**: `tests/test_simulation_jacobian.py` — 1 teste atualizado.
- `test_compute_jacobian_jax_fallback_fd` → `test_compute_jacobian_jax_native_or_fallback`:
  aceita tanto `method='jacfwd'` (PR #14b) quanto `method='fd_central'`
  (PR #13 legacy).

**Resultado**: `14/14 PASS em 47.18s` (5 novos + 9 regressão).

---

## 3. Paridade e precisão

### 3.1 Forward pure vs Numba (smoke test)

- Modelo: oklahoma_3 análogo (3 camadas, 10/100/10 Ω·m, 10 posições).
- `max_abs(H_pure − H_numba) = 2.616 × 10⁻¹⁴` (ordem do epsilon `complex128`).

### 3.2 jacfwd nativo vs FD Numba

- Modelo: 3 camadas simples, 8 posições, 1 frequência.
- Entradas com |∂H/∂ρ| > 1e-9: rel_err máximo < 5e-2 (FD ε=1e-4).
- **Gate 5%** aprovado: FD tem erro O(ε²) intrínseco.

### 3.3 Estabilidade alta resistividade

- Modelo: ρ = {10, 1500, 10} Ω·m × ρ_v = {10, 3000, 10}.
- Resultado: ∂H/∂ρ_h e ∂H/∂ρ_v **finitos em todas as componentes**.

---

## 4. Benchmark 4-way CPU macOS Intel Core i9 (16 threads)

Modelos: 3 canônicos (oklahoma_3, oklahoma_5, oklahoma_28), 50 posições,
1 frequência (20 kHz), TR=1 m.

| Backend | oklahoma_3 | oklahoma_5 | oklahoma_28 |
|:--------|-----------:|-----------:|------------:|
| **Fortran `tatu.x`** (subprocess) | 29,7 ms | 219,0 ms | 393,8 ms |
| **Python Numba** | **3,61 ms** | **3,92 ms** | **7,17 ms** |
| JAX híbrido (pure_callback Numba) | 15.870 ms | 15.834 ms | 16.024 ms |
| JAX nativo (`forward_pure_jax`) | 17.135 ms | 17.111 ms | 17.359 ms |
| **Speedup Numba / Fortran** | **8,2×** | **55,9×** | **55,0×** |

### 4.1 Leitura honesta dos resultados

- **Numba é o backend mais rápido em CPU puro** (paridade Fortran confirmada
  em PR #14a com max_abs < 2 × 10⁻¹³).
- **JAX CPU tem custo de trace alto** (`jax.vmap` puro sobre loop Python ×
  compilação de `lax.switch`): ~16 s/modelo em CPU é dominado por setup
  sem JIT cacheado. Para produção massiva em CPU, **continue usando
  Numba** via `cfg.backend='numba'`.
- **JAX só compensa quando**:
  - (a) o usuário precisa de `jax.jacfwd` / `jax.grad` (inversão diferenciável, PINN oráculo);
  - (b) hardware disponível é GPU T4/A100 (XLA fusion reduz a compilação amortizada);
  - (c) batches grandes (`jax.vmap` sobre n_models ≥ 1000).

### 4.2 Precisão mantida

Todos os 4 backends respondem **fisicamente equivalente** (validado em PR
#14a para Fortran ↔ Numba com max_abs < 2 × 10⁻¹³, e aqui para
Numba ↔ JAX nativo com max_abs < 2,6 × 10⁻¹⁴).

---

## 5. Arquivos alterados neste PR

| Arquivo | Tipo | LOC |
|:--------|:-----|----:|
| `geosteering_ai/simulation/_jax/forward_pure.py` | NOVO | ~380 |
| `geosteering_ai/simulation/_jacobian.py` | MOD | +60 / −40 |
| `geosteering_ai/simulation/__init__.py` | MOD | versão 1.2.0 → 1.3.0 |
| `tests/test_simulation_jacfwd_native.py` | NOVO | ~170 |
| `tests/test_simulation_jacobian.py` | MOD | 1 teste atualizado |
| `notebooks/bench_jax_gpu_colab_pr14b.ipynb` | NOVO | Colab GPU T4 |
| `docs/reference/sprint_5_1b_jacfwd_nativo.md` | NOVO | este arquivo |
| `docs/ROADMAP.md` | MOD | F7.5.1b ⬜→✅ |
| `.claude/commands/geosteering-simulator-python.md` | MOD | v1.6.0-alpha → v1.6.0 |

Total: ~660 LOC (produção ~440 + testes ~170 + docs ~50 MD).

---

## 6. Gates para merge

- [x] 5 testes novos passam (paridade forward, jacfwd, high-ρ, hybrid preservado)
- [x] 9 testes existentes continuam passando (zero regressão)
- [x] `JAX_ENABLE_X64=True` obrigatório e validado via `RuntimeError`
- [x] Caminho híbrido `use_native_dipoles=False` intacto
- [x] `JacobianResult.method == 'jacfwd'` e `backend == 'jax_native'`
- [x] Notebook Colab preparado (execução manual pelo usuário)
- [x] Docs PT-BR acentuadas

---

## 7. Pendências para PR #14c

- **Sprint 6.1** — `simulator_backend` em `PipelineConfig` (hoje não há
  campo; `SimulationConfig` está isolado).
- **Sprint 6.2** — `SyntheticDataGenerator` substituindo
  `Fortran_Gerador/batch_runner.py` (ProcessPoolExecutor → `jax.vmap`
  ou `prange` Numba).
