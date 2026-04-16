# Sprint 12 — `find_layers_tr_jax` + vmap real multi-TR/multi-ângulo (PR #25 / v1.6.0)

> **Status**: ✅ CONCLUÍDO em 2026-04-16
> **PR**: #25 → v1.6.0 estável
> **Meta atingida**: vmap real sobre `(iTR, iAng)` com paridade bit-exata (0.000e+00)
> vs Python loop + `find_layers_tr_jax` tracer-safe (20+ testes PASS)

---

## 1. Motivação

Após a consolidação 44→1 XLA program da Sprint 10 Phase 2, a validação GPU T4
revelou duas limitações em `simulate_multi_jax`:

1. **Python loop sobre (iTR, iAng)** — cada combinação gera 1 dispatch JAX
   separado. Para poços verticais (dip=0°, dedup por hordist 100%) isso é
   eficiente. Para multi-dip real, o overhead de dispatch Python domina.

2. **`find_layers_tr` é Numba/NumPy puro** — impede `camad_t`/`camad_r`
   serem tracers JAX, bloqueando refatoração para vmap real.

**Sprint 12** entrega:

- ✅ `find_layers_tr_jax` tracer-safe via `jnp.searchsorted`
- ✅ `_simulate_multi_jax_vmap_real` com `jax.vmap` aninhado
- ✅ Dispatcher opt-in via `cfg.jax_vmap_real=True` (default False)
- ✅ 21 testes: paridade + shape + jacfwd + high-ρ + backward-compat
- ✅ Benchmark matrix scaffold (`bench_sprint12_regression.py`)

---

## 2. Arquitetura

### 2.1 `find_layers_tr_jax`

Port JAX tracer-safe de `_numba/geometry.py::find_layers_tr:211`:

```python
def find_layers_tr_jax(h0, z, prof_array, n):
    idx_r = jnp.clip(
        jnp.searchsorted(prof_array, z, side="right") - 1, 0, n - 1
    ).astype(jnp.int32)
    idx_t = jnp.clip(
        jnp.searchsorted(prof_array, h0, side="left") - 1, 0, n - 1
    ).astype(jnp.int32)
    return idx_t, idx_r
```

**Convenção assimétrica** (paridade Fortran `utils.f08:63`):

| Papel | Operador | searchsorted side | Comportamento em fronteira |
|:------|:--------:|:-----------------:|:--------------------------:|
| Receptor | `z >= prof[i]` (inclusivo) | `"right"` | `z==prof[i]` → camada ABAIXO |
| Transmissor | `h0 > prof[j]` (estrito) | `"left"` | `h0==prof[j]` → camada ACIMA |

Validada via sweep brute-force de 1000+ `(h0, z)` pairs com gate `diff == 0`
(inteiros devem ser idênticos, não < 1e-12).

### 2.2 Fluxo `_simulate_multi_jax_vmap_real`

```
simulate_multi_jax(..., cfg=SimulationConfig(jax_vmap_real=True))
      ↓
_simulate_multi_jax_vmap_real
      ├─ Pré-computa filtros Hankel + prof_arr + h_arr      [NumPy, 1x]
      ├─ Converte rho_h, rho_v, positions_z, freqs → jnp   [1x]
      ├─ Obtém jitted = _get_unified_jit(n, npt)           [cache LRU]
      │
      └─ vmap sobre (L, θ) flat:
          _one_config(L, θ, rho_h, rho_v):
              cos_t, sin_t = jnp.cos/sin(deg2rad(θ))        [tracers]
              dz_half = 0.5·L·cos_t                          [tracer]
              r_half  = 0.5·L·sin_t                          [tracer]
              Tz = positions_z + dz_half                     [tracer (n_pos,)]
              Rz = positions_z - dz_half                     [tracer (n_pos,)]
              camad_t_arr, camad_r_arr =
                  find_layers_tr_jax_vmap(Tz, Rz, prof, n)   [tracers int32]
              return jitted(..., camad_t_arr, camad_r_arr,
                            dz_half, r_half, theta_rad, ...)
      ↓
H_tensor shape (nTR, nAngles, n_pos, nf, 9) complex128
```

### 2.3 Pontos de mudança no código (PR #25)

| Arquivo | LOC | Mudança |
|:--------|:---:|:--------|
| `_jax/geometry_jax.py` (novo) | +180 | `find_layers_tr_jax` + `find_layers_tr_jax_vmap` com D1 mega-header, docstring Google-style completa, convenção assimétrica documentada |
| `_jax/__init__.py` | +10 | Exports de `find_layers_tr_jax`, `find_layers_tr_jax_vmap` |
| `_jax/multi_forward.py` | +210 | `_simulate_multi_jax_vmap_real` helper + dispatcher em `simulate_multi_jax` honrando `cfg.jax_vmap_real` |
| `config.py` | +15 | Campo `jax_vmap_real: bool = False` com docstring D4 explicando trade-off |
| `tests/test_simulation_jax_sprint12.py` (novo) | +475 | 21 testes (10 find_layers_tr + 11 vmap_real) |
| `benchmarks/bench_sprint12_regression.py` (novo) | +280 | CLI `--matrix {short,full,critical}` × CPU/GPU + CSV output |
| `__init__.py` (simulation) | +3 | Bump `__version__ = "1.6.0"` + entry log |

---

## 3. Resultados validados

### 3.1 Paridade `find_layers_tr_jax` vs Numba (CPU local)

| Teste | Modelo / configuração | Amostras | Max diff (int) |
|:------|:---------------------|:--------:|:--------------:|
| Sweep random | 5-camadas, ±5/±15 m | 1000 | **0** |
| Boundaries exatas | 3-camadas (z=0, z=5) | 8 | **0** |
| Semi-espaços | 5-camadas, ±1e5 m | 4 | **0** |
| vmap batch | 5-camadas, 60 pos | 60 | **0** |
| Sob `@jit` | 3-camadas | 1 | **0** |
| Tracer via vmap | 3-camadas | 3 | dtype int32 ✓ |
| Modelos canônicos | oklahoma_3/5/15/28, 200 pos | 800 | **0** |

**Total: 1876+ pontos, 0 divergências.**

### 3.2 Paridade vmap_real vs Python loop (CPU local)

| Modelo | nTR × nAngles × nf | `max|H_loop - H_vmap|` | Gate |
|:-------|:------------------:|:----------------------:|:----:|
| oklahoma_3 | 2×3×1 | **0.000e+00** | <1e-10 |
| oklahoma_5 | 2×3×1 | **0.000e+00** | <1e-10 |
| oklahoma_5 (multi-dip exótico) | 4×4×1 | **0.000e+00** | <1e-10 |
| oklahoma_5 (multi-freq) | 2×2×4 | **0.000e+00** | <1e-10 |
| oklahoma_28 | 2×2×1 | **0.000e+00** | <1e-10 |
| Vertical well (dip=0°) | 3×1×1 | **0.000e+00** | <1e-10 |
| Single (1×1×1) | 1×1×1 | `finite` | sanity ✓ |

Paridade é **bit-exata** porque ambos caminhos chamam internamente o mesmo
`_get_unified_jit(n, npt)` — apenas a forma de iteração difere (Python loop vs
vmap). Operações de ponto flutuante executadas em ordem idêntica.

### 3.3 Estabilidade alta resistividade

`test_vmap_real_high_rho_stability`: oklahoma_28 com `rho_h_high = rho_h × 15`
(≈ 1500 Ω·m), 2 dips (0°, 30°), 30 posições. Resultado: `np.all(np.isfinite)`
TRUE — zero NaN/Inf.

### 3.4 Testes (gate obrigatório pré-merge)

```
tests/test_simulation_jax_sprint12.py: 21/21 PASS em 250s
  1-10  test_find_layers_tr_jax_*          (paridade + boundaries + semi + jit + tracer + diverse)
  11-12 test_vmap_real_parity_vs_python_loop[oklahoma_3/5]
  13    test_vmap_real_shape_is_correct
  14    test_vmap_real_backward_compat_default_false
  15    test_vmap_real_multi_dip_exotic
  16    test_vmap_real_high_rho_stability
  17    test_vmap_real_vertical_well_still_works
  18    test_vmap_real_multi_freq_parity
  19    test_vmap_real_single_tr_single_dip_single_freq
  20    test_vmap_real_parity_oklahoma_28
  21    test_vmap_real_bucketed_strategy_also_works
```

---

## 4. Trade-offs CPU vs GPU

### 4.1 CPU

Em CPU puro (JAX XLA CPU), `vmap_real` é aproximadamente equivalente ao
Python loop em latência, já que ambos caminhos executam a mesma sequência de
operações XLA. O ganho real do vmap vem da eliminação do overhead de
**dispatch Python entre configurações**, que é negligível em CPU.

**Recomendação CPU**: manter `jax_vmap_real=False` (default) — o Python loop
com dedup por hordist é clareza-amigável e tem paridade perfeita.

### 4.2 GPU (esperado, validação manual pendente em Colab T4)

Em T4/A100, o dispatch Python entre configurações introduz sincronização
CPU↔GPU a cada iteração. Com vmap_real, todas as configs `(L, θ)` são fundidas
em **1 único kernel XLA** — eliminando sincronizações e maximizando ocupação
dos SMs.

**Esperado em T4**:

| Config | Python loop | vmap_real | Speedup estimado |
|:-------|:-----------:|:---------:|:----------------:|
| Vertical well (dip=0°) | baseline | ~baseline | 1.0× (dedup já ótimo) |
| Multi-dip (3 dips, 1 TR) | baseline | ~1.5× | 1.5× |
| Multi-TR × multi-dip (3×3) | baseline | ~2.5× | 2.5× |
| Stress (3×5 = 15 configs) | baseline | ~3.0× | 3.0× |

**Validação GPU**: pendente em Colab Pro+ (E11 do plano). Re-rodar
`benchmarks/bench_sprint12_regression.py --matrix full --backend gpu` após
conectar T4/A100.

---

## 5. Backward compatibility

- `cfg.jax_vmap_real = False` permanece **default** em `SimulationConfig`
- `cfg.jax_strategy = "bucketed"` também permanece **default**
- `test_backward_compat_bucketed_default` continua PASS (not altered)
- `test_vmap_real_backward_compat_default_false` (novo) protege a flag
- Nenhum símbolo público foi removido ou renomeado
- API de `simulate_multi_jax` 100% retrocompatível com v1.5.0

---

## 6. Pendências pós-v1.6.0

| Item | Prioridade | Quando |
|:-----|:----------:|:------:|
| **Validação GPU manual Colab T4/A100** (produção 600×3 ≥2×) | 🔴 Alta | Pré-v1.6.1 |
| Re-rodar diagnóstico regressão em GPU (E2 do plano) | 🔴 Alta | Pré-v1.6.1 |
| Flip `jax_strategy` default → `"unified"` | 🟡 Média | v1.6.1 (após soak) |
| Flip `jax_vmap_real` default → `True` | 🟡 Média | v1.6.1 ou v1.6.2 |
| complex64 mixed precision GPU (VRAM ÷ 2) | 🟡 Média | Sprint 13 (v1.7.0) |
| Deprecação path bucketed | 🟡 Média | v1.7.0+ |
| pmap multi-GPU para nTR grande | 🟢 Baixa | v2.0 |

---

## 7. Referências

- Código principal:
  - `geosteering_ai/simulation/_jax/geometry_jax.py` (novo, E3)
  - `geosteering_ai/simulation/_jax/multi_forward.py:_simulate_multi_jax_vmap_real` (E7-E8)
  - `geosteering_ai/simulation/config.py:jax_vmap_real` (flag)
- Testes: `tests/test_simulation_jax_sprint12.py` (21 testes, 100% PASS)
- Benchmark: `benchmarks/bench_sprint12_regression.py`
- Plano: `/Users/daniel/.claude/plans/cosmic-riding-garden.md`
- Sprint anterior: `docs/reference/sprint_10_phase2_unified_jit.md`
