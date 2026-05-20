# Sprint A1.5 — API Batched JAX GPU (simulate_multi_jax por batch)

> Este arquivo contém o plano detalhado da sprint **em execução**.
> Após o merge, deve ser renomeado para snapshot imutável (convenção: `v2.42.md`
> ou nome equivalente atribuído no primeiro commit) e este arquivo fica vazio.

---

## Cabeçalho

| Campo | Valor |
|:--|:--|
| **Código backlog** | `A-jax-gpu-batched-api` |
| **Versão alvo** | Atribuída no primeiro commit da sprint (ADR-0001 R2) |
| **Branch** | `feat/a15-jax-gpu-batched-api` (criar antes do primeiro commit) |
| **Trilha** | A (Simulador) |
| **Esforço estimado** | 8-12h implementação Claude + testes locais CPU |
| **Data início** | 2026-05-19 |
| **Modelo Claude** | Sonnet 4.6 |
| **Executor Colab** | Daniel Leal (valida gate pós-A1.6) |

---

## Contexto

Sprint A1 (`A-jax-gpu-validate`) concluiu com **DONE-PARTIAL**: paridade Fortran
<1e-12 confirmada em GPU real (163/163 pytest PASS), mas gate de performance falhou
nos três cenários obrigatórios:

| Cenário | Medido T4 | Gate (≥1.5×) | Ratio |
|:-:|:-:|:-:|:-:|
| A | 448k mod/h | 1.77M mod/h | 0.38× |
| B | 119k mod/h | 480k mod/h | 0.37× |
| E | 75k mod/h | 183k mod/h | 0.61× |

**Causa-raiz confirmada**: ausência de API batched sobre o eixo de modelos.
`simulate_multi_jax` opera em um modelo por vez via loop Python serial
(`multi_forward.py:400-422`). Para n_pos ≤ 600, o compute JAX na GPU é
**<1ms**, mas o overhead Python por modelo é **5-20ms** (build_static_context
+ np.asarray sync). A GPU fica ociosa >90% do tempo.

Esta sprint implementa a API batched que corrige a causa-raiz arquitetural.

---

## Dependências Satisfeitas

| Dependência | Status |
|:--|:-:|
| `A-jax-gpu-validate` concluída (paridade GPU confirmada) | DONE-PARTIAL ✓ |
| `_UNIFIED_JIT_CACHE` em `forward_pure.py` (keyed `(n, npt)`, invariante a modelos) | OK |
| `_simulate_multi_jax_vmap_real` prova que a função JAX é vmappable | OK |
| `jax.vmap` sobre eixo de modelos — prova de conceito disponível via vmap_real | OK |

---

## Escopo da Sprint A1.5

### Análise técnica do problema

```
# Situação atual (multi_forward.py:400-422) — PROBLEMA
for model in models:                           # loop Python → GIL preso
    ctx = build_static_context(model)          # 5-20ms: Hankel I/O + 8× jnp.asarray
    H = forward_pure_jax(ctx, pos_m, ...)      # JAX GPU: <1ms para n_pos≤600
    result = np.asarray(H)                     # GPU→CPU sync → kill async pipeline
    results.append(result)                     # GPU ociosa >90% do tempo total
```

```
# Situação alvo (Sprint A1.5) — SOLUÇÃO
ctx = build_static_context_shared(cfg)         # 1× por batch (shape invariante)
rho_h_b = jnp.stack([m["rho_h"] for m in models])   # (n_models, n)
rho_v_b = jnp.stack([m["rho_v"] for m in models])   # (n_models, n)
esp_b   = jnp.stack([m["esp"]   for m in models])   # (n_models, n-1)
batched_fn = _get_batched_unified_jit(n, npt)  # jax.vmap sobre eixo 0
H_batch = batched_fn(rho_h_b, rho_v_b, esp_b, ctx_shared, ...)  # GPU full-utilization
H_batch.block_until_ready()                    # 1 sync por batch (não por modelo)
```

### O que fazer (checklist)

```
Fase 0 — Análise de viabilidade (~1h)
  [ ] Ler multi_forward.py:400-422 (loop serial atual)
  [ ] Ler multi_forward.py:497-650 (_simulate_multi_jax_vmap_real — referência)
  [ ] Ler forward_pure.py:424-573 (_BUCKET_JIT_CACHE e _UNIFIED_JIT_CACHE)
  [ ] Identificar quais parâmetros do context são invariantes a modelos vs por-modelo
  [ ] Confirmar que rho_h, rho_v, esp são os únicos parâmetros por-modelo

Fase 1 — Separar contexto estático de dados por-modelo (~2h)
  [ ] Criar `_build_shared_static_context(cfg: SimulationConfig)` em multi_forward.py
      → retorna contexto com: freqs, spacings, dips, filtros Hankel, pos_m
      → NÃO inclui: rho_h, rho_v, esp (variam por modelo)
  [ ] Criar `_get_batched_unified_jit(n: int, npt: int) → callable`
      → chama `_get_unified_jit(n, npt)` e aplica `jax.vmap(..., in_axes=(0,0,0,None,...))`
      → cacheia em novo dict `_BATCHED_JIT_CACHE: dict[(n, npt), callable]`
  [ ] Verificar que assinatura do jit unificado aceita rho_h/rho_v/esp como eixo vmappável

Fase 2 — Nova função pública batched (~3h)
  [ ] Criar `simulate_multi_jax_batched(models: list[dict], cfg: SimulationConfig)`
      em multi_forward.py
      → empilha rho_h/rho_v/esp de todos os modelos: jnp.stack shape (n_models, n)
      → chama _get_batched_unified_jit(n, npt)
      → retorna H_batch shape (n_models, n_pos, nf, 9) complex128 + block_until_ready
  [ ] Manter `simulate_multi_jax` legada com comportamento inalterado (backward-compat)
  [ ] Exportar `simulate_multi_jax_batched` em `simulation/_jax/__init__.py`
  [ ] Adicionar campo `cfg.jax_use_batched_api: bool = False` em config.py (opt-in)
  [ ] Dispatcher em `simulate_multi_jax`: se `cfg.jax_use_batched_api=True` e
      `len(models) > 1` → rota para `simulate_multi_jax_batched`

Fase 3 — Testes (~2h)
  [ ] Criar `tests/test_simulation_jax_batched_api.py`
  [ ] Paridade: simulate_multi_jax (loop serial) vs simulate_multi_jax_batched
      → max abs diff < 1e-10 (paridade numérica; não necessariamente bit-exato)
  [ ] Shape: H_batch.shape == (n_models, n_pos, nf, 9)
  [ ] Correctness: sem NaN/Inf em nenhum elemento
  [ ] Regressão: simulate_multi_jax sem flag inalterada (backward-compat)
  [ ] GPU marker: @pytest.mark.gpu nos testes que requerem GPU
  [ ] CPU path: simulate_multi_jax_batched funciona em CPU (fallback)

Fase 4 — Benchmark local (~1h)
  [ ] Medir speedup batched vs serial em CPU local (cenários A, B, E)
  [ ] Confirmar ausência de regressão em suite completa
  [ ] pytest tests/ -v --tb=short (0 FAIL obrigatório)
```

### O que NÃO fazer nesta sprint

- Não modificar `forward_pure.py` (apenas consumir `_get_unified_jit`)
- Não implementar o dispatcher unificado `simulation.simulate()` (Sprint A2)
- Não reescrever o notebook (Sprint A1.6)
- Não medir benchmark em GPU real (Sprint A1.6 fará isso com notebook corrigido)
- Não modificar `SyntheticDataGenerator` (Sprint A3)

---

## Configurações Técnicas Relevantes

```python
# _UNIFIED_JIT_CACHE — já existe em forward_pure.py
# Chave: (n, npt) — invariante a valores de modelo
# Usado por _simulate_multi_jax_vmap_real (Sprint 12)
# Sprint A1.5 consome este cache via jax.vmap sobre eixo de modelos

# Nova entrada em SimulationConfig (config.py):
jax_use_batched_api: bool = False  # Opt-in para Sprint A1.5
# True ativa simulate_multi_jax_batched em vez do loop serial
# Default False para backward-compat durante período de validação
```

---

## Interface Alvo

```python
# Pública (nova)
def simulate_multi_jax_batched(
    models: list[dict],          # lista de {"rho_h": np.ndarray, "rho_v": ..., "esp": ...}
    cfg: SimulationConfig,
) -> np.ndarray:                 # shape (n_models, n_pos, nf, 9) complex128
    """Simula batch de modelos via jax.vmap sobre eixo n_models.

    Elimina overhead do loop Python serial (5-20ms/modelo).
    Usa _UNIFIED_JIT_CACHE (n, npt) — sem cold-start por modelo.
    Um único block_until_ready() ao final (não por modelo).
    """

# Retrocompatível (inalterada)
def simulate_multi_jax(
    models: list[dict],
    cfg: SimulationConfig,
) -> list[np.ndarray]:           # comportamento atual preservado
```

---

## Commits Planejados

| # | Commit | Quando | Responsável |
|--:|:--|:--|:--|
| C01 | `feat(sim): _build_shared_static_context + _get_batched_unified_jit` | Fase 1 | Claude |
| C02 | `feat(sim): simulate_multi_jax_batched via vmap eixo n_models` | Fase 2 | Claude |
| C03 | `feat(config): cfg.jax_use_batched_api opt-in + dispatcher` | Fase 2 | Claude |
| C04 | `test(sim): suite paridade + shape + regressão batched API` | Fase 3 | Claude |
| C05 | `docs(sprint): snapshot A1.5 → v2.42.md` | Ao final | Claude |

O número da versão (`v2.X`) é atribuído no commit C01.

---

## Critérios de Aceitação

- [ ] `simulate_multi_jax_batched(models, cfg)` retorna shape `(n_models, n_pos, nf, 9)` complex128
- [ ] Paridade vs loop serial: `max(|batched - serial|) < 1e-10` em cenários A, B, E
- [ ] `simulate_multi_jax` (sem flag) preserva comportamento atual bit-exato
- [ ] `cfg.jax_use_batched_api = False` por default (opt-in)
- [ ] pytest 100% PASS (0 FAIL, 0 regressões)
- [ ] Nenhum import novo de produção (sem PyTorch, sem TF)
- [ ] `block_until_ready()` chamado exatamente uma vez por batch (não por modelo)

---

## Critérios de Aprovação do Gate (pós-A1.6 no Colab)

Após Sprint A1.6 (notebook corrigido + baseline Numba medido no T4):

| Cenário | Gate |
|:-:|:--|
| A | JAX GPU ≥ 1.5× Numba T4 |
| B | JAX GPU ≥ 1.5× Numba T4 |
| E | JAX GPU ≥ 1.5× Numba T4 |

---

## Logs em Andamento

| Data | Ação | Responsável |
|:--|:--|:--|
| 2026-05-19 | Sprint A1.5 criada; ROADMAP §0 atualizado com A-jax-gpu-batched-api (CANDIDATE) + A-jax-gpu-benchmark-redesign (BACKLOG) | Claude Sonnet 4.6 |
| — | Implementação `simulate_multi_jax_batched` | Claude |
| — | Testes de paridade + shape + regressão | Claude |
| — | Sprint A1.6: notebook rewrite | Claude |
| — | Revalidação Colab T4 | Daniel Leal |

---

*Template alinhado com ADR-0001. Versão atribuída no primeiro commit da sprint.*
