---
Spec: 0011c-sm-stochastic
Titulo: SM app MVVM — Fatia 3 (geração estocástica de modelos TIV, paridade com o monólito; pool adiado p/ Fatia 5)
Backlog-Code: F-mvc-split
Trilha-Dominante: F
Produtos: [SM, STU]
Converge-Em: simulate_batch  # física + agrupamento por n_layers; geração = port puro do gerador do monólito
Status: implementado
Released-As: v2.57
Constituicao: 1.0
Autor: Daniel Leal
Data: 2026-06-06
---

# Spec 0011c — Geração estocástica do SM MVVM (Fatia 3 da 0011)

## 0. Nota de escopo

Evolui a Fatia 2 (0011b): troca a **geologia FIXA** de `_build_batch` (3 camadas determinísticas)
por **geração estocástica** de N modelos TIV com **paridade de recursos do SM monolítico** (7 geradores
QMC/PRNG, distribuição loguni/uniforme, λ controlado, n_layers fixo/amostrado, stick-breaking, seed
reprodutível). Decisões do usuário (2026-06-06): (1) **gerador PURO em `gui/services`** com todos os
recursos do monólito → obtido via **extração Strangler** do core puro de `sm_model_gen.py` (DRY, sem
duplicar); (2) **ProcessPool/EphemeralProcessRunner ADIADO p/ Fatia 5** (multi-sim concorrente + JAX GPU).
Mantidos intactos: `positions_z` Fortran (Fatia 2), contrato de retorno (`H6`/`positions_z`/`info`),
física só via `simulate_batch` (<1e-12). Monólito: comportamento preservado (re-importa o core extraído).

## 1. Contexto e Problema

| Estado | Onde | Evidência |
|:--|:--|:--|
| gerador estocástico (core PURO) | `simulation/tests/sm_model_gen.py:128-462` | `GenConfig` + `generate_models` + 7 samplers + stick-breaking — numpy/scipy.qmc/secrets, SEM Qt |
| parte Qt (assíncrona) | `simulation/tests/sm_model_gen.py:500-664` | `ModelGenerationThread(QThread)` — fica no monólito |
| geologia FIXA (a substituir) | `gui/services/sim_request.py::_build_batch` | `ρₕ=[1,10,100]·(1+0.1·i)`, λ²=2, esp=8m |
| seam de física | `gui/services/sim_request.py::_run_simulation` | 1 chamada `simulate_batch` (retangular, n_layers uniforme) |

O `simulate_batch` exige batch **retangular** (n_layers uniforme); o gerador produz modelos **ragged**
(n_layers variável quando amostrado). Logo a Fatia 3 agrupa por `n_layers` e chama `simulate_batch` por
grupo, reassemblando `H6` na ordem original — sem pool (cada chamada é o caminho Numba serial da Fatia 2).

## 2. Requisitos Funcionais (RF)

| ID | Requisito | MoSCoW | Cobertura |
|:--|:--|:--:|:--|
| RF-1 | **Extrair** o core puro do gerador (`GenConfig`, `generate_models`, samplers, stick-breaking, `_resolve_rng_seed`, `MODEL_KEYS`, `GENERATORS_AVAILABLE`) p/ `gui/services/stochastic_geology.py` (PURO, sem Qt) | Must | extração |
| RF-2 | **Rewire** `sm_model_gen.py`: re-importa o core de `stochastic_geology` + mantém `ModelGenerationThread`/`DEFAULT_GEN_CHUNK_SIZE` (Qt). Todos os consumidores (`simulation_manager`, `sm_benchmark`, `test_simulation_random_seed`) continuam funcionando | Must | rewire/regressão |
| RF-3 | `SimRequest` += campos de geologia (`geology_mode` fixed/stochastic, `n_layers_min/max/fixed`, `rho_h_min/max`, `rho_h_distribution`, `anisotropic`, `lambda_min/max`, `min_thickness`, `generator`, `normal_mu_log/sigma_log`, `rng_seed`); `total_depth`=`tj` | Must | extensão |
| RF-4 | `_build_batch`/`_run_simulation`: modo `stochastic` gera modelos via `stochastic_geology.generate_models` → **agrupa por n_layers** → `simulate_batch` por grupo → reassembla `H6` na ordem original. Modo `fixed` = comportamento Fatia 2 | Must | NOVO/seam |
| RF-5 | `SimulationViewModel`: properties de geologia + validação (espelha `GenConfig.validate` + errata); `run()` monta `SimRequest` completo | Must | evolução |
| RF-6 | `SimulatorView`: widgets de geologia (combo gerador, ranges ρₕ/λ, distribuição, n_layers fixo/range, min_thickness, anisotropic, seed) + binding | Must | evolução |
| RF-7 | Testes: extração (paridade bit-a-bit do `generate_models` extraído vs comportamento conhecido) + agrupamento (ordem/shape) + VM + e2e estocástico | Must | testes |

### Critérios de Aceite
- [ ] **AC-1** (extração pura): `from geosteering_ai.gui.services.stochastic_geology import GenConfig, generate_models` funciona SEM importar Qt; `generate_models(GenConfig(n_layers_fixed=5), n_models=10, rng_seed=42)` retorna 10 dicts `MODEL_KEYS` determinísticos.
- [ ] **AC-2** (paridade monólito/regressão): `test_simulation_random_seed` 7/7; SM GUI 16/16; `sm_model_gen` re-exporta `GENERATORS_AVAILABLE`/`GenConfig`/`generate_models`/`_resolve_rng_seed`/`DEFAULT_GEN_CHUNK_SIZE`/`ModelGenerationThread`; mesma seed → mesmos modelos (bit-a-bit) antes/depois da extração.
- [ ] **AC-3** (geração estocástica): `_run_simulation(SimRequest(geology_mode="stochastic", n_layers_fixed=5, n_models=8, rng_seed=42))` → `H6 (8, nTR, nAng, n_pos, nf, 9)` finito; mesma seed → mesmo H6.
- [ ] **AC-4** (agrupamento variável): com `n_layers` amostrado (min≠max), `_run_simulation` agrupa por n_layers, chama `simulate_batch` por grupo e reassembla na ordem original (modelo i no índice i do H6) — verificado.
- [ ] **AC-5** (VM puro + validação): VM valida geologia (ρₕ_min>0, ρₕ_max>min, λ_min≥1, generator∈lista, distribution∈{loguni,uniform}, n_layers≥3) SEM Qt; `run()` monta SimRequest; testável com stub.
- [ ] **AC-6** (fronteira/fidelidade): `stochastic_geology` PURO (sem Qt); VM importável sem Qt; `positions_z` Fortran intacto; física só `simulate_batch`; regressão da fundação verde.

## 3. RNF

| ID | Requisito | Limite |
|:--|:--|:--|
| RNF-1 | **DRY**: gerador NÃO duplicado — extração única reusada por app + monólito | AC-2 |
| RNF-2 | **Pureza**: `stochastic_geology` + VM sem Qt (Princípio X) | AC-1/AC-5 |
| RNF-3 | **Fidelidade**: física só `simulate_batch` (<1e-12); `positions_z` Fortran; geologia bit-paridade pré/pós extração | AC-2/AC-6 |
| RNF-4 | **Monólito**: shell GUI intocado; `sm_model_gen` só rewire de import | AC-2 |
| RNF-5 | **Pool adiado**: nenhum ProcessPool nesta fatia (Fatia 5) | declarado |

## 4. Escopo

### IN
- NOVO `gui/services/stochastic_geology.py` (core extraído); rewire `simulation/tests/sm_model_gen.py`;
  `sim_request.py` (campos + seam estocástico + agrupamento); `viewmodel.py` (geologia + validação);
  `view.py` (widgets); `tests/test_sim_app_skeleton.py` (+ novos) e/ou novo `tests/test_sm_stochastic.py`.

### OUT (fatias futuras)
- **ProcessPool/EphemeralProcessRunner + refactor cross-process do BaseService** → **Fatia 5 / 0012** (multi-sim concorrente + JAX GPU). Decisão do usuário 2026-06-06.
- ResultsView galeria/cache/.session → Fatia 4.
- Geração assíncrona com barra de progresso no app (ModelGenerationThread no app) → futuro (o app gera síncrono off-thread no Worker; N moderado).
- Editor de geologia manual (LayersManualDialog) → futuro.

## 5. [NEEDS CLARIFICATION]
- [x] ~~Reusar SyntheticDataGenerator vs gerador do monólito vs novo?~~ → **RESOLVIDO** (usuário): gerador PURO em `gui/services` com paridade do monólito → **extração Strangler** do core de `sm_model_gen` (DRY).
- [x] ~~ProcessPool agora?~~ → **RESOLVIDO** (usuário): **adiar p/ Fatia 5**.
- [x] ~~n_layers variável no app?~~ → **RESOLVIDO**: suportado via agrupamento por n_layers (paridade); `simulate_batch` por grupo.

**GATE-S: PASSOU** — 0 marcadores.

## 6. Dependências e Riscos

| Tipo | Item | Impacto/Mitigação |
|:--|:--|:--|
| Dep | 0011b (SimRequest/VM/seam) | evolui o SimRequest/_build_batch/VM/View |
| Risco | extração quebrar o monólito | mitigado: re-export completo + regressão (test_simulation_random_seed 7/7 + SM GUI 16/16 + sm_benchmark) + paridade bit-a-bit da seed |
| Risco | agrupamento embaralhar ordem dos modelos no H6 | mitigado: reassembla por índice original; AC-4 testa |
| Risco | scipy.qmc ausente em algum ambiente | mitigado: já há fallback `_HAS_SCIPY_QMC`→np.random no core (preservado) |
| Risco | n_layers grande × n_pos grande ⇒ memória | guardrail `_N_POS_MAX` (Fatia 2) já limita n_pos; n_models moderado no app |

## 7. GATE-S
- [x] 0 marcadores; todo RF→AC; IN/OUT explícito; pool adiado declarado; fidelidade (positions_z Fortran, simulate_batch, bit-paridade da extração) fixada; DRY via extração.
