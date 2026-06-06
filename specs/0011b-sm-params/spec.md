---
Spec: 0011b-sm-params
Titulo: SM app MVVM — Fatia 2 (params completos: freqs/dips/TRs + h1/tj/p_med + positions_z Fortran)
Backlog-Code: F-mvc-split
Trilha-Dominante: F
Produtos: [SM, STU]
Converge-Em: simulate_batch  # SÓ orquestra; física + positions_z = convenção Fortran replicada
Status: planejado
Released-As:
Constituicao: 1.0
Autor: Daniel Leal
Data: 2026-06-06
---

# Spec 0011b — Params completos do SM MVVM (Fatia 2 da 0011)

## 0. Nota de escopo

Evolui o walking skeleton (0011a): o `SimulationViewModel` passa de params ESCALARES (1 freq, 1 dip)
para os **params completos da ParametersPage** — listas de freqs/dips/TRs + `h1`/`tj`/`p_med`, com
validação completa (errata por elemento) e o `n_pos`/`positions_z` derivados pela **convenção Fortran
exata do monólito**. A GEOLOGIA segue fixa (3-camadas TIV); geração estocástica = Fatia 3. Monólito
intocado.

## 1. Contexto e Problema

O VM do skeleton aceita 1 freq + 1 dip e usa um `positions_z` genérico (`linspace(-1, total+1, 50)`).
O SM real coleta **listas** multi-config e calcula `positions_z` pela convenção Fortran
(`positions_z = linspace(-h1, tj-h1, n_pos)`, `n_pos = ceil(tj/(p_med·cos(dip0)))`). Sem isso, o app
não reproduz o comportamento do monólito.

| Estado | Onde | Evidência |
|:--|:--|:--|
| fórmula canônica | `simulation_manager.py:~8221-8225` | `cos_d=max(1e-6,cos(rad(|dip0|))); n_pos=max(1,ceil(tj/(p_med·cos_d))); positions_z=linspace(-h1,tj-h1,n_pos)` |
| escalar (a evoluir) | `apps/.../viewmodel.py` (0011a) | `frequency_hz`/`dip_deg` escalares |
| genérico (a substituir) | `gui/services/sim_request.py::_build_batch` | `positions_z=linspace(-1,total+1,50)` |

## 2. Requisitos Funcionais (RF)

| ID | Requisito | MoSCoW | Cobertura |
|:--|:--|:--:|:--|
| RF-1 | `SimRequest`: + `h1`/`tj`/`p_med` (floats); `frequencies_hz`/`tr_spacings_m`/`dip_degs` já são tuplas | Must | extensão |
| RF-2 | `_compute_positions_z(request)` (PURO): replica a fórmula Fortran EXATA (`cos_d`/`n_pos`/`linspace(-h1,tj-h1,n_pos)`); `_build_batch` a usa | Must | NOVO/FIX |
| RF-3 | `SimulationViewModel`: estado multi-valor (`frequencies`/`dips`/`tr_spacings` tuplas) + `h1`/`tj`/`p_med`/`n_models`; `validate()` completa (errata do SIMULADOR POR ELEMENTO + guardrail anti-OOM de `n_pos`); `n_pos` derivado (read-only p/ exibir) | Must | evolução |
| RF-4 | `SimulatorView`: inputs de listas (CSV parse) + spinboxes (h1/tj/p_med/n_models) + label de `n_pos` derivado; binding ao VM | Must | evolução |
| RF-5 | Atualizar testes do skeleton (API escalar→multi) + novos (positions_z fidelidade, validação por elemento, e2e multi-config) | Must | testes |

### Critérios de Aceite
- [ ] **AC-1** (fidelidade positions_z): `_compute_positions_z(SimRequest(h1,tj,p_med,dip_degs))` == `np.linspace(-h1, tj-h1, max(1,ceil(tj/(p_med·max(1e-6,cos(rad(|dip0|)))))))` — igualdade exata com a fórmula do monólito.
- [ ] **AC-2** (VM multi-valor, puro): `SimulationViewModel` valida pela errata do **SIMULADOR** (não do pipeline DL) — cada freq∈[10,2e6] Hz (`config.py:158`), cada dip∈[0,90]° (`multi_forward.py:620`), cada TR∈[0.1,50] m (`multi_forward.py:617`), h1>0, tj>0, p_med>0, n_models≥1, **e** o `n_pos` derivado ≤ `_N_POS_MAX` (guardrail anti-OOM p/ geometria degenerada dip≈90°); `run()` monta `SimRequest` com tudo; testável com stub SEM Qt.
  > **Correção (revisão adversarial)**: a versão inicial herdou a errata do **pipeline DL** ([100,1e6]/[0.1,10]) — estreita demais p/ um app de SIMULAÇÃO, que rejeitaria configs físicas válidas (ex.: TR=25 m deep-reading). O VM espelha agora o que `simulate_batch` realmente aceita.
- [ ] **AC-3** (e2e multi-config): `simulate_batch` com 2 freqs × 2 dips × 1 TR retorna H6 `(n_models, nTR, nAng, n_pos, nf, 9)` com `nf=2, nAng=2, n_pos` derivado — finito.
- [ ] **AC-4** (n_pos derivado): a property `n_pos` do VM == `max(1, ceil(tj/(p_med·cos(dip0))))`.
- [ ] **AC-5** (regressão/fronteira): VM importável sem Qt; suíte GUI do SM 16/16; regressão da fundação verde.

## 3. RNF

| ID | Requisito | Limite |
|:--|:--|:--|
| RNF-1 | **Fidelidade**: `positions_z` IDÊNTICO à fórmula do monólito (paridade de comportamento) | AC-1 |
| RNF-2 | VM PURO (multi-valor sem Qt) | AC-2 |
| RNF-3 | física `simulate_batch` intocada (<1e-12) | declarado |
| RNF-4 | Monólito intocado | AC-5 |

## 4. Escopo

### IN
- `sim_request.py` (+ h1/tj/p_med + `_compute_positions_z`), `viewmodel.py` (multi-valor + validação), `view.py` (inputs), `tests/test_sim_app_skeleton.py` (atualizar + novos).

### OUT (fatias futuras)
- Geração estocástica de modelos + pool efêmero → **Fatia 3** (0011c).
- ResultsView galeria/cache/.session → **Fatia 4**.
- Backend auto/jax/fortran → **Fatia 5 / 0012**.
- Editor de geologia (camadas TIV manuais, LayersManualDialog) → futuro.

## 5. [NEEDS CLARIFICATION]
- [x] ~~Inputs de lista: CSV ou widget dedicado?~~ → **RESOLVIDO**: CSV (QLineEdit parse "20000, 40000") no skeleton; widget rico é fatia futura.
- [x] ~~positions_z genérico ou Fortran?~~ → **RESOLVIDO**: Fortran EXATO (RF-2/AC-1) — fidelidade.

**GATE-S: PASSOU** — 0 marcadores.

## 6. Dependências e Riscos

| Tipo | Item | Impacto/Mitigação |
|:--|:--|:--|
| Dep | 0011a (VM/Service/threading) | evolui o VM/SimRequest |
| Risco | divergir de positions_z do monólito | mitigado: replica byte-a-byte; AC-1 compara com a fórmula |
| Risco | quebrar testes do skeleton (API escalar→multi) | atualizar os testes (parte do escopo, RF-5) |
| Risco | parse de CSV inválido | validação no VM (erro claro, sem crash) |

## 7. GATE-S
- [x] 0 marcadores; todo RF→AC; IN/OUT explícito; fidelidade `positions_z` fixada (fórmula do monólito); física intocada.
