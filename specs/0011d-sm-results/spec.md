---
Spec: 0011d-sm-results
Titulo: SM app MVVM — Fatia 4 (ResultsView: galeria do ensemble + seletores + cache LRU + .session)
Backlog-Code: F-mvc-split
Trilha-Dominante: F
Produtos: [SM, STU]
Converge-Em: gui.plot_backends + gui.persistence  # reuso; sem física nova
Status: planejado
Released-As:
Constituicao: 1.0
Autor: Daniel Leal
Data: 2026-06-06
---

# Spec 0011d — ResultsView do SM MVVM (Fatia 4 da 0011)

## 0. Nota de escopo

A View atual (Fatia 2/3) plota só ``h6[0,0,0,:,0,0]`` (modelo 0, canal 0). Com geração estocástica
de N modelos diversos (Fatia 3), o usuário não vê o ENSEMBLE. A Fatia 4 entrega a **galeria/grade** dos
N modelos + **seletores** (componente Hxx..Hzz, plot-kind Re/Im/|H|/fase, índices TR/dip/freq) +
**cache LRU** de curvas (reusa ``gui.persistence.plot_cache.LRUPlotCache``) + **persistência ``.session``**
dos params (reusa ``gui.persistence.session.SessionDocument``). Reusa ``PlotCanvas.add_subplot_grid``
(galeria). Sem física nova; sem ProcessPool/progresso/cancel (Fatia 5). Monólito intocado.

## 1. Contexto e Problema

| Estado | Onde | Evidência |
|:--|:--|:--|
| plot único modelo 0 | ``apps/.../view.py::_on_result_ready`` | ``np.real(h6[0,0,0,:,0,0])`` |
| galeria + seletores (gap) | monólito ``simulation_manager.py`` (COMPONENT_NAMES/PLOT_KINDS/seletor de modelo) | parity-gaps G5/G6/G8 |
| LRU cache pronto | ``gui/persistence/plot_cache.py::LRUPlotCache`` | PURO, ``put/get/total_bytes`` |
| .session pronto | ``gui/persistence/session.py::SessionDocument`` | PURO, ``save/load`` JSON (sem pickle) |
| grade pronta | ``gui/plot_backends::PlotCanvas.add_subplot_grid(rows, cols)`` | matriz de subplots |

## 2. Requisitos Funcionais (RF)

| ID | Requisito | MoSCoW | Cobertura |
|:--|:--|:--:|:--|
| RF-1 | ``ResultsViewModel`` (PURO, sem Qt): ``set_result(dict)``; seletores ``component_index`` (0-8), ``plot_kind`` (re/im/mag/phase), ``tr_index``/``dip_index``/``freq_index``, ``page``; deriva ``curve_for(model)`` (transform do plot-kind sobre ``H6[m,tr,dip,:,freq,comp]``) e ``depth``; paginação (``page_size``, ``n_pages``, ``page_models``); **LRU cache** de curvas | Must | NOVO |
| RF-2 | ``ResultsView`` (Qt): galeria via ``add_subplot_grid(rows, cols)`` dos modelos da página; barra de seletores (combo componente, combo plot-kind, spinboxes TR/dip/freq quando dim>1); paginação (◀ ▶ + label) | Must | NOVO |
| RF-3 | ``SimulationViewModel`` compõe ``self.results: ResultsViewModel`` e o alimenta em ``_on_sim_finished``; ``SimulatorView`` embute a ``ResultsView`` (substitui o plot único); ``result_ready`` preservado | Must | evolução |
| RF-4 | Persistência ``.session``: salvar/abrir os PARAMS (``SimRequest`` serializável) via ``SessionDocument``; resultado reproduzível pela seed. Botões na View | Must | reuso |
| RF-5 | Testes: ``ResultsViewModel`` puro (curvas Re/Im/Mag/Phase, clamp de seletores, paginação, cache); galeria render (gui); ``.session`` roundtrip; regressão | Must | testes |

### Critérios de Aceite
- [ ] **AC-1** (curvas): ``ResultsViewModel.curve_for(m)`` == ``{re:np.real, im:np.imag, mag:np.abs, phase:deg(angle)}[kind](H6[m, tr, dip, :, freq, comp])`` — finita, shape ``(n_pos,)``, para cada plot-kind e componente.
- [ ] **AC-2** (clamp): ``set_result`` ajusta seletores aos ``dims`` (component∈[0,9), 0≤tr<nTR, 0≤dip<nAng, 0≤freq<nf); setar índice fora de range é clampado, sem crash.
- [ ] **AC-3** (paginação): ``n_pages == ceil(n_models/page_size)``; ``page_models`` da última página é parcial; ``page`` fora de range clampada.
- [ ] **AC-4** (cache LRU): a 2ª chamada de ``curve_for(m)`` (mesmos seletores) vem do cache; o cache é bounded (reusa ``LRUPlotCache``).
- [ ] **AC-5** (galeria, gui): ``ResultsView`` renderiza uma grade com os modelos da página após ``set_result``; trocar componente/plot-kind/página re-renderiza sem crash.
- [ ] **AC-6** (.session): salvar→abrir um ``.session`` reconstrói os params do ``SimulationViewModel`` (roundtrip via ``SessionDocument``); JSON sem pickle.
- [ ] **AC-7** (pureza/regressão): ``ResultsViewModel`` importável sem Qt; ``SimulationViewModel`` ainda emite ``result_ready``; regressão da fundação + Fatias anteriores verde.

## 3. RNF

| ID | Requisito | Limite |
|:--|:--|:--|
| RNF-1 | **Pureza**: ``ResultsViewModel`` sem Qt (Princípio X) | AC-7 |
| RNF-2 | **DRY**: reusa ``LRUPlotCache``/``SessionDocument``/``add_subplot_grid`` (não duplica) | reuso |
| RNF-3 | **Segurança**: ``.session`` JSON, NUNCA pickle (já garantido por SessionDocument) | AC-6 |
| RNF-4 | **Sem física nova**: só re-exibe o H6 já computado | declarado |
| RNF-5 | Monólito intocado | — |

## 4. Escopo

### IN
- NOVO ``apps/sim_manager/perspectives/simulation/results_viewmodel.py`` (PURO) + ``results_view.py`` (Qt);
  evolui ``viewmodel.py`` (compõe ``results``) e ``view.py`` (embute galeria + botões .session);
  novo ``tests/test_sm_results.py``.

### OUT (fatias futuras)
- Barra de progresso/cancel/pause durante a simulação, ProcessPool, backend auto/jax → **Fatia 5 / 0012**.
- Export ``.dat``/``.out``/figura, tipos de plot especializados (resistividade/geosinais), perfis canônicos,
  editor manual de camadas, preferências de estilo → futuro.
- Persistir o H6 no ``.session`` (grande/complexo) → fora; persistimos PARAMS (reproduzível pela seed).

## 5. [NEEDS CLARIFICATION]
- [x] ~~Galeria (grade) vs seletor de 1 modelo?~~ → **RESOLVIDO**: galeria (grade paginada) + seletores compartilhados (componente/plot-kind/config) — vê o ensemble.
- [x] ~~.session guarda o H6?~~ → **RESOLVIDO**: guarda só os PARAMS (SimRequest); o H6 é reproduzível pela seed (e grande/complexo p/ JSON).
- [x] ~~Onde vive a ResultsView?~~ → **RESOLVIDO**: embutida na perspectiva de Simulação (SimulationViewModel compõe um ResultsViewModel); sem cross-perspective bus.

**GATE-S: PASSOU** — 0 marcadores.

## 6. Dependências e Riscos

| Tipo | Item | Impacto/Mitigação |
|:--|:--|:--|
| Dep | 0011c (result_ready + H6) | a galeria consome o H6 |
| Dep | gui.plot_backends / gui.persistence | reuso (já testados) |
| Risco | N grande ⇒ grade gigante | paginação (page_size fixo); galeria mostra 1 página |
| Risco | ordem dos 9 componentes do H6 | COMPONENT_NAMES = [Hxx..Hzz] (axis −1, index 0=Hxx..8=Hzz); testar |
| Risco | curva cara ⇒ recomputo ao paginar | LRU cache de curvas (AC-4) |

## 7. GATE-S
- [x] 0 marcadores; todo RF→AC; IN/OUT explícito; reuso (DRY) declarado; pureza do VM (Princípio X); sem física nova.
