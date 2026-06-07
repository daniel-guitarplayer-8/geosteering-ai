---
Spec: 0017-sm-plots-complete
Titulo: Fatia 6d — Plots completos (geosinais byte-fiéis + perfis ρ/λ + PyQtGraph default + animação)
Backlog-Code: F-mvc-split
Trilha-Dominante: F
Produtos: [SM, STU]
Converge-Em: simulate_batch  # física intocada; só visualização + geologia no resultado
Status: implementado
Released-As: v2.59
Constituicao: 1.0
Autor: Daniel Leal
Data: 2026-06-07
---

# Spec 0017 — Fatia 6d — Plots completos

## 0. Escopo

Plots científicos completos na galeria: **geosinais** (USD/UAD/UHR/UHA/U3DF, byte-fiéis),
**perfis ρ/λ**, **heatmap de ensemble** (imagem multidimensional), **animação**, com
**PyQtGraph como backend default** (interativo/dinâmico). Física intocada — `_run_simulation`
só passa a EXPOR a geologia já gerada (paridade `<1e-12` preservada).

## 1. Requisitos Funcionais (RF)

| ID | Requisito | MoSCoW |
|:--|:--|:--:|
| RF-1 | `gui/services/derived.py` (PURO): `compute_geosignal(name,h9)` byte-fiel (USD=Hxx/Hyy, UAD=Hxx−Hyy, UHR=Hxz/Hzz, UHA=Hxz−Hzz, U3DF=(Hxy−Hyx)/(Hxy+Hyx+ε)) + `rho_profile`/`lambda_profile` (step via `cumsum(esp)`/`√(ρᵥ/ρₕ)`) | Must |
| RF-2 | `_run_simulation` expõe `"geology": [{rho_h,rho_v,thicknesses}]` + `n_models` (sem tocar `simulate_batch`) | Must |
| RF-3 | **Backend default = PyQtGraph**; ABC `PlotCanvas` ganha `plot_step`/`plot_image`/`set_colorbar`; impls em pyqtgraph_canvas (stepMode/ImageItem) + mpl_canvas (step/imshow) | Must |
| RF-4 | Galeria: canal = 9 componentes **+** 5 geosinais; plot-kinds `perfil ρ`/`anisotropia λ`; `heatmap ensemble` (imagem n_models×n_pos); animation bar (varre o ensemble) | Must |
| RF-5 | Fidelidade: geosinais byte-fiéis; perfis corretos; paridade geosinal numba×jax `<1e-12` (gated) | Must |

## 2. Critérios de Aceite

- [x] **AC-1** `compute_geosignal` byte-fiel (índices 0=Hxx..8=Hzz; ε nas razões); valores conhecidos.
- [x] **AC-2** `rho_profile`/`lambda_profile` step corretos (interfaces `cumsum`, λ≥1).
- [x] **AC-3** `_run_simulation` retorna geologia por modelo (fixed/stochastic/manual); H6 de run completa inalterado.
- [x] **AC-4** Backend default PyQtGraph; `plot_step`/`plot_image` nos 2 backends reais; fallback matplotlib OK.
- [x] **AC-5** Galeria computa geosinais/perfis/heatmap (PURO no VM, testável); animation bar varre modelos.
- [x] **AC-6** Paridade geosinal numba×jax `<1e-12` (gated GPU); suíte não-regredida.

## 3. Riscos/Fidelidade
- Geosinais derivam do H6 (se H6 paridade `<1e-12`, geosinal idem); perfis exigem geologia no resultado (já gerada, zero física); PyQtGraph headless (offscreen) — verificar render + fallback.

## 4. GATE-S
- [x] 0 marcadores; RF→AC; física intocada; geosinais byte-fiéis fixados.
