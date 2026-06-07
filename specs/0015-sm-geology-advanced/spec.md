---
Spec: 0015-sm-geology-advanced
Titulo: Fatia 6b — Geologia avançada (filtro de Hankel + perfis canônicos + editor manual)
Backlog-Code: F-mvc-split
Trilha-Dominante: F
Produtos: [SM, STU]
Converge-Em: simulate_batch  # física intocada; só amplia a entrada de modelo
Status: implementado
Released-As: v2.58
Constituicao: 1.0
Autor: Daniel Leal
Data: 2026-06-06
---

# Spec 0015 — Fatia 6b — Geologia avançada

## 0. Escopo

Amplia a entrada de geologia do SM MVVM (paridade com o monólito), sem tocar a
física: **filtro de Hankel** selecionável, **perfis canônicos** (7) e **editor
manual de camadas** (N camadas). Reusa o que já existe — zero mudança de assinatura
na física.

## 1. Requisitos Funcionais (RF)

| ID | Requisito | MoSCoW |
|:--|:--|:--:|
| RF-1 | **Filtro de Hankel**: `SimRequest.hankel_filter` repassado a `simulate_batch` (que JÁ aceita o parâmetro — `dispatch.py:248`). Combo na View (catálogo `FilterLoader.available()`) | Must |
| RF-2 | **Geologia manual**: `ManualLayersModel` PURO (n_layers/thicknesses/rho_h/rho_v + `validate`/`from_canonical`/`to_model_dict`); `geology_mode="manual"` em `_run_simulation` (replica n_models× → `_simulate_grouped`, NÃO toca `_build_batch`) | Must |
| RF-3 | **Editor manual** (`LayersDialog`): tabela ρₕ/ρᵥ/espessura (espessura só nas internas); devolve `ManualLayersModel` validado | Must |
| RF-4 | **Perfis canônicos**: reusa `simulation.validation.canonical_models` (7 perfis); VM `apply_canonical_profile(name, auto_geometry)` → manual + tj/h1 | Must |
| RF-5 | VM: properties `hankel_filter`/`manual_layers`; validação manual (espelha `ManualLayersModel.validate`); sessão persiste hankel + manual_layers | Must |
| RF-6 | Fidelidade: paridade `<1e-12` (manual numba vs jax); filtro só troca pesos quasi-estáticos (resultado de run completa correto) | Must |

## 2. Critérios de Aceite

- [x] **AC-1** `_run_simulation(manual)` produz `H6` finito; modelo replicado n_models×.
- [x] **AC-2** Filtro selecionado flui (resultados de filtros distintos diferem).
- [x] **AC-3** `ManualLayersModel.validate` pega tamanhos errados + ρᵥ<ρₕ (TIV λ≥1).
- [x] **AC-4** `apply_canonical_profile` → geology_mode="manual" + manual_layers do perfil + auto-geometria; `validate()==[]`.
- [x] **AC-5** Sessão roundtrip: hankel_filter + manual_layers restaurados.
- [x] **AC-6** Paridade `max|Δ|<1e-12` (manual numba vs jax, gated GPU).

## 3. Riscos/Fidelidade (testes-guarda)
- `ρᵥ ≥ ρₕ` (TIV) validado; `Σ(esp)` consistente; abscissas do filtro monotônicas (garantido por `FilterLoader`); manual reusa o caminho ragged (`_simulate_grouped`) — sem novo kernel.

## 4. GATE-S
- [x] 0 marcadores; RF→AC; física intocada (só `simulate_batch`, parâmetro já existente); paridade fixada.
