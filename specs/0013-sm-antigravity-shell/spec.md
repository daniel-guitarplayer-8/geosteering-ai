---
Spec: 0013-sm-antigravity-shell
Titulo: Shell Antigravity — activity rail + perspectivas empilhadas + secondary sidebar + tema profissional
Backlog-Code: F-mvc-split
Trilha-Dominante: F
Produtos: [SM, STU]
Converge-Em: gui/shell + gui/theme  # fundação compartilhada; sem física
Status: implementado
Released-As: v2.58
Constituicao: 1.0
Autor: Daniel Leal
Data: 2026-06-06
---

# Spec 0013 — Shell Antigravity & Tema

## 0. Nota de escopo

Redesign da **casca (shell)** e do **tema** da GUI MVVM para um look & feel **integralmente
inspirado no Google Antigravity IDE** — não só cores, mas layout, estética dos elementos e
organização dos recursos. Fundação **reusável** (SM + futuro Studio). NÃO toca física nem
ViewModels (Princípio X preservado). É o pré-requisito visual das Fatias 6a/6b.

**Anatomia Antigravity (pesquisada):** activity rail vertical (nav por ícones) · perspectivas
empilhadas no centro (`QStackedWidget`) · secondary sidebar à direita (Histórico/Log/Artifacts) ·
status bar com accent. Dark índigo/roxo (`#4f46e5`); cards (superfície elevada + sombra) e
botões pill. Limites do QtWidgets aceitos (sem blur; sombra via `QGraphicsDropShadowEffect`).

## 1. Requisitos Funcionais (RF)

| ID | Requisito | MoSCoW |
|:--|:--|:--:|
| RF-1 | `MainWindowBase` refatorada com **hooks de host** (`_init_host`, `_host_count/_host_widget/_host_current_index/_host_insert`) — comportamento de abas INALTERADO (default) | Must |
| RF-2 | `AntigravityMainWindow(MainWindowBase)`: layout activity rail + `QStackedWidget` + secondary sidebar + status bar; reusa `add_perspective`/lazy-build | Must |
| RF-3 | Widgets de shell reusáveis em `gui/shell/widgets/`: `ActivityBar`, `SecondarySidebar`, helpers `make_card`/`make_pill_button`/`apply_shadow` | Must |
| RF-4 | Activity rail expõe TODAS as perspectivas (Simulação ativa; Resultados/Benchmark/Análise/Preferências = scaffold desabilitado "em breve") → organização visível | Must |
| RF-5 | Tema: tokens ampliados (escala spacing, radius_md/lg, font_size_sm/lg, accent_light, contraste WCAG) + QSS Antigravity (cards, pill `[role]`, focus, statusbar accent, rail, dock, scrollbars, tabela) | Must |
| RF-6 | `app.py` usa `AntigravityMainWindow`; registra Simulação + scaffolds | Must |

## 2. Critérios de Aceite

- [x] **AC-1** Comportamento de abas de `MainWindowBase` inalterado (tests de fundação verdes; `perspective_count`/`add_perspective` preservados).
- [x] **AC-2** `AntigravityMainWindow` registra perspectiva, troca via rail, lazy-build (testável).
- [x] **AC-3** ViewModels intocados (Princípio X) — nenhuma lógica de domínio no shell.
- [x] **AC-4** Tokens puros (sem Qt); QSS gerado cobre cards/pill/rail/sidebar/statusbar; `test_gui_theme` verde.
- [x] **AC-5** Screenshot offscreen do shell renderiza (rail + stack + sidebar + status bar).
- [x] **AC-6** Sem regressão: suites GUI/SM verdes; PyQt6 primário + PySide6 fallback (sem código binding-específico).

## 3. Escopo

### IN
- `gui/shell/main_window_base.py` (hooks), `gui/shell/antigravity_window.py` (NOVO), `gui/shell/widgets/` (NOVO), `gui/theme/{tokens,stylesheet}.py`, `apps/sim_manager/{app,main_window}.py`, scaffolds de perspectiva, testes.

### OUT (futuro)
- Perspectivas Resultados/Benchmark/Análise/Preferências **completas** → Fatias 6c-6i.
- Animações/transições ricas, browser embutido (artifacts ao vivo).

## 4. GATE-S
- [x] 0 marcadores; RF→AC; IN/OUT explícito; fidelidade N/A (sem física); Princípio X preservado.
