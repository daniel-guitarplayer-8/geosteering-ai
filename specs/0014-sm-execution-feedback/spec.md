---
Spec: 0014-sm-execution-feedback
Titulo: Fatia 6a — Execução & feedback (progresso, cancelamento, pause/resume, log, status)
Backlog-Code: F-mvc-split
Trilha-Dominante: F
Produtos: [SM, STU]
Converge-Em: simulate_batch  # física intocada; só feedback de execução
Status: implementado
Released-As: v2.58
Constituicao: 1.0
Autor: Daniel Leal
Data: 2026-06-06
---

# Spec 0014 — Fatia 6a — Execução & feedback

## 0. Escopo

Feedback de execução para sims longas (GPU/ensembles grandes), no shell Antigravity:
**progresso** (barra), **cancelamento**, **pause/resume**, **log** (secondary sidebar)
e **status** (estado/elapsed/throughput). Física intocada — só instrumentação.

## 1. Requisitos Funcionais (RF)

| ID | Requisito | MoSCoW |
|:--|:--|:--:|
| RF-1 | `_run_simulation`/`_simulate_grouped` aceitam `progress_callback`/`cancel_event`/`pause_event` (kwargs opcionais); progresso emitido **por-grupo**; cancel checado **entre grupos** → retorna `{"cancelled": True}` (resultado parcial descartado, não corrompido) | Must |
| RF-2 | `Worker` (flag `report_progress`) injeta `progress_callback=self.signals.progress.emit` → cross-thread via QueuedConnection. `BaseService._run_async(report_progress=...)` | Must |
| RF-3 | `SimulationService`: `_cancel_event`/`_pause_event` (threading.Event); `numba` in-thread recebe events + report_progress; `jax`/auto subprocesso = SEM progresso/cancel (v1, no-op); `request_cancel/pause/resume` | Must |
| RF-4 | `SimulationViewModel` (PURO): estado `progress_done/total`, `status_display` (estado/elapsed/throughput), `log_entry` VMSignal, `request_cancel/pause/resume`; trata resultado `cancelled`; guard de reentrância | Must |
| RF-5 | `SimulatorView`: botões Pausar/Cancelar + `QProgressBar` + labels elapsed/throughput; estados habilitado/desabilitado por status. Log/Histórico → secondary sidebar (via perspective + ctx.extras) | Must |

## 2. Critérios de Aceite

- [x] **AC-1** Progresso emitido por-grupo (numba); `progress_done/total` no VM avança.
- [x] **AC-2** Cancelamento (numba) entre grupos → `{"cancelled": True}` → status `"cancelled"`; resultado parcial descartado.
- [x] **AC-3** jax/subprocesso: caminho inalterado (sem progress/cancel intra-run; finished/error normais).
- [x] **AC-4** VM puro testável sem Qt; `status_display` (estado/elapsed/throughput); guard reentrância.
- [x] **AC-5** Fidelidade `<1e-12` preservada (progresso/cancel NÃO alteram a ordem de ops nem o resultado de uma run completa).
- [x] **AC-6** Sem regressão: caminho numba completo idêntico; suites verdes.

## 3. Riscos (testes-guarda)
- Sem pool persistente (regressão warmup); JAX só no subprocesso (TLS); sem `.wait()` na main (deadlock); progresso só via QueuedConnection.

## 4. GATE-S
- [x] 0 marcadores; RF→AC; fidelidade preservada (resultado de run completa inalterado).
