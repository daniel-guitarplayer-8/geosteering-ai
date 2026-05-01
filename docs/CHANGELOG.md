# Changelog — Geosteering AI Simulation Manager

Todas as mudanças notáveis do Simulation Manager são documentadas aqui.

O formato segue [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/) e
o projeto usa [Versionamento Semântico](https://semver.org/lang/pt-BR/).

---

## [2.12] — 2026-04-30

### Workers Nativos no `simulate_multi`

Implementação da Sprint 12 (relatório técnico
[v2.11_simulador_python_analise_paralelismo_2026-04-30.md](reports/v2.11_simulador_python_analise_paralelismo_2026-04-30.md))
que migra o suporte a paralelismo inter-modelo da camada UI
(`tests/sm_workers.py`) para o **core do simulador**
(`geosteering_ai/simulation/_workers.py`).

### Adicionado

- **Novo módulo `geosteering_ai/simulation/_workers.py`** (~530 LOC)
  com 4 modos de execução:
  - **A** (Single, 1w × 1t) — debug/pequeno
  - **B** (Multi-Thread, 1w × Nt) — 1 simulação grande
  - **C** (Workers, Mw × 1t) — batch n_pos baixo
  - **D** (Hybrid, Mw × Kt) — ★ DEFAULT PRODUÇÃO
- `MultiSimulationResultBatch` (frozen dataclass) em
  `geosteering_ai.simulation` com campos:
  `H_stack`, `z_obs`, `elapsed_s`, `throughput_mod_per_h`,
  `backend`, `n_workers`, `n_threads`, `mode`.
- `release_pool()` exportado em `geosteering_ai.simulation` para
  cleanup explícito.
- 3 novos kwargs em `simulate_multi`:
  - `models: Optional[List[Dict]]` — batch de modelos
  - `n_workers: Optional[int]` — número de processos do pool
  - `threads_per_worker: Optional[int]` — threads Numba por worker
    (`None` = auto anti-oversubscription)
- 2 novos campos em `SimulationConfig`:
  `n_workers`, `threads_per_worker` (defaults `None`).
- **Anti-oversubscription automático**: quando
  `threads_per_worker is None`, usa `eff = max(1, cpu // n_workers)`.
- **Benchmark** `benchmarks/bench_v212_workers.py` com 4 modos
  comparativos.
- **17 novos testes** em `tests/test_simulation_workers.py`
  (paridade A/B/C/D, validação de input, métricas).
- **9 novos testes** em `tests/test_simulation_config.py` para
  validação dos novos campos.

### Migração Simulation Manager

- `closeEvent` agora libera tanto `release_numba_pool()` (UI)
  quanto `release_pool()` (core) — cleanup completo de ambos pools.
- API nativa **disponível** mas UI mantém pool próprio para
  preservar Pause/Cancel cooperativo (v2.11) com checkpoints
  `_wait_if_paused` + `_cancel_requested`.

### Métricas de paridade

- Modo A vs Modo B: **bit-exato** (`assert_allclose rtol=1e-14`).
- Modo A vs Modo C: **<1e-12** (tolerância FP entre processos).
- Modo A vs Modo D: **<1e-12** (tolerância FP entre processos).

### Backward-compat

- `simulate_multi(rho_h=..., rho_v=..., esp=..., positions_z=...)`
  retorna `MultiSimulationResult` (v2.11) — comportamento atual.
- `simulate_multi(models=[...], n_workers=N)` retorna
  `MultiSimulationResultBatch` (v2.12) — caminho novo.
- API existente intocada. **Zero regressão** em pytest
  (734 simulation tests passing, 0 failed).

### Smoke tests

- 197 → 202 (+5: T24-T28).

---

## [2.11] — 2026-04-29

### Análise Causa-Raiz do Freezing GUI

Identificados 5 gargalos `O(N)` na main thread via profiling instrumentado
(`MainThreadHeartbeat`):

1. `generate_models(N)` — loop síncrono na main thread (3-30s para 30k modelos)
2. `appendPlainText` log — `O(N²/100²)` cumulativo (5-30s acumulativos)
3. `_refresh_keys` combo populate — `O(min(100, N))` síncrono
4. `_append_simulation_snapshot` JSON serialize — `O(N)` na main thread
5. Pool spawn first-time — `O(n_workers)` na worker thread (UX gap)

### Adicionado

- `ModelGenerationThread` — geração de modelos assíncrona em `QThread` separada
- `PhaseTimer` — instrumentação permanente com sinais Qt (`phase_started`, `phase_completed`)
- `WorkerProgressWidget` — barras individuais por worker com health status
- `CorrelationBySlice` — p-values granulares por frequência + UI tabbed
- `SnapshotPersistThread` — persistência de snapshot em `QThread`
- `MainThreadHeartbeat` — sentinel de gaps na main thread (debug)
- Painel "Cronologia da Simulação" com tempos exatos de cada fase
- Botões de Pause/Resume/Cancel com sinais cooperativos

### Corrigido

- GUI travava por tempo proporcional a N (qualquer quantidade de modelos)
- Buffer de log com flush throttled (1 Hz) substitui `appendPlainText` direto
- Combo populate usa `setModel(QStringListModel)` em batch
- Cancelamento limpo do pool persistente em `closeEvent`

### Métrica de sucesso

- `max_gap_ms < 50ms` na main thread para qualquer N (100, 1k, 10k, 30k)
- Latência click → primeiro feedback < 200ms

### Smoke tests

- 156 → 166 (+10: T17-T26)

---

## [2.10] — 2026-04-28

### Adicionado

- Pool persistente Numba (`_acquire_numba_pool`, `_numba_init_worker`,
  `release_numba_pool`) — workers spawn/import/JIT 1× por sessão
- Defer mechanism: `_pending_sim_trigger` + `_prewarm_numba_pool`
  auto-disparam simulação ao concluir warmup

### Corrigido

- 1ª simulação 3× mais lenta que subsequentes (overhead de spawn/import/JIT)
- Fallback p-value combinado em `_compute_pvalues` agora retorna `1.0`
  (conservador, assume não-significância) em vez de `np.mean(pvals)` inválido

### Smoke tests: 148 → 156 (+8: T15-T16)

---

## [2.9] — 2026-04-28

### Adicionado

- `NumbaPrimer(QThread)` — pré-aquecimento assíncrono de cache JIT na startup
- Status bar label "🔥 JIT Numba…" → "✓ JIT (Xs)"

### Corrigido

- Race condition de timing no `crosshair` matplotlib — `CrosshairManager`
  removido completamente (234 LOC)
- Cache JIT invalidado após atualização Numba (1ª simulação 3× mais lenta)
- Aviso `OMP: omp_set_nested deprecated` suprimido (`KMP_WARNINGS=FALSE`)
- Tema do canvas agora persiste corretamente entre sessões (`canvas/theme`
  unificado em `_qsettings()`)
- `NumbaPrimer` lazy start em `showEvent` + cleanup em `closeEvent`

### Removido

- `sm_crosshair.py` (234 LOC) + 11 seções em `simulation_manager.py`
- Shortcut `Ctrl+Shift+C` + botão de toolbar

### Smoke tests: 142 → 148 (+6: T13-T14)

---

## [2.8] — anterior

- Plot kinds dinâmicos (`_on_kind_mode_changed`)
- `CorrelationAnalysisDialog` com método selecionável (Pearson/Spearman/Kendall)
- Export CSV de matriz de correlação
- Smoke tests T9-T12

---

## [2.7a] — 2026-04-25 (PR #29)

- Migração `PyQt6` + `PySide6` (compatibilidade dupla via `sm_qt_compat.py`)
- Bug fixes diversos + polimento de UX
- Smoke tests T1-T5 (binding, ALIGN_*/ORIENT_*, dark mode,
  `CollapsibleGroupBox`, `PyQtGraphCanvas`)

---

## [2.6b] — anterior

- Bug fix A1 (referenciado em `feat/simulation-manager-v2.6b`)
- Multi-backend foundation
- Smoke tests T6-T8

---

## [2.5] — 2026-04-25 (PR #26)

- `PlotComposerDialog`
- Fix cache LRU multi-freq × angle
- Fortran multi-TR + JAX `chunk_size`
- 70/70 smoke tests; 1464/0 pytest
