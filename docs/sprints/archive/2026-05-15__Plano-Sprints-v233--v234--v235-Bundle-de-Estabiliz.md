# Plano: Sprints v2.33 + v2.34 + v2.35 (Bundle de Estabilização & Observabilidade)

## Context

O Sprint v2.32 (`geosteering-warmup` entry point) está em `main` (commits
`e5a4dca…8a9f15c`, 2026-05-13). Três frentes não-bloqueantes mas de alto valor
foram listadas no `docs/ROADMAP.md` e priorizadas:

- **v2.33** — Testes pytest-qt da GUI do Simulation Manager. Hoje **não há
  cobertura automatizada** do componente Qt (~10.7k linhas, 1 MainWindow + 1
  SimulationThread). Bug em UI só é descoberto manualmente.
- **v2.34** — Integrar `geosteering-warmup` no GitHub Actions para isolar
  cold-start JIT do tempo medido de benchmarks e atualizar o `PERFORMANCE_BASELINE`
  com referência "warm cache" comparável entre PRs.
- **v2.35** — Adicionar Cenário H (8×8×8 = 512 combos) ao benchmark CLI como
  stress-test multi-core, complementando o catálogo A–G atual.

Os 3 sprints são **independentes** (sem dependências cruzadas em código).
Ordem proposta: **v2.35 → v2.34 → v2.33** (menor → maior risco), mas pode ser
qualquer ordem desde que sejam **3 PRs separados** (1 PR por sprint) — facilita
review e bisect futuro.

---

## Sprint v2.35 — Cenário H (8×8×8 = 512 combos)

### Arquivos a modificar

| Arquivo | Ação | LOC estimado |
|:--------|:-----|:-------------|
| [geosteering_ai/cli/benchmark.py:105-128](geosteering_ai/cli/benchmark.py#L105-L128) | Adicionar entrada `"H"` ao dict `SCENARIOS` | ~5 |
| [geosteering_ai/cli/main.py:290-302](geosteering_ai/cli/main.py#L290-L302) | Estender `choices=["A","B","C","D","E","F","G"]` para incluir `"H"` | 1 |
| [tests/test_cli_mvp.py](tests/test_cli_mvp.py) | Adicionar 3 testes (espelho dos testes de G): help text contém H, dict tem H, smoke run com n=2 workers=2 threads=2 timeout=600s | ~60 |
| [geosteering_ai/cli/main.py:81](geosteering_ai/cli/main.py#L81) | Bump `SIMULATION_MANAGER_VERSION = "v2.35"` (após o sprint completar) | 1 |
| [docs/CHANGELOG.md](docs/CHANGELOG.md) | Bloco `[v2.35]` curto: motivação + parâmetros do H + nota sobre timeout | ~25 |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Marcar v2.35 como done | 1 |
| [CLAUDE.md](CLAUDE.md) | Atualizar linha "Simulation Manager" v2.32 → v2.35 | 1 |

### Parâmetros do Cenário H (proposta)

```python
"H": {
    "n_pos": 100,
    "freqs": (1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5),  # 8 freq log-spaced
    "trs":   (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5),  # 8 TR
    "dips":  (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 105.0),  # 8 dip
},
```

### Riscos e mitigações

- **5.12M forward calls com `--n 100`** (100 modelos × 100 pos × 512 combos)
  pode levar >5min em single-thread. **Mitigação**: smoke test no CI usa
  `--n 2 --workers 2 --threads 2 --timeout 600`. Documentar no help.
- **Não há persistência CSV/MD do CLI** — o handle_benchmark atual só imprime
  stdout (sm_output/benchmark_summary.csv é legado). **Decisão**: NÃO criar
  persistência neste sprint (escopo creep). Adiar para v2.36.

### Funções a reutilizar

- `handle_benchmark()` em [geosteering_ai/cli/benchmark.py:168-301](geosteering_ai/cli/benchmark.py#L168-L301) — sem alteração estrutural; só leitura do dict expandido.
- `simulate_multi()` (lazy import linha 221) — já suporta listas N-dim de freqs/trs/dips.

---

## Sprint v2.34 — `geosteering-warmup` no CI/CD

### Arquivos a modificar

| Arquivo | Ação | LOC estimado |
|:--------|:-----|:-------------|
| [.github/workflows/ci.yml](.github/workflows/ci.yml) | Adicionar 2 novos steps após install: (1) `geosteering-warmup --verbose` (2) `geosteering-cli benchmark --scenario E --n 200` (capturando rate em log) | ~15 |
| [.github/workflows/ci.yml:13-14](.github/workflows/ci.yml#L13-L14) | Manter matriz `python-version: ["3.13", "3.12"]` | 0 |
| [docs/PERFORMANCE_BASELINE.md](docs/PERFORMANCE_BASELINE.md) | Nova seção "Warm-Cache Baseline (CI)" com instruções para interpretar números pós-warmup; bump versão para v2.34 | ~30 |
| [.claude/perf_baseline.json](.claude/perf_baseline.json) | Adicionar chave `scenarios.E_n200_warm` com baseline "warm" (a ser medido na primeira run do CI pós-merge); bump `version: "v2.34"` | ~10 |
| [docs/CHANGELOG.md](docs/CHANGELOG.md) | Bloco `[v2.34]` curto | ~20 |
| [CLAUDE.md](CLAUDE.md) | Atualizar linha "Simulation Manager" para v2.34 | 1 |

### Patch proposto em `ci.yml` (após o step de install, antes do pytest)

```yaml
      - name: Warm up JIT/LLVM cache (Sprint v2.32+)
        run: geosteering-warmup --verbose
        timeout-minutes: 5

      - name: Benchmark smoke (cenário E n=200)
        run: |
          python -m geosteering_ai.cli benchmark --scenario E --n 200 \
            | tee benchmark_ci.log
        timeout-minutes: 10
        continue-on-error: true  # não bloqueia merge — só observabilidade
```

### Decisões intencionais

- **Sem `actions/cache` para NUMBA_CACHE_DIR**: cache em `/tmp` é efêmero por
  design — adicionar `actions/cache` introduz inconsistência (cache hit vs miss
  produz tempos drasticamente diferentes). Warm-up explícito é mais robusto.
- **`continue-on-error: true` no benchmark**: o objetivo é observabilidade
  histórica, não gating. Hook `check-perf-regression.sh` já existe localmente
  para devs.

---

## Sprint v2.33 — pytest-qt Suite para GUI Simulation Manager

### Arquivos a criar/modificar

| Arquivo | Ação | LOC estimado |
|:--------|:-----|:-------------|
| [pyproject.toml](pyproject.toml) | Adicionar `pytest-qt>=4.4` a `[project.optional-dependencies] dev` | 1 |
| [tests/conftest_qt.py](tests/conftest_qt.py) **NOVO** | Fixture `qt_app` (singleton QApplication com headless support via `QT_QPA_PLATFORM=offscreen`) + fixture `mock_simulation_thread` (substitui `SimulationThread` por stub que emite sinais sintéticos) | ~80 |
| [tests/test_simulation_manager_gui.py](tests/test_simulation_manager_gui.py) **NOVO** | 15–20 testes pytest-qt cobrindo: abertura MainWindow, WelcomeWidget → SimulatorPage, validação de parâmetros, start/pause/cancel via `qtbot.mouseClick`, sinais `simulation_complete`/`error`/`progress_update`, `Preferences` salva/restaura QSettings, error path quando `SimulationThread` emite `.error.emit("...")` | ~250 |
| [.github/workflows/ci.yml](.github/workflows/ci.yml) | Adicionar `xvfb-run -a pytest tests/test_simulation_manager_gui.py` (Linux headless) — coordenar com mudanças do v2.34 | ~5 |
| [docs/CHANGELOG.md](docs/CHANGELOG.md) | Bloco `[v2.33]` | ~25 |
| [CLAUDE.md](CLAUDE.md) | Atualizar linha "Simulation Manager" e nota sobre cobertura GUI | 2 |

### Pontos de extensão da GUI a testar

| Componente | Arquivo:linha | Sinal/Botão | Teste |
|:-----------|:-------------|:------------|:------|
| `MainWindow` | [simulation_manager.py:7370](geosteering_ai/simulation/tests/simulation_manager.py#L7370) | `show()` | T1: janela abre sem warnings |
| `SimulatorPage` | [simulation_manager.py:2297](geosteering_ai/simulation/tests/simulation_manager.py#L2297) | `request_start` | T2: clicar `btn_start` emite signal |
| `btn_pause` | [simulation_manager.py:2526](geosteering_ai/simulation/tests/simulation_manager.py#L2526) | checkable QPushButton | T3: toggle muda estado |
| `btn_cancel` | [simulation_manager.py:2539](geosteering_ai/simulation/tests/simulation_manager.py#L2539) | `request_cancel` | T4: emite cancel signal |
| `SimulationThread` | [sm_workers.py:470](geosteering_ai/simulation/tests/sm_workers.py#L470) | `progress_update`, `finished_all`, `error` | T5–T7: mock emite sinais → UI atualiza |
| `ParametersPage.to_dict/from_dict` | [simulation_manager.py:2229](geosteering_ai/simulation/tests/simulation_manager.py#L2229) | persistência | T8: roundtrip preserva valores |
| `QSettings cache` (v2.29.2) | preferências | T9: limites de cache salvos/restaurados |

### Padrões a reutilizar

- **`sm_qt_compat.QT_BINDING`** em `geosteering_ai/simulation/tests/sm_qt_compat.py` — usar para imports agnósticos PyQt6/PySide6 nos testes.
- **`SimulationThread` mockado** — não rodar simulação real em testes GUI; usar `MagicMock` que emite sinais via `QtCore.QTimer.singleShot(50, lambda: thread.finished_all.emit({...}))`.

### Riscos

- **Linux headless**: requer `xvfb-run` no CI. Já é prática comum; documentar em CHANGELOG.
- **PyQt6 vs PySide6**: testes devem usar `sm_qt_compat` para evitar lock-in. Validar em ambas em uma run local antes de PR.
- **MainWindow é gigantesca (~10.7k linhas)**: NÃO refatorar nesse sprint. Apenas adicionar testes em pontos de entrada estáveis.

---

## Estratégia de Implementação

### Ordem recomendada

1. **v2.35** (menor, ~2–3h) — adiciona Cenário H, valida com smoke local.
2. **v2.34** (médio, ~3–4h) — depende de v2.35 estar em main para usar H opcionalmente (ou ficar com E).
3. **v2.33** (maior, ~6–8h) — adiciona pytest-qt + suite completa.

Cada sprint = 1 PR separado. Não bundle-ar — facilita review e rollback.

### Para cada sprint

1. **Backup** dos arquivos a modificar em `.backups/v{sprint}_$(date +%Y-%m-%d_%H%M%S)/`.
2. **Implementar** mudanças seguindo D1–D14.
3. **Rodar local**:
   - v2.35: `pytest tests/test_cli_mvp.py -v -k benchmark`
   - v2.34: validar YAML sintaxe (`yamllint`); push para branch + verificar CI verde.
   - v2.33: `xvfb-run -a pytest tests/test_simulation_manager_gui.py -v` em Linux; sem `xvfb` em macOS.
4. **`/geosteering-code-reviewer`** — revisão estrutural local.
5. **`/code-review`** (CodeRabbit `cr review --agent -t uncommitted`) — revisão de PR.
6. **Aplicar findings** críticos/major antes de commitar.
7. **Commits granulares** (1 por mudança lógica).
8. **Validação final**:
   - `pytest tests/test_simulation_compare_fortran.py -v` (paridade Fortran <1e-12 sagrada)
   - Suite completa do sprint correspondente

### Critérios de aceitação por sprint

| Sprint | Critério |
|:------:|:---------|
| v2.35 | 3 novos testes PASS; smoke `geosteering-cli benchmark --scenario H --n 2 --workers 2 --threads 2` completa em <10min em laptop M-series |
| v2.34 | CI verde após merge; baseline `E_n200_warm` registrado em `.claude/perf_baseline.json`; throughput warm ≥ 110% do throughput cold atual |
| v2.33 | 15+ testes pytest-qt PASS local + CI (xvfb); zero `QWarning`/`QCritical` em stderr; cobertura ≥80% de `SimulatorPage.btn_*` |

---

## Verificação End-to-End

```bash
# Após v2.35 mergeado
geosteering-cli benchmark --scenario H --n 2 --workers 2 --threads 2

# Após v2.34 mergeado
gh run watch  # confirmar CI verde com warmup + benchmark steps

# Após v2.33 mergeado
xvfb-run -a pytest tests/test_simulation_manager_gui.py -v
pytest tests/ -v --tb=short  # full regression — esperado 1700+ PASS

# Paridade Fortran (sagrada — após cada sprint)
pytest tests/test_simulation_compare_fortran.py -v
```

---

## Funções/Padrões Reutilizados

- **Dict `SCENARIOS`** em `cli/benchmark.py` — extensão por chave (v2.35)
- **`build_parser()` argparse** em `cli/main.py` — adicionar choice "H" (v2.35)
- **`geosteering-warmup` entry point** — disponível no PATH após `pip install -e .` (v2.34)
- **`.claude/hooks/check-perf-regression.sh`** — já consulta `perf_baseline.json` (v2.34)
- **`sm_qt_compat.QT_BINDING`** — agnóstico PyQt6/PySide6 (v2.33)
- **`SimulationThread` sinais** (`progress_update`, `finished_all`, `error`) — mockáveis via `MagicMock` (v2.33)

---

## Riscos Cruzados

- **Erratas físicas inviolada**: nenhum sprint toca `simulation/forward.py` ou
  `simulation/multi_forward.py`. Paridade Fortran <1e-12 preservada por construção.
- **Performance baseline**: v2.34 introduz métrica "warm" que pode mascarar
  regressões reais. Manter também a métrica "cold" no histórico (`E_n200` vs
  `E_n200_warm`).
- **Quality Mesh L1/L2/L5**: hooks de pre-commit (`check-anti-patterns.sh`,
  `validate-no-pytorch.sh`, ruff, mypy) já cobrem os 3 sprints — sem mudanças.

---

## Commits Granulares Sugeridos (por sprint)

### v2.35
1. `feat(cli): v2.35 add Cenário H to benchmark (8×8×8 = 512 combos)`
2. `test(cli): v2.35 add 3 tests for benchmark scenario H`
3. `docs(sm): v2.35 CHANGELOG + CLAUDE.md + ROADMAP`

### v2.34
1. `ci(workflows): v2.34 add geosteering-warmup pre-step + benchmark smoke`
2. `docs(perf): v2.34 add warm-cache baseline section + perf_baseline.json bump`
3. `docs(sm): v2.34 CHANGELOG + CLAUDE.md`

### v2.33
1. `build(dev): v2.33 add pytest-qt>=4.4 to dev dependencies`
2. `test(gui): v2.33 add conftest_qt.py fixtures (qt_app + mock_sim_thread)`
3. `test(gui): v2.33 add 15+ pytest-qt tests for Simulation Manager GUI`
4. `ci(workflows): v2.33 add xvfb-run for headless GUI tests on Linux`
5. `docs(sm): v2.33 CHANGELOG + CLAUDE.md`
