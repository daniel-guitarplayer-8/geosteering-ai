> **Blueprint de Arquitetura do Geosteering AI — CI/CD.** Índice: [README.md](README.md) · Constituição SDD: [../../specs/CONSTITUTION.md](../../specs/CONSTITUTION.md) · Roadmap: [../../specs/ROADMAP.md](../../specs/ROADMAP.md). Gerado 2026-06-05 (workflow multi-agente + revisão crítica).

## CI/CD — Pipeline por Produto, Gates, Release Automation

> Base auditada: `.github/workflows/ci.yml:1-97` (1 workflow monolítico) + `.github/workflows/docker.yml:1-125` (build+smoke API). Pre-commit em `.pre-commit-config.yaml:1-90` já enforça ruff/mypy/anti-patterns/errata-física/**paridade Fortran full** (hook `run-fortran-parity.sh`, modo `full` = 7 modelos ~146 s). Perf-gate em `.claude/perf_baseline.json` (`threshold_pct=90`; gate JAX-GPU A6000 oficial `jax_gpu_a6000_gate._meta.status="GATE OFICIAL"`). Markers existentes: `slow`, `gui` (`pyproject.toml:127-130`). Entry points: `geosteering-cli`, `geosteering-warmup` (`pyproject.toml:108-112`). `__version__="2.0.0"` em `geosteering_ai/__init__.py:63`.

### 1. Diagnóstico do estado atual (o que aproveitar / corrigir)

| Item | Onde | Status | Ação no redesign |
|:-----|:-----|:------:|:-----------------|
| Warmup JIT no CI | `ci.yml:62-68` (`geosteering-warmup --verbose --auto`) | implementado | Manter; isola cold-start antes do perf-gate |
| pytest-qt headless xvfb | `ci.yml:79-84` | parcial (`continue-on-error: true`, sem threshold de cobertura) | Promover a **gate** ≥80% cov GUI |
| tatu.x portável (anti-SIGILL AVX-512) | `ci.yml:52-56` (`make portable`, `-march=x86-64-v2`) | implementado | Reusar como composite action `setup-fortran` |
| Paridade Fortran <1e-12 | pre-commit `fortran-parity-full` (`.pre-commit-config.yaml`) + `tests/test_simulation_compare_fortran.py` | parcial (só pre-commit local, **não roda no CI runner**) | **GATE bloqueante** em job dedicado no PR |
| Perf não-regressão | `.claude/hooks/check-perf-regression.sh` (WARN-only, n=200) | parcial (não-bloqueante, fora do CI) | Job `perf-gate` bloqueante via `perf_baseline.json` |
| mypy / benchmark | `ci.yml:87-96` (`continue-on-error: true`) | parcial | mypy → gate; benchmark → observabilidade |
| Docker API | `docker.yml` (build+/health+/predict 503) | implementado | Reusar; adicionar push GHCR no release |
| Release automation | — | ausente | Novo `release.yml` (semver+tags+wheels+constructor+docker) |
| Branch strategy / environments | — | ausente | Trunk-based + 3 environments |
| GPU runner (A6000) | gate JAX existe em JSON, **sem runner** (`E-github-gpu`) | ausente | Self-hosted A6000 `[gpu, a6000, self-hosted]` |
| pip-audit / segurança | — | ausente | Job `security` (pip-audit + detect-secrets + Trivy) |
| Golden-tests modelo (MLOps) | — | ausente | Novo `tests/test_golden_models.py` + MLflow registry |

---

### 2. Estratégia de Branch, Ambientes e Mapa de Workflows

**Decisão: Trunk-based modificado** (1 trunk `main` protegido + branches efêmeras `feat/*`, `fix/*`, `chore/*` + `release/vX.Y` curta para freeze). Justificativa: solo-dev de alta cadência (≥5 commits/sprint — vide regra de relatórios), versões atribuídas no 1º commit da sprint (ADR-0001 R2). `develop` mantido como integração contínua (já referenciado em `ci.yml:5`).

```
┌──────────────────────────────────────────────────────────────────────┐
│  feat/v2.XX-*  ──PR──▶  develop  ──PR(squash)──▶  main  ──tag──▶ release │
│       │                    │                       │                    │
│   PR checks            nightly slow            release.yml             │
│   (fast gates)         + GPU gate              (semver/wheels/docker)   │
└──────────────────────────────────────────────────────────────────────┘
```

| Ambiente | Branch/Trigger | Deploy alvo | Aprovação |
|:---------|:---------------|:------------|:---------:|
| **dev** | push `feat/*`, `fix/*` | nada (só CI) | automático |
| **staging** | merge → `develop` | GHCR `:develop`, TestPyPI, nightly slow+GPU | automático |
| **prod** | tag `vX.Y.Z` em `main` | PyPI, GHCR `:vX.Y.Z`+`:latest`, GH Release (wheels+installers) | **manual** (`environment: prod` protegida) |

**Mapa de 5 workflows** (substitui o `ci.yml` monolítico):

| Arquivo | Trigger | Jobs principais | Bloqueante |
|:--------|:--------|:----------------|:----------:|
| `pr-checks.yml` | `pull_request` → main/develop | lint, typecheck, **fortran-parity**, test-fast, gui-cov, golden-models, security, build-validate | SIM |
| `ci-full.yml` | push main/develop | tudo do PR + test-slow (matriz 3.12/3.13) + docker-smoke | SIM |
| `nightly.yml` | `schedule` (cron 03:00) + `workflow_dispatch` | test-slow completo, **perf-gate** (Numba), **gpu-gate** (A6000 self-hosted), pip-audit fresh | report-only (cria issue) |
| `release.yml` | `push tags: v*.*.*` | semver-validate, build-wheel, conda-constructor (SM/Studio), docker-push, changelog, gh-release | SIM (gate prod) |
| `docker.yml` | paths API (existente) | build + /health + /predict smoke | SIM |

---

### 3. Pipeline por Produto (jobs × gates × matriz)

Cada produto compartilha a fundação (`geosteering_ai/simulation/` + `geosteering_ai/gui/`), então a **paridade Fortran e o perf-gate do simulador são gates transversais** que protegem todos. Diferenciação por *path-filter* + jobs específicos.

#### 3.1 Lib/API Python (wheel PyPI + REST FastAPI + Docker)

| Job | Trigger | Matriz | Gate |
|:----|:--------|:-------|:-----|
| `lint` | PR | 3.13 | ruff check + ruff format --check (bloqueia) |
| `typecheck` | PR | 3.13 | mypy `geosteering_ai/` (promovido a bloqueante; hoje `continue-on-error`, `ci.yml:94-96`) |
| `test-fast` | PR | **3.12 + 3.13** | `pytest -m "not slow and not gui" --cov=geosteering_ai --cov-fail-under=85` |
| `golden-models` | PR (path `models/`,`losses/`,`inference/`) | 3.13 | golden snapshots de inferência (MLOps) — ver §5 G7 |
| `api-contract` | PR (path `api/`) | 3.13 | schema OpenAPI diff + `test_api_*` (health/predict/schemas) |
| `docker-smoke` | push + tag | — | `docker.yml` (/health ok, /predict 503) |
| `build-wheel` | tag | 3.13 | `python -m build` + `twine check` + import-smoke do sdist |
| `test-slow` | ci-full/nightly | 3.13 | `pytest -m slow` (PINNs, training E2E) |

#### 3.2 CLI `geosteering-cli` / `geosteering-warmup`

| Job | Trigger | Gate |
|:----|:--------|:-----|
| `cli-smoke` | PR (path `cli/`) | `geosteering-cli version`, `--help`, `benchmark --scenario E --n 200`, `simulate --backend numba` e `--backend jax` |
| `warmup-gate` | PR + ci-full | `geosteering-warmup --verbose --auto` < 5 min (reusa `ci.yml:62-68`); valida 1 `.nbc`/função (test_simulation_numba_specializations) |
| `cli-backend-auto` | PR | `test_cli_backend_table` + (planejado) `--backend auto` quando gap fechar |

#### 3.3 Simulation Manager (SM = `gui/` + `simulation/`, GUI Qt6)

| Job | Trigger | Gate |
|:----|:--------|:-----|
| `fortran-parity` | PR (path `simulation/_numba/`,`forward.py`,`multi_forward.py`) | **<1e-12 sagrado, BLOQUEANTE** — roda `tests/test_simulation_compare_fortran.py` no runner com tatu.x portável |
| `jax-parity` | PR (path `simulation/_jax/`) | **<1e-10 JAX-vs-Numba** — `test_simulation_jax_fortran_parity` + `test_simulation_jax_complex64_parity` |
| `perf-gate` | ci-full/nightly | throughput ≥90% baseline (`perf_baseline.json` Numba, cenário E n=200) — bloqueia em ci-full |
| `gui-cov` | PR (path `gui/`,`simulation/tests/sm_*`) | `xvfb-run pytest -m gui --cov=geosteering_ai/gui --cov-fail-under=80` (promove `ci.yml:79-84` a gate; hoje 25% → meta 80%) |
| `gpu-gate` | nightly (self-hosted A6000) | `jax_gpu_a6000_gate` ≥90% (5 membros A/B/E/G/DG hot) |

#### 3.4 Geosteering AI Studio (FLAGSHIP, MVVM Qt6, 0% código → planejado)

| Job | Trigger | Status | Gate |
|:----|:--------|:------:|:-----|
| `studio-cov` | PR (path `studio/`) | planejado | pytest-qt MVVM (ViewModels testáveis sem display) cov ≥80% |
| `ingest-contract` | PR (path `ingest/`) | planejado (gap WITSML/ETP/LAS/DLIS) | round-trip parse→22-col contra fixtures golden |
| `studio-installer` | tag | planejado | conda-constructor build + launch-smoke headless |
| `mvvm-arch-lint` | PR | planejado | custom check: View não importa biblioteca DL direto (só via ViewModel) |

**Path-filters (resumo de roteamento):**

```
geosteering_ai/simulation/_numba/  ──▶ fortran-parity (BLOQUEIA, <1e-12)
geosteering_ai/simulation/_jax/    ──▶ jax-parity (<1e-10) + gpu-gate(nightly)
geosteering_ai/gui/ , sm_*.py      ──▶ gui-cov ≥80% (xvfb)
geosteering_ai/{models,losses}/    ──▶ golden-models (MLOps)
geosteering_ai/api/                ──▶ api-contract + docker-smoke
geosteering_ai/cli/                ──▶ cli-smoke + warmup-gate
geosteering_ai/ingest/ (planejado) ──▶ ingest-contract
geosteering_ai/studio/ (planejado) ──▶ studio-cov + mvvm-arch-lint
```

---

### 4. GATES OBRIGATÓRIOS — Tabela mestre (job × trigger × bloqueio × fonte)

| # | Gate | Job | Trigger | Bloqueia? | Implementação / fonte |
|:-:|:-----|:----|:--------|:---------:|:----------------------|
| G1 | **Paridade Fortran <1e-12 (SAGRADO)** | `fortran-parity` | PR (path simulator) + ci-full | **SIM** | `test_simulation_compare_fortran.py` + tatu.x portável (`ci.yml:52-56`); `FORTRAN_PARITY_MODE=full` |
| G2 | Paridade JAX <1e-10 | `jax-parity` | PR (path _jax) | SIM | `test_simulation_jax_fortran_parity.py`, `_complex64_parity.py` |
| G3 | pytest fast | `test-fast` | PR | SIM | `-m "not slow and not gui"` cov ≥85% (matriz 3.12/3.13) |
| G4 | pytest slow | `test-slow` | ci-full/nightly | SIM (ci-full) | `-m slow` (PINNs, training) |
| G5 | **pytest-qt headless ≥80% GUI** | `gui-cov` | PR (path gui) | SIM | `xvfb-run pytest -m gui --cov-fail-under=80` (promove `ci.yml:79`) |
| G6 | mypy | `typecheck` | PR | SIM | `mypy geosteering_ai/` (remover `continue-on-error`) |
| G7 | ruff (lint+format) | `lint` | PR | SIM | `ruff check` + `ruff format --check` (já em pre-commit) |
| G8 | **Golden-tests modelo (MLOps)** | `golden-models` | PR (path models/losses) | SIM | snapshots SavedModel/TFLite/ONNX + tol numérica — ver §5 |
| G9 | **Throughput não-regressão** | `perf-gate` | ci-full/nightly | SIM (ci-full) | `perf_baseline.json` ≥90% (`check-perf-regression.sh` → bloqueante) |
| G10 | GPU throughput (A6000) | `gpu-gate` | nightly self-hosted | SIM (nightly) | `jax_gpu_a6000_gate` 5 membros ≥90% |
| G11 | pip-audit / segurança | `security` | PR + nightly | SIM (CVE HIGH) | pip-audit + detect-secrets + Trivy (Docker) |
| G12 | Warmup JIT | `warmup-gate` | PR + ci-full | SIM (timeout 5min) | `geosteering-warmup --auto` (`ci.yml:62-68`) |
| G13 | Errata física | (pre-commit `validate-physics`) + `physics-guard` job | PR (path config.py) | SIM | `validate-physics.sh` (FREQUENCY_HZ/eps_tf/INPUT_FEATURES) |
| G14 | Anti-patterns / no-PyTorch | `policy-guard` | PR | SIM | `check-anti-patterns.sh` + `validate-no-pytorch.sh` (KB-013 prange aninhado, print, globals.get) |
| G15 | Acentuação PT-BR docs | `docs-guard` | PR (path *.md, docstrings) | WARN→SIM | `check-ptbr-accentuation.sh` |
| G16 | Build wheel + sdist | `build-wheel` | tag | SIM | `python -m build` + `twine check` |

**Notas de calibração herdadas (preservar):**
- `WARMUP_BLOCKING_THRESHOLD_S=5.0` no runner (mais lento que dev local — `ci.yml:73-76`).
- tatu.x **deve** ser recompilado portável no runner (binário commitado é `-march=native` AVX-512 → SIGILL — `ci.yml:47-51`).
- `libegl1 libgl1 libxkbcommon0 libdbus-1-3` obrigatórios mesmo sem rodar GUI (plugin pytest-qt importa PyQt6.QtGui no `pytest_configure` — `ci.yml:37-45`).

**Como o gate G9 (perf) é tornado determinístico:** runner público GH é compartilhado (variância alta) → perf-gate roda só em **ci-full/nightly** com `warmup` prévio (isola cold-start) e mediana de 5 runs, `stdev<5%`; bloqueia apenas se mediana <90% baseline. GPU-gate (G10) só em self-hosted A6000 (determinístico).

---

### 5. Golden-tests de modelo (MLOps) e MLflow registry (gap-fechamento)

Hoje **ausente**. Proposta acionável:

| Artefato novo | Conteúdo | Gate |
|:--------------|:---------|:-----|
| `tests/test_golden_models.py` | Para cada Tier-1 (ResNet_18, WaveNet, TCN): carrega pesos fixos (seed), roda forward num input canônico → compara vs snapshot `tests/golden/*.npz` (tol 1e-5 fp32) | G8 bloqueia se drift |
| `tests/golden/export_parity.py` | SavedModel vs TFLite vs ONNX no mesmo input (tol export 1e-4) | G8 |
| MLflow registry (`mlruns/` + `MLFLOW_TRACKING_URI`) | Registro de modelos por versão semver; CI faz `mlflow models validate` no tag de release | release.yml |

Pipeline MLOps no release: `train (manual/GPU local) → log MLflow → tag modelo → release.yml valida golden + export parity → anexa SavedModel/TFLite/ONNX ao GH Release`.

---

### 6. Release Automation (`release.yml`)

Trigger: `push: tags: ['v*.*.*']`. `environment: prod` (aprovação manual). Versão: **única fonte** = `geosteering_ai/__init__.py:63` (`__version__`); tag deve casar (gate `semver-validate`).

```
┌─ semver-validate ─┐   tag vX.Y.Z == __version__ ? senão FAIL
        │
        ├─▶ build-wheel ──────▶ PyPI (Lib/API)      [twine upload]
        │                       TestPyPI se prerelease (vX.Y.Z-rc.N)
        │
        ├─▶ conda-constructor-sm ──▶ installer SM (Linux .sh + Win .exe)
        ├─▶ conda-constructor-studio ▶ installer Studio (Phase-gated)
        │
        ├─▶ docker-push ─────▶ GHCR :vX.Y.Z + :latest  (reusa Dockerfile.cpu)
        │
        ├─▶ changelog-gen ───▶ git-cliff/commitizen a partir de Conventional Commits
        │                       (feat→minor, fix→patch, feat!→major)
        │
        └─▶ gh-release ──────▶ cria Release, anexa wheels + installers +
                               SavedModel/TFLite/ONNX (golden-validados)
```

| Produto | Formato de release | Ferramenta | Status |
|:--------|:-------------------|:-----------|:------:|
| Lib/API | wheel + sdist | `python -m build` + `twine` | planejado |
| API (deploy) | imagem OCI | `docker/build-push-action` → GHCR (reusa `Dockerfile.cpu`) | parcial (build existe, falta push) |
| CLI | incluso no wheel (entry points `pyproject.toml:108-112`) | — | implementado (entry points) |
| SM | installer conda (`.sh`/`.exe`) | `conda-constructor` | planejado |
| Studio | installer conda Phase-1 | `conda-constructor` | planejado |
| Changelog | `docs/CHANGELOG.md` (Keep-a-Changelog) | `git-cliff` + Conventional Commits | parcial (manual hoje) |

**Semver mapeado a Conventional Commits** (já em uso parcial — vide `feat(simulation):`, `docs(...)`):
`fix:`→patch · `feat:`→minor · `feat!:`/`BREAKING CHANGE`→major. Hook valida mensagem no PR (`policy-guard`).

---

### 7. GPU Runner — decisão A6000 self-hosted (item E-github-gpu)

| Critério | Self-hosted A6000 (local) | Cloud GPU (ephemeral) |
|:---------|:--------------------------|:----------------------|
| Custo | já adquirida (sunk) | $1-3/h ondemand |
| Determinismo perf-gate | **alto** (HW fixo = baseline `jax_gpu_a6000_v244`) | variável (HW heterogêneo) |
| Disponibilidade | nightly OK; PR pode esperar | sempre |
| Segurança | runner privado, sem secrets em PR de fork | isolado |

**Decisão: self-hosted A6000** com label `[self-hosted, gpu, a6000]`, usado **só em nightly** (`gpu-gate`), nunca em PR de fork (risco RCE em self-hosted). Baseline já é `jax_gpu_a6000_v244` (vide memória `gpu-dev-local-a6000`). Hardening: `if: github.event.pull_request.head.repo.full_name == github.repository`. Colab descontinuado (v2.44) — não há fallback cloud no fluxo padrão.

---

### 8. Diagrama ASCII — commit → PR → merge → release

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ DEV LOCAL                                                                     │
│  git commit ──▶ pre-commit (ruff·mypy·anti-patterns·errata·FORTRAN-PARITY    │
│                  full <1e-12 ·detect-secrets)                                 │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                 │ git push feat/v2.XX
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PR CHECKS  (pr-checks.yml — todos BLOQUEANTES p/ merge)        matriz 3.12+3.13│
│  ┌──────────┬──────────┬───────────────┬───────────┬──────────┬────────────┐ │
│  │ lint     │typecheck │ fortran-parity│ test-fast │ gui-cov  │ golden-    │ │
│  │ (ruff)   │ (mypy)   │  <1e-12 G1 ★  │ cov≥85 G3 │ ≥80% G5  │ models G8  │ │
│  ├──────────┼──────────┼───────────────┼───────────┼──────────┼────────────┤ │
│  │ security │warmup-   │ jax-parity    │ cli-smoke │ api-     │ policy-    │ │
│  │ G11      │gate G12  │ <1e-10 G2     │           │ contract │ guard G14  │ │
│  └──────────┴──────────┴───────────────┴───────────┴──────────┴────────────┘ │
│   branch protection: 1 review (CodeRabbit) + status checks verdes obrigatório │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                 │ squash-merge ──▶ main
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ CI-FULL (ci-full.yml)  +  NIGHTLY (cron 03:00, self-hosted A6000)             │
│   test-slow G4  ·  perf-gate G9 (Numba ≥90%)  ·  docker-smoke                 │
│   ──── nightly: gpu-gate G10 (A6000 ≥90%) · pip-audit fresh · cria issue ─────│
└───────────────────────────────┬─────────────────────────────────────────────┘
                                 │ bump __version__ + git tag vX.Y.Z  (1º commit sprint, ADR-0001 R2)
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ RELEASE (release.yml)  environment: prod  [APROVAÇÃO MANUAL]                  │
│  semver-validate ─▶ build-wheel→PyPI · conda-constructor(SM,Studio)          │
│                   · docker-push→GHCR :latest · changelog · gh-release         │
└─────────────────────────────────────────────────────────────────────────────┘
   ★ G1 paridade Fortran <1e-12 = SAGRADO. Quebra → merge BLOQUEADO, sem exceção.
```

---

### 9. Resumo job × trigger × gate (matriz consolidada)

| Job | PR | push main | nightly | tag | Matriz | Bloqueia |
|:----|:--:|:--------:|:-------:|:---:|:------:|:--------:|
| lint (G7) | ✓ | ✓ | | | 3.13 | SIM |
| typecheck (G6) | ✓ | ✓ | | | 3.13 | SIM |
| fortran-parity (G1★) | ✓ | ✓ | ✓ | | 3.13 | **SIM** |
| jax-parity (G2) | ✓ | ✓ | | | 3.13 | SIM |
| test-fast (G3) | ✓ | ✓ | | | **3.12+3.13** | SIM |
| test-slow (G4) | | ✓ | ✓ | | 3.13 | SIM (ci-full) |
| gui-cov (G5) | ✓ | ✓ | | | 3.13 | SIM |
| golden-models (G8) | ✓ | ✓ | | ✓ | 3.13 | SIM |
| perf-gate (G9) | | ✓ | ✓ | | 3.13 | SIM (ci-full) |
| gpu-gate (G10) | | | ✓ | | self-hosted | SIM (nightly) |
| security (G11) | ✓ | ✓ | ✓ | | 3.13 | SIM (HIGH) |
| warmup-gate (G12) | ✓ | ✓ | | | 3.13 | SIM |
| physics/policy (G13/14) | ✓ | ✓ | | | 3.13 | SIM |
| docker-smoke | path | ✓ | | ✓ | — | SIM |
| build/release | | | | ✓ | 3.13 | SIM |

★ = sagrado/inviolável.
