# Roadmap Completo — Geosteering AI v2.0+

**Versão do documento:** 1.3 (Maio 2026)
**Autor:** Daniel Leal
**Projeto:** Inversão 1D de Resistividade via Deep Learning para Geosteering
**Framework:** TensorFlow 2.x / Keras (exclusivo)
**Repositório:** `github.com/daniel-guitarplayer-8/geosteering-ai`

---

## Marcos recentes do Simulation Manager

| Versão  | Data       | Tema central                                                  | Smoke tests | Branch / PR                          |
|:-------:|:-----------|:-------------------------------------------------------------|:-----------:|:-------------------------------------|
| v2.5    | 2026-04-25 | PlotComposerDialog + cache LRU multi-freq×angle + JAX chunk  | 70/70       | PR #26 (mergeado)                    |
| v2.6b   | -          | Bug fix A1 + multi-backend foundation                        | -           | PR (mergeado)                        |
| v2.7a   | 2026-04-27 | Migração PyQt6+PySide6 + bug fixes + UX polish               | -           | PR #29 (mergeado)                    |
| v2.8    | -          | Plot kinds dinâmicos + CorrelationAnalysisDialog             | 142         | -                                    |
| v2.9    | 2026-04-28 | NumbaPrimer + remoção crosshair + QSettings fix              | 148         | -                                    |
| v2.10   | 2026-04-28 | Pool persistente Numba + p-value fallback conservador        | 156         | -                                    |
| v2.11   | 2026-04-29 | Eliminação freezing GUI + Pause/Cancel cooperativo + PhaseTimer | 197         | `feat/simulation-manager-v2.11`      |
| v2.12   | 2026-04-30 | Workers Nativos no `simulate_multi` (4 modos A/B/C/D) + anti-oversubscription | 202         | `feat/simulation-manager-v2.12`      |
| v2.13   | 2026-05-01 | Otimizações Numba JIT — prange(nf) + cache cross-call + nogil universal (Sprints 13.1+13.2+13.4) | 165 (152+13 novos)     | `feat/simulation-manager-v2.13` (mergeado) |
| v2.14   | 2026-05-01 | prange combinado TR×ângulo (Sprint 13.3) + fastmath seletivo hankel.py (Sprint 13.4) + benchmark formal CLI 4 cenários | 27 novos (8+6+13) | PR #33 (mergeado) |
| v2.15   | 2026-05-01 | Hardware validation 4 cenários, JIT cache observability (`get_jit_cache_info`), code review P1, fix CI tatu.x exec format Linux | 31 novos (4 JIT + 27 v2.14) | `feat/simulation-manager-v2.15` |
| v2.16 | 2026-05-01 | Fix regressão crítica de threading masking (4–8× em produção GUI) + Cenário E benchmark production scale (600 pts) + I/O `write_dat_from_tensor` vetorizado ≥3× | 7 novos (3 threading + 4 I/O) | `feat/simulation-manager-v2.16` |
| v2.17 | 2026-05-02 | Fix regressão de oversubscrição em CPUs Hyperthreading/SMT (3× em produção GUI): `detect_cpu_topology()` + `recommend_default_parallelism()` + warning visual GUI + logging diagnóstico. Defaults baseados em cores físicos (não lógicos): 8C/16T HT → (4w × 2t = 8) em vez de (4w × 4t = 16) | 19 novos (5 detection + 6 recommendation + 6 simulated hardware + 2 no-regression) | `feat/simulation-manager-v2.17` |
| v2.18 | 2026-05-02 | Fix throughput reportado erroneamente (38k→85k): `t0_sim` definido após pool warm (via `_noop` tasks pós-`_acquire_numba_pool`) + `PoolWarmupThread` pré-aquece pool em background ao abrir GUI → primeira simulação sem cold-start overhead | 4 novos smoke (T29–T32) | `feat/simulation-manager-v2.17` |
| v2.19 | 2026-05-02 | Fix bug funcional do gerador aleatório (`rng_seed=42` hardcoded) + UI control de semente (default aleatório) + `nogil=True` universal no hot path Numba + benchmark CLI CPU-aware defaults + warning de oversubscrição. Cenário A: 189k→802k mod/h (4×) com defaults corretos. Auditoria PyQt6 descarta como causa-raiz | 12 novos (7 random seed + 5 pool warmup) | `feat/simulation-manager-v2.19` |
| v2.20 | 2026-05-02 | Investigação empírica rigorosa (5 runs × 2 configs) refuta hipótese "HT-aware = logical cores" e confirma estratégia v2.17 (phys_cores) como ótima. 4w × 2t mediana 46k mod/h em E vs 4w × 4t mediana 38k mod/h (-25%). Cenário A com defaults v2.20: 1.18M mod/h | 1 novo (M2/M3 Pro topology) | `feat/simulation-manager-v2.20` |
| **v2.21** | **2026-05-02** | **CAUSA-RAIZ DA REGRESSÃO HISTÓRICA encontrada via análise comparativa `old_geosteering_ai/`: Sprint 13.1 (v2.13) introduziu `@njit(parallel=True)` em `_fields_in_freqs_kernel_cached` (chamada milhões de vezes de prange outer) → overhead de paralelismo aninhado pago em cada chamada (~14s acumulado em Cenário E). Fix: remove `parallel=True`, range serial. Cenário E saltou 46k → 122k mod/h (2.65× ganho), atingindo meta histórica >120k. Paridade Fortran <1e-12 preservada.** | **0 novos (regressão é fix arquitetural; 68/68 testes existentes PASS)** | **`feat/simulation-manager-v2.21`** |
| **v2.22** | **2026-05-08** | **FLAT prange 4D (`nTR×nAng×n_pos×nf`) elimina `range(nf)` serial em `_fields_in_freqs_kernel_cached`. Cenário B +11%, F +9% single-process; E sem regressão. 27 testes paridade FLAT vs legacy bit-exato. Suite total 1597 PASS / 295 SKIP / 0 FAIL.** | 27 novos (paridade FLAT) | `feat/simulation-manager-v2.22-flat-prange` |
| **v2.22.4** | 2026-05-09 | **Promoção `cfg.use_flat_prange=True` a default** (após validação 1 semana opt-in). Backward-compat preservada via flag. Sem regressão Cenário E. Desbloqueia Sprint v2.23. | mantém | `main` (mergeado) |
| **v2.22.5** | 2026-05-09 | **Skills agent-config-override**: physics-reviewer Sonnet→Opus 4.7 extra-high; simulator-fortran+python+jax+pinns model+effort campos; orquestrador override conforme §19. 13 skills configuradas. 0 CodeRabbit findings. | — | `main` (mergeado) |
| **v2.22.6** | **2026-05-09** | **Fase 1 Fundação Multi-Agente COMPLETA (§22.1)**: I1.2 skill `geosteering-simulator-numba.md` (Opus 4.7 extra-high, 384 LOC) + I1.9 MCP `physics-validator` handlers REAIS (6 tools, 15 testes) + I1.10 MCP `numba-profiler` handlers REAIS (6 tools, 17 testes) + `get_jit_cache_info()` em `multi_forward.py`. Fix dos 3 erros pré-existentes ruff/mypy (F821 + 2× type narrowing). Working tree clean: 0 SKIP em pre-commit. | 30 novos (15+15) | `feat/fase1-fundacao-multiagente` |
| **v2.22.7-docs** | **2026-05-09** | **Pré-Mortem inaugural + atualizações arquiteturais**: relatório `premortem_geosteering_ai_2026-05-09.md` + §74 (Backends de Inversão Alternativa: Occam + LUT + Tikhonov) + §75 (Framework-Agnostic Core) + §24.4 (cadência de pré-mortem) + skill `geosteering-premortem-analyst`. Não muda código de produção. | — | `feat/premortem-analysis-artifacts` |
| **v2.23** | **2026-05-10** | **Fastmath Dual-Mode + Threads Adaptativos** (multi-agente Hub-and-Spoke): A.1 `@njit(fastmath=True)` em `_numba/geometry.py` (3 funções) + `_numba/rotation.py` (2 funções) — paridade Fortran <1e-12 PRESERVADA em 7 modelos canônicos / 10 testes (oklahoma_3, 5, 28, hou_7, devine_8, oklahoma_15, viking_graben_10). A.2 auto-detect via `recommend_default_parallelism()` no `__post_init__` quando `n_workers=None AND threads_per_worker=None` — backward-compat preservada (n_workers=1 explícito = single-process). Defesa em camadas KB-019: warning se `n_workers × threads > 4× physical_cores`. Tolerante a falhas: try-except no lazy import. Hook `setup-environment.sh` exibe topologia + defaults v2.23. Skill v2.23: `geosteering-physics-reviewer` (Opus 4.7) + `geosteering-perf-reviewer` (Haiku 4.5) + `geosteering-code-reviewer` (Sonnet 4.6) em paralelo. | 7 novos (v2.23) + 1 atualizado (test_defaults_*) | `feat/sprint-v2.23-fastmath-adaptive-threads` |
| **v2.29.3** | **2026-05-11** | **Investigação de regressão + infraestrutura anti-regressão**: usuário relatou regressão pós-v2.29.2 em Cenário E n=2000 (config padrão). Benchmarks lado-a-lado (commit `24fba72` vs `e3a0617`, cache limpo, 3 runs) CONFIRMAM: v2.29.2 é **+3.8% MELHOR** que v2.29.1 (mediana 95,124 vs 91,620 mod/h). **Não há regressão.** Análise estática (2 Explore agents) confirma que todas as 4 mudanças v2.29.2 são periféricas ao caminho crítico Numba JIT (STARTUP only + UI Preferences + signal não-disparado durante simulação). **Mudanças aplicadas**: (1) Fix `is_paused()` em `_resume_simulation` (era `sim.is_paused` sem parênteses — referência do método, sempre truthy; não afetava throughput por idempotência de `request_resume`). (2) Hook `.claude/hooks/check-perf-regression.sh` (anti-regressão automático Cenário E n=200 vs `.claude/perf_baseline.json`). (3) `docs/PERFORMANCE_BASELINE.md` (tabela canônica de cenários + processo de atualização). (4) Skill `.claude/commands/geosteering-perf-baseline.md` (reviewer perf-aware com heurística de severidade INALTERADO/MELHORIA/MINOR/MAJOR/CRÍTICA). **Aprimoramentos arquiteturais**: 5 níveis de defesa anti-regressão (hooks + baseline doc + skills + 37 tests + processo humano). Paridade Fortran <1e-12 PRESERVADA. CodeRabbit: 6 findings → todos aplicados. Limitação: 150k mod/h histórico não atingido em medições atuais (~95k em n=2000); investigar Sprint v2.30+ (condições de medição). | 0 novos | `main` |
| **v2.29.2** | **2026-05-11** | **Cache LRU configurável + varredura GUI**: Resolve feature pendente desde v2.5 ("Aumente o limite de cache em Preferências (futuro)"). `LRUPlotCache` tinha limite hardcoded 500 MB em 2 locais (`sm_plot_cache.py:73` e `simulation_manager.py:7368`). Em multi-freq × multi-angle com `complex128`, tensor histórico facilmente excede 5× esse limite (calculado: 1000 mod × 2 TR × 4 dips × 600 pos × 4 freq × 9 comp × 16 B = **2.77 GB**). **Solução**: (1) `default_max_bytes()` em `sm_plot_cache.py` — auto-detect 10% RAM via psutil, piso 500 MB, teto 4 GB, fallback robusto em qualquer Exception. (2) `LRUPlotCache.__init__(max_bytes: Optional[float] = None)` ativa auto-detect. (3) Persistência QSettings (`cache/max_bytes_mb`, `cache/maxlen`) — lê em `MainWindow.__init__` com validação. (4) UI Preferências group "Cache LRU de Tensores": 2× QSpinBox (limite MB range [100,16384], snapshots [1,20]) + botão Auto-detect + label informativo. Signal `cache_settings_changed(int, int)` reinstancia cache preservando snapshots e atualiza widgets dependentes (`_update_cache_status`, `_refresh_results_experiment_combo`, `mark_history_out_of_cache`). (5) Mensagem dinâmica do diálogo (mostra limite atual + linka "Preferências → Cache LRU"). **Varredura GUI**: smoke test reportava 3 falhas pré-existentes desde v2.29 (`is_paused`/`is_cancelled` viraram métodos mas teste tratava como properties) — corrigidas. Confirmado que GUI v2.21 está integralmente adaptada (12 classes preservadas) — nada para adaptar. Suite total 30 PASS + 7 novos LRU PASS. Paridade Fortran <1e-12 PRESERVADA. CodeRabbit: 4 findings na 1ª passada + 0 findings após fixes. | 7 novos (LRU cache) | `feat/simulation-manager-v2.21` (branch reutilizado) |
| **v2.29.1** | **2026-05-11** | **Fix `NUMBA_NUM_THREADS` no PAI + restauração de 150k mod/h**: a v2.29 reintroduziu bug latente — setar `os.environ["NUMBA_NUM_THREADS"]` dentro de `run_numba_chunk` (worker) falha quando ambiente do pai tem `NUMBA_NUM_THREADS` setado para valor distinto. Em modo spawn, o módulo `sm_workers` é importado durante bootstrap (para unpickle de `run_numba_chunk`), encadeando o import de numba ANTES de qualquer linha da função rodar. Setar env var dentro do worker é tardio demais → `RuntimeError: Cannot set NUMBA_NUM_THREADS to a different value`. **Fix**: mover setting para `SimulationThread.run` e `_run_numba_parallel_sync` (PAI) com `try/finally` para restaurar env vars (evita vazamento). Workers spawned herdam env var correto; `multi_forward.py:1010-1028` chama `numba.set_num_threads(req.n_threads)` para mascarar threads ativas. **Throughput restaurado**: reprodutor empírico (cenário do bug — 2 freqs × 2 dips × 1 TR + 4w×4t + `NUMBA_NUM_THREADS=2` herdado) atinge **149,984 mod/h ≈ 150k** em warm cache (meta empírica). CLI single-process Cenário E: 145k mod/h. Paridade Fortran <1e-12 PRESERVADA. 3 novos guard tests v2.29.1 (worker NÃO seta env var; pai SETA antes do pool; try/finally restaura). CodeRabbit: 0 findings. | 3 novos (total 9 ephemeral) | `feat/simulation-manager-v2.21` (branch reutilizado) |
| **v2.29** | **2026-05-11** | **Back to Basics — reversão arquitetural definitiva**: após 4 tentativas (v2.25–v2.28) sem resolver a regressão de throughput (75–107k vs >150k esperado), warmup visível ~33s, Python travando ao fechar. Análise comparativa com `old_geosteering_ai/simulation/tests/sm_workers.py` (869 LOC, comprovadamente funcional) revelou que a infra v2.18+ (pool persistente + PoolWarmupThread + t0_sim + NOOPs) é **incompatível** com o cold-start JIT do Numba. **Solução**: reverter para arquitetura OLD — pool EFÊMERO (`with ProcessPoolExecutor`), warmup INLINE em `run_numba_chunk` com dados REAIS de `chunk[0]` (cobre TODOS paths Numba automaticamente), `t0 = time.perf_counter()` único antes do pool. **Removidos**: `_PERSISTENT_POOL`, `_acquire_numba_pool`, `release_numba_pool`, `PoolWarmupThread`, `_run_numba_warmup_task`, `_numba_init_worker`, `_noop`, `_WORKER_INITIALIZED`, `t0_sim`, `_prewarm_numba_pool`, `lbl_warmup_status`, `_warmup_thread`, 4 métodos GUI `_start_background_warmup`/`_on_warmup_*`/`_on_backend_changed_warmup`. **Preservado**: NumbaPrimer (v2.9), pause/cancel cooperativo (v2.11), `os.environ["NUMBA_NUM_THREADS"]` setup (v2.16), `detect_cpu_topology` (v2.17). sm_workers.py: **1478→1062 LOC (-416)**. Paridade Fortran <1e-12 PRESERVADA. Hang ao fechar **resolvido por design** (context manager `__exit__`). 6 novos testes guard ephemeral + 1 realocação seed. Deletado: `test_simulation_pool_warmup.py` (12 testes obsoletos). | 7 novos / 12 deletados | `feat/simulation-manager-v2.21` (branch reutilizado) |
| **v2.28** | **2026-05-11** | **Fix warmup incompleto: Warmups C+D para cobertura total** (INEFICAZ — regressão persistiu, requereu v2.29): v2.27 eliminou saturação LLVM mas introduziu regressão de throughput (~55k mod/h vs 800k–1.4M esperado). Análise de timestamps de `__pycache__` revelou 5 funções JIT críticas compiladas APÓS `t0_sim`: `propagation.common_arrays`, `propagation.common_factors`, `geometry.find_layers_tr`, `geometry._sanitize_profile_kernel`, `kernel.precompute_common_arrays_cache`. Causa: warmup A+B usava `rho_h==rho_v` (isotrópico) + `dip=0°` (path simplificado) — não acionava especializações reais. Workers 1..N compilavam essas funções INLINE no warmup secundário, após `t0_sim`. Fix: adicionados Warmup C (anisotrópico `_rho_v = _rho * 0.3` + dip=0°) e Warmup D (anisotrópico + dip=30°) em `_run_numba_warmup_task`, garantindo cobertura completa de paths antes de `t0_sim`. Paridade Fortran <1e-12 PRESERVADA. 2 novos testes T11/T12 (total 12 warmup PASS). | 2 novos (total 12 warmup) | `feat/simulation-manager-v2.21` (branch reutilizado) |
| **v2.27** | **2026-05-10** | **Fix warmup LLVM saturation (1 worker → cache compartilhado)**: Bug v2.26 diagnosticado — `PoolWarmupThread.run()` submetia `n_workers` futures de warmup simultaneamente, causando saturação LLVM: N × LLVM_threads disputam os mesmos cores físicos → cada compilação 3–4× mais lenta → 35 s/worker → **110+ s** total. Fix: 1 único future → 1 worker compila JIT e grava cache em disco; workers 2..N carregam bytecode do cache (< 2 s) via warmup secundário em `run_numba_chunk`. Segundo bug: `_WORKER_INITIALIZED` nunca setado após warmup secundário → rodava em todo chunk. 2 novos testes + 10/10 warmup PASS + paridade Fortran <1e-12. | 2 novos (total 10 warmup) | `feat/simulation-manager-v2.21` (branch reutilizado) |
| **v2.26** | **2026-05-10** | **Move warmup do inicializador para Future (elimina hang no shutdown)**: `_numba_init_worker` simplificado para apenas env vars (< 100 ms). Novo `_run_numba_warmup_task` com shapes corretos (n=10, esp 8 elem, 50 posições) submetido como Future cancelável. `PoolWarmupThread.run()` aguarda futures. `release_numba_pool(cancel_futures=True)` efetivo (workers não travados em init). 3 novos testes + paridade Fortran <1e-12. | 3 novos | `main` |
| **v2.25** | **2026-05-10** | **Fix warmup silencioso em `_numba_init_worker` (regressão 50k→150k)**: corrigiu shape `esp=[5.0, 5.0]` (inválida para 3 camadas) → ValueError silencioso → `_WORKER_INITIALIZED=True` falso → cold-start em produção. Fix: shape correto (n=10, esp 8 elem). Warmup com 600 posições + 4 combos em inicializador. | — | `main` |
| **v2.24** | **2026-05-10** | **Débitos Técnicos + I2.5 + I2.6** (multi-agente): **Frente 1**: 1.1 `scripts/count_pytest_pass.py` automatiza contagem PASS (M2 recorrente fechado). 1.2 `setup-environment.sh` totalmente em PT-BR acentuado (11+ ocorrências). 1.3 `cache=True` explícito em 13 decoradores `@njit` em `_numba/` (propagation 2 + dipoles 2 + rotation 2 + hankel 4 + geometry 3 — incluindo `_sanitize_profile_kernel` que já tinha cache=True desde v2.23). 1.4 `use_fastmath` documentado como status documental (dispatcher real **deferido v2.25** e substituído por estratégia superior `fastmath` seletivo + pairwise summation — ver Sprint v2.25 PLANEJADA). **I2.5**: hooks `check-ptbr-accentuation.sh` (catálogo `.claude/ptbr-words.txt` 60+ pares, registrado em settings.json PostToolUse) + `generate-pr-description.sh` (template `.claude/templates/pr_description_template.md`). **I2.6**: CLI MVP `geosteering-cli` (subcomandos `simulate`/`benchmark`/`version`) em `geosteering_ai/cli/` + entry point `pyproject.toml`. Lazy imports nos handlers para `--help` rápido. Reutiliza `simulate_multi` + `recommend_default_parallelism`. Paridade Fortran <1e-12 PRESERVADA em 7 modelos canônicos (10/10 testes). | 41 novos (11 v2.24 + 17 I2.5 + 13 CLI) | `feat/sprint-v2.24-debits-i25-i26` |
| **v2.25 (PLANEJADO)** | TBD | **O3 Seguro (Selective Fastmath + Pairwise Summation) + Alta-ρ Canônicos + KB-020 + Kong 61pt CLI** (alinhado §7.4, §7.6, §8.3, §8.5 do `analise_cenarios_otimizacao_simulador_numba.md`). **Decisão técnica** (auditoria 2026-05-10): substituir proposta original (dispatcher dual-mode `hmd_tiv_fast`/`hmd_tiv_safe`) por **`fastmath={'nnan','ninf','contract','arcp','afn'}`** (5 flags SEGUROS — exclui `reassoc` e `nsz` que causam cancelamento catastrófico em alta ρ) + **pairwise summation manual** em 7 reduções `np.sum` de `hmd_tiv` (linhas 665-671) e 2 reduções de `vmd` (linhas 961-962). Erro O(ε·log 201) ≈ O(ε·7.65) vs O(ε·201) naive. **Defesa em profundidade**: 3 modelos canônicos alta ρ (`carbonato_seco_5c` 5kΩ·m, `evaporita_3c` 100kΩ·m, `gas_seco_8c` 10kΩ·m) adicionados ao gate Fortran <1e-12 (`tests/test_simulation_high_resistivity_parity.py`). **KB-020** documenta proibição de `reassoc`/`nsz` em `dipoles.py`. **DEPRECATE** flag `use_fastmath` com `DeprecationWarning` (remoção v2.27). **Kong 61pt CLI**: flag `--filter {werthmuller,kong,anderson}` em `geosteering-cli simulate/benchmark` (§8.5 doc análise; +230% mod/h projetado para geração de treinamento). **Ganho esperado**: +6-10% global (hmd_tiv+vmd) preservando paridade <1e-12 inclusive em ρ=10⁵ Ω·m. **Multi-agente**: physics-reviewer (Opus 4.7) valida gate alta-ρ + perf-reviewer (Haiku 4.5) valida +6-10% + code-reviewer (Sonnet 4.6) valida pairwise. **Limpeza Numba cache automatizada** em `setup-environment.sh` (ressalva physics-reviewer v2.24). | TBD (~30 novos: 3 high-ρ + ~10 pairwise unit + ~5 selective fastmath + ~5 Kong CLI + ~5 deprecation + ~2 cache cleanup) | TBD |
| **v2.26 (PLANEJADO)** | TBD | **O1 Adaptive Thread Count Dinâmico + 3 modelos alta-ρ adicionais (opcional)** (§8.1 doc análise). Adiciona `cfg.adaptive_threads: bool = False` (opt-in) e `cfg.adaptive_min_tasks_per_thread: int = 4`. Helper `_apply_adaptive_thread_mask(cfg, n_combos, n_pos, nf)` em `multi_forward.py:1010` computa `effective_tasks = n_combos*n_pos*nf` (FLAT mode) ou `n_combos*n_pos` (legacy) e ajusta `numba.set_num_threads(min(max_threads, max(1, effective_tasks // min_tasks)))`. `try/finally` para restaurar estado anterior. Bypass para `n_pos ≤ 1`. **Ganho esperado**: +30-60% em Cenário 1 com `n_pos=10` (evita 6 threads ociosas em CPU 8C); ~0% em cenários já saturados (`n_pos ≥ 100`). 5 novos testes (default OFF preserva paridade bit-exata, redução para n_pos baixo, cap em cfg.num_threads, restore on exception, paridade Fortran <1e-12 com adaptive ON). Esforço ~1.5-2 dias. **NÃO** ativar default — após validação em CI por 1 sprint, considerar promoção em v2.27. | TBD (~5 novos) | TBD |
| **v2.27 (PLANEJADO)** | TBD | **O4 Cache Contexto + O6 F7 einsum + GUI testing (pytest-qt) + Remoção `use_fastmath`** (§8.4, §8.6 doc análise). O4: cache LRU de `precompute_common_arrays_cache` por `(hordist, freq, rho_h, rho_v, esp, filter_name)` — beneficia inversão iterativa e geosteering em tempo real (+30% projetado para esses casos). O6: vetorização `apply_tilted_antennas` via `np.einsum` (+3× em F7 para `n_tilted > 5`, <2% impacto global). **GUI testing**: `pytest-qt` integrado + "Golden Path" smoke test do Simulation Manager (`cmb_scenario=E`, `spn_n_models=10`, `btn_run` → `qtbot.waitSignal(simulation_finished, timeout=60s)` → throughput ≥ 50k mod/h). **Remoção `use_fastmath`** flag (já deprecated em v2.25). Consolida overhead do hook `check-ptbr-accentuation.sh` (105 greps → 1 `grep -f`, ressalva perf-reviewer v2.24). | TBD (~12 novos: 4 O4 + 3 O6 + 5 GUI smoke) | TBD |
| **v2.28 (PLANEJADO)** | TBD | **Adapter opt-in para Datasets LWD Reais** (validação suplementar antecipada): `geosteering_ai/data/loaders/real_data_adapter.py` + SDAR/SPWLA canonical models + Volve/Teapot suplementares. **NÃO-BLOQUEANTE** — prioridade primária permanece simuladores 1D→2D→2.5D→3D (§21). | — | TBD |
| **v2.29 (PLANEJADO)** | TBD | **Métodos Alternativos de Inversão**: Occam regularizado + Look-up Table + Tikhonov + benchmark comparativo (§74 doc arquitetura). Baseline científico para paper de validação. | — | TBD |
| **v2.30 (PLANEJADO)** | TBD | **Framework-Agnostic Core (Tier 2)**: `BaseInversionModel` + adapters TF/PyTorch/ONNX (§75 doc arquitetura). TF default exclusivo no pipeline; PyTorch opt-in via adapter para pesquisa; ONNX para deploy em campo. Hook `validate-no-pytorch.sh` ATIVO desde v2.22.7-docs. | — | TBD |
| **v2.31 (PLANEJADO)** | TBD | **Backend-Agnostic Code Hygiene (Tier 1 — §75.10)**: auditoria `tf.*` → `keras.ops.*` em modelos custom + losses (excluindo PINNs com GradientTape, que permanecem TF-specific por design). Habilita JAX backend naturalmente e prepara terreno para Tier 3 (multi-backend) condicional. **Sem mudança de default** — TF continua exclusivo no pipeline de produção. | — | TBD |

Documentação detalhada em `docs/reports/v2.{N}_2026-04-{D}.md` e
`docs/CHANGELOG.md`. Relatório técnico Fase 1 completa em
`docs/reports/fase1_fundacao_multiagente_completa_2026-05-09.md`.
Pré-mortem inaugural em
`docs/reports/premortem_geosteering_ai_2026-05-09.md`.

**Planejamento técnico das Sprints v2.25-v2.27** (alinhamento com
`docs/reference/analise_cenarios_otimizacao_simulador_numba.md`):
ver `docs/reference/sprints_v2_25_v2_27_planning.md` — define escopo,
critérios de aceitação e estratégia técnica para fechar 6/6 otimizações
(O1-O6) do documento de análise, incluindo a decisão multi-agente
(2026-05-10) de substituir o dispatcher dual-mode `use_fastmath` por
estratégia superior `fastmath` seletivo + pairwise summation manual.

**Cadência transversal (F-cross — fase cross-cutting de governança)**:
Pré-mortem trimestral via skill `geosteering-premortem-analyst` (Opus 4.7,
effort extra-high). Detalhes em §24.4 do
[documento de aprofundamento](reports/arquitetura_multiagente_geosteering_ai_aprofundamento_2026-05-02.md).
Próximo: ~2026-08-09 OU sob gatilho explícito (release major, mudança de
fase F).

---

## Sumário

1. [Status Atual do Projeto](#1-status-atual-do-projeto-abril-2026)
2. [Embasamento Científico](#2-embasamento-científico-literatura-2024-2026)
3. [Roadmap de Desenvolvimento (F1-F6)](#3-roadmap-de-desenvolvimento-fases-f1-f6)
4. [Propostas de Novos Recursos](#4-propostas-de-novos-recursos)
5. [Otimizações de Código](#5-otimizações-de-código-recomendadas)
6. [Próximos Passos](#6-próximos-passos-requisições-futuras)
7. [Referências Bibliográficas](#7-referências-bibliográficas)
8. [Arquivos Críticos](#8-arquivos-críticos-para-consulta)

---

## 1. Status Atual do Projeto (Abril 2026)

### 1.1 Métricas Globais

| Métrica | Valor |
|:--------|:------|
| Linhas de código (produção) | 44.762 LOC |
| Linhas de testes | 9.024 LOC |
| Arquivos Python (src) | 73 |
| Arquivos Python (testes) | 15 |
| Testes totais | 744 (CPU) / 1011+ (GPU) |
| Campos PipelineConfig | 246 |
| Arquiteturas (ModelRegistry) | 48 (9 famílias) |
| Funções de perda | 26 (4 categorias) |
| Tipos de ruído | 34 implementados |
| Feature Views | 7 (identity, raw, H1_logH2, logH1_logH2, 3× razão/fase, second_order) |
| Famílias de Geosinais | 5 (USD, UAD, UHR, UHA, U3DF) |
| Cenários PINN | 8 (oracle, surrogate, maxwell, smoothness, skin_depth, continuity, variational, self_adaptive) |
| Presets YAML | 7 (baseline, robusto, nstage_n2, nstage_n3, geosinais_p4, dtb_p5, realtime_causal) |
| Callbacks Keras | 17+ |
| **Simulador Fortran** | 6.859 LOC (6 módulos F08) + gerador Python (~900 LOC) + f2py wrapper + batch runner |
| **Documentação Simulador** | ~8.000+ LOC (20 seções + 6 pipelines, v8.0 com Multi-TR + f2py + batch + análise novos recursos) |
| **Formulação Teórica TeX** | `Tex_Projects/TatuAniso/FormulaçãoTatuAnisoTIV.tex` (~960 LOC TeX) |
| Métricas customizadas | 3 (R2Score, PerComponentMetric, AnisotropyRatioError) |
| Formatos de exportação | 3 (SavedModel, TFLite, ONNX) |

### 1.2 Arquitetura do Pacote

```
geosteering_ai/                           # 73 arquivos, 44.762 LOC
├── config.py                             # PipelineConfig (246 campos, 7 presets)
├── data/ (12 módulos, ~6.700 LOC)
│   ├── loading.py                        # Carregamento .dat 22-col Fortran
│   ├── splitting.py                      # Split por modelo geológico (P1)
│   ├── feature_views.py                  # 7 Feature Views (NumPy + TF)
│   ├── geosignals.py                     # 5 famílias Geosinais (NumPy + TF)
│   ├── scaling.py                        # 8 scalers + per-group P3
│   ├── pipeline.py                       # DataPipeline on-the-fly (noise→FV→GS→scale)
│   ├── boundaries.py                     # DTB labels (P5)
│   ├── sampling.py                       # Oversampling alta rho (Estratégia B)
│   ├── second_order.py                   # Features 2º grau (Estratégia C)
│   ├── surrogate_data.py                 # Extração pares SurrogateNet (Modo A/B/C)
│   └── inspection.py                     # Diagnóstico pré-treinamento
├── noise/ (2 módulos, ~2.600 LOC)
│   ├── functions.py                      # 34 tipos de ruído
│   └── curriculum.py                     # Curriculum 3-phase (clean→ramp→stable)
├── models/ (13 módulos, ~8.800 LOC)
│   ├── blocks.py                         # 23+ blocos Keras reutilizáveis
│   ├── cnn.py                            # ResNet-18/34/50, ConvNeXt, Inception, CNN_1D, ResNeXt
│   ├── tcn.py                            # TCN, TCN_Advanced, ModernTCN
│   ├── rnn.py                            # LSTM, BiLSTM
│   ├── hybrid.py                         # CNN_LSTM, CNN_BiLSTM_ED, ResNeXt_LSTM
│   ├── unet.py                           # 14 variantes U-Net
│   ├── transformer.py                    # 6 Transformers
│   ├── decomposition.py                  # N-BEATS, N-HiTS
│   ├── advanced.py                       # DNN, FNO, DeepONet, Geophysical_Attention
│   ├── geosteering.py                    # WaveNet, Causal_Transformer, Informer, Mamba_S4
│   ├── surrogate.py                      # SurrogateNet TCN (campo receptivo ~127m)
│   └── registry.py                       # ModelRegistry — 48 entradas + build()
├── losses/ (3 módulos, ~3.600 LOC)
│   ├── catalog.py                        # 26 funções de perda
│   ├── factory.py                        # LossFactory.build_combined()
│   └── pinns.py                          # 8 cenários PINN + TIV constraint + lambda schedule
├── training/ (7 módulos, ~8.100 LOC)
│   ├── loop.py                           # TrainingLoop.run()
│   ├── callbacks.py                      # 17+ callbacks Keras
│   ├── metrics.py                        # R2Score, PerComponentMetric, AnisotropyRatioError
│   ├── nstage.py                         # NStageTrainer (N=2,3,4)
│   ├── adaptation.py                     # DomainAdapter (fine-tune + progressive)
│   └── optuna_hpo.py                     # HPO com Optuna
├── inference/ (4 módulos, ~1.800 LOC)
│   ├── pipeline.py                       # InferencePipeline serializável (save/load)
│   ├── realtime.py                       # Sliding window (causal)
│   ├── export.py                         # SavedModel, TFLite, ONNX
│   └── uncertainty.py                    # MC Dropout + Ensemble
├── evaluation/ (11 módulos, ~7.300 LOC)
│   ├── metrics.py                        # MetricsReport (R², RMSE, MAE, MBE, MAPE)
│   ├── comparison.py                     # Ranking multi-modelo
│   ├── advanced.py                       # InterfaceReport, CoherenceReport, StabilityReport
│   ├── dod.py                            # Picasso DOD analítico (6 métodos)
│   ├── geosteering_metrics.py            # DTB error, look-ahead accuracy
│   ├── realtime_comparison.py            # Offline vs Realtime
│   ├── report.py                         # Relatório Markdown automático
│   ├── manifest.py                       # Manifesto JSON reprodutibilidade
│   └── config_report.py                  # Relatório pré-treinamento
├── visualization/ (11 módulos, ~2.500 LOC)
│   ├── eda.py                            # EDA avançada (6 funções)
│   ├── holdout.py                        # True vs predicted (clean + noisy)
│   ├── training.py                       # Curvas loss, R², LR, noise
│   ├── error_maps.py                     # Heatmap 2D, barras, perfil espacial
│   ├── geosteering.py                    # Curtain, DTB, dashboard
│   ├── uncertainty.py                    # Histogramas, bandas, calibração
│   ├── optuna_viz.py                     # Optuna (4 visualizações)
│   └── export.py                         # Exportação batch (PNG, PDF)
└── utils/ (6 módulos, ~2.500 LOC)
    ├── logger.py                         # ColoredFormatter, setup_logger
    ├── timer.py                          # ProgressTracker, elapsed_since
    ├── validation.py                     # ValidationTracker, validate_shape
    ├── formatting.py                     # format_number, log_header
    ├── system.py                         # is_colab, get_gpu_info, set_seed
    └── io.py                             # load_json, save_json, safe_mkdir
```

### 1.3 Funcionalidades Implementadas

| Categoria | Funcionalidade | Status |
|:----------|:---------------|:------:|
| **Dados** | Carregamento .dat 22-col Fortran | ✅ |
| **Dados** | Split por modelo geológico (P1) — evita data leakage | ✅ |
| **Dados** | 7 Feature Views (NumPy + TF consistentes, SEMPRE log10) | ✅ |
| **Dados** | 5 famílias Geosinais — USD, UAD, UHR, UHA, U3DF (NumPy + TF) | ✅ |
| **Dados** | 8 scalers (Standard, Robust, MinMax, etc.) + per-group P3 | ✅ |
| **Dados** | DataPipeline on-the-fly — GS veem ruído (fidelidade LWD) | ✅ |
| **Dados** | DTB labels — detect_boundaries + compute_dtb (P5) | ✅ |
| **Dados** | Oversampling alta rho (Estratégia B) | ✅ |
| **Dados** | Features 2º grau — |H|², d|H|/dz, Re/Im ratio (Estratégia C) | ✅ |
| **Dados** | SurrogateDataset — extração pares Modo A/B/C (K componentes) | ✅ |
| **Ruído** | 34 tipos (9 CORE + 13 EXTENDED + 8 geofísicos + 4 geosteering) | ✅ |
| **Ruído** | Curriculum 3-phase (clean → ramp → stable) | ✅ |
| **Modelos** | 48 arquiteturas em 9 famílias (CNN, TCN, RNN, Hybrid, U-Net, Transformer, Decomposition, Advanced, Geosteering) | ✅ |
| **Modelos** | 23+ blocos Keras reutilizáveis | ✅ |
| **Modelos** | SurrogateNet TCN dilatado (6 blocos, campo receptivo ~127m) | ✅ |
| **Modelos** | SurrogateNet v2 ModernTCN (4 blocos DWConv, campo receptivo ~204m) | ✅ |
| **Modelos** | ModernTCN — convolução pura moderna para inversão seq2seq (Luo 2024) | ✅ |
| **Modelos** | INN — Invertible Neural Network para UQ probabilística (Ardizzone 2019) | ✅ |
| **Modelos** | ResNeXt — grouped convolutions 32×4d bottleneck (Xie 2017) | ✅ |
| **Modelos** | ResNeXt_LSTM — híbrido ResNeXt encoder + LSTM temporal | ✅ |
| **Inference** | UQ via INN (sampling da posterior, alternativa ao MC Dropout) | ✅ |
| **Modelos** | Static injection P2/P3 (broadcast, dual_input, FiLM) | ✅ |
| **Modelos** | TIVConstraintLayer (hard constraint rho_v ≥ rho_h via softplus) | ✅ |
| **Losses** | 26 funções de perda (13 genéricas + 4 geofísicas + 2 geosteering + 7 avançadas) | ✅ |
| **Losses** | 8 cenários PINN + 4 lambda schedules (fixed, linear, cosine, step) | ✅ |
| **Losses** | Forward analítico 1D (magnitude log10|H| + complexo Re/Im) | ✅ |
| **Losses** | TIV constraint loss (penaliza rho_h > rho_v) | ✅ |
| **Losses** | LossFactory.build_combined() — base + look_ahead + DTB + PINNs | ✅ |
| **Treinamento** | TrainingLoop.run() + N-Stage (N=2,3,4) com mini-curriculum | ✅ |
| **Treinamento** | 17+ callbacks (DualValidation, PINN schedule, Causal, LR schedules) | ✅ |
| **Treinamento** | HPO Optuna com visualizações | ✅ |
| **Treinamento** | DomainAdapter (fine-tune + progressive) | ✅ |
| **Treinamento** | GradientMonitor (normas por layer via GradientTape) | ✅ |
| **Inferência** | InferencePipeline serializável (model + scalers + config → save/load) | ✅ |
| **Inferência** | Realtime sliding window (causal, buffer FIFO) | ✅ |
| **Inferência** | Export SavedModel / TFLite (com quantização) / ONNX | ✅ |
| **Inferência** | UQ — MC Dropout (30 passes) + Ensemble + CI 95% | ✅ |
| **Avaliação** | Métricas (R², RMSE, MAE, MBE, MAPE por componente) | ✅ |
| **Avaliação** | DOD Picasso analítico (6 métodos: standard, contrast, SNR, frequency, anisotropy, dip) | ✅ |
| **Avaliação** | Relatórios Markdown automáticos + Manifesto JSON | ✅ |
| **Avaliação** | Comparação offline vs realtime + geosteering metrics | ✅ |
| **Visualização** | EDA, holdout, training curves, error maps, geosteering, UQ, Optuna | ✅ |
| **Compatibilidade** | Keras 3.x (8 fixes: KerasTensor, DepthwiseConv1D, FourierLayer, etc.) | ✅ |
| **CI/CD** | GitHub Actions (compile + pytest + mypy, Python 3.10/3.11) | ✅ |
| **Validação GPU** | 824 passed no Colab Pro+ (TF 2.19, Keras 3.x, GPU T4) | ✅ |
| **Documentação** | Padronização seq_len — remoção de hardcoded 600 em docstrings e comentários (~30 ocorrências corrigidas em 15 módulos) | ✅ |
| **Documentação** | Mega-headers atualizados com contagens corretas (8 CNN, 3 TCN, 3 Hybrid, 5 Advanced) | ✅ |
| **Documentação** | Manual técnico de ruídos — `docs/documentacao_noises.md` (34 tipos, curriculum, API) | ✅ |
| **Documentação** | Manual técnico de losses — `docs/documentacao_losses.md` (26 losses, 8 PINN, TIV) | ✅ |
| **Documentação** | Manual técnico de modelos — `docs/documentacao_models.md` (48 arqs, 9 famílias, blocos) | ✅ |
| **Documentação** | Manual técnico de inferência offline — `docs/documentacao_inferencia_offline.md` (pipeline, UQ, export) | ✅ |
| **Documentação** | Manual técnico de geosteering — `docs/documentacao_geosteering.md` (5 arqs causais, realtime) | ✅ |
| **Documentação** | Manual técnico de PINNs — `docs/documentacao_pinns.md` (8 cenários, TIV, SurrogateNet) | ✅ |
| **Documentação** | Apresentação geral do projeto — `docs/documentacao_apresentacao_geosteering_ai.md` | ✅ |

### 1.4 Lacunas Reconhecidas

| Lacuna | Impacto | Dependência |
|:-------|:--------|:------------|
| Treino do SurrogateNet com dados reais | Alto — PINNs modo surrogate neural | Dados Fortran |
| Surrogate Modo C (tensor completo 9 componentes, 18 canais) | Médio — digital twin Fortran | Fortran multi-dip |
| Notebook de treinamento end-to-end | Alto — pipeline completo validado | Dados disponíveis |
| Multi-Task Loss (Kendall et al. 2018) — placeholder | Baixo — auto-balanceamento | Implementação |
| Cenário PINN petrofísica (Archie + Klein) | Baixo — requer parâmetros reservatório | Dados de campo |
| Evidential Regression (alternativa a MC Dropout para UQ) | Baixo — MC Dropout funciona | Pesquisa |

---

## 2. Embasamento Científico (Literatura 2024-2026)

### 2.1 Artigos-Chave

| # | Referência | Contribuição para o Projeto |
|:-:|:-----------|:---------------------------|
| 1 | Morales et al. (2025) — *Anisotropic resistivity estimation and uncertainty quantification from borehole triaxial electromagnetic induction measurements: Gradient-based inversion and physics-informed neural network.* Computers & Geosciences, 196, 105786. | Base teórica dos cenários PINN. PINN estima propriedades petrofísicas em 0,5 ms com 91-99% de acurácia vs minutos com inversão baseada em gradiente. Validado com medidas triaxiais. **Já referenciado no projeto.** |
| 2 | INN-UDAR (2025) — *Invertible neural network for real-time inversion and uncertainty quantification of ultra-deep resistivity measurements.* Computers & Geosciences. | Redes invertíveis (INN) fornecem distribuição posterior completa para quantificação de incerteza probabilística em tempo real, sem necessidade de múltiplas forward passes (MC Dropout). **Implementado: INN como arquitetura #45 em `models/advanced.py` + UQ em `inference/uncertainty.py`.** |
| 3 | SPE/SPWLA (2021) — *Real-Time 2.5D Inversion of LWD Resistivity Measurements Using Deep Learning for Geosteering Applications Across Faulted Formations.* | Deep learning para inversão 2.5D com custo online desprezível. Validado com ferramenta LWD triaxial em formações anisotrópicas com falhas. **Referência para extensão futura (v3.0).** |
| 4 | Li et al. (2025) — *Self-Supervised Deep Learning Inversion Incorporating a Fast Forward Network for Transient Electromagnetic Data.* JGR: Machine Learning and Computation. | Inversão auto-supervisionada que incorpora rede forward rápida diretamente na função de perda. **Alinha com cenário "surrogate" PINN do projeto.** |
| 5 | Jiang et al. (2025) — *One-Fit-All Transformer for Multimodal Geophysical Inversion.* JGR: Machine Learning and Computation. | Framework G-Query — Transformer unificado que adapta entre múltiplas modalidades geofísicas via query tokens. **Proposta: avaliar como arquitetura #46 (G_Query_Transformer).** |
| 6 | Frontiers (2025) — *Fast forward modeling and response analysis of extra-deep azimuthal resistivity measurements in complex model.* | Modelagem direta rápida para UDAR (Ultra-Deep Azimuthal Resistivity) em modelos complexos. **Complementa o surrogate neural do projeto.** |
| 7 | ModernTCN (ICLR 2024) — *A Modern Pure Convolution Structure for General Time Series Analysis.* | Modernização do TCN clássico com patch embedding e channel mixing, superando TCN em benchmarks de séries temporais. **Implementado: ModernTCN em `models/tcn.py` + SurrogateNet v2 em `models/surrogate.py`.** |
| 8 | Physics-guided AEM inversion (GJI 2024) — *Physics-guided deep learning-based inversion for airborne electromagnetic data.* | PGNN incorpora leis físicas governantes diretamente na função de perda para inversão EM aerotransportada. **Valida a abordagem PINN adotada pelo projeto.** |
| 9 | Noh, K., Pardo, D., Torres-Verdín, C. (2023) — *Physics-guided deep-learning inversion method for the interpretation of noisy logging-while-drilling resistivity measurements.* Geophysical Journal International, 235. | Combina constraints físicas com DL para inversão de medidas LWD ruidosas. Diretamente aplicável aos cenários PINN do projeto com injeção de ruído. **PDF disponível em `PDFs/ggad217.pdf`.** |
| 10 | Puzyrev, V. (2019) — *Deep learning electromagnetic inversion with convolutional neural networks.* Geophysical Journal International, 218. | Demonstra viabilidade de CNNs para inversão EM em tempo real — base histórica para a abordagem DL do projeto. **PDF disponível em `PDFs/ggz204.pdf`.** |
| 11 | Guo, W. et al. (2024) — *Efficient 1D Modeling of Triaxial Electromagnetic Logging in Uniaxial and Biaxial Anisotropic Formations Using Virtual Boundaries and Equivalent Resistivity Schemes.* Journal of Geophysics and Engineering. | Algoritmo de propagação eficiente para forward 1D triaxial em formações TIV — relevante para o surrogate analítico e validação do SurrogateNet. **PDF disponível em `PDFs/gxag017.pdf`.** |
| 12 | Liu, W. et al. (2022) — *Physics-Driven Deep Learning Inversion with Application to Magnetotelluric.* Remote Sensing, 14, 3218. | Propõe operador físico forward integrado ao treinamento DL — valida a abordagem physics-driven do projeto. **PDF disponível em `PDFs/remotesensing-14-03218-v2.pdf`.** |
| 13 | Liu, W. et al. (2024) — *Physics-Informed Deep Learning Inversion with Application to Noisy Magnetotelluric Measurements.* Remote Sensing, 16, 62. | Estende abordagem com Swin Transformer e estratégias de noise injection para dados EM ruidosos de campo — alinha diretamente com o curriculum de ruído do projeto. **PDF disponível em `PDFs/remotesensing-16-00062-v2.pdf`.** |
| 14 | Constable, M. V. et al. (2016) — *Looking Ahead of the Bit While Drilling: From Vision to Reality.* Petrophysics, 57(5). | Apresenta a ferramenta EMLA para detecção de contrastes de resistividade à frente da broca, com DOI > 75 m. Contexto aplicado da ferramenta LWD que o projeto pretende substituir por DL. **PDF disponível em `PDFs/Constable_et_al_2016_Petrophysics.pdf`.** |
| 15 | Wang, L. et al. (2018) — *Sensitivity analysis and inversion processing of azimuthal resistivity logging-while-drilling measurements.* Journal of Geophysics and Engineering, 15, 2339. | Inversão Gauss-Newton com janela deslizante 1D para medidas ARM. Referência de algoritmo convencional que o projeto busca substituir via DL em tempo real. **PDF disponível em `PDFs/Wang_2018_J._Geophys._Eng._15_2339.pdf`.** |
| 16 | Guoyu Li et al. (2025) — *Optimization and Analysis of Sensitive Areas for Look-Ahead Electromagnetic Logging-While-Drilling Based on Geometric Factors.* Energies, 18, 3014. | Análise de áreas sensíveis para look-ahead EM LWD via fatores geométricos — relevante para design de features e avaliação DOD do projeto. **PDF disponível em `PDFs/Guoyu_et_al_2025_FG.pdf`.** |
| 17 | Bai, J. et al. (2022) — *An Introduction to Programming Physics-Informed Neural Networks for Computational Solid Mechanics.* arXiv:2210.09060. | Tutorial prático de implementação de PINNs com TensorFlow/PyTorch — referência pedagógica para os cenários PINN implementados em `losses/pinns.py`. **PDF disponível em `PDFs/2210.09060v4.pdf`.** |

### 2.2 Tendências Identificadas

1. **Redes Invertíveis (INN / Normalizing Flows)** para quantificação de incerteza probabilística — alternativa superior ao MC Dropout por fornecer a distribuição posterior completa em uma única forward pass.

2. **Auto-supervisão com forward model neural** integrado na função de perda — elimina necessidade de labels supervisionados, alinhando com o cenário "surrogate" PINN já implementado no projeto.

3. **Transformers multimodais** para inversão geofísica — framework G-Query demonstra que um único modelo pode adaptar entre EM, sísmica e gravimetria via tokens de query especializados.

4. **ModernTCN** supera o TCN clássico em benchmarks de séries temporais, mantendo a arquitetura puramente convolucional (sem atenção) com menor custo computacional.

5. **Inversão 2.5D/3D via Deep Learning** em formações complexas (falhas, intrusões salinas, dip variável) — próxima fronteira para o pipeline, atualmente limitado a 1D.

6. **Quantificação de incerteza como requisito obrigatório** para decisões operacionais em geosteering — não é mais opcional, é mandatório para uso em tempo real.

7. **Robustez a ruído como critério de avaliação primário** — Liu et al. (2024) e Noh et al. (2023) mostram que modelos treinados sem estratégias de noise injection falham em dados de campo reais, reforçando o curriculum de 34 tipos de ruído do projeto.

8. **Forward modeling eficiente como habilitador** — algoritmos rápidos para 1D triaxial (Guo et al. 2024) são pré-requisito para treinar surrogates em escala e para o ciclo de auto-supervisão.

---

## 3. Roadmap de Desenvolvimento (Fases F1-F6)

### Visão Geral

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ROADMAP GEOSTEERING AI v2.0+                                              │
│                                                                             │
│  F1 ─ Consolidação e Commit (imediato)                                     │
│  ├── Commit pendente (Passos 1+2 surrogate)                               │
│  ├── Atualização MEMORY.md                                                 │
│  └── Documentação roadmap (este documento)                                 │
│                                                                             │
│  F2 ─ Treinamento e Validação GPU (curto prazo)                           │
│  ├── Notebook train_surrogate.ipynb                                        │
│  ├── Notebook train_baseline.ipynb                                         │
│  ├── Benchmark 48 arquiteturas no Colab                                    │
│  ├── Validação end-to-end com dados Fortran                               │
│  └── Preset surrogate_mode_a.yaml                                          │
│                                                                             │
│  F3 ─ Otimização de Performance (médio prazo)                             │
│  ├── XLA/JIT compilation (2-3× speedup)                                   │
│  ├── Mixed precision float16 (1.5-2× speedup)                             │
│  ├── Profiling TensorBoard                                                 │
│  ├── Benchmark latência por arquitetura                                    │
│  └── Otimização DataPipeline (prefetch, cache)                            │
│                                                                             │
│  F4 ─ Expansão Científica (médio prazo)                                   │
│  ├── INN (Invertible Neural Network) — arquitetura #45                    │
│  ├── G-Query Transformer multimodal — arquitetura #46                     │
│  ├── ModernTCN para SurrogateNet v2                                        │
│  ├── Multi-Task Loss (Kendall et al. 2018)                                │
│  ├── Cenário PINN petrofísica (Archie)                                    │
│  └── Evidential Regression para UQ                                         │
│                                                                             │
│  F5 ─ Dados Multi-Dip e Modo C (longo prazo)                             │
│  ├── Re-simulação Fortran multi-dip (0°-90°)                             │
│  ├── Treino SurrogateNet Modo B/C                                          │
│  ├── Surrogate Transformer (se TCN insuficiente)                          │
│  └── Validação física H_surrogate vs H_Fortran                            │
│                                                                             │
│  F6 ─ Produção e Deploy (longo prazo)                                     │
│  ├── API REST (FastAPI/gRPC)                                               │
│  ├── Integração WITSML/OPC-UA                                             │
│  ├── Dashboard web (Streamlit/Grafana)                                     │
│  ├── Containerização (Docker + K8s)                                        │
│  └── Monitoramento MLOps (MLflow/Weights&Biases)                          │
│                                                                             │
│  F7 ─ Simulador Python Otimizado (médio-longo prazo)                       │
│  ├── Reimplementação do PerfilaAnisoOmp em Python (Numba JIT)             │
│  ├── Kernels CUDA via Numba para GPU (commonarraysMD + Hankel)            │
│  ├── Integração direta com geosteering_ai/simulation/                     │
│  ├── Geração on-the-fly de dados sintéticos durante treinamento           │
│  └── Validação numérica Python vs Fortran (erro < 1e-10)                  │
│                                                                             │
│  v3.0 ─ Inversão 2D/3D (futuro)                                           │
│  ├── Extensão para formações com falhas                                    │
│  ├── Múltiplas frequências simultâneas                                     │
│  └── Digital twin completo do simulador Fortran                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### F1 — Consolidação e Commit (Imediato)

**Objetivo**: Commitar trabalho pendente e estabilizar a base de código.

| Passo | Descrição | Prioridade |
|:------|:----------|:----------:|
| F1.1 | Rodar `pytest tests/ -v --tb=short` — validar 802+ testes | Alta |
| F1.2 | Commit dos Passos 1+2 (surrogate_data.py + surrogate.py + testes) | Alta |
| F1.3 | Reescrever MEMORY.md (<150 linhas, atualizado) | Média |
| F1.4 | Gerar `docs/ROADMAP.md` (este documento) | Média |

**Verificação**: `pytest` green, `git log --oneline -3` mostra commit.

---

### F2 — Treinamento e Validação GPU (Curto Prazo)

**Objetivo**: Validar o pipeline end-to-end com treinamento real no Colab Pro+ GPU.

| Passo | Descrição | Artefato | Prioridade |
|:------|:----------|:---------|:----------:|
| F2.1 | Criar `notebooks/train_surrogate.ipynb` | Notebook Colab | Alta |
| | — Carregar dados .dat (Inv0Dip 0°, existentes) | | |
| | — `extract_surrogate_pairs()` Modo A (XX, ZZ → 4 canais) | | |
| | — Split por modelo geológico (P1) | | |
| | — `build_surrogate(config)` → `model.compile()` → `model.fit()` | | |
| | — Salvar como `surrogate_mode_a.keras` | | |
| F2.2 | Criar `configs/surrogate_mode_a.yaml` | YAML | Alta |
| F2.3 | Criar `notebooks/train_baseline.ipynb` | Notebook Colab | Alta |
| | — Pipeline completo: load → split → train → eval → report | | |
| | — Preset `robusto.yaml` (curriculum 3-phase, noise 8%) | | |
| | — Gerar MetricsReport + figuras (EDA, holdout, training) | | |
| F2.4 | Criar `notebooks/benchmark_architectures.ipynb` | Notebook Colab | Média |
| | — Forward + backward pass para 44 arquiteturas no GPU | | |
| | — Medir: latência (ms), memória (MB), parâmetros (M) | | |
| | — Ranking por trade-off R² × latência × parâmetros | | |
| F2.5 | Validação surrogate analítico vs Fortran | Métricas | Alta |
| | — Comparar H_analítico vs H_Fortran para 100+ modelos | | |
| | — RMSE por componente (Re/Im × XX/ZZ) | | |

#### F2.5.1 — CPU Optimization Roadmap (Simulador Fortran PerfilaAnisoOmp)

Roteiro de 6 fases para otimização do simulador Fortran conforme [`docs/reference/analise_paralelismo_cpu_fortran.md`](reference/analise_paralelismo_cpu_fortran.md) §7. Relatórios de execução:
- Fases 0 e 1: [`docs/reference/relatorio_fase0_fase1_fortran.md`](reference/relatorio_fase0_fase1_fortran.md)
- **Fase 2 + Débitos**: [`docs/reference/relatorio_fase2_debitos_fortran.md`](reference/relatorio_fase2_debitos_fortran.md)
- **Fase 3**: [`docs/reference/relatorio_fase3_fortran.md`](reference/relatorio_fase3_fortran.md)
- **Fase 4**: [`docs/reference/relatorio_fase4_fortran.md`](reference/relatorio_fase4_fortran.md)
- **Validação Final (Fases 0→4)**: [`docs/reference/relatorio_validacao_final_fortran.md`](reference/relatorio_validacao_final_fortran.md)
- **Fase 5 + PR Débitos B1/B3/B5/B7**: [`docs/reference/relatorio_fase5_debitos_fortran.md`](reference/relatorio_fase5_debitos_fortran.md)
- **Análise de Evolução (Fortran → Python/JAX/Numba)**: [`docs/reference/analise_evolucao_simulador_fortran_python.md`](reference/analise_evolucao_simulador_fortran_python.md)

| Fase | Descrição | Status | Ganho Real / Esperado |
|:----:|:----------|:------:|:----------------------|
| **Fase 0** | Benchmark Baseline (wall-time, MD5, infra `bench/`) | ✅ **Concluída 2026-04-04** | Baseline publicado: **0,1047 s/modelo**, **~34.400 mod/h** (i9-9980HK, 8 threads, AVX-2, CPU fria, n=10) |
| **Fase 1** | SIMD Hankel Reduction (`!$omp simd` em `commonarraysMD`) | ⏭️ **Pulada 2026-04-04** | Δ +0,96 % (Welch *t*=+0,425, não-significativo). Causa: gfortran 15.x já auto-vetoriza em AVX-2 32-byte. Experimento arquivado em `Fortran_Gerador/bench/attic/`. Re-avaliar em AVX-512. |
| **Fase 2** | Hybrid Scheduler + particionamento multiplicativo + correção Débitos 1/2/3 | ✅ **Concluída 2026-04-04** | Bug 2-thread corrigido: −41 % (speedup 1,07× → 1,60×). Baseline atual (n=29): **0,383 s/modelo**, **9.399 mod/h** a 8 threads. |
| **Fase 3** | Workspace Pre-allocation (`type :: thread_workspace` + `ws_pool`) | ✅ **Concluída 2026-04-05** | +30,1 % serial, +11,5 % 8 threads. Elimina 99,92 % dos mallocs do hot path. `max\|Δ\|=3,4e-14`. Pós-Fase 3: **0,343 s/modelo**, **10.485 mod/h** a 8 threads. |
| **PR1 Hygiene** | Correção dos Débitos **D4 (firstprivate), D5 (barrier), D6 (tid global)** | ✅ **Concluída 2026-04-05** (`db997d2`) | Pré-requisito estrutural para multi-ângulo. Zero impacto em runtime (ntheta=1), MD5 bit-exato em 1/2/4/8 threads. |
| **Fase 4** | Cache de `commonarraysMD` por `(r, freq)` — 1.200 → 2 chamadas/modelo | ✅ **Concluída 2026-04-05** | **Speedup 5,54× em 8 threads** (0,343 s → 0,062 s). Throughput **58.064 mod/h** (**242 % da meta original** de 24k mod/h). Eliminação de 99,83 % das chamadas. `max\|Δ\|=3,97e-13`. Débito B2 (hoist de `eta`) resolvido junto. |
| **Validação Final** | Bateria de testes de identidade numérica + revisão de código Fases 0→4 | ✅ **Concluída 2026-04-05** | **Fase 4 @ -O0 bit-exata** vs Fase 2 @ -O0 em `model.in` n=15 e n=10 sintético (`max\|Δ\|=0`). Fase 4 @ -O3 vs Fase 2 @ -O3: `max\|Δ\|=1,96e-13` (n=15), `6,11e-14` (n=10) — ambos sub-ULP, zero NaN/Inf. Determinismo MD5 idêntico em 1/2/4/8 threads. Nenhum bug encontrado. |
| **Fase 3b** | `thread_workspace` estendido: 6 novos campos `Mxdw..FEupz` (fatores de onda) | ✅ **Concluída 2026-04-05** | Arrays `Mxdw, Mxup, Eudw, Euup, FEdwz, FEupz` movidos de stack para heap via workspace. Elimina ~19 KB/thread de pressão de stack. Robustez para `n ≥ 30` camadas + muitos threads. Bit-exato @ -O0 vs Fase 5. MD5 determinístico 1/2/4/8 threads. |
| **Fase 2b** | Chunk tuning: `schedule(guided, 16)` no inner parallel do | ✅ **Concluída 2026-04-05** | **+9,3 %** vs Fase 5b (0,0668 → 0,0612 s/modelo). Throughput **58.856 mod/h** (**245 % da meta original**). `guided` com chunk=16 melhora balanceamento em regimes degradados e prepara para `ntheta>1` com `nmed(k)` variável. Bit-exato @ -O0 (`97123697...`). Determinismo 1/2/4/8 threads. |
| **PR Débitos B1/B3/B5/B7** | OpenMP hygiene secundária: cópia redundante eliminada (B1), `firstprivate(zrho,cH)` inner (B3/D7), `krwJ0J1` leak corrigido (B5), `contiguous` em `hmd/vmd_ws` (B7) | ✅ **Concluída 2026-04-05** | Sem impacto em performance (semanticamente neutro). MD5 bit-exato @ -O0 vs Fase 4. Código mais robusto e portável. |
| **Fase 5** | Single-level parallel para `ntheta=1` | ✅ **Concluída 2026-04-05** | Performance neutra. Bit-exato @ -O0 vs Fase 4. `max\|Δ\|=4,26e-13` @ -O3. |
| **Fase 5b** | Paralelismo adaptativo: `if(ntheta>1)` no outer `!$omp parallel do` | ✅ **Concluída 2026-04-05** | `ntheta=1`: single-level com `maxthreads`. `ntheta>1`: nested com `num_threads_k × num_threads_j`. `tid` adaptativo (`omp_get_thread_num()` vs `omp_get_ancestor_thread_num(1)*nj+omp_get_thread_num()`). Bit-exato @ -O0. **0,0668 s/modelo, 53.865 mod/h** (8 threads). Determinismo 1/2/4/8. |
| **Fase 6** | Cache de `commonfactorsMD` por `camadT` | ⚠️ **Revisada** | **Erro conceitual na proposta original** (§7.6): `commonfactorsMD` depende de `h0` (variável em `j`), não apenas de `camadT`. Cache por sentinela **não preserva fidelidade física**. |
| **Fase 6b** | Fatoração dos termos invariantes em `h0` (semi-cache) | ❌ **Descartada** | Instabilidade numérica fatal. |
| **Feature 1** | **Múltiplos pares T-R** (`nTR` espaçamentos simultâneos) | ✅ **Concluída 2026-04-06** | `model.in` estendido com `nTR` + lista de espaçamentos. Loop `do itr = 1, nTR` em `perfila1DanisoOMP`. Cache Fase 4 recomputado por par. Saída: `_TR{itr}.dat` separados para `nTR > 1`, sem sufixo para `nTR = 1` (backward-compatible). Bit-exato @ -O0 para `nTR=1` (`97123697...`). `_TR2.dat` com `dTR=1.0` = baseline `3d3c309f...`. Zero NaN/Inf. |
| **Feature 2** | **Tensor completo (9 componentes)** | ✅ **Já implementado** | `cH(i,1..9)` = `tH(1,1)..tH(3,3)` — Hxx, Hxy, Hxz, Hyx, Hyy, Hyz, Hzx, Hzy, Hzz. Escrito no `.dat` como 18 valores real/imag. Documentação atualizada. |
| **Feature 3** | **f2py wrapper** (`tatu_f2py_wrapper.f08`) | ✅ **Concluída 2026-04-06** | Módulo `tatu_wrapper` com sub-rotina `simulate()` que retorna `zrho_out(nTR, ntheta, nmmax, nf, 3)` e `cH_out(nTR, ntheta, nmmax, nf, 9)` diretamente em Python. Sem I/O de disco. Target `make f2py_wrapper` no Makefile. |
| **Feature 4** | **Batch paralelo de modelos** (`batch_runner.py`) | ✅ **Concluída 2026-04-06** | `ProcessPoolExecutor` com N workers × M OMP threads. Cada worker em diretório temporário isolado. CLI: `--models N --workers W --omp-threads T`. `fifthBuildTIVModels.py` atualizado para formato multi-TR. |
| **Feature 5** | **Frequências arbitrárias (nf > 2)** | 📋 **Planejada** | Suporte a 4-8 frequências (10-400 kHz) para DOI variável. Extensão do cache Fase 4 para `nf × nTR` combinações. ~50 LOC. |
| **Feature 6** | **Compensação de poço (midpoint multi-TR)** | 📋 **Planejada** | Pós-processamento: `H_comp = (H_T1R + H_T2R) / 2` com simetrização e phase-difference/attenuation. ~300 LOC. |
| **Feature 7** | **Antenas inclinadas (tilted coils)** | 📋 **Planejada** | Combinação linear: `H_tilted = cos(β)×H_axial + sin(β)×H_transverse`. Pós-processamento no tensor H(3×3). ~50 LOC. |
| **Feature 8** | **Correção 1.5D (relative dip)** | 📋 **Planejada** | Modificação dos coeficientes de reflexão para camadas inclinadas. Essencial para geosteering real. ~500 LOC. |
| **Feature 9** | **Efeito de invasão (mud filtrate)** | 📋 **Planejada** | Modelo de invasão radial step/gradient. Melhora fidelidade dos dados de treinamento. ~400 LOC. |
| **Feature 10** | **Sensibilidades ∂H/∂ρ (Jacobiano)** | ✅ **Implementada (v10.0 — 2026-04-10)** | Via diferenças finitas centradas — **Estratégias B (Python Workers) + C (Fortran OpenMP)**. Refactor de revisão aplicado: assumed-shape (elimina copy-in/out), allocate out-of-parallel (95% menos allocs), guard δ robusto, header .jac v2 com `nmeds`, ordem correta de flags no `batch_runner`. Validado via `validate_jacobian.py` (4/4 testes PASS — ordem O(δ²) = 2.00, bit-exato Fortran ↔ Python manual). ~1.250 LOC. |

**Débitos técnicos — status atualizado**:

1. ✅ **Débito 1 — `writes_files` append bug** — **CORRIGIDO na Fase 2**. Abertura condicional com `inquire()` + detecção de `modelm==1` OR arquivo ausente. Ver [`PerfilaAnisoOmp.f08:230-258`](../Fortran_Gerador/PerfilaAnisoOmp.f08).
2. ✅ **Débito 2 — `omp_set_nested` depreciado** — **CORRIGIDO na Fase 2**. Migração para `omp_set_max_active_levels(2)` (OpenMP 5.0+). Ver [`PerfilaAnisoOmp.f08:100-103`](../Fortran_Gerador/PerfilaAnisoOmp.f08).
3. ✅ **Débito 3 — `num_threads_j` degenera em 2 threads** — **CORRIGIDO na Fase 2**. Particionamento multiplicativo `num_threads_k × num_threads_j`. Ver [`PerfilaAnisoOmp.f08:105-109`](../Fortran_Gerador/PerfilaAnisoOmp.f08). Validado empiricamente: speedup em 2 threads 1,07× → 1,60× (+41 %).
4. ✅ **Débito 4 — `private(z_rho1, c_H1)` com `allocatable`** — **CORRIGIDO na PR1-Hygiene (pós-Fase 3, 2026-04-05)**. Arrays `allocatable` privativados migrados para `firstprivate(z_rho1, c_H1)`, garantindo herança de alocação + valores do master. Ver [`PerfilaAnisoOmp.f08:180-182`](../Fortran_Gerador/PerfilaAnisoOmp.f08).
5. ✅ **Débito 5 — `!$omp barrier` órfão** — **CORRIGIDO na PR1-Hygiene**. Diretiva redundante + fora de região paralela removida. A barreira implícita do `!$omp end parallel do` é suficiente. Ver [`PerfilaAnisoOmp.f08:217`](../Fortran_Gerador/PerfilaAnisoOmp.f08).
6. ✅ **Débito 6 — `tid` local do inner team** — **CORRIGIDO na PR1-Hygiene**. Cálculo global `tid = omp_get_ancestor_thread_num(1) * num_threads_j + omp_get_thread_num()`. Backward-compat: com `num_threads_k=1`, `ancestor(1)=0` e `tid` permanece idêntico. Pré-requisito para ativar multi-ângulo. Ver [`PerfilaAnisoOmp.f08:207-210`](../Fortran_Gerador/PerfilaAnisoOmp.f08).

7. ✅ **Débito B1 — Cópia redundante `krJ0J1/wJ0/wJ1 = krwJ0J1(:,1..3)`** — **CORRIGIDO na PR Débitos (2026-04-05)**. Slices passadas diretamente para `hmd_TIV_optimized_ws` e `vmd_optimized_ws`, eliminando 3 cópias de `npt` doubles por chamada.
8. ✅ **Débito B3/D7 — `private(zrho, cH)` com `allocatable` no inner parallel** — **CORRIGIDO na PR Débitos (2026-04-05)**. Migração para `firstprivate(zrho, cH)`, mesmo padrão de D4.
9. ✅ **Débito B5 — `krwJ0J1` alocado sem deallocate** — **CORRIGIDO na PR Débitos (2026-04-05)**. Adicionado `if (allocated(krwJ0J1)) deallocate(krwJ0J1)` após o loop paralelo. Leak de ~4,8 KB/modelo eliminado.
10. ✅ **Débito B7 — Dummy arguments sem `contiguous`** — **CORRIGIDO na PR Débitos (2026-04-05)**. Atributo `contiguous` adicionado a `krJ0J1(:)`, `wJ0(:)`, `wJ1(:)`, `Mxdw(:)`, `Mxup(:)`, `Eudw(:)`, `Euup(:)`, `FEdwz(:)`, `FEupz(:)` em `hmd_TIV_optimized_ws` e `vmd_optimized_ws`. Garante que o compilador não gere cópia temporária para slices column-major.

**Débitos pendentes:**
- 📋 **B4** — Semântica de `deallocate(zrho, cH)` após região paralela (cleanup cosmético).
- 📋 **B6** — Stride inconveniente em `zrho1(ntheta, nmmax, nf, 3)` para `writes_files` (otimização cache L1).

**Verificação**: Notebooks executam sem erro no Colab Pro+ (T4/A100). SurrogateNet Mode A converge com val_loss decrescente. Simulador Fortran com **Fase 5 + Débitos em produção**: compilação limpa, MD5 bit-exato @ -O0 vs Fase 2, `max|Δ|=4,26e-13` @ -O3, determinismo 1/2/4/8 threads. Infra `bench/` operacional com validador numérico automatizado.

---

### F3 — Otimização de Performance (Médio Prazo)

**Objetivo**: Maximizar velocidade de treinamento e inferência.

| Passo | Descrição | Ganho Esperado | Prioridade |
|:------|:----------|:---------------|:----------:|
| F3.1 | **XLA/JIT compilation** | 2-3× speedup treinamento | Alta |
| | — `model.compile(jit_compile=True)` | | |
| | — Verificar compatibilidade com as 48 arquiteturas | | |
| | — Campo `config.use_xla` já existe no PipelineConfig | | |
| F3.2 | **Mixed precision (float16)** | 1.5-2× speedup + 50% menos memória GPU | Alta |
| | — `tf.keras.mixed_precision.set_global_policy('mixed_float16')` | | |
| | — Verificar estabilidade numérica (EPS=1e-12, gradientes, losses) | | |
| | — Campo `config.use_mixed_precision` a adicionar | | |
| F3.3 | **Profiling TensorBoard** | Identificação de gargalos | Média |
| | — `tf.profiler.experimental.start()` em sessão de treino | | |
| | — Análise de utilização GPU (kernels, memória, I/O) | | |
| F3.4 | **Benchmark latência de inferência** | Tabela por arquitetura | Média |
| | — Medir tempo/amostra (ms) para 27 arquiteturas causais | | |
| | — Identificar candidatas para realtime (<10ms) | | |
| | — Comparar TFLite vs SavedModel vs ONNX | | |
| F3.5 | **Otimização DataPipeline** | Reduzir I/O overhead | Média |
| | — `tf.data.Dataset.cache()` para datasets que cabem em RAM | | |
| | — `prefetch(tf.data.AUTOTUNE)` para sobreposição I/O-compute | | |
| | — `interleave()` para carregamento paralelo | | |
| F3.6 | **Quantização TFLite** | 4× redução de tamanho | Baixa |
| | — Post-training quantization (int8 pesos, float32 ativações) | | |
| | — Validar RMSE degradation < 5% pós-quantização | | |

**Verificação**: Speedup mensurável documentado. Latência < 10ms para top-5 arquiteturas causais.

---

### F4 — Expansão Científica (Médio Prazo)

**Objetivo**: Incorporar avanços recentes da literatura (2024-2026) ao pipeline.

#### F4.1 — Invertible Neural Network (INN) — Arquitetura #45

**Base científica**: INN-UDAR (Computers & Geosciences, 2025). Redes invertíveis fornecem distribuição posterior completa para UQ probabilística, sem necessidade de MC Dropout.

| Item | Descrição |
|:-----|:----------|
| Módulo | `models/advanced.py` — adicionar `build_inn()` |
| Registro | `registry.py` → entrada "INN" na ModelRegistry |
| Testes | `test_models.py` → forward pass + verificação de invertibilidade |
| Config | `model_type: "INN"` aceito pelo PipelineConfig |
| Referência | Ardizzone et al. (2019) + INN-UDAR (2025) |

#### F4.2 — G-Query Transformer Multimodal — Arquitetura #46

**Base científica**: Jiang et al. (2025) JGR:ML. Framework "One-Fit-All" que adapta entre modalidades geofísicas via query tokens especializados.

| Item | Descrição |
|:-----|:----------|
| Módulo | `models/transformer.py` — adicionar `build_gquery_transformer()` |
| Registro | `registry.py` → entrada "G_Query_Transformer" |
| Diferencial | Multi-head cross-attention entre componentes EM (XX, ZZ, XZ, ZX) |

#### F4.3 — ModernTCN para SurrogateNet v2

**Base científica**: ModernTCN (ICLR 2024). Modernização do TCN com patch embedding + depthwise separable + channel mixing.

| Item | Descrição |
|:-----|:----------|
| Módulo | `models/surrogate.py` — adicionar `build_surrogate_modern()` |
| Ganho | Melhor captura de dependências longas com menos parâmetros |
| Config | `surrogate_architecture: "tcn"` ou `"modern_tcn"` |

#### F4.4 — Multi-Task Loss (Kendall et al. 2018) — Completar #22

**Base científica**: "Multi-Task Learning Using Uncertainty to Weigh Losses" — pesos automáticos via variáveis treináveis log_sigma por tarefa.

| Item | Descrição |
|:-----|:----------|
| Módulo | `losses/catalog.py` → completar `make_multitask()` (atualmente placeholder) |
| Fórmula | `L = Σ (0.5/σ_i²) × L_i + 0.5 × log(σ_i²)` |
| Aplicação | Balancear rho_h + rho_v + DTB automaticamente |

#### F4.5 — Cenário PINN Petrofísica (Archie + Klein)

**Base científica**: Morales et al. (2025) + Lei de Archie (1942). Integrar constraints petrofísicas na função de perda.

| Item | Descrição |
|:-----|:----------|
| Módulo | `losses/pinns.py` → cenário #9 "petrophysics" |
| Constraint | `rho_t = a × Rw / (phi^m × Sw^n)` |
| Dependência | Parâmetros do reservatório (Rw, phi, m, n) como inputs adicionais |

#### F4.6 — Evidential Regression para UQ

**Base científica**: Amini et al. (2020) — "Deep Evidential Regression". UQ em uma única forward pass sem MC Dropout.

| Item | Descrição |
|:-----|:----------|
| Módulo | `inference/uncertainty.py` → método "evidential" |
| Output | (μ, σ², ν, α) por ponto — incerteza aleatória + epistêmica |
| Vantagem | ~30× mais rápido que MC Dropout (1 pass vs 30 passes) |

**Verificação**: Cada nova feature tem testes unitários + validação no Colab GPU.

---

### F5 — Dados Multi-Dip e Modo C (Longo Prazo)

**Objetivo**: Treinar SurrogateNet com tensor EM completo para múltiplos ângulos de dip.

**Contexto**: As componentes off-diagonais do tensor H (Hxz, Hzx, Hxy, etc.) são estruturalmente zero em dip=0° por simetria TIV. Para treinar o Modo C (9 componentes, 18 canais), é necessário re-executar o simulador Fortran com múltiplos ângulos.

| Passo | Descrição | Dependência | Prioridade |
|:------|:----------|:------------|:----------:|
| F5.1 | Re-executar PerfilaAnisoOmp com multi-dip | Fortran + HPC | Alta |
| | — dip = [0°, 5°, 10°, 15°, 20°, 30°, 45°, 60°, 75°, 90°] | | |
| | — ~40.000 modelos × 600 pontos × 22 colunas por ângulo | | |
| | — Estimativa: ~10 GB de dados brutos | | |
| F5.2 | Treinar SurrogateNet Modo B (XX, ZZ, XZ, ZX → 8 canais) | F5.1 | Alta |
| | — Requer dados com dip ≥ 5° (componentes cruzadas ≠ 0) | | |
| | — `configs/surrogate_mode_b.yaml` | | |
| F5.3 | Treinar SurrogateNet Modo C (9 comp → 18 canais) | F5.1 | Média |
| | — Avaliar se TCN atual (~2M params) é suficiente ou precisa escalar | | |
| | — Alternativa: `build_surrogate_modern()` (ModernTCN, F4.3) | | |
| | — `configs/surrogate_mode_c.yaml` | | |
| F5.4 | Validação física H_surrogate vs H_Fortran | F5.2/F5.3 | Alta |
| | — RMSE < 1% por componente diagonal, < 5% por cruzada | | |
| | — Gráficos de comparação (Re/Im × componente × dip) | | |
| F5.5 | Integrar surrogate treinado nas PINNs | F5.4 | Alta |
| | — `make_surrogate_physics_loss()` com modo `"neural_external"` | | |
| | — Carregar SavedModel do surrogate treinado | | |

**Verificação**: Surrogate Modo B/C convergem. RMSE vs Fortran < 1% para diagonais, < 5% para cruzadas.

---

### F6 — Produção e Deploy (Longo Prazo)

**Objetivo**: Levar o pipeline para ambiente de produção industrial para uso em operações de geosteering em tempo real.

| Passo | Descrição | Tecnologia | Prioridade |
|:------|:----------|:-----------|:----------:|
| F6.1 | **API REST** para inferência | FastAPI + TF Serving | Alta |
| | — `POST /predict` — inversão batch | | |
| | — `POST /predict/realtime` — streaming com sliding window | | |
| | — `POST /uncertainty` — predição com MC Dropout / Ensemble | | |
| | — Documentação automática OpenAPI/Swagger | | |
| F6.2 | **Integração WITSML/OPC-UA** | python-witsml + opcua | Média |
| | — Consumir dados LWD em tempo real de sistemas SCADA | | |
| | — Converter formato WITSML → layout 22 colunas | | |
| F6.3 | **Dashboard web** | Streamlit ou Grafana | Média |
| | — Visualização realtime (curtain plot, DTB, bandas de incerteza) | | |
| | — Seleção de presets e configurações | | |
| | — Histórico de predições e métricas | | |
| F6.4 | **Containerização** | Docker + Kubernetes | Média |
| | — Dockerfile multi-stage (build + runtime, CPU + GPU) | | |
| | — Helm chart para deploy em cluster | | |
| | — Auto-scaling horizontal por demanda | | |
| F6.5 | **MLOps** | MLflow ou W&B | Baixa |
| | — Tracking de experimentos (hiperparâmetros, métricas, artefatos) | | |
| | — Model registry (versionamento de modelos treinados) | | |
| | — Monitoramento de data drift em produção | | |
| F6.6 | **Edge deployment** | TFLite + NVIDIA Jetson | Baixa |
| | — Inferência on-rig sem conectividade cloud | | |
| | — Latência-alvo: < 5ms em hardware embarcado | | |

**Verificação**: API responde em < 100ms. Dashboard renderiza em realtime. Container passa health checks.

---

### F7.1 — Evolução do Simulador Fortran (Curto-Médio Prazo)

**Objetivo**: Expandir as capacidades físicas do simulador PerfilaAnisoOmp para gerar dados de treinamento mais realistas e diversos.

**Base Técnica**: [`docs/reference/analise_novos_recursos_simulador_fortran.md`](reference/analise_novos_recursos_simulador_fortran.md)

| Passo | Descrição | Complexidade | Prioridade | LOC |
|:------|:----------|:-------------|:----------:|:---:|
| F7.1.1 | ✅ **Frequências arbitrárias (nf > 2)** — suporte a 1-16 frequências simultâneas | Baixa | Alta | ~80 |
| | — Flag `use_arbitrary_freq` no model.in (0=off, 1=on) | | | |
| | — Validação nf ∈ [1, 16], guard para nf > 2 quando desabilitado | | | |
| | — Frequências típicas: 10, 20, 40, 100, 200, 400 kHz | | | |
| F7.1.2 | **Compensação de poço (midpoint)** — pós-processamento multi-TR | Média | Alta | ~300 |
| | — Phase difference, attenuation, symmetrized ratios | | | |
| | — Borehole compensation via dual-transmitter averaging | | | |
| F7.1.3 | ✅ **Antenas inclinadas (tilted coils)** — combinação linear pós-tensor | Baixa | Média | ~120 |
| | — Flag `use_tilted_antennas` + n_tilted configs (β, φ) no model.in | | | |
| | — `H_tilted(β,φ) = cos(β)·Hzz + sin(β)·[cos(φ)·Hxz + sin(φ)·Hyz]` | | | |
| | — Saída estendida: 22 + 2×n_tilted colunas, wrapper f2py `simulate_v8` | | | |
| F7.1.4 | **Correção 1.5D (relative dip)** — camadas inclinadas | Alta | Alta | ~500 |
| | — Modificação de coeficientes de reflexão TE/TM | | | |
| | — Essencial para geosteering em poços horizontais | | | |
| F7.1.5 | **Efeito de invasão (mud filtrate)** — perfil radial de resistividade | Média | Média | ~400 |
| | — Modelo step + gradient para zona lavada | | | |
| | — Dados mais realistas para treinamento DL | | | |
| F7.1.6 | ✅ **Sensibilidades ∂H/∂ρ (Jacobiano)** — via FD centradas, Estratégias B+C | Alta | Alta | ~760 |
| | — Estratégia B: Python Workers (expande 1+4n sub-modelos, ~1.930 mod+J/h) | | | |
| | — Estratégia C: Fortran OpenMP interno (compute_jacobian_fd, ~12.900 mod+J/h) | | | |
| | — Novo wrapper f2py `simulate_v10_jacobian` retorna `dH_dRho_h/v` | | | |
| | — Flags model.in: use_jacobian, jacobian_method, jacobian_fd_step | | | |
| F7.1.7 | **Modelo de rocha (Archie)** no gerador Python | Média | Média | ~200 |
| | — Input: (φ, Sw, salinidade) → output: (ρ_h, ρ_v) | | | |
| | — fifthBuildTIVModels.py com parametrização petrofísica | | | |

**Verificação**: Cada feature validada bit-exato para `nTR=1` backwards-compatible. RMSE < 1e-10 vs baseline para features que não alteram o forward pass. F5/F7 implementadas em v8.0, F6/Filtro Adaptativo em v9.0 e F10 (Jacobiano) em v10.0 (Abril 2026) — todas desabilitadas por padrão, backward compatible.

---

### v3.0 — Inversão 2D/3D (Futuro)

| Área | Descrição | Base Científica |
|:-----|:----------|:---------------|
| Inversão 2D | Formações com falhas + dip variável | SPE/SPWLA (2021) — "Real-Time 2.5D Inversion" |
| Multi-frequência | Múltiplas frequências simultâneas (2-96 kHz) | Perspectiva P3 + ferramentas UDAR |
| Digital twin | Surrogate neural substitui completamente o Fortran | Li et al. (2025) — self-supervised forward |
| Geomecânica | Integrar pressão de poros + estabilidade de poço | Extensão petrofísica além de resistividade |
| Transfer learning | Pré-treinar em sintéticos, fine-tune em dados de campo | DomainAdapter já implementado (v2.0) |
| Born 2D approximation | Correção 2D via integral de espalhamento sobre modelo 1D | F7.1 + 1.5D completo |
| Anisotropia ortorrômbica | σ_x ≠ σ_y ≠ σ_z (além de TIV) | Reformulação TE/TM |
| Inversão conjunta EM+Sônica | Multi-física: resistividade + velocidade | Dados sônicos |

---

## 4. Propostas de Novos Recursos

### 4.1 Alta Prioridade

| # | Recurso | Módulo | Justificativa | Esforço |
|:-:|:--------|:-------|:-------------|:-------:|
| 1 | Notebook `train_baseline.ipynb` | `notebooks/` | Pipeline end-to-end validado em GPU | 1 sessão |
| 2 | Notebook `train_surrogate.ipynb` | `notebooks/` | Treinar SurrogateNet Modo A | 1 sessão |
| 3 | XLA compilation | `training/loop.py` | 2-3× speedup de treinamento | 1 sessão |
| 4 | Mixed precision (float16) | `training/loop.py` | 1.5-2× speedup + 50% menos memória | 1 sessão |
| 5 | Benchmark 48 arquiteturas | `notebooks/` | Ranking R² × latência × parâmetros | 2 sessões |
| 6 | Preset `surrogate_mode_a.yaml` | `configs/` | Reprodutibilidade do surrogate | 0.5 sessão |

### 4.2 Média Prioridade

| # | Recurso | Módulo | Justificativa | Esforço |
|:-:|:--------|:-------|:-------------|:-------:|
| 7 | INN — arquitetura #45 | `models/advanced.py` | UQ probabilística (state-of-art 2025) | 2 sessões |
| 8 | ModernTCN SurrogateNet v2 | `models/surrogate.py` | Supera TCN clássico em benchmarks | 1 sessão |
| 9 | Multi-Task Loss (Kendall) | `losses/catalog.py` | Auto-balanceamento rho + DTB | 1 sessão |
| 10 | Profiling TensorBoard | `training/` | Identificar gargalos GPU | 1 sessão |
| 11 | DataPipeline cache + prefetch | `data/pipeline.py` | Reduzir overhead de I/O | 0.5 sessão |
| 12 | Quantização TFLite (int8) | `inference/export.py` | 4× redução tamanho para edge | 1 sessão |

### 4.3 Baixa Prioridade (Pesquisa)

| # | Recurso | Módulo | Justificativa | Esforço |
|:-:|:--------|:-------|:-------------|:-------:|
| 13 | G-Query Transformer (#46) | `models/transformer.py` | Multimodal geofísico (2025) | 3 sessões |
| 14 | Evidential Regression | `inference/uncertainty.py` | UQ single-pass (~30× mais rápido) | 2 sessões |
| 15 | PINN Petrofísica (Archie) | `losses/pinns.py` | Constraint de reservatório | 2 sessões |
| 16 | API REST (FastAPI) | Novo módulo | Produção industrial | 3 sessões |
| 17 | Dashboard Streamlit | Novo módulo | Visualização realtime | 2 sessões |
| 18 | Containerização Docker | Infraestrutura | Deploy padronizado | 2 sessões |

---

## 5. Otimizações de Código Recomendadas

| # | Otimização | Arquivo(s) | Impacto Esperado |
|:-:|:-----------|:-----------|:-----------------|
| 1 | Lazy import consistente em TODOS os módulos | Vários (`__init__.py`) | Startup ~50% mais rápido |
| 2 | `@tf.function` em funções de loss críticas | `losses/catalog.py` | 10-30% speedup em treinamento |
| 3 | Batch processing para `predict()` via `tf.data` | `inference/pipeline.py` | Melhor utilização GPU |
| 4 | Pré-compilação de scalers em TF ops | `data/scaling.py` | Eliminar overhead NumPy→TF |
| 5 | Paralelização de Feature Views + Geosinais no `tf.data.map` | `data/pipeline.py` | Reduzir latência on-the-fly |
| 6 | Memoização de blocos Keras comuns | `models/blocks.py` | Reduzir memory footprint |
| 7 | Gradient checkpointing para variantes U-Net | `models/unet.py` | ~40% menos memória GPU |

---

## 6. Próximos Passos (Requisições Futuras)

As seguintes requisições podem ser utilizadas em futuras sessões de desenvolvimento:

### F1 — Consolidação (Imediato)
```
"Commitar as mudanças pendentes dos Passos 1+2 (surrogate) e atualizar MEMORY.md"
```

### F2 — Treinamento GPU (Curto Prazo)
```
"Criar notebook train_baseline.ipynb para treinamento completo no Colab"
"Criar notebook train_surrogate.ipynb para treinar SurrogateNet Modo A"
"Criar configs/surrogate_mode_a.yaml com hiperparâmetros otimizados"
"Criar notebook benchmark_architectures.ipynb para ranking das 48 arquiteturas"
```

### F3 — Otimização (Médio Prazo)
```
"Implementar XLA compilation e mixed precision no TrainingLoop"
"Otimizar DataPipeline com cache, prefetch e interleave"
"Criar benchmark de latência de inferência para as 27 arquiteturas causais"
```

### F4 — Expansão Científica (Médio Prazo)
```
"Implementar INN (Invertible Neural Network) como arquitetura #45"
"Implementar ModernTCN para SurrogateNet v2"
"Completar Multi-Task Loss (Kendall et al. 2018) — atualmente placeholder"
"Adicionar cenário PINN petrofísica com constraints de Archie"
"Implementar Evidential Regression como método alternativo de UQ"
```

### F5 — Dados Multi-Dip (Longo Prazo)
```
"Criar script de processamento para dados multi-dip do Fortran"
"Treinar SurrogateNet Modo B com dados de dip 5°-90°"
"Validar SurrogateNet vs Fortran — RMSE por componente e dip"
```

### F6 — Deploy (Longo Prazo)
```
"Criar API REST com FastAPI para inferência batch e realtime"
"Criar Dockerfile multi-stage para deploy CPU e GPU"
"Implementar dashboard Streamlit para visualização realtime"
```

### F7 — Simulador Python Otimizado (Médio-Longo Prazo)

**Contexto**: O simulador Fortran `PerfilaAnisoOmp` (6.859 LOC, 6 módulos) resolve o forward EM 1D TIV
via decomposição TE/TM + transformada de Hankel (filtro Werthmuller 201 pts) + coeficientes de reflexão
recursivos. A análise de viabilidade (doc v2.0) demonstra que o gargalo computacional está em
`commonarraysMD` (propagação) e na convolução de Hankel — ambos altamente paralelizáveis.

**Branch de desenvolvimento**: `feature/simulator-python`
**Plano detalhado**: [`docs/reference/plano_simulador_python_jax_numba.md`](reference/plano_simulador_python_jax_numba.md)
**Sub skill**: `.claude/commands/geosteering-simulator-python.md`

#### Decisões arquiteturais (fixadas em 2026-04-11)

| Questão                    | Decisão                                                      |
|:---------------------------|:-------------------------------------------------------------|
| Ordem de implementação     | **Numba primeiro** (paridade CPU), depois JAX                |
| Precisão default           | **complex128** + `complex64` via config (prod. GPU)          |
| Filtro default             | **werthmuller_201pt** (paridade filter_type=0 Fortran)       |
| Dependência empymod        | **Incluída** (validação cruzada, 3ª fonte independente)      |
| Backend PipelineConfig     | Permanece **`fortran_f2py`** até Fase 6 concluída            |
| Branch                     | **`feature/simulator-python`** (criada em 2026-04-11)        |

#### Sprints

```
F7.0   ✅ — Setup (branch, deps JAX+Numba+empymod, estrutura simulation/)   [CONCLUÍDA]
F7.1.1 ✅ — Extração dos pesos Hankel Fortran → .npz (Kong/Wer/And)         [CONCLUÍDA]
F7.1.2 ✅ — SimulationConfig dataclass (errata validation, YAML roundtrip)  [CONCLUÍDA]
F7.1.3 ✅ — Teste de referência analítico (half-space homogêneo, 5 casos)   [CONCLUÍDA]
F7.2.1 ✅ — _numba/propagation.py: common_arrays + common_factors           [CONCLUÍDA]
F7.2.2 ✅ — _numba/dipoles.py: hmd_TIV + vmd                                 [CONCLUÍDA]
F7.2.3 ✅ — _numba/hankel.py + rotation.py + geometry.py                     [CONCLUÍDA]
F7.2.4 ✅ — _numba/kernel.py: orquestrador forward                            [CONCLUÍDA]
F7.2.5 ✅ — forward.py: API simulate() + backend dispatch                     [CONCLUÍDA]
F7.2.6 ✅ — Validação Numba vs 5 casos analíticos (gate < 1e-10)            [CONCLUÍDA]
F7.2.7 ✅ — Benchmark meta ≥ 40 000 mod/h (gate fim Fase 2)                 [CONCLUÍDA (small+medium)]
F7.2.8 ✅ — ThreadPool paralelização + utilitário visualização GridSpec(3,7) [CONCLUÍDA]
F7.2.9 ✅ — fields_in_freqs→@njit + prange (speedup 6.6× medium) + 7 modelos [CONCLUÍDA]
F7.2.10 ✅ — Cache common_arrays (Fase 4 Fortran) + 16 plots (4 categorias)  [CONCLUÍDA]
F7.3.1 ✅ — Backend JAX fundação: hankel + rotation (diferenciáveis)         [CONCLUÍDA]
F7.3.2 ✅ — _jax/propagation.py: common_arrays com jax.lax.scan              [CONCLUÍDA]
F7.3.3 ✅ — _jax/kernel.py híbrido (JAX prop + Numba dipoles via callback)  [CONCLUÍDA]
F7.3.3.1 ✅ — JAX dipolos nativo parcial (decoupling + phases diferenciáveis) [CONCLUÍDA]
F7.3.3.2 ✅ — HMD native ETAPA 5 via `lax.switch` (6 casos) + 4 plots LWD/PINN [CONCLUÍDA 2026-04-13]
F7.3.3.3 ✅ — VMD native ETAPA 5 via `lax.switch` (6 casos) + 26 PASS         [CONCLUÍDA 2026-04-13]
F7.3.3.4 ✅ — ETAPAS 3+6 nativas end-to-end (prop + assembly + jax.grad)      [CONCLUÍDA 2026-04-13]
F7.3.4 ✅ — GPU support (jax[cuda12]) + notebook Colab T4/A100               [CONCLUÍDA]
F7.3.5 ✅ — Notebook Colab GPU T4 sprint_3_3_4_2 (HMD+VMD bench + empymod)   [CONCLUÍDA 2026-04-13]
F7.4.1 ✅ — Validação cruzada empymod opt-in (Hzz VMD axial)                 [CONCLUÍDA]
F7.4.2 ✅ — empymod TIV INFRA 9 componentes (mapa AB + λ² + dataclass)       [CONCLUÍDA 2026-04-13]
F7.4.3 ✅ — empymod bit-exactness: fator ρ + ab codes 1-3 corrigidos          [CONCLUÍDA 2026-04-13]
F7.I/O ✅ — read_model_in + simulate_from_model_in → .dat/.out              [CONCLUÍDA 2026-04-13]
F7.4.4 ✅ — TIV analítico (half_space): wavenumber_tiv + VMD/HMD TIV         [CONCLUÍDA 2026-04-13 PR #13]
F7.5.1 ✅ — JAX jacfwd dispatcher + fallback FD (experimental)               [CONCLUÍDA 2026-04-13 PR #13]
F7.5.2 ✅ — Numba FD centrada (política δ=clip(ε·|ρ|,1e-6,0.1·|ρ|))          [CONCLUÍDA 2026-04-13 PR #13]
F7.4.5 ✅ — Fortran↔Python direto .dat comparison (tatu.x subprocess)        [CONCLUÍDA 2026-04-13 PR #14a]
F7.BENCH ✅ — Benchmarks reais CPU Intel i9: 239–1566% Fortran (3 perfis)    [CONCLUÍDA 2026-04-13 PR #14a]
F7.5.1b ✅ — jax.jacfwd end-to-end nativo via forward_pure_jax + notebook    [CONCLUÍDA 2026-04-13 PR #14b]
F7.6.1 ✅ — Integração simulator_backend em PipelineConfig (5 campos+validação) [CONCLUÍDA 2026-04-13 PR #14c]
F7.6.2 ✅ — SyntheticDataGenerator substitui Fortran batch_runner (log-uniform) [CONCLUÍDA 2026-04-13 PR #14c]
F7.7.1 ✅ — Performance JAX: bucketing + jit + vmap (1.723× speedup CPU)     [CONCLUÍDA 2026-04-13 PR #14d]
F7.7.2 ✅ — compute_case_index_jax JAX-friendly (jnp.where para tracers z)  [CONCLUÍDA 2026-04-13 PR #14d]
F7.7.3 ✅ — Notebook Colab GPU T4 revisado + monitoramento VRAM             [CONCLUÍDA 2026-04-13 PR #14d]
F7.7.4 ✅ — Cache LRU bounded (VRAM fix): clear_jit_cache+set_maxsize       [CONCLUÍDA 2026-04-14 PR #14e]
F7.7.5 ✅ — warmup_all_buckets + forward_pure_jax_chunked (Sprint 8)        [CONCLUÍDA 2026-04-14 PR #14f]
F7.9.1 ✅ — forward_pure_jax_pmap multi-GPU (Sprint 9)                      [CONCLUÍDA 2026-04-14 PR #14f]
F7.14  ✅ — Multi-TR + Multi-Ângulo Numba nativo + F6/F7 wiring + .dat       [CONCLUÍDA 2026-04-14 PR #15]
        (simulate_multi + MultiSimulationResult + dedup hordist + paridade Fortran < 2e-13 + 17 testes + 1.16-2.14× vs OpenMP)
F7.7   — Otimizações finais restantes (pmap multi-GPU, JAX híbrido, etc.)   [PENDENTE]
F7.10  ✅ — JAX unified JIT (lax.fori_loop): 44 buckets → 1 XLA program     [CONCLUÍDA 2026-04-15 PR #24 v1.5.0]
        (Phase 1 PR #23 unified HMD + Phase 2 PR #24 VMD unified + wrappers cabeados
         + cfg.jax_strategy + 7 testes E2E + paridade 3.5e-14 + docs completos)
F7.11  ✅ — Multi-TR + multi-ângulo propagando cfg.jax_strategy (part2)     [CONCLUÍDA 2026-04-15 PR #24]
        (vmap REAL sobre iTR/iAng diferido para Sprint 12 — requer find_layers_tr_jax)
F7.17  ✅ — Sprint 12: find_layers_tr_jax tracer-safe + vmap real           [CONCLUÍDA 2026-04-16 PR #25 v1.6.0]
        (`_jax/geometry_jax.py` via jnp.searchsorted + `_simulate_multi_jax_vmap_real`
         opt-in via cfg.jax_vmap_real=True; 21 testes PASS; paridade bit-exata
         0.000e+00 vs Python loop; bench_sprint12_regression.py scaffold)
F7.18  🔜 — Flip cfg.jax_strategy default → "unified" + jax_vmap_real True    [PENDENTE v1.6.1 após soak]
F7.19  🔜 — complex64 mixed precision GPU (VRAM ÷ 2)                         [PENDENTE Sprint 13 v1.7.0]
F7.20  🔜 — Deprecação path bucketed                                          [PENDENTE v1.7.0+]
```

**Fase 1 concluída** ✅ em 2026-04-11: todos os entregáveis de Foundations
(filtros + config + testes analíticos) foram implementados, revisados
(7 correções pós-review) e validados com **153/153 testes PASS** em 1.81 s.

#### Sprint 1.1 — Extração dos Pesos Hankel (concluída 2026-04-11)

- ✅ `scripts/extract_hankel_weights.py` — parser do Fortran `filtersv2.f08`
  (regex robusta aceitando `1.23D+02`, `0.21D-28`, `.21D-28`, `1D0` +
  conversão D→E + validação + SHA-256 auditável).
- ✅ `geosteering_ai/simulation/filters/*.npz` — 3 artefatos:
  - `werthmuller_201pt.npz` (5.8 KB, filter_type=0, default)
  - `kong_61pt.npz` (2.6 KB, filter_type=1, rápido)
  - `anderson_801pt.npz` (19.6 KB, filter_type=2, preciso)
- ✅ `geosteering_ai/simulation/filters/loader.py` — `FilterLoader` com
  **cache classe-level thread-safe** (double-checked locking via
  `threading.Lock` — seguro para Fase 2 com workers paralelos) +
  `HankelFilter` (`@dataclass(frozen=True)`, arrays read-only).
- ✅ `tests/test_simulation_filters.py` — **53 testes** (após revisão):
  - 11 de bit-exactness (Kong, Werthmüller, Anderson: primeiro/meio/último)
  - 13 de API (canônico, aliases, filter_type numérico, cache)
  - 4 de imutabilidade (arrays read-only, dataclass frozen)
  - 4 de sincronia SHA-256 (gate de auditoria entre .f08 e .npz)
  - 21 restantes (shapes, semântica, Anderson expandido)
- ✅ **7 correções pós-review** aplicadas (1 race condition, 2 regex,
  1 assertion, 1 spot-check Anderson, 1 fixture scope, 1 header D1).
- ✅ Sub skill `.claude/commands/geosteering-simulator-python.md`.

#### Sprint 1.2 — SimulationConfig dataclass (concluída 2026-04-11)

- ✅ `geosteering_ai/simulation/config.py` — `SimulationConfig`
  (`@dataclass(frozen=True)`, 13 campos):
  - Validação de errata em `__post_init__` (ranges físicos, enums,
    conflitos mútuos backend × device).
  - **4 presets** @classmethod: `default()`, `high_precision()`,
    `production_gpu()`, `realtime_cpu()`.
  - **YAML roundtrip** via `to_yaml/from_yaml` (lazy import PyYAML).
  - **Dict roundtrip** via `to_dict/from_dict` (ignora chaves extras).
- ✅ `tests/test_simulation_config.py` — **62 testes**:
  - 11 defaults (frequency=20000, tr_spacing=1.0, backend=fortran_f2py, ...)
  - 8 ranges numéricos (frequência, spacing, posições)
  - 12 enums (backend, dtype, device, hankel_filter)
  - 4 mutual exclusivity (fortran+gpu, numba+gpu inválidos)
  - 6 listas opcionais (multi-freq, multi-TR)
  - 4 num_threads
  - 5 presets (todos passam na validação)
  - 3 imutabilidade (frozen, replace revalida)
  - 6 serialização (dict + YAML)
  - 3 igualdade + hash

#### Sprint 1.3 — Soluções analíticas half-space (concluída 2026-04-11)

- ✅ `geosteering_ai/simulation/validation/__init__.py` — fachada.
- ✅ `geosteering_ai/simulation/validation/half_space.py` — **5 funções
  puras NumPy** com ground-truth analítico para validação dos backends:
  1. `static_decoupling_factors(L)` → (ACp, ACx). Bit-exato com CLAUDE.md.
  2. `skin_depth(f, rho)` → δ em metros (Nabighian 1988).
  3. `wavenumber_quasi_static(f, rho)` → k complexo (Ward-Hohmann 1988).
  4. `vmd_fullspace_axial(L, f, rho, m)` → Hz em (0,0,L).
  5. `vmd_fullspace_broadside(L, f, rho, m)` → Hz em (L,0,0).
  - Convenção temporal **e^(-iωt)** (geofísica / Moran-Gianzero 1979).
  - Convenção quasi-estática **k² = iωμ₀σ** → Im(k) > 0 (atenuação).
- ✅ `tests/test_simulation_half_space.py` — **38 testes**:
  - 7 decoupling factors (bit-exato vs CLAUDE.md, sinal, escalamento L³)
  - 8 skin depth (fórmula, 1/√f, √ρ, array broadcast)
  - 7 wavenumber (Im(k)>0, Re(k)=Im(k), |k|·δ=√2)
  - 7 VMD axial (limite estático, linearidade, skin effect)
  - 5 VMD broadside (limite estático ACp, sinal negativo)
  - 4 cross-cutting (razão axial/broadside=-2, consistência entre casos)

#### Estimativas de desempenho (doc seção 12.6)

| Implementação        | Tempo/modelo | vs Fortran  | Meta              |
|:---------------------|:------------|:------------|:------------------|
| NumPy vetorizado     | ~4 s        | 10× lento   | Não será usado    |
| Numba CPU (JIT)      | ~0.5 s      | ~1.2× lento | ≥ 40k mod/h       |
| JAX CPU (JIT)        | ~0.5 s      | ~1.2× lento | ≥ 40k mod/h       |
| JAX GPU T4           | ~0.02 s     | ~3× rápido  | ≥ 200k mod/h      |
| JAX GPU A100         | ~0.008 s    | ~8× rápido  | ≥ 500k mod/h      |
| Jacobiano GPU (jacfwd)| -          | autodiff    | ≥ 10× JAX forward |

#### Módulos propostos (após Fase 2)

```
geosteering_ai/simulation/
├── __init__.py          ← fachada pública (★ já criado, Sprint 1.1)
├── config.py            ← SimulationConfig [Sprint 1.2]
├── forward.py           ← API simulate() [Fase 2-3]
├── _numba/              ← backend CPU [Fase 2]
├── _jax/                ← backend CPU/GPU/TPU [Fase 3]
├── filters/             ← ★ IMPLEMENTADO (Sprint 1.1)
│   ├── loader.py        ← FilterLoader, HankelFilter
│   └── *.npz            ← 3 filtros Hankel extraídos
├── geometry.py          ← [Fase 2]
├── postprocess.py       ← [Fase 2]
├── validation/          ← [Fase 4]
└── benchmarks/          ← [Fase 4-7]
```

Ref: `docs/reference/documentacao_simulador_fortran.md` (6.558 LOC, v4.0) — documentação completa incluindo formulação teórica via Potenciais de Hertz (Moran & Gianzero 1979), análise avançada de paralelismo OpenMP (3-level collapse, NUMA), viabilidade CUDA/Python, **Pipeline A Fortran** (otimização CPU+GPU em 3 fases), **Pipeline B Fortran** (novos recursos), **Pipeline A Python** (conversão Numba JIT CPU+GPU), **Pipeline B Python** (novos recursos), **Pipeline C Python** (vantagens sobre Fortran), e **Pipeline D Python** (avaliação comparativa Fortran vs Python).
Ref: `Tex_Projects/TatuAniso/FormulaçãoTatuAnisoTIV.tex` — artigo com a formulação matemática fundamental do simulador.

---

## 7. Referências Bibliográficas

### 7.1 Artigos Fundamentais do Pipeline

- Morales, M. et al. (2025). Anisotropic resistivity estimation and uncertainty quantification from borehole triaxial electromagnetic induction measurements: Gradient-based inversion and physics-informed neural network. *Computers & Geosciences*, 196, 105786.
- He, K. et al. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
- Bai, S. et al. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. *arXiv:1803.01271*.
- Oord, A. et al. (2016). WaveNet: A Generative Model for Raw Audio. *arXiv:1609.03499*.

### 7.2 Artigos Recentes (2024-2026) — Expansões Propostas

- INN-UDAR (2025). Invertible neural network for real-time inversion and uncertainty quantification of ultra-deep resistivity measurements. *Computers & Geosciences*.
- Jiang, H. et al. (2025). One-Fit-All Transformer for Multimodal Geophysical Inversion. *JGR: Machine Learning and Computation*.
- Li, X. et al. (2025). A Novel Self-Supervised Deep Learning Inversion Method Incorporating a Fast Forward Network for Transient Electromagnetic Data. *JGR: ML&C*.
- ModernTCN (2024). A Modern Pure Convolution Structure for General Time Series Analysis. *ICLR*.
- SPE/SPWLA (2021). Real-Time 2.5D Inversion of LWD Resistivity Measurements Using Deep Learning for Geosteering Applications Across Faulted Formations.

### 7.3 Referências dos PDFs do Projeto (`PDFs/`)

Os artigos abaixo estão disponíveis localmente na pasta `PDFs/` e devem ser consultados nas fases de expansão científica (F4) e validação física (F5).

#### 7.3.1 Deep Learning para Inversão EM

- **Noh, K., Pardo, D., Torres-Verdín, C.** (2023). Physics-guided deep-learning inversion method for the interpretation of noisy logging-while-drilling resistivity measurements. *Geophysical Journal International*, 235, ggad217. DOI: 10.1093/gji/ggad217
  → Arquivo: `PDFs/ggad217.pdf`
  → Uso: Validação científica dos cenários PINN (`losses/pinns.py`) com ênfase em robustez a ruído LWD.

- **Puzyrev, V.** (2019). Deep learning electromagnetic inversion with convolutional neural networks. *Geophysical Journal International*, 218, 817–832.
  → Arquivo: `PDFs/ggz204.pdf`
  → Uso: Base histórica para inversão EM via DL; comparação com arquiteturas CNN do projeto.

- **Liu, W., Wang, H., Xi, Z., Zhang, R., Huang, X.** (2022). Physics-Driven Deep Learning Inversion with Application to Magnetotelluric. *Remote Sensing*, 14(13), 3218. DOI: 10.3390/rs14133218
  → Arquivo: `PDFs/remotesensing-14-03218-v2.pdf`
  → Uso: Valida integração de operador forward físico no loop de treinamento (cenário "surrogate" PINN).

- **Liu, W., Wang, H., Xi, Z., Wang, L.** (2024). Physics-Informed Deep Learning Inversion with Application to Noisy Magnetotelluric Measurements. *Remote Sensing*, 16(1), 62. DOI: 10.3390/rs16010062
  → Arquivo: `PDFs/remotesensing-16-00062-v2.pdf`
  → Uso: Estratégias de noise injection para dados EM de campo; alinha com curriculum 3-phase do projeto.

#### 7.3.2 Forward Modeling e Ferramenta LWD

- **Guo, W., Wang, L., Wang, N., Qiao, P., Zeng, Z., Yang, K.** (2024). Efficient 1D Modeling of Triaxial Electromagnetic Logging in Uniaxial and Biaxial Anisotropic Formations Using Virtual Boundaries and Equivalent Resistivity Schemes. *Journal of Geophysics and Engineering*, gxag017. DOI: 10.1093/jge/gxag017
  → Arquivo: `PDFs/gxag017.pdf`
  → Uso: Algoritmo eficiente para forward 1D triaxial TIV — base para treinar SurrogateNet e validar surrogate analítico.

- **Wang, L., Li, H., Fan, Y., Wu, Z.** (2018). Sensitivity analysis and inversion processing of azimuthal resistivity logging-while-drilling measurements. *Journal of Geophysics and Engineering*, 15, 2339–2352. DOI: 10.1088/1742-2140/aac5b1
  → Arquivo: `PDFs/Wang_2018_J._Geophys._Eng._15_2339.pdf`
  → Uso: Algoritmo de inversão convencional Gauss-Newton + janela deslizante 1D — baseline para comparação com o pipeline DL.

- **Guoyu Li, Zhenguan Wu, Xiaoqiao Liao, Xizhou Yue, Xiang Zhang, Tianlin Liu, Yunxin Zeng** (2025). Optimization and Analysis of Sensitive Areas for Look-Ahead Electromagnetic Logging-While-Drilling Based on Geometric Factors. *Energies*, 18, 3014. DOI: 10.3390/en18123014
  → Arquivo: `PDFs/Guoyu_et_al_2025_FG.pdf`
  → Uso: Análise de sensibilidade e DOD para look-ahead EM LWD — referência para avaliação de DOD Picasso (`evaluation/dod.py`).

- **Constable, M. V., Antonsen, F., Stalheim, S. O.** et al. (2016). Looking Ahead of the Bit While Drilling: From Vision to Reality. *Petrophysics*, 57(5), 426–438.
  → Arquivo: `PDFs/Constable_et_al_2016_Petrophysics.pdf`
  → Uso: Contexto aplicado da ferramenta EMLA (look-ahead EM) com DOI > 75 m; justifica o foco em geosteering de longo alcance.

#### 7.3.3 PINNs e Métodos Físico-Informados

- **Bai, J., Jeong, H., Batuwatta-Gamage, C. P.** et al. (2022). An Introduction to Programming Physics-Informed Neural Networks for Computational Solid Mechanics. *arXiv:2210.09060*.
  → Arquivo: `PDFs/2210.09060v4.pdf`
  → Uso: Tutorial de implementação de PINNs (TensorFlow/JAX) — referência pedagógica para `losses/pinns.py`.

#### 7.3.4 Documentação Técnica de Ferramentas

- **SDAR Work Group** (2020). Benchmark Models for Look-Ahead Applications — Standardization of LWD Deep Azimuthal Resistivity. Industry Technical Document.
  → Arquivo: `PDFs/Benchmark_Look_Ahead_Models.pdf`
  → Uso: Modelos benchmark padrão para look-ahead com LWD — referência para validação de cenários de geosteering.

- **University of Houston** (2016). A Survey on Definitions of Some Key Service Specs of LWD Deep Azimuthal Resistivity Tools. Technical Report.
  → Arquivo: `PDFs/Key_Specs_Survey_UH.pdf`
  → Uso: Definições de DOI (Depth of Investigation), DOD (Depth of Detection) e especificações de ferramentas — base conceitual para `evaluation/dod.py` (Picasso plots).

- **Schlumberger** (2020). GeoSphere HD — High-Definition Reservoir Mapping-While-Drilling Service. Internal Documentation Rev. 1.0.
  → Arquivos: `PDFs/arquivo.md` (texto) e `PDFs/Schlumberger_GeoSphere HD 1.0.pdf`
  → Uso: Especificações da ferramenta GeoSphere HD — física dos geosinais USD/UAD/UHR/UHA/U3DF, configuração de antenas, interpretação de medidas, Picasso plots.

- **CNOOC/COSL** (internal). GeoSphere vs. TatuAniso1D — Convenções de Sinais. Technical Note.
  → Arquivo: `PDFs/GeoSphereXTatu.pdf`
  → Uso: Tabela de conversão de sinais USD/UAD/U3DF entre GeoSphere e simulador TatuAniso1D — essencial para validar `data/geosignals.py`.

#### 7.3.5 Novos Recursos do Simulador (Análise Abril 2026)

- **Documento de Análise**: `docs/reference/analise_novos_recursos_simulador_fortran.md` — Análise detalhada de estratégias 1.5D, 2D, compensação de poço, e novos recursos para o simulador Fortran. Abril 2026.
- **Noh, K., Pardo, D., Torres-Verdín, C.** (2021). Real-Time 2.5D Inversion of LWD Resistivity Measurements Using Deep Learning for Geosteering Applications. *Petrophysics*, 63(4), 506-524. → Base para estratégia 2.5D via DL.
- **Rabinovich, M. B.** et al. (2004). Effect of relative dip angle on electromagnetic measurements and formation boundary detection. *Petrophysics*, 45(6), 518-532. → Efeito do dip relativo nas medições EM — motivação para Feature 8 (1.5D).
- **Li, H. & Wang, H.** (2016). Investigation of eccentricity effect on induction response in horizontal wells using 3D FEM. *J. Pet. Sci. Eng.*, 143, 211-225. → Extensão 1.5D para poços horizontais.
- **Frontiers** (2025). Fast forward modeling and response analysis of extra-deep azimuthal resistivity measurements in complex model. *Front. Earth Sci.* → Modelagem rápida para UDAR em modelos complexos.

### 7.4 Referências do Simulador Fortran

- Liu, C. (2017). *Theory of Electromagnetic Well Logging*. Elsevier. — Rotação do tensor triaxial (eq. 4.80), utilizada em `RtHR()`.
- Werthmuller, D. (2006). "EMMOD — Electromagnetic Modelling". *Report, TU Delft*. — Filtro digital de 201 pontos para transformada de Hankel.
- Werthmuller, D. (2017). "An open-source full 3D electromagnetic modeller for 1D VTI media in Python: empymod". *Geophysics*, 82(6), WB9-WB19. — Referência Python para reimplementação.
- Chew, W.C. (1995). *Waves and Fields in Inhomogeneous Media*. IEEE Press. — Formulação TE/TM para meios estratificados.
- Anderson, W.L. (1982). "Fast Hankel Transforms Using Related and Lagged Convolutions". *ACM TOMS*, 8(4), 344-368.

### 7.5 Livros-Texto

- Ellis, D. V. & Singer, J. M. (2008). *Well Logging for Earth Scientists* (2nd ed.). Springer.
  → Arquivo: `PDFs/Darwin V Ellis Julian M Singer - Well Logging for Earth Scientists 2008 Springer - libgenli.pdf`
- Zhang, W. (2013). *Principles and Applications of Well Logging*. Springer.
  → Arquivo: `PDFs/Principles and Applications of Well Logging.pdf`
- Misra, S., Li, H. & He, J. (2019). *Machine Learning for Subsurface Characterization*. Elsevier.
- Zhang, A. et al. (2023). *Dive into Deep Learning*. D2L.ai.
  → Arquivo: `PDFs/[Livro] Dive Into Deep Learning_compressed.pdf`

---

## 8. Arquivos Críticos para Consulta

| Arquivo | Propósito |
|:--------|:----------|
| `geosteering_ai/config.py` | PipelineConfig (246 campos) — ponto único de verdade |
| `geosteering_ai/models/registry.py` | ModelRegistry — 48 arquiteturas |
| `geosteering_ai/losses/factory.py` | LossFactory.build_combined() |
| `geosteering_ai/losses/pinns.py` | 8 cenários PINN + forward analítico |
| `geosteering_ai/models/surrogate.py` | SurrogateNet TCN |
| `geosteering_ai/data/surrogate_data.py` | Extração de pares para surrogate |
| `geosteering_ai/training/loop.py` | TrainingLoop |
| `geosteering_ai/training/callbacks.py` | 17+ callbacks Keras |
| `geosteering_ai/inference/pipeline.py` | InferencePipeline (save/load) |
| `docs/ARCHITECTURE_v2.md` | Especificação arquitetural completa |
| `docs/physics/errata_valores.md` | Constantes físicas imutáveis |
| `CLAUDE.md` | Regras e proibições de desenvolvimento |
| `docs/ROADMAP.md` | Este documento |
| `docs/reference/documentacao_simulador_fortran.md` | Documentação completa do simulador Fortran PerfilaAnisoOmp |
| `Fortran_Gerador/PerfilaAnisoOmp.f08` | Simulador EM 1D TIV — módulo principal |
| `Fortran_Gerador/fifthBuildTIVModels.py` | Gerador de modelos geológicos (Sobol QMC) |
| `Tex_Projects/TatuAniso/FormulaçãoTatuAnisoTIV.tex` | Formulação teórica (Potenciais de Hertz, TIV) |

---

*Documento gerado em Abril 2026. Geosteering AI v2.0.*
*Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch proibido).*
*Configuração: PipelineConfig dataclass (ponto único de verdade).*
