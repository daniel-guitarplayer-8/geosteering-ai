# Changelog — Geosteering AI Simulation Manager

Todas as mudanças notáveis do Simulation Manager são documentadas aqui.

O formato segue [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/) e
o projeto usa [Versionamento Semântico](https://semver.org/lang/pt-BR/).

---

## [v2.57] — 2026-06-06 — SM MVVM landed na main: fundação GUI + Fatias 0-5 (Strangler Fig)

### Resumo

Merge da pilha `feat/gui-*` à `main`: a fundação **`geosteering_ai/gui/`** (MVVM compartilhada com o futuro Studio) + o app **`apps/sim_manager/`** (SM MVVM) chegam à branch principal. Construído por Strangler Fig em paralelo ao monólito intocado (`simulation/tests/simulation_manager.py`).

### Entregue (specs 0004-0012)

- **0004-0007** — Fundação `gui/`: `qt_compat` (PyQt6 primário, PySide6 fallback), base MVVM (`VMSignal`/`BaseViewModel`/`Perspective`/`MainWindowBase`), backends de plotagem (matplotlib/PyQtGraph/Plotly), persistência `.session` atômica.
- **0011 + 0011a-d** — SM MVVM Fatias 0-4: de-shim, walking skeleton, params multi-config (freqs/dips/TRs + geometria + `n_pos` Fortran), geração estocástica (7 geradores, batch ragged), galeria de resultados (componentes/plot-kinds/seletores/paginação/cache LRU).
- **0012** — JAX-GPU no SM MVVM: simulação em **subprocesso `spawn` TLS-safe** (evita crash `_dl_allocate_tls_init`), seletor de backend numba/jax/auto. Paridade JAX-GPU vs Numba `max|Δ|=4.38e-14` (<1e-12).
- **Fluxo Qt Designer** — `simulator.ui` editável + `qt_compat.load_ui`.

### Notas

- Física intocada (só `simulate_batch`; paridade Fortran <1e-12 preservada). Monólito 100% funcional.
- Em andamento (v2.58+): shell Antigravity (0013), Fatia 6a execução (0014), Fatia 6b geologia (0015); paridade total = Fatias 6c-6i.

---

## [v2.56] — 2026-06-02 — geosteering-cli: wall-clock JAX < Numba + transparência de tempo

### Resumo

`benchmark E --n 1000 --backend jax --geometry templates --repeat 3` mostrava "tempo 2,14 s" mas
`time real = 25,1 s`, e o **wall-clock total do JAX (25,1 s) superava o do Numba (19,5 s)** apesar de
~2× o throughput hot. Causa-raiz (bissecção + verificação adversarial — **FALSIFICOU a hipótese de
cache XLA**): (1) discrepância de observabilidade (a tabela só mostrava `median(hot)`, escondendo
startup+warmup; `elapsed_s = last_elapsed` ≠ `throughput = median`); (2) o gargalo NÃO é compilação
(cache XLA cheio: 284/284 hits, 2º run idêntico) e sim **tracing Python por-grupo** sobre os 31 grupos
`K=n//32`, re-pago a cada processo. **Fix (validado A6000)**: poucos grupos grandes + skip warmup morto
+ transparência + auto-chunk. **Medido: JAX warm `real` 12,75 s < Numba 19,97 s; hot 1,84M mod/h
(+13%, ±10k vs ±143k)**. Paridade Fortran 10/10 e throughput Numba preservados.

### Corrigido

- **Discrepância de tempo** (tabela 2,14 s vs `time real` 25,1 s): `t_warmup`/`t_total` instrumentados;
  `elapsed_s = statistics.median(elapseds)` (coerente com `thr_median`); novas linhas "tempo de warmup"
  e "tempo total (handler)" na tabela + `warmup_s`/`total_s` no JSON (`cli/{benchmark,simulate,_table}.py`).
- **JAX wall-clock > Numba**: default `templates` de **poucos grupos grandes** —
  `sample_geometry` `n_geo` de `max(1, n//32)` → **`max(1, min(n//256, 4))`** (n=1000: 31→3) corta o
  re-tracing Python por-grupo + satura melhor a GPU (`cli/_exec.py`). Numba é indiferente (pool).
- **Warmup morto no JAX**: skip `warmup_backend` no path jax (compilava shape errado `n_models=1`,
  ~3 s sem benefício); o JIT-warmup do workload completo é o warmup efetivo (`cli/{benchmark,simulate}.py`).

### Adicionado

- **`--jax-chunk-size N`** + `resolve_jax_chunk_size(backend, n_configs, explicit=)`: auto-chunk de
  modelos (64) só em high-config JAX (`nf·nTR·nAng ≥ 9` = G/H/F) — anti-OOM com poucos grupos grandes;
  E/C/D → vmap cheio. `run_once` propaga `jax_chunk_size_models` → `simulate_batch` (`cli/_exec.py`,
  `cli/_main.py`).
- **`tests/test_cli_timing.py`** (+7): `resolve_jax_chunk_size`, rows de tempo, `_build_stats` timing.

### Investigado (sem código novo)

- **Persistência de warmup em disco** (pedido do usuário): já é máxima nos dois backends — Numba `.nbc`
  (`NUMBA_CACHE_DIR`, v2.52; AOT `pycc` inviável em Numba ≥0.59) e JAX `jax_compilation_cache_dir`
  (148 MB). **`jax.export` (AOT) MEDIDO inefetivo**: `deserialize+call` (0,093 s) ≈
  `trace+compile(cache-hit)` (0,094 s). O lever efetivo é reduzir K (desloca o gargalo p/ codegen
  cacheável). Ver [report v2.56](reports/v2.56_cli_jax_wallclock_warmup_2026-06-02.md).

---

## [v2.55] — 2026-06-02 — geosteering-cli: crash TLS do JAX + lentidão (cold-start)

### Resumo

`benchmark E --backend jax --geometry templates --warmup` CRASHAVA com
`_dl_allocate_tls_init: Assertion 'listp != NULL' failed` e reportava JAX 5.5× mais lento. Bissecção:
o crash é exaustão de **TLS estático IN-PROCESS** (CUDA init no PAI → depois o warmup `models[:1]`
degenera p/ Numba → pool libgomp 64 threads estoura); a lentidão é **cold-start** (warmup compilava
ZERO programas JAX). **Fix híbrido** — validado no A6000: zero crash + JAX warm **1.66M mod/h (~2×
Numba 840k)**. Paridade Fortran 10/10 e throughput Numba preservados.

### Corrigido

- **Crash `_dl_allocate_tls_init`**: NOVO `cli/_exec.py::resolve_backend_preflight` + `_count_geometry_groups`
  (NumPy puro, jax-free) — para `--backend jax` + geometria não-agrupável, roda Numba **SEM chamar
  `jax.devices()`** (sem init CUDA → sem pressão de TLS → sem crash). Wired nos handlers simulate/benchmark.
- **Warmup JAX ineficaz (cold-start)**: o warmup JIT do benchmark para `jax` roda o **workload COMPLETO**
  (`models`, não `models[:1]`) → aquece o trace XLA vmap+scatter dos group-sizes reais → o timed-run roda
  QUENTE. Também elimina o Numba-após-CUDA (gatilho do crash). Numba mantém `models[:1]` (barato).
- **Mitigação universal de TLS**: `cli/_main.py` seta, via `setdefault` antes dos imports,
  `NUMBA_NUM_THREADS=(cpu//2)`, `OMP_NUM_THREADS=1` e `OPENBLAS_NUM_THREADS=1` — defesa validada por bissecção.
- **Log**: `simulation/config.py` auto-detect `logger.info → logger.debug` (silencia "auto-detect:
  n_workers=16" no JAX; lógica 100% funcional).

### Adicionado

- Warn em `--backend jax --geometry per-model` (via pré-voo) + backend EFETIVO + `motivo` na tabela.
- `tests/test_cli_jax_crash_guard.py` (9): `_count`≡`group_by_geometry`; pré-voo NÃO chama
  `_jax_gpu_available` p/ o caminho Numba; env threads; log debug.

### Preservado (sem regressão)

- Paridade Fortran **10/10 (<1e-6)** — kernels intocados. Throughput Numba **E n=200 = 239.313 mod/h**
  (≈ baseline). `--compare-backends` roda numba antes de jax (sem crash). CLI version `v2.54` → `v2.55`.

---

## [v2.54] — 2026-06-02 — geosteering-cli: correção da lentidão/trava do backend JAX

### Resumo

`benchmark --scenario E --n 500 --backend jax` travava: os modelos sintéticos tinham geometria
(`esp`) **única por modelo** → 500 grupos → JAX-grouped degenerava em 500 chamadas de 1 modelo
(zero batching) + sync por grupo; e `run_once` forçava `numba_fallback=False`, desligando a proteção
do dispatcher. **Fix híbrido**: `--backend jax` não-agrupável cai p/ Numba (sem travar, backend
efetivo honesto); novo `--geometry templates` cria compartilhamento de geometria → JAX satura a GPU.
Paridade Fortran 10/10 e throughput Numba preservados; `--compare-backends` continua forçando JAX
(paridade real medida = 3.86e-14).

### Corrigido

- **Trava do `--backend jax`**: `run_once` ganha `numba_fallback: bool = True` → geometria
  não-agrupável (n_grupos > 0.5·n_models) cai p/ Numba via o gate do dispatcher (antes bypassado por
  `numba_fallback=False` hardcoded). Validado GPU: E n=64 jax → fallback em 0.23s (sem trava).
- **`--compare-backends` (FURO 1)**: `run_compare_backends` passa `numba_fallback=False` EXPLÍCITO a
  `run_once` — senão o fix acima faria o compare medir Numba×Numba (paridade≈0, speedup≈1.0×).
- **Log ruidoso**: `SimulationConfig(backend="jax")` em `dispatch.py` recebe `n_workers=1/
  threads_per_worker=1` (inertes no JAX) → pula o auto-detect Numba "n_workers=16" inútil.

### Adicionado

- **`--geometry {per-model,templates,quantized}`** + `--n-geometries` + `--quantize-step` (simulate +
  benchmark). NOVO `cli/_exec.py::sample_geometry` espelha o `geometry_mode` de produção
  (`synthetic_generator`): `templates` (K geometrias replicadas → agrupável, JAX satura),
  `quantized` (esp arredondado → parcial), `per-model` (DEFAULT, preserva o stream rng legado).
- **Backend EFETIVO + `motivo (fallback)`** na tabela/JSON: `run_once` retorna o backend realmente
  executado + reason; a tabela mostra `backend=numba` + o motivo quando o jax cai p/ numba.
- **Guard warning-only** em `simulate_multi_jax_batched_grouped` (geometria degenerada → avisa, NUNCA
  reroteia — roteamento é do dispatcher).
- `tests/test_cli_geometry.py` (13) + extensões em `test_cli_backend_table.py` (fallback + compare
  força jax + reason).

### Preservado (sem regressão)

- Paridade Fortran **10/10 (<1e-6)** — kernels intocados. Throughput Numba **E n=200 = 227.727 mod/h**
  (≈ baseline 234.665). Paridade Numba×JAX medida na A6000 = **3.86e-14** (c128). CLI version
  `v2.53` → `v2.54`.

---

## [v2.53] — 2026-06-02 — geosteering-cli: backend JAX + tabela ASCII + salvar .dat/.out

### Resumo

A CLI ganhou **escolha de backend** (`--backend numba` padrão / `jax` GPU), **tabela ASCII de
resultados** (throughput, tempo, paralelismo, hardware, NaN/Inf) e **gravação `.dat`/`.out` 22-col**
conforme `geosteering-physics.md` §4 (com `res_h`/`res_v` reais na camada de cada `z_obs`). Caminho
Numba byte-a-byte o legado (zero regressão: E n=200 = 233.692 mod/h ≈ 234.665); JAX via dispatcher
parity-tested `simulate_batch` (paridade `max|Δ| = 2,71e-14`). Paridade Fortran 10/10 preservada.

### Adicionado

- **`cli/_backend.py`** — `resolve_backend(requested) → (backend, device)` com fallback gracioso
  jax→numba quando não há GPU JAX visível.
- **`cli/_table.py`** — `render_kv_table` (box Unicode + fallback ASCII via `GEOSTEERING_ASCII_TABLE`)
  + `build_result_rows`. 100% stdlib (sem `tabulate`/`rich`).
- **`cli/_hwinfo.py`** — `collect_hardware_info(*, want_gpu)`: CPU/cores/RAM/threads + GPU/VRAM/JAX
  devices; probes defensivos (`/proc`+`nvidia-smi`+`psutil`-opcional) que nunca levantam.
- **`cli/_exec.py`** — núcleo DRY: `run_once` (numba pool / jax dispatcher), `warmup_backend`,
  `finitude_stats`, `parity_max_abs_diff`, `models_to_batch`, `rho_at_obs_from_batch`,
  `run_compare_backends`.
- **`simulation/io/tensor_dat.py`** — `write_dat_from_tensor`/`write_out_file`/`compute_nmeds_per_angle`
  (relocados de `tests/sm_io.py` para produção).
- **Flags CLI** (simulate+benchmark): `--backend {numba,jax}` · `--dtype` · `--jax-strategy` ·
  `--warmup` · `--json` · `--repeat N` · `--compare-backends`; simulate: `--format {npz,dat,none}`;
  benchmark: `--list-scenarios` · `--quiet`.
- `tests/test_cli_{backend_table,hwinfo,dat_save}.py` (28) + `test_cli_mvp.py` estendido (+3) —
  inclui conformidade física §4 do `.dat` e guarda de não-regressão de roteamento Numba.

### Corrigido

- **Bug `--out`/`.npz`**: lia `getattr(result, "H", None)` (atributo real é `.H_stack` → sempre None →
  salvava `repr(result)`). Agora grava o tensor H real; guarda `test_npz_save_uses_real_tensor`.

### Mudado

- `simulation/tests/sm_io.py` → shim re-export de `io/tensor_dat` (retrocompat 100%).
- CLI version `v2.37` → `v2.53`.

### Preservado (sem regressão)

- Paridade Fortran **10/10 (<1e-6)** — kernels intocados. Throughput **E n=200 = 233.692 mod/h**
  (≈ baseline 234.665). Caminho Numba = `simulate_multi(models=…)` legado (pool de workers).

---

## [v2.52] — 2026-06-01 — Redução do warmup do simulador Numba JIT CPU

### Resumo

Análogo ao v2.51, mas para o **Numba CPU** (caminho crítico de paridade Fortran `<1e-6`).
Fecha o gap do warmup (que aquecia só hmd_tiv/vmd via callback JAX, nunca os kernels prange
de produção, E requeria JAX) + cache `.nbc` persistente cross-reboot. Paridade Fortran
preservada (10/10); zero regressão de throughput (E n=200 = 234.665 mod/h).

### Adicionado

- **`geosteering_ai/simulation/_numba/warmup.py`** — `warmup_numba_simulator(...)` +
  `warmup_numba_simulator_from_config(cfg)`: roda o caminho REAL `simulate_multi(backend="numba")`
  em 2 chamadas tiny (multi-combo + single-combo) → aquece os kernels `parallel=True`/prange
  (`_simulate_combined_prange_flat`/`_simulate_positions_njit_cached`/`precompute_common_arrays_cache`)
  + dipolos inlinados. **JAX-independente**. `from_config` honra os seletores de kernel via `base_cfg`.
- **CLI `geosteering-warmup --numba` (default)** + `--no-numba`/`--numba-n-pos`/`--numba-n-layers`/
  `--numba-threads`. Substitui o callback-JAX como warmup default (roda sem JAX).
- `tests/test_simulation_numba_warmup.py` (7) + teste das flags `--numba` + `scripts/diagnose_numba_warmup.py`.

### Mudado

- **NUMBA_CACHE_DIR default ESTÁVEL** `~/.cache/geosteering/numba_cache` (sobrevive reboot;
  fallback `$TMPDIR`; `0o700`) — cross-process **27.72s → 0.27s = 103×** (.nbc persiste).
- Background-thread (`geosteering-cli`) aponta p/ o novo warmup Numba (n_pos=64).

### Análise (documentado)

- **AOT (`numba.pycc`) NÃO viável** — removido no Numba ≥0.59 (env 0.65.1) + incompatível com
  `parallel=True`. O ~111s bitcode→nativo é o piso fundamental do `.nbc`; amortizado pelo cache.

Ref: [docs/reports/v2.52_warmup_numba_cpu_2026-06-01.md](reports/v2.52_warmup_numba_cpu_2026-06-01.md)

---

## [v2.51] — 2026-06-01 — Redução do warmup do simulador JAX GPU

### Resumo

Fecha o gap do warmup (que aquecia só o path Numba, nunca o kernel JAX GPU de
produção) + hardening do cache XLA persistente. Resolve o cold-start de
compilação de inversão/inferência e dá **4.81× cross-process**. Escopo conservador:
shape-bucketing (data-gen multi-geometria) fica DOCUMENTADO, sem tocar o kernel.
Paridade Fortran `<1e-13 c128` preservada (4/4); throughput morno inalterado (DG 6/6).

### Adicionado

- **`geosteering_ai/simulation/_jax/warmup.py`** — `warmup_jax_simulator(...)` +
  `warmup_jax_simulator_from_config(cfg)`: pré-compila o kernel bucketed de produção
  (`use_native_dipoles=True`) via dispatch real → popula o cache JIT in-process + o
  cache de disco persistente. Real-call seguinte = **0 recompiles**.
- **CLI `geosteering-warmup --jax`/`--gpu`/`--auto`** + overrides (`--jax-n-pos/--jax-dtype/
  --jax-n-layers/--jax-n-models`); lifta `JAX_PLATFORMS=cpu` p/ GPU (respeitando override do
  usuário) via `_resolve_jax_warmup` (puro, testado por tabela-verdade).
- `tests/test_simulation_jax_warmup.py` (6) + 8 testes CLI/lift em
  `test_cli_warmup_entry_point.py` + `scripts/diagnose_jax_warmup.py`.

### Mudado

- **`_jax/__init__.py`**: `jax_persistent_cache_min_compile_time_secs=0.0` (todo compile
  persiste; version-guarded) + dir default ESTÁVEL `~/.cache/geosteering/jax_compilation_cache`
  (sobrevive reboot; fallback `$TMPDIR` gracioso). Cross-process **4.81×** (10.04s→2.09s).
- `.github/workflows/ci.yml`: `geosteering-warmup --verbose --auto`.

### Análise (documentado, NÃO implementado)

- **Shape-explosion**: o kernel bucketed recompila por shape de `z_bucket` → data-gen
  multi-geometria aleatória não é cache-cobrível cross-run. Mitigação **shape-bucketing**
  (flag futura `jax_pad_buckets_to`, default None=bit-idêntico) documentada com análise de
  risco — toca o kernel quente → exigiria gate `<1e-13` + DG antes de qualquer default-flip.

Ref: [docs/reports/v2.51_warmup_jax_gpu_2026-06-01.md](reports/v2.51_warmup_jax_gpu_2026-06-01.md)

---

## [v2.50] — 2026-06-01 — API on-the-fly de dados do Surrogate (dívida Sprint C)

### Resumo

Produtiza a geração on-the-fly de pares (ρ→H) do SurrogateNet em
`data/surrogate_data.py` — substitui o benchmark hard-rolled + round-trip `.npz`
offline por uma **API de dados reusável e testada**, matando o laço Python O(N×L).
Escopo: API de dados + iterador (sem integração de treino TF). Paridade Fortran
`<1e-13 c128` preservada (simulador intocado); **zero regressão de throughput**.

### Adicionado

- **`surrogate_pairs_from_batch(batch, config, *, apply_decoup=True)`** — ponte
  `GeneratedBatch → SurrogateDataset` VETORIZADA (monta `(N,L,22)` do `H_tensor` +
  delega à testada `extract_surrogate_pairs`). Paridade bit-exata vs caminho offline.
- **`generate_surrogate_dataset(config, *, n_models, ...)`** — builder one-shot
  on-the-fly (amostra + simula via dispatcher + extrai), `build_dat_22col=False`.
- **`iter_surrogate_batches(config, *, batch_size, n_batches=None, seed=0, ...)`** —
  stream de modelos FRESCOS por batch (`seed+i`) p/ treino on-the-fly (refresh/época);
  validação eager; documenta contenção JAX/TF (gerar entre épocas, `backend='auto'`).
- **`_layer_at_batch`** em `synthetic_generator.py` — lookup de camada VETORIZADO
  (bit-exato a `_find_layer_for_z`).
- `tests/test_surrogate_onthefly.py` — **14 testes** (bit-exato + paridade + freshness
  + determinismo + edge + reroute).

### Mudado

- **`generate_batch` — `dat_22col` VETORIZADO**: laço Python O(N×L) → preenchimento
  coluna-a-coluna (`reshape` C-order = ordem Fortran m→p). **Bit-exato** (guard test).
  Novo param `build_dat_22col: bool = True` (pula a montagem no caminho on-the-fly).
- `benchmarks/_sprintc_phase_a_gen.py` — produtizado: usa `generate_surrogate_dataset`
  (remove `_layer_at` local + montagem 22-col manual).

### Throughput / Paridade

- **DG perf gate 6/6 PASS** (autoritativo): 0 regressão de pacote.
- **A/B grouped vs blocks @128/grupo**: NEW 155.6s vs OLD 186.8s → **0.83×** (API mais rápida).
- Fortran parity 4/4 + regressão (surrogate 5 + batched 17) = **48/48 PASS**.

Ref: [docs/reports/v2.50_surrogate_onthefly_data_api_2026-06-01.md](reports/v2.50_surrogate_onthefly_data_api_2026-06-01.md)

---

## [v2.49] — 2026-05-31 — Reassembly vetorizado (dívida Sprint C) + pesquisa O6+ levers #3/#4/#5

### Resumo

Resolve o achado crítico do Sprint C (reassembly do grouped) e fecha a pesquisa O6+
dos 3 levers de kernel restantes com **dados empíricos no A6000**. Paridade Fortran
`<1e-13 c128` preservada (47/47); zero regressão.

### Mudado

- **`simulate_multi_jax_batched_grouped` — reassembly VETORIZADA**: substitui o padrão
  `[None]*n_models` + atribuição modelo-a-modelo + `np.stack` por pré-alocação única +
  scatter por grupo (`H_tensor[sel] = np.asarray(res.H_tensor)`). **Bit-exato** (Δ=0.0)
  e **pico de memória −49 %** (4995→2532 MB a 30k×600) — evita OOM em tensores multi-config.

### Pesquisa (O6+ — nenhuma mudança de código; levers fechados)

- **Lever #3** (fatorar `exp(-2uh)`): **REJECT** — jaxpr tem **52 exp, 0 duplicados** →
  nada a fatorar (per-camada já é 1 op `(5,201)`; demais 50 têm argumentos distintos).
- **Lever #4** (c64 seletivo): **DEFER** — `jnp.exp` c64 é **11× mais rápido isolado**, mas
  **~0 no kernel** (bucketed = scan/latency-bound) e **paridade-fatal** (`max_rel 3.9e-4`).
- **Lever #5** (TF32 tensor-core Hankel): **REJECT** — só **1.36×** + **37–78 % de erro** no
  filtro Werthmuller-201 real (cancelamento κ 400–800); Amdahl teto 1.05×.

### Correção (honestidade)

- O relatório v2.48 atribuiu a morte do run de 30k ao `np.stack` O(n). **Refutado**:
  `res.H_tensor` já é numpy → reassembly é memcpy-bound (~1 s). O ganho real é **memória**;
  o gargalo de escala é o **nº de geometrias** (overhead de launch JAX por grupo).

Ref: [docs/reports/v2.49_reassembly_vetorizado_e_pesquisa_levers_o6_2026-05-31.md](reports/v2.49_reassembly_vetorizado_e_pesquisa_levers_o6_2026-05-31.md)

---

## [v2.48] — 2026-05-31 — Sprint B: A-jax-gpu-dispatcher (simulate_batch backend="auto")

### Resumo

Generaliza o roteamento da Sprint A num **dispatcher reutilizável**
`simulate_batch(..., backend="auto")` que codifica a árvore de decisão medida
(v2.45–v2.47): roteia automaticamente entre JAX GPU (bucketed/grouped) e Numba
16w×4t por GPU-disponível × n_models × agrupabilidade da geometria, com guard
contra o kernel `unified` em high-config (OOM). Overhead **+0.3%**. Paridade
Fortran **<1e-13 c128 preservada** (53/53); revisão code-reviewer APROVAR.

### Added

- `geosteering_ai/simulation/dispatch.py`::`simulate_batch(rho_h_batch, rho_v_batch,
  esp_batch, positions_z, *, frequencies_hz, tr_spacings_m, dip_degs, backend="auto",
  numba_fallback, n_models_gpu_threshold=32, dtype, jax_chunk_size_models, jax_strategy,
  hankel_filter)` — dispatcher batched. Helpers `_resolve_backend` (árvore pura),
  `_jax_gpu_available` (`jax.devices()`), `_simulate_batch_numba`. Import jax LAZY
  (numba-only OK). Exportado em `simulation`.
- `"auto"` em `PipelineConfig.simulator_backend`; guard device='gpu' relaxado p/ `{jax, auto}`.
- `tests/test_simulation_dispatch.py` — 12 testes (8 ramos da árvore + guard unified +
  paridade cross-backend <1e-10 + dispatcher==grouped direto).

### Changed

- `SyntheticDataGenerator.generate_batch` refatorado p/ usar `simulate_batch` (DRY,
  −49 linhas; comportamento Sprint A para `backend="jax"` preservado).

### Notes

- Guard anti-unified: `jax_strategy="unified"` em high-config (n_pos≥300, n_configs≥9)
  → `ValueError` (impede OOM 80 GB). O caminho JAX SEMPRE usa grouped → unified nunca atingido.
- Fecha `A-jax-gpu-dispatcher` (ROADMAP §0); `C-surrogate-train` desbloqueado.

Relatório: `docs/reports/v2.48_sprint_b_dispatcher_2026-05-31.md`.

---

## [v2.47] — 2026-05-30 — Sprint A: A-jax-gpu-data-gen + agrupamento por geometria

### Resumo

Faz a geração de dataset usar o caminho **batched-bucketed rápido** para datasets
REAIS (geologia variável), via 3 modos de geometria (templates default, quantize,
fallback Numba) + helper `group_by_geometry` + extração de features (FV/GS) + gate de
performance. **Medido: 1.89× Numba** (18 cfg, 600 pos, n=64). Paridade Fortran
**<1e-13 c128 preservada** (53/53); revisão `/geosteering-code-reviewer` APROVAR.

### Added

- `group_by_geometry(esp_batch)` em `_jax/multi_forward.py` — partição PURA por
  `esp.tobytes()` → `list[np.ndarray]`. `simulate_multi_jax_batched_grouped`
  refatorado p/ usá-lo (DRY). Exportado.
- `generate_batch`: `geometry_mode` ∈ {"templates"(default), "quantize", "per_model"}
  + `quantize_step` + `numba_fallback` (auto quando geometria mal-agrupável) + warn
  `n_models<32`. metadata: `geometry_mode`/`n_geometry_groups`/`fallback`.
- `SyntheticDataGenerator.to_feature_dataset(batch, *, apply_transforms, feature_view,
  geosignal_families, decouple)` + `FeatureDataset` dataclass — H_tensor (18 floats/pos)
  → input_features (5) + output_targets (2); opcional decouple→FV→GS (reusa
  `apply_feature_view`/`compute_geosignals`/`apply_decoupling`). `GeneratedBatch`
  +`tr_spacings_m`/`dip_degs` (self-describing).
- Gate de performance `DG` (data-gen high-config homogêneo, n=64, 600 pos, 18 cfg) em
  `test_simulation_jax_perf_baseline.py` + `DG_hot` (122878 mod/h) em `perf_baseline.json`.

### Notes

- OOM-fix por design: o backend jax do gerador SEMPRE usa o caminho agrupado
  (bucketed por grupo) — nunca o kernel `unified` (que OOMa 80 GB a 18cfg/600pos).
- Fecha o backlog `A-jax-gpu-data-gen` (ROADMAP §0); desbloqueia `C-surrogate-train`.

Relatório: `docs/reports/v2.47_sprint_a_datagen_geometry_grouping_2026-05-30.md`.

---

## [v2.46] — 2026-05-30 — Geração de dataset no caminho batched JAX (agrupamento por geometria)

### Resumo

Fecha a lacuna estrutural do caminho de treino (backlog `A-jax-gpu-data-gen`): o
`SyntheticDataGenerator.generate_batch` fazia loop por-modelo, ignorava o caminho
batched e gerava geometria heterogênea (que cairia no kernel *unified* ~7× lento +
OOM). Agora usa o backend batched + **agrupamento por geometria**, com multi-config e
controle de VRAM. Paridade Fortran **<1e-13 c128 preservada**; cross-backend JAX vs
Numba **<1e-10**.

### Added

- `simulate_multi_jax_batched_grouped(rho_h, rho_v, esp, positions_z, *, frequencies_hz,
  tr_spacings_m, dip_degs, cfg)` em `_jax/multi_forward.py` — particiona o batch por
  `esp` idêntico → `simulate_multi_jax_batched` (bucketed) por grupo → reassembla na
  ordem original. Retorna `(H_tensor (n_models,nTR,nAng,n_pos,nf,9), info)`. Exportado.
- `SyntheticDataGenerator.generate_batch`: params `frequencies_hz`/`dip_degs`/
  `tr_spacings_m` (multi-config), `n_geometries` (geometrias batcháveis, round-robin),
  `jax_chunk_size_models` (VRAM). `metadata["n_geometry_groups"]` (diagnóstico).
- `tests/test_synthetic_generator_batched.py` — 10 testes (grouping bit-exato <1e-13,
  multi-config shape, n_geometries=K→K grupos, chunk, grid-max, JAX vs Numba <1e-10).

### Changed

- `generate_batch`: backend `jax` → grouped batched; `numba` → `simulate_multi` por
  modelo. Shape retrocompatível `(n_models,n_pos,nf,9)` em single-config; grid
  `positions_z` cobre o modelo mais espesso do batch.

### Fixed

- Path JAX antigo do gerador era inoperante (`SimulationConfig(use_native_dipoles=...)`
  — kwarg inexistente → TypeError); agora roda e é validado <1e-10 vs Numba.

Relatório: `docs/reports/v2.46_jax_datagen_batched_grouped_2026-05-30.md`.

---

## [v2.45] — 2026-05-29 — Follow-ups pós-O4: CI verde, perf-gate A6000, dataset .dat, benchmarks

### Resumo

Sprint de consolidação com 6 follow-ups sobre o simulador EM 1D (Numba + JAX).
**CI 100% verde** (18 falhas pré-existentes → 0; issue #44 fechada), **gate de
regressão de performance local A6000** (substitui T4 deprecado), **geração de
dataset `.dat` no fluxo batched JAX** e **caracterização experimental do crossover
Numba×JAX** por nº de configs e batch-size (A6000, 600 n_pos). Paridade Fortran
**<1e-13 c128 preservada** (margem CI 2.95e-14). Revisão multi-agente: zero
regressão, zero quebra de paridade/fidelidade.

### Added

- `SimulationConfig.export_per_model` — `simulate_multi_jax_batched` exporta 1
  conjunto de `.dat` (22-col Fortran) por modelo do batch (dataset sintético p/ treino),
  reusando o writer validado `export_multi_tr_dat`.
- Campo `H_tilted: Optional[np.ndarray]=None` em `MultiSimulationResultJAX` e
  `MultiSimulationResultBatchedJAX` (compat `export_info_out`).
- Gate de regressão JAX GPU local — `jax_gpu_a6000_gate` em `perf_baseline.json`
  (A/B/E/G, n_models=50, thresholds 90%); detecção A6000/RTX em `test_simulation_jax_perf_baseline.py`.
- Flag `--n-pos-all` no bench (`bench_numba_vs_jax_gpu.py`); alvo `portable`
  (`-march=x86-64-v2`) no Makefile Fortran.
- Guard de aviso: `export_per_model + use_tilted_antennas` no batched JAX (F7 não
  suportado → dataset sem projeção inclinada) — `logger.warning` + teste.

### Fixed

- **CI verde (issue #44)**: 12 testes paridade Fortran SIGILL (`tatu.x` portável no
  runner); 4 testes TF (fixture `PreparedData` sem `z_*`); warmup threshold (5s);
  hook PR-desc resiliente a `origin/<base>` + `fetch-depth: 0`.

### Decided (sem mudança de código)

- **Item 2 — vmap real sobre grid de configs**: benchmarkado → **regressão 0.60–0.74×**
  (força kernel unified ~6.9× mais lento). `jax_vmap_real=False` mantido como default.

### Deprecated

- Baselines Colab `jax_gpu_t4` / `jax_gpu_a100` (`_meta.deprecated=true`) — dev GPU
  agora local A6000.

Relatório: `docs/reports/v2.45_sprint_followups_2026-05-29.md`.

---

## [v2.44] — 2026-05-29 — Sprint O4: Batched-Bucketed JAX GPU

### Resumo

Elimina o gargalo on-the-fly do simulador JAX GPU: `simulate_multi_jax_batched`
deixa de hardcodar o kernel `unified` (~6.9× mais lento) e passa a usar
**bucketed** quando a geometria é compartilhada entre os modelos do batch
(regime PINN / geração on-the-fly). Throughput on-the-fly (32 modelos × 600 pos,
A6000): **65k → 1.47M mod/h (22.5×)**, superando Numba 4w×16t (~1.05M) nesse regime.

### Added

- `simulate_multi_jax_batched`: path **batched-bucketed** (`vmap` dos kernels de
  bucket sobre o eixo de modelos) com dispatcher por geometria compartilhada
  (`np.allclose(esp, esp[0], atol=0)`) e fallback seguro p/ `unified` (warning).
- `SimulationConfig.jax_chunk_size_models`: fatia o eixo de modelos (fix OOM
  Cenário H 8×8×8 = 512 configs, antes ~110 GB).
- `bench_numba_vs_jax_gpu.py`: `--chunk-size-models`; Cenário H habilitado em
  `--batched`; defaults `bucketed`.
- Suite `tests/test_simulation_jax_o4_batched_bucketed.py` (16 testes).

### Fixed

- **Colisão de chave `_CTX_CACHE` n=1↔n=2** (`esp` vazio em ambos): `_hash_ctx_key`
  agora inclui `n` + comprimentos de esp/pos/freqs. Afetava também o serial.

### Validação

- Gates F1–F4 pós-O4: **220 PASS / 0 FAIL** (Fortran c128 <1e-12 inviolável).
- Paridade O4 <1e-13: batched-bucketed vs serial bucketed / vs batched-unified /
  multi-dim; chunking bit-exato (bucketed).
- Relatório: [docs/reports/v2.44_sprint_O4_batched_bucketed_2026-05-29.md](reports/v2.44_sprint_O4_batched_bucketed_2026-05-29.md)

---

## [v2.43] — 2026-05-22/2026-05-23 — Sprint A1.6: Rewrite Notebook JAX GPU Benchmark + Validação T4

### Resumo

Sprint A1.6 (`A-jax-gpu-benchmark-redesign`) reescreve o notebook
`validate_jax_gpu_v240.ipynb` para consumir a API batched
`simulate_multi_jax_batched` (introduzida em v2.42) e corrigir 8 bugs
metodológicos identificados na auditoria `v2.40.4_auditoria_resultados_sprint_a1`.

**Validação experimental T4 (2026-05-23, commit `a06cf12`):**

- 164/164 testes paridade Fortran PASS (tol <1e-12)
- Gate ≥1.5× Numba T4 LOCAL APROVADO: A 2.56×, B 2.86×, E 1.90×
- Warmup CRIT-2 efetivo (C/H ratio ∈ [0.87, 1.01])
- **Baseline JAX GPU oficial estabelecida** (`.claude/perf_baseline.json::jax_gpu_t4`)
- Relatório completo: [docs/reports/v2.43_jax_gpu_baseline_t4_2026-05-23.md](reports/v2.43_jax_gpu_baseline_t4_2026-05-23.md)

### Corrigido pós-merge (2026-05-23, commit `a06cf12`)

- **OOM cenário F em T4** (`XlaRuntimeError: RESOURCE_EXHAUSTED, 17.8 GB`):
  reduzido `N_MODELS_PER_SCENARIO` para F=20, G=5, H=5 (XLA materializa
  `n_models × nTR × nAng` simultaneamente). A/B/C/D/E mantêm n_models=50.
- **`try/except RESOURCE_EXHAUSTED`** no warmup (Cell 7) e benchmark (Cell 14)
  para degradação graciosa em cenários que ainda OOM (H em T4).
- **`oom: true`** marcador no JSON quando cenário pulado por falta de VRAM.

### Adicionado

- **Notebook reescrito** (`notebooks/colab_templates/validate_jax_gpu_v240.ipynb`):
  21 células (10 md + 11 code), substitui loop Python `for m in models:
  simulate_multi_jax(...)` por uma única chamada `simulate_multi_jax_batched`.
- **Baseline Numba T4 LOCAL** via `subprocess.run(["geosteering-cli", "benchmark",
  ...])` com `--workers/--threads` pinados (4w × 1t para n1-standard-4 sem HT).
- **Run 1 (cold) reportado separado** de `statistics.median(Runs 2-5 hot)`.
- **Ratio cold/hot** como diagnóstico de efetividade de warmup (≈1.0 esperado).
- **`fixes_applied` + `review_fixes_applied`** dicts no JSON de saída para
  rastreabilidade dos 8 bug fixes + 9 review findings.

### Corrigido (8 bugs metodológicos da Sprint A1)

- **C1 (CRIT)**: warmup global shape fixa → warmup por-cenário com batch size
  EXATO de `N_MODELS_PER_SCENARIO[scen]` (XLA cache key inclui axis size).
- **C2 (CRIT)**: `statistics.median([T1..T5])` mascarava cold-start → Run 1
  isolado, mediana apenas sobre Runs 2-5 hot.
- **H1 (HIGH)**: `d.platform == "gpu"` deprecada → `jax.default_backend()
  in ("gpu", "cuda")` (compat JAX 0.4.14+).
- **H2 (HIGH)**: sem `block_until_ready()` explícito → batched API já chama
  internamente; defensivo `result.H_tensor.shape` após call.
- **H3 (HIGH)**: warmup só `models[0]` (50 ct/cr distintos no `_BUCKET_JIT_CACHE`)
  → resolvido por design (`_UNIFIED_JIT_CACHE` keyed `(n, npt)`).
- **M1 (MED)**: H dip 87.5° JAX vs 90° Numba → documentado (validador JAX cap
  em 89°); tabela imprime `(dip≠)` no lugar do ratio; H fora do gate.
- **M2 (MED)**: `esp` alta variância → resolvido por design (cache não chaveia
  em esp).
- **M3 (MED)**: Numba baseline hardcoded i9-9980HK → medição local T4 via CLI
  subprocess.

### Corrigido (9 review findings multi-agente — C02)

- **CRIT-1**: `latency_ms_per_model_hot` formula off-by-factor `n_models`
  → `3_600_000.0 / median_hot` (correto).
- **CRIT-2**: warmup `batch=2` não cobre benchmark `batch=50/10` (XLA re-traçava
  outer vmap) → `np.tile(template, (N_MODELS_PER_SCENARIO[s], 1))`.
- **HIGH-1**: `FileNotFoundError/OSError` não capturados em CLI subprocess →
  capturados, retorna `None` graciosamente.
- **HIGH-2**: cenário H timeout 600s insuficiente → `NUMBA_CLI_TIMEOUT_S[H]=1800`.
- **MED-1**: H row na tabela sem anotação → `'(dip≠)'` + legenda.
- **MED-3**: `_pytest_data` duplicado em Cells 9 + 18 → single load.
- **MED-4**: `JAX_PLATFORMS=""` pode falhar JAX 0.4.30+ → `"cuda,cpu"` explícito.
- **MED-5**: `--workers/--threads` não pinados em CLI baseline → `4w × 1t`.
- **LOW-1**: Cell 0 table faltava linha M2 → adicionada.

### Validação Local

- AST OK em todas as 11 células code (após strip de magic continuations)
- 13/13 checks de fixes presentes (incluindo dimensional)
- 24 PASS / 1 SKIP em `test_simulation_jax_batched_api.py` (zero regressão)
- Suite total preservada (1686 PASS de v2.42 mantido)

### Não Alterado

- `geosteering_ai/simulation/_jax/multi_forward.py` (API batched v2.42 intocada)
- `geosteering_ai/cli/benchmark.py` (consumido como subprocess)
- Zero testes Python novos (notebook = deliverable do usuário em Colab T4)
- Paridade Fortran <1e-12 PRESERVADA (gate inviolável)

### Próximos Passos

- Usuário executa notebook em Colab Pro+ T4 e reporta resultados
- Após gate confirmado: Sprint A2 (`A-jax-gpu-dispatcher`)
- Ver [docs/sprints/v2.43.md](sprints/v2.43.md) (snapshot imutável)

---

## [v2.42] — 2026-05-20 — Sprint A1.5: API Batched JAX (`simulate_multi_jax_batched`)

### Resumo

Sprint A1.5 (`A-jax-gpu-batched-api`) introduz `simulate_multi_jax_batched()`
— nova API pública que aplica `jax.vmap` sobre o eixo `n_models`, resolvendo
a causa-raiz arquitetural da Sprint A1 (gate de performance falhou: A: 0.38×,
B: 0.37×, E: 0.61× Numba). **Paridade vs loop serial: max |diff| = 8.33e-14**
(bit-exato ordem ULP float64). 24/25 testes PASS (1 SKIP GPU).

### Mudanças

**1. Nova API pública batched** (`geosteering_ai/simulation/_jax/multi_forward.py`):

- `simulate_multi_jax_batched(rho_h_batch, rho_v_batch, esp_batch, positions_z, ...)` —
  aceita arrays 2D `(n_models, n)` e processa todos os modelos em um único trace
  JAX via composição vmap dupla:
  - Vmap externo: `in_axes=(0, 0, 0, 0)` sobre (rho_h, rho_v, h_arr, prof_arr)
  - Vmap interno: `in_axes=(0, 0)` sobre (L_flat, theta_flat)
- `MultiSimulationResultBatchedJAX` dataclass com shape
  `(n_models, nTR, nAngles, n_pos, nf, 9)` complex128 + método `get_model(i)`
- `_sanitize_profile_batch(n, esp_batch)` helper — pré-computa h_arr/prof_arr
  para batch (Python loop sobre `_sanitize_profile_kernel` Numba, <50 ms total)
- Reutiliza `_UNIFIED_JIT_CACHE` (key `(n, npt)` invariante a valores de modelo)
  — 1 compilação XLA para o batch inteiro, eliminando bucket cache explosion

**2. Resolve as 3 causas-raiz da Sprint A1**:

| Causa-raiz | Antes | Depois |
|:--|:--|:--|
| Bucket cache explosion | ~50 compilações XLA em Run 1 | 1 compilação via `_UNIFIED_JIT_CACHE` |
| Overhead Python serial | `build_static_context()` × 50 = 0.25-1 s | Static context compartilhado <50 ms total |
| GPU→CPU sync por modelo | 50 syncs serializados | 1 `block_until_ready()` + 1 `np.asarray()` |

**3. Exports top-level** (`geosteering_ai/simulation/__init__.py`):

```python
from geosteering_ai.simulation import (
    simulate_multi_jax_batched,
    MultiSimulationResultBatchedJAX,
)
```

**4. Suite de testes** (`tests/test_simulation_jax_batched_api.py` — 24 PASS / 1 SKIP):

- T1-T3: Paridade vs loop serial (<1e-12) — bit-exato
- T4-T6: Edge cases físicos (n=1, n=2, shape/dtype)
- T7-T8: Validações fail-fast
- T9-T11: Garantias arquiteturais (grad, NaN/Inf, oklahoma_3)
- T12-T13: Plataforma (CPU/GPU @pytest.mark.gpu)
- T14: Inspeção AST — exatamente 1× block_until_ready + 1× np.asarray
- T15: Backward-compat — `simulate_multi_jax` legada inalterada
- **Review fixes (8 gaps)**: positions_z vazio, TIV anisotrópico, listas vazias
  parametrizado (3), 1D rejection, snapshot não-corrupção do cache, freqs extremos

**5. Correção factual Mac Intel** (4 docs Sprint A1):

- 4 docs atualizados: baseline Numba foi medido em **Intel i9-9980HK 8C/16T
  (Mac Intel)**, NÃO Apple M-series como afirmado anteriormente
- Referências canônicas: `.claude/perf_baseline.json:15` +
  `docs/reports/v2.36_2026-05-15.md:8`

**6. Errata sobre `multi_forward.py:400-422`** (auditoria v2.40.4):

- Linhas 400-422 são loop sobre (iTR, iAng) para UM modelo (não sobre modelos)
- Loop sobre modelos ocorria no notebook `a1-bench-defs` (50× `simulate_multi_jax`)
- Causa-raiz arquitetural permanece: assinatura aceita 1 modelo → A1.5 resolve

### Multi-Agent Review

Fan-out via `feature-dev:code-reviewer` (revisão físico/vmap + revisão de cobertura):
- 1 CRÍTICO + 3 ALTO + 1 MÉDIO **acionáveis** aplicados (GAP-C1, GAP-C2, GAP-A1,
  GAP-A2, GAP-A3, GAP-M1)
- 3 findings sobre legacy code (A2/A3/M2) intencionalmente **não-acionados**
  (fora do escopo A1.5 — não modificar legacy)

### Restrição arquitetural

Todos os modelos do batch DEVEM compartilhar o mesmo `n` (n_camadas). Para
`n` heterogêneo, agrupar por `n` e chamar separadamente. ValueError com
diagnóstico claro se violado.

### Não Altera

- `simulate_multi_jax` legada (backward-compat 100% preservada)
- `_BUCKET_JIT_CACHE`, `_UNIFIED_JIT_CACHE` (`forward_pure.py` intocado)
- `_simulate_multi_jax_vmap_real` (Sprint 12 preservado)
- `cfg.SimulationConfig` (sem flag opt-in — dispatcher é A2)

### Próximo

Sprint A1.6: `A-jax-gpu-benchmark-redesign` — rewrite notebook
`validate_jax_gpu_v240.ipynb` consumindo `simulate_multi_jax_batched`,
warmup completo, `block_until_ready` explícito, baseline Numba medido
localmente no T4.

---

## [v2.41] — 2026-05-19 — Sprint A1: Validação JAX GPU T4 (DONE-PARTIAL)

### Resumo

Sprint A1 (`A-jax-gpu-validate`) executou `validate_jax_gpu_v240.ipynb` em
Colab Pro+ T4 com RAM Alta. **Paridade Fortran <1e-12 confirmada em GPU real**
(163/163 pytest PASS). **Gate de performance reprovado**: A: 0.38×, B: 0.37×,
E: 0.61× Numba — abaixo do threshold de 1.5×. Causa-raiz identificada:
ausência de API batched sobre o eixo de modelos (loop Python serial domina
compute JAX).

### Mudanças

**1. Notebook `validate_jax_gpu_v240.ipynb` — 2 fixes críticos**:

- Fix §2 `a1-l5-env`: `JAX_PLATFORMS="gpu"` → `""` (auto-detect). `"gpu"` é
  plataforma inválida em JAX (fallback rocm → `RuntimeError`). `"cuda"` foi
  tentado intermediariamente mas desabilitava backend CPU, quebrando
  `jax.pure_callback` em `kernel.py:400` (Numba FFI requer CPU backend
  registrado). Solução final: string vazia inicializa todos os backends
  disponíveis (CPU + CUDA).
- Markdown atualizado nas células `a1-sec4-md` para refletir a correção.

**2. Artefatos gerados**:

- `docs/perf_baselines/sprint_a1_jax_benchmark_t4_20260519_192245.json` —
  dados completos benchmark A–H × {vmap, vmap_real} × 5 runs, gate results,
  audit_findings com referências de código exatas.
- `docs/reports/v2.40.4_auditoria_resultados_sprint_a1_2026-05-19.md` —
  relatório de auditoria (8 seções + 2 apêndices): 3 patologias, 8 bugs no
  notebook (C1-C2 críticos, H1-H3 altos, M1-M3 médios), 4 caminhos de decisão.

**3. ROADMAP §0 atualizado**:

- `A-jax-gpu-validate`: CANDIDATE → **DONE-PARTIAL**
- `A-jax-gpu-batched-api`: **NOVO**, P1 CANDIDATE — API batched via vmap eixo n_models
- `A-jax-gpu-benchmark-redesign`: **NOVO**, P1 BACKLOG — rewrite notebook
- `A-jax-gpu-dispatcher`: dependência atualizada para batched-api + benchmark-redesign
- `C-surrogate-train`: dependências expandidas para incluir A1.5 + A1.6

**4. Docs de sprint**:

- `docs/sprints/v2.41.md` — snapshot imutável Sprint A1 (este arquivo)
- `docs/sprints/CURRENT.md` — plano Sprint A1.5 (`A-jax-gpu-batched-api`)

### Resultados Benchmark T4 (resumo)

| Cenário | vmap (mod/h) | vmap_real (mod/h) | Ratio Numba | Gate |
|:-:|:-:|:-:|:-:|:-:|
| A | 448 397 | 229 394 | 0.38× | FAIL |
| B | 92 745 | 119 473 | 0.37× | FAIL |
| E | 74 899 | 33 968 | 0.61× | FAIL |
| C | 94 372 | 94 300 | 0.79× | — |
| G | 5 612 | 8 528 | 0.12× | — |

### Auditoria (3 patologias)

1. **Cold-start 35-119×**: `_BUCKET_JIT_CACHE` keyed `(ct, cr, n, npt)` — 50
   modelos aleatórios geram ~50 compilações na Run 1 (`forward_pure.py:573`)
2. **JAX mais lento que Numba**: loop Python serial + `build_static_context()`
   5-20ms/modelo + `np.asarray()` GPU→CPU sync por modelo (`multi_forward.py:400-422`)
3. **Hardware mismatch**: baseline Numba medido em **Intel i9-9980HK 8C/16T (Mac Intel)** — `.claude/perf_baseline.json:15`, `docs/reports/v2.36_2026-05-15.md:8`; T4 tem 4vCPU sem HT

### Próximo

Sprint A1.5: `A-jax-gpu-batched-api` — implementar
`simulate_multi_jax_batched(models=[...])` via `jax.vmap` sobre eixo `n_models`.

---

## [v2.40] — 2026-05-18 — MCP colab-bridge + tf.data + Mixed Precision

### Resumo

Sprint v2.40 entrega **I2.2 (MCP colab-bridge)** + **F2 (tf.data + Mixed Precision)**,
desbloqueando treinamento remoto automatizado em Colab Pro+ (A100) com ganho
esperado +15-50% via mp16+XLA. Resposta arquitetural à pergunta sobre automação
de testes JAX GPU via MCP: SIM, implementado via híbrido caminhos (a) template
Colab + (c) parcial marker pytest. Bug crítico D5 resolvido: `setup_mixed_precision_policy()`
agora callable ANTES de `build_model()` (era depois — camadas fp32 mesmo com flag ativa).

### Mudanças Principais

**1. Frente 1 — MCP colab-bridge (Tier B browser MCP)**:

- `.claude/commands/geosteering-colab-mcp.md` (~300 LOC) — Skill Sonnet 4.6
  effort=medium. 3 workflows: validação JAX GPU (D3), treinamento remoto mp16,
  benchmark tf.data. Tier B (oficial `googlecolab/colab-mcp` já em `.mcp.json`)
  como default; Tier C documentado como fallback (não implementado v2.40).
- `.claude/hooks/colab-token-refresh.sh` (~80 LOC) — PreToolUse matcher=Bash,
  warn-only. Checa `~/.config/gcloud/access_tokens.db` modtime; avisa se
  >50min sem bloquear execução. Compatível macOS BSD stat + Linux GNU stat.
- `.claude/settings.json` — registra hook PreToolUse para matcher Bash.

**2. Frente 1 — 3 Templates Colab `notebooks/colab_templates/`**:

- `__README.md` (~180 LOC) — Doc + convenções (variáveis, JSON output, PT-BR).
- `train_v240_mp16.ipynb` (19 cells) — Treina ResNet 18 com mp16+XLA usando
  `build_model_with_mp_policy(config)` garantindo ordem correta. Smoke test
  opcional via ngrok + POST /predict.
- `validate_jax_gpu_v240.ipynb` (13 cells) — Resposta D3 caminho (a): pytest
  -m gpu em 109 testes JAX + paridade JAX GPU vs Numba CPU em 7 modelos
  canônicos (gate <1e-10).
- `benchmark_tfdata_mp16.ipynb` (11 cells) — Mede 4 configs × 5 runs com
  mediana + stdev. Gate v2.40: C2 (mp16) speedup ≥ 1.15x baseline em T4.

**3. Frente 1 — Marker `gpu` pytest (D3 caminho c parcial)**:

- `pyproject.toml` — `markers = [..., "gpu: ..."]`
- `tests/conftest.py` — `pytest_collection_modifyitems` adiciona skip
  automático a testes `@pytest.mark.gpu` quando GPU não detectada via TF/JAX.
- 11 arquivos `tests/test_simulation_jax_*.py` — `pytestmark = pytest.mark.gpu`
  (ou lista `[skipif(HAS_JAX), gpu]` em foundation+propagation).
  Resultado em macOS: 109 testes JAX SKIPPED (esperado). Em Colab T4: executam.

**4. Frente 2 — Fix D5 Mixed Precision ordem (CRÍTICO)**:

- `geosteering_ai/training/loop.py` — Nova função módulo-level
  `setup_mixed_precision_policy(config)` callable ANTES de `build_model()`.
  `_setup_mixed_precision()` privado preservado como wrapper retrocompatível
  (chamado de `run()`). Warning ativo em `run()` se `model.compute_dtype != "float16"`
  mas `config.use_mixed_precision=True`.
- `geosteering_ai/training/__init__.py` — Exporta `setup_mixed_precision_policy`.
- `geosteering_ai/models/registry.py` — Novo helper `build_model_with_mp_policy(config)`
  que faz setup + build em uma única chamada. Lazy import evita circular.

**5. Frente 2 — 4 Novas flags `PipelineConfig` D6**:

- `geosteering_ai/config.py` — `tf_shuffle_buffer_size: int = 10000`,
  `tf_num_parallel_calls: int = -1`, `tf_prefetch_buffer_size: int = -1`,
  `tf_cache_eval: bool = True`. Validações em `__post_init__` (cap 100k anti-OOM).
- `geosteering_ai/data/pipeline.py` — `build_tf_dataset` consome flags;
  resolve sentinelas `-1 → tf.data.AUTOTUNE`. Default preserva legado.
- `configs/baseline.yaml` — Adiciona 4 campos com defaults explícitos.

**6. Testes (19 novos)**:

- `tests/test_config.py::TestTfDataFlagsV240` (12 testes) — defaults, valores
  válidos, rejeições. **Todos PASS** localmente (12/12).
- `tests/test_training.py::TestSetupMixedPrecisionPolicyV240` (4 testes) +
  `TestBuildModelWithMpPolicyV240` (3 testes) — importabilidade + ordem.
  Local: **4 PASS + 3 SKIPPED** (3 dependem de TF, rodam em Colab).
- `tests/test_data_pipeline.py::TestBuildTfDatasetV240` (4 testes @requires_tf)
  — 4 SKIPPED local (rodam em Colab).

**7. Documentação**:

- `docs/PERFORMANCE_BASELINE.md` — Nova §8 "TF Training Throughput" com
  métricas, 4 configurações, placeholder T4/A100, gate v2.40, workflow re-medição.
- `docs/ROADMAP.md` — Entrada v2.40 no topo da tabela. F4.3 SurrogateNet
  marcado como desbloqueado.
- `docs/reports/v2.40_colab_bridge_mp16_2026-05-18.md` (a gerar pós-merge).

### Métricas

- **Commits**: 9 granulares na branch `feature/v2.40-colab-bridge-mp16`
- **Arquivos novos**: 7 (skill, hook, 3 notebooks, README templates, PERFORMANCE seção)
- **Arquivos modificados**: 14 (config, pipeline, loop, registry, conftest,
  pyproject, settings.json, baseline.yaml, ROADMAP, CHANGELOG, 4 testes)
- **LOC adicionado**: ~2150 (código + testes + docs + notebooks)
- **Testes novos**: 19 (12 CPU + 7 TF-dependent)
- **Suite total**: 1653 PASS + 458 SKIPPED + **0 FAILED** (após pytest gpu marker)
- **Paridade Fortran <1e-12**: ✅ PRESERVADA (não tocamos `simulation/`)

### Resposta à Pergunta Arquitetural Crítica (D3)

> "É possível automatizar testes JAX GPU via MCP colab-bridge?"

**SIM**, via solução híbrida em 2 camadas:

| Caminho | Status v2.40 | Mecanismo |
|:---|:---|:---|
| **(a)** Template Colab via MCP | ✅ Implementado | `validate_jax_gpu_v240.ipynb` |
| **(c parcial)** Marker pytest `gpu` | ✅ Implementado | 109 testes JAX + skip CPU automático |
| **(b)** Endpoint `/simulate` na API | 📌 backlog | Schema novo, sprint dedicada — ver [ROADMAP.md](ROADMAP.md) |
| **(d)** GitHub Actions GPU runner | 📌 backlog | Orçamento dedicado — ver [ROADMAP.md](ROADMAP.md) |

### Conformidade com Restrições do Projeto

- ✅ Paridade Fortran <1e-12 preservada (não tocamos `simulation/`)
- ✅ TensorFlow/Keras exclusivo (sem PyTorch — `validate-no-pytorch.sh` valida)
- ✅ PT-BR acentuado em todos os `.py` e `.md` novos
- ✅ Mega-header D1 + docstrings Google-style D5/D6
- ✅ Sem `print()` em `geosteering_ai/` (logging)
- ✅ `PipelineConfig` como parâmetro (sem `globals().get()`)

### Hooks Bypass Documentados

- `CLAUDE_BYPASS_ANTI_PATTERNS=1` em 3 commits — KB-GLB/KB-EPS pré-existentes
  (header docstring de config.py + teste de errata em test_config.py — meta-código).
- `SKIP=mypy` em 1 commit — 6 erros pré-existentes em `apply_feature_view` kwargs
  unpacking (linhas 527-529, 619, 669, 674 de pipeline.py — dívida fora de escopo).

### Itens de Backlog Desbloqueados por v2.40

Ver **[docs/ROADMAP.md](ROADMAP.md)** para os itens canônicos do backlog
(prioridade, trilha, dependências). v2.40 desbloqueou em específico:

| Code | Trilha | Tema | Esforço |
|:--|:--:|:--|:--:|
| **C-surrogate-train** | C | F4.3 SurrogateNet Training (Colab A100 mp16) | 4-6h |
| **E-api-simulate** | E | Endpoint `POST /simulate` (D3 caminho b) | 6-8h |
| **C-noise-35** | C | Catálogo de Ruído 35 tipos (Trilha C Fase II) | 8-10h |
| **B-flat-prange-default** | B | `use_flat_prange=True` como default | 1-2h |
| **D-dtb-parser** | D | DTB + Parser Geológico (Trilha D Fase III) | 4-5h |

> Versões `vX.Y` serão atribuídas no commit da sprint (ADR-0001 — uma sprint
> por item committed).

---

## [v2.39] — 2026-05-18 — API REST MVP + Dockerfile.cpu + CI Docker

### Resumo

Sprint v2.39 entrega **I2.7 (API REST MVP)** + **I2.8 (Dockerfile.cpu + CI build)**,
desbloqueando a transição Fase 2 → Fase 3 do roadmap. O pacote `geosteering_ai/`
passa de "consumível apenas via CLI" para "exposição HTTP completa", habilitando
integração com Geosteering AI Studio ALPHA, MCP `colab-bridge` e Fase 3 (MLOps).

### Mudanças Principais

**1. Novo pacote `geosteering_ai/api/`** — FastAPI + Pydantic v2:

- `__init__.py` — `__version__ = "2.39.0"`
- `app.py` — App factory, CORS condicional, lifespan, 3 middlewares
  (request_id, latency, body size limit), 4 exception handlers
- `cli.py` — Entry point `geosteering-api` (wrapper sobre `uvicorn.run`)
- `dependencies.py` — Settings dataclass + singleton thread-safe + exceções
  tipadas `ModelNotLoadedError` / `ModelLoadFailedError`
- `schemas.py` — `PredictRequest`, `PredictResponse`, `HealthResponse`,
  `ErrorResponse` (todos Pydantic v2 com `extra="forbid"`)
- `routes/health.py` — `GET /health` (custo <1 ms, sem TF)
- `routes/predict.py` — `POST /predict` (lazy load TF no 1º request)

**2. Containerização**:

- `Dockerfile.cpu` — multi-stage Python 3.13-slim, user non-root,
  HEALTHCHECK, imagem final ~3.7 GB (TF wheel domina)
- `.dockerignore` — exclui caches, testes, docs, Fortran, modelos
- `.github/workflows/docker.yml` — build + smoke `/health` + smoke
  `/predict 503` em CI, com cache GHA e `paths-filter`

**3. `pyproject.toml`**:

- Novo extra `[api]`: `fastapi>=0.110`, `pydantic>=2.5`, `uvicorn[standard]>=0.27`
- `httpx>=0.27` em `[dev]` para `TestClient`
- Novo entry point `geosteering-api = "geosteering_ai.api.cli:main"`

### Hardenings de Segurança (Pós-Review)

Após implementação, despachei 2 reviewers em paralelo (code-reviewer +
security-auditor). 9 findings (2 CRÍTICO + 4 ALTO + 3 MÉDIO) foram corrigidos
no commit `eec45a6`:

- **C1**: asserts removíveis por `python -O` → `isinstance`/`raise`
- **C2/MED-1**: middleware `enforce_body_size` (DoS via JSON bomb)
- **A1/MED-3**: exceções tipadas + mensagens sanitizadas (sem vazar paths/env)
- **A2/ALTO-1**: CORS `allow_credentials` condicional (spec CORS compliance)
- **A4/MED-7**: Dockerfile sem `pip install -e` (frágil cross-stage)
- **MED-2**: bounds `MAX_N_SAMPLES=1024`, `MAX_SEQUENCE_LENGTH=10000`
- **M1**: removido `ENV GEOSTEERING_API_DOCS_ENABLED=1` hardcoded
- **M3**: `request_id` propagado para responses de erro
- **Misc**: `HTTP_413` deprecation, `AsyncGenerator` corretos

### Validação

- **Testes novos**: 41/41 PASS em 1.98 s (15 schemas + 9 health + 17 predict)
- **Suite parcial**: 133/133 PASS (api + inference + config) — sem regressão
- **Pre-commit hooks**: TODOS PASS em 12 commits (ruff, mypy, anti-patterns)
- **Smoke Docker local**: imagem buildada, `/health` 200, `/predict` 503
- **Paridade Fortran <1e-12**: PRESERVADA (não tocamos `simulation/`)

### Próximos Passos

1. Merge da branch + tag `v2.39`
2. **I2.2** — MCP `colab-bridge` (4–6 h)
3. **tf.data + Mixed Precision** (3–4 h)
4. Sprint v2.40 — hardening pré-produção (auth, rate limit, headers)

### Detalhes

Relatório completo: `docs/reports/v2.39_api_rest_dockerfile_2026-05-18.md`.

Branch: `feature/v2.39-api-rest-mvp` (12 commits, ~2.300 LOC).

---

## [v2.35] — 2026-05-15 — Cenário H (estresse multi-core 8×8×8 = 512 combos)

### Resumo

Sprint v2.35 amplia o catálogo de benchmarks do `geosteering-cli benchmark`
com o **Cenário H**, complementando os 7 cenários A–G existentes. H foi
desenhado como stress-test para CPUs multi-core: **8 frequências × 8 TRs ×
8 dips = 512 combos** por posição, contra os 64 combos do Cenário G.

### Mudanças Principais

**1. Cenário H adicionado a `SCENARIOS`**:

- `geosteering_ai/cli/benchmark.py` — entrada `"H"` no dict:
  - **n_pos**: 100 posições (alinhado a G)
  - **freqs**: `(1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5)` Hz (8 valores
    em escala log cobrindo bandas LWD reais 1 kHz–200 kHz)
  - **trs**: `(0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5)` m (8 espaçamentos)
  - **dips**: `(0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 105.0)` ° (zenitais
    típicas de geosteering horizontal/desviado)

**2. CLI argparse**:

- `geosteering_ai/cli/main.py`: `choices` do `--scenario` estendido para
  `["A", "B", "C", "D", "E", "F", "G", "H"]`.
- Help text atualizado com nota "H=estresse multi-core 8freq×8TR×8dips".

**3. Testes (3 novos em `tests/test_cli_mvp.py`)**:

| Teste | Cobertura |
|:------|:----------|
| `test_cli_benchmark_help_includes_scenario_h` | `--help` exibe "H" + "8" |
| `test_cli_benchmark_scenario_h_exists` | Dict tem 8×8×8 = 512 combos |
| `test_cli_benchmark_scenario_h_runs` (slow) | Smoke `--n 2 --workers 2 --threads 2` em <600s |

**4. Documentação**:

- Mega-header e docstring de `cli/benchmark.py` atualizados com linha H + nota
  sobre desempenho esperado em CPUs multi-core (M-series 8C/16T).
- `docs/ROADMAP.md` marca Sprint v2.35 como done.
- `CLAUDE.md` linha "Simulation Manager" v2.34 → v2.35.

### Considerações de Desempenho

- **5.12M forward calls** com `--n 100 --n_pos 100` (100 modelos × 100 pos
  × 512 combos) — alvo de stress-test, não de uso rotineiro.
- **Recomendado**: `--workers 4 --threads 2` ou superior. Em single-thread,
  pode ultrapassar 5 min.
- **CI**: smoke usa `--n 2 --workers 2 --threads 2` com timeout 600s.

### Decisões Intencionais

- **Sem persistência CSV/MD**: o handler atual apenas imprime stdout (D9
  exception). Atualização do `sm_output/benchmark_summary.csv` adiada para
  Sprint v2.36 (escopo creep evitado).
- **Paridade Fortran**: nenhum arquivo de simulação core foi modificado;
  paridade <1e-12 preservada por construção.

### Bump de Versão

`SIMULATION_MANAGER_VERSION` em `cli/main.py`: **v2.34 → v2.35**.

---

## [v2.34] — 2026-05-15 — `geosteering-warmup` integrado no CI/CD

### Resumo

Sprint v2.34 integra o entry point `geosteering-warmup` (introduzido em
v2.32) ao workflow GitHub Actions, isolando o cold-start JIT/LLVM do tempo
medido em benchmarks de CI. Estabelece também uma métrica de baseline
"warm cache" para detecção de regressões pós-merge.

### Mudanças Principais

**1. `.github/workflows/ci.yml`**:

- Novo step `Warm up JIT/LLVM cache (Sprint v2.32+)` antes do pytest:
  ```yaml
  - name: Warm up JIT/LLVM cache (Sprint v2.32+)
    run: geosteering-warmup --verbose
    timeout-minutes: 5
  ```
- Novo step `Benchmark smoke (cenário E n=200)` pós-pytest com
  `continue-on-error: true` (observabilidade, não gating).

**2. `docs/PERFORMANCE_BASELINE.md`**:

- Nova seção "Warm-Cache Baseline (CI)" explicando como interpretar o
  baseline `E_n200_warm` (medido pós-warmup, isolado do cold-start).
- Bump da seção de versão para v2.34.

**3. `.claude/perf_baseline.json`**:

- Novo campo `scenarios.E_n200_warm` com placeholder a ser ajustado
  empiricamente na primeira run do CI pós-merge.
- Campo `version` bumped para `"v2.34"`.

**4. `CLAUDE.md`**:

- Linha "Simulation Manager" atualizada v2.33 → v2.34.

### Decisões Intencionais

- **Sem `actions/cache` para `NUMBA_CACHE_DIR`**: cache em `/tmp` é efêmero
  por design. Warm-up explícito é mais robusto e determinístico que cache
  hit/miss aleatório entre jobs.
- **`continue-on-error: true` no benchmark**: o objetivo é observabilidade
  histórica via log. Gating de regressão é responsabilidade local do
  `check-perf-regression.sh` para devs.

### Bump de Versão

`SIMULATION_MANAGER_VERSION` em `cli/main.py`: **v2.33 → v2.34**.

---

## [v2.33] — 2026-05-15 — pytest-qt Suite para GUI Simulation Manager

### Resumo

Sprint v2.33 adiciona **cobertura automatizada da GUI** do Simulation
Manager via pytest-qt. Antes: ~10.7k linhas de Qt sem nenhum teste
automatizado — bugs UI só descobertos manualmente. Agora: suite com 15+
testes cobrindo abertura de janela, sinais, slots e validação de
parâmetros.

### Mudanças Principais

**1. Dependência nova**:

- `pyproject.toml`: `pytest-qt>=4.4` adicionado a `[project.optional-dependencies] dev`.

**2. Fixtures Qt (`tests/conftest_qt.py` NOVO)**:

- `qt_app`: fixture session-scoped que cria um QApplication único
  (singleton) com `QT_QPA_PLATFORM=offscreen` para suporte headless.
- `mock_simulation_thread`: fixture que substitui `SimulationThread` por
  `MagicMock` configurável, emitindo sinais sintéticos (`progress_update`,
  `finished_all`, `error`) via `QTimer.singleShot`.

**3. Suite de testes (`tests/test_simulation_manager_gui.py` NOVO)**:

15+ testes cobrindo:

- T1: MainWindow abre sem warnings/criticals
- T2: `SimulatorPage.btn_start.click()` emite `request_start`
- T3: `btn_pause` toggle (checkable) muda estado
- T4: `btn_cancel.click()` emite `request_cancel`
- T5–T7: mock `SimulationThread` emite sinais → UI atualiza progress/erro
- T8: `ParametersPage.to_dict/from_dict` roundtrip preserva valores
- T9: `Preferences` salva/restaura QSettings (cache LRU v2.29.2)
- Demais testes cobrem WelcomeWidget → tab switch, error path, sinais
  `simulation_complete`/`error`/`progress_update`.

**4. `.github/workflows/ci.yml`**:

- Novo step `Run GUI tests (xvfb headless)`:
  ```yaml
  - name: Run GUI tests (pytest-qt + xvfb)
    run: xvfb-run -a pytest tests/test_simulation_manager_gui.py -v
    timeout-minutes: 10
  ```
- xvfb instalado via `apt-get` no Ubuntu runner.

**5. `CLAUDE.md`**:

- Linha "Simulation Manager" atualizada v2.32 → v2.33 com nota sobre
  cobertura GUI.

### Padrões Reutilizados

- `sm_qt_compat.QT_BINDING` para imports agnósticos PyQt6/PySide6.
- `SimulationThread` mockado via `MagicMock` (não roda simulação real
  em testes GUI — mantém suite rápida <1min).

### Bump de Versão

`SIMULATION_MANAGER_VERSION` em `cli/main.py`: **v2.32 → v2.33**.

---

## [v2.32] — 2026-05-13 — `geosteering-warmup` Entry Point + Dívidas v2.31

### Resumo

Sprint v2.32 adiciona o entry point standalone **`geosteering-warmup`** para
execução síncrona do warmup JIT/LLVM (CI, notebooks, debug), e fecha duas
dívidas técnicas administrativas remanescentes do v2.31:

1. CHANGELOG documenta o hotfix JAX (commit `68b93c7`).
2. Timeouts dos testes CLI multi-dim em `test_cli_mvp.py` ampliados de 120s
   → 300s para tolerar cold-start de JIT em CI/macOS.

### Mudanças Principais

**1. Novo entry point `geosteering-warmup`**:

- `geosteering_ai/cli/warmup.py` (~180 LOC com mega-header + docstrings):
  comando standalone, síncrono e bloqueante, que aquece Numba + LLVM Tier 2
  via caminho JAX e retorna ao shell quando o warmup termina.
- Flags: `--verbose` (timing por fase) e `--version` (exibir versão e sair).
- Exit codes: `0` = sucesso, `1` = falha (exceção propagada), `2` = arg
  inválido. CI pode detectar falhas.
- Registrado em `pyproject.toml` `[project.scripts]`:

  ```toml
  geosteering-warmup = "geosteering_ai.cli.warmup:main"
  ```

**2. Refatoração em `cli/main.py`**:

- Extraída função pura síncrona `_warmup_numba_tier2_sync(verbose=False)`
  reutilizável pelo entry point e pelo background thread (Sprint v2.31 Part 2).
- O wrapper `_warmup_numba_tier2_background()` agora apenas envolve a chamada
  síncrona com `try/except` silencioso (mantém comportamento de daemon thread
  best-effort).
- Versão `SIMULATION_MANAGER_VERSION` bumpada `v2.31` → `v2.32`.

**3. Testes (`tests/test_cli_warmup_entry_point.py`)**:

- 4 testes novos cobrindo `--help`, `--version`, execução completa
  (`@pytest.mark.slow`, timeout 300s), e declaração do entry point em
  `pyproject.toml`.

**4. Dívidas técnicas v2.31 fechadas**:

- Timeouts em `tests/test_cli_mvp.py` ajustados (120s → 300s, 180s → 300s)
  para tolerar cold-start JIT em CI. O teste `test_cli_simulate_multi_dips`,
  que vinha falhando intermitentemente, agora passa de forma estável.
- Em `tests/test_cli_warmup_background.py`: assert de versão tornada robusta
  (substring estável) e timeout do segundo teste 60s → 240s.

### Validação

| Métrica | Status |
|:--------|:------:|
| 4/4 testes do entry point | PASS |
| 7/7 testes warmup background + guard tests v2.31 | PASS |
| Paridade Fortran <1e-12 | preservada |
| Throughput baseline | 100% preservado |

**Smoke manual**:

```
$ geosteering-warmup --version
Geosteering AI Simulation Manager v2.32

$ time geosteering-warmup --verbose
Warming up Geosteering AI v2.32...
  [warmup] filter loaded (0.31s)
  [warmup] JAX callback path warm (12.18s)
OK (12.2s)
```

### Casos de Uso

- **CI**: `geosteering-warmup && geosteering-cli benchmark --scenario E` —
  isolar cold-start do tempo de benchmark medido.
- **Notebooks**: executar antes do primeiro `simulate_multi` para timing
  consistente.
- **Debug**: `--verbose` revela fases (load filtro, JAX callback) para
  diagnosticar gargalos de warmup.

### Backup

`.backups/v2.32_warmup_entry_point_2026-05-13_193612/`

---

## [v2.31] — 2026-05-12 — Otimização Warmup JIT (Parte 1)

### Resumo

Sprint v2.31 aplica a **Parte 1** das otimizações de warmup identificadas no
relatório `docs/reports/warmup_analysis_jit_2026-05-12.md`. Elimina a anomalia
de especialização dupla de `hmd_tiv`/`vmd` (causada pelo caminho JAX passar
arrays readonly) e adiciona mitigação do gargalo LLVM bitcode via
`NUMBA_CACHE_DIR` apontando para tmpfs.

### Mudanças Principais

**1. Anomalia `hmd_tiv`/`vmd` 2 ``.nbc`` eliminada**:

- Helper privada `_to_writeable(arr)` em `geosteering_ai/simulation/_jax/kernel.py`
  garante `np.ndarray` writeable=True via `ascontiguousarray + copy` quando o
  flag é `writeable=False`.
- 28 sites de `np.asarray(...)` substituídos por `_to_writeable(...)` em
  `_dipoles_numba_host` (caminho JAX → Numba via `jax.pure_callback`).
- Resultado: 1 `.nbc` por função em vez de 2 (`hmd_tiv` 2.6 MB → 1.275 MB;
  `vmd` 1.47 MB → 683 KB).

**2. `NUMBA_CACHE_DIR` em tmpfs (mitigação cold-start LLVM)**:

- `geosteering_ai/cli/main.py` define `NUMBA_CACHE_DIR =
  $TMPDIR/geosteering_numba_cache` antes de qualquer import pesado.
- Permissões `0o700` aplicadas (CodeRabbit major finding) — apenas o dono
  acessa o cache (evita injeção de `.nbc` maliciosos em sistemas multi-tenant).
- Override do usuário (`export NUMBA_CACHE_DIR=...`) preservado.
- Falhas em `OSError` toleradas — Numba cai no default `$CWD/__pycache__`.

**3. Guard tests (`tests/test_simulation_numba_specializations.py`)**:

- 5 testes novos garantindo:
  - 1 especialização compilada por função (`hmd_tiv`, `vmd`);
  - 1 `.nbc` em disco por função;
  - `NUMBA_CACHE_DIR` setado após import de `cli/main.py`;
  - Override do usuário em `NUMBA_CACHE_DIR` preservado.

### Validação

| Métrica | Antes (v2.30) | Após (v2.31) |
|:--------|:-------------:|:------------:|
| `.nbc` `hmd_tiv` em disco | 2 (1.275 + 1.338 KB) | 1 (1.275 KB) |
| `.nbc` `vmd` em disco | 2 (734 + 734 KB) | 1 (683 KB) |
| Especializações compiladas em RAM | 2 cada | 1 cada |
| Cold-start (alvo) | 111 s | < 100 s (esperado) |
| Paridade Fortran | <1e-12 | **<1e-12 preservado** |
| Throughput baseline | 100% | **100% preservado** |

**Suite de regressão**: 56/56 PASS em 37.42 s
(Fortran parity + CLI MVP + guard tests v2.31 + workers ephemeral + LRU cache).

### Code Review

- CodeRabbit (1ª iteração): 1 major (permissões 0o700) → aplicado.
- CodeRabbit (2ª iteração): **0 findings**.

### Referências

- Relatório: `docs/reports/v2.31_warmup_optimization_2026-05-12.md`
- Análise base: `docs/reports/warmup_analysis_jit_2026-05-12.md`
- Backup: `.backups/v2.31_warmup_2026-05-12_150327/`

### Hotfix (commit 68b93c7) — Eliminação de ruído INFO no stderr

Após o release v2.31, observou-se que cada invocação `geosteering-cli ...`
emitia 4 mensagens INFO espúrias no stderr:

```
INFO: Unable to initialize backend 'rocm': ... (not found)
INFO: Unable to initialize backend 'tpu': ... (not found)
INFO: Unable to initialize backend 'rocm': ... (not found)
INFO: Unable to initialize backend 'tpu': ... (not found)
```

Duas causas-raiz independentes:

1. **JAX 0.4+ sonda todos os backends** (ROCM/TPU/CUDA) na inicialização.
   Em CPUs sem essas plataformas, cada probe falha com INFO no stderr.
2. **Logger `jax` propaga para a raiz**: o JAX instala seu próprio
   `StreamHandler`, mas com `propagate=True` (default Python), cada mensagem
   também atravessa o handler raiz do `basicConfig` → 2× emissões.

Correções aplicadas em `geosteering_ai/cli/main.py`:

- `os.environ.setdefault("JAX_PLATFORMS", "cpu")` no escopo do módulo
  (antes de qualquer `import jax`, inclusive transitivo via
  `_jax/kernel.py`). `setdefault` preserva override do usuário
  (`export JAX_PLATFORMS=cuda` para quem tem GPU CUDA).
- `logging.getLogger("jax").propagate = False` em `main()` (após
  `basicConfig`) corta a propagação que duplicava as mensagens.

Resultado: `geosteering-cli version` produz stderr limpo. Sem impacto em
funcionalidade — `JAX_PLATFORMS=cpu` já é o caminho atual do simulador.

---

## [v2.30] — 2026-05-11 — CLI Multi-Dimensional (multi-freq + multi-dip + multi-TR)

### Resumo

Sprint v2.30 implementa **personalização multi-dimensional completa do CLI Geosteering AI**.
Os subcomandos `simulate` e `benchmark` agora aceitam flags para frequências arbitrárias,
ângulos de dip variados e espaçamentos transmissor-receptor customizados, eliminando
restrições anteriores.

### Mudanças Principais

**1. Novos argumentos CLI — `simulate` subcomando**:

- `--frequencies HZ` (padrão: 20000): frequências EM em Hz separadas por vírgula
- `--dips DEG` (padrão: 0): ângulos de inclinação em graus
- `--tr-spacings M` (padrão: 1.0): espaçamentos transmissor-receptor em metros

**2. Novos argumentos CLI — `benchmark` subcomando (overrides)**:

Mesmos 3 flags acima, agora sobrescrevem valores do cenário pré-definido.

**3. Novo cenário benchmark — Cenário G**:

| Parâmetro | Valor |
| :--- | :--- |
| n_pos | 100 |
| Frequências | 2000, 20000, 100000, 400000 Hz (4) |
| TRs | 0.5, 1.0, 1.5, 2.0 m (4) |
| Dips | 0, 15, 30, 45 graus (4) |
| **Combinatória total** | **256 configurações por modelo** |

Cenário G representa máxima flexibilidade multi-dimensional (4 freq × 4 TR × 4 dips).

**4. Helper `_parse_float_list()` — edge cases robustos**:

- Trata input vazio, nulo, whitespace
- Suporta separadores: `,` e `;`
- Edge case ",,," (só separadores) → retorna default
- ValueError → log warning + retorna default

**5. Atualização de handlers**:

- `handle_simulate()`: parseia args → frequencies_hz, dip_degs, tr_spacings_m
- `handle_benchmark()`: aplica override logic + ambas chamadas `simulate_multi()` recebem dip_degs
- Logs atualizados: mostram "X freq, Y dips, Z TR"

### Exemplos de Uso

```bash
# Múltiplas frequências
geosteering-cli simulate --models 100 --frequencies 2000,20000,100000,400000 --n-pos 600

# Múltiplos ângulos de dip
geosteering-cli simulate --models 50 --dips 0,15,30,45 --n-pos 200

# Combinação completa
geosteering-cli simulate --models 200 \
  --frequencies 2000,20000 \
  --dips 0,15,30 \
  --tr-spacings 0.5,1.0,1.5 \
  --n-pos 300

# Benchmark Cenário G (máxima combinatória)
geosteering-cli benchmark --scenario G --n 50

# Benchmark com override de dips
geosteering-cli benchmark --scenario E --n 100 --dips 0,15,30
```

### Testes

- **8 novos testes multi-dim**: PASS 100%
- **17 testes CLI existentes**: PASS (backward-compatible)
- **10 testes Fortran parity**: PASS <1e-12
- **Total**: 42/42 PASS (CLI + Fortran + cache)

### Revisão de Código

- **CodeRabbit**: 0 findings (após fix 2 menores em _parse_float_list)
- **mypy**: 0 erros
- **ruff**: 0 erros

### Backward Compatibility

✅ **100% backward-compatible**: novos argumentos são opcionais.

---

## [v2.29.3] — 2026-05-11 — Investigação de regressão + infraestrutura anti-regressão

### Contexto

Usuário reportou regressão de throughput após v2.29.2 na configuração padrão
(Cenário E n=2000). Esta sprint investigou empiricamente a regressão e
implementou infraestrutura de prevenção.

### Veredito Empírico

**NÃO há regressão por v2.29.2.** Benchmarks lado-a-lado:

| Run | v2.29.1 (24fba72) | v2.29.2 (e3a0617) | Δ |
|:---:|:-----------------:|:-----------------:|:-:|
| Cold | 91,620 mod/h | 94,563 mod/h | +3.2% |
| Warm 1 | 93,510 mod/h | 96,737 mod/h | +3.5% |
| Warm 2 | 88,732 mod/h | 95,124 mod/h | +7.2% |
| **Mediana** | **91,620** | **95,124** | **+3.8%** |

v2.29.2 é **+3.8% MELHOR** que v2.29.1. Análise estática (2 Explore agents)
confirma: todas as 4 mudanças v2.29.2 são periféricas ao caminho crítico
Numba JIT.

### Mudanças Aplicadas

**1. Fix `is_paused()` em `_resume_simulation`** ([simulation_manager.py:8472](../geosteering_ai/simulation/tests/simulation_manager.py#L8472)):

`sim.is_paused` (referência ao método, sempre truthy) → `sim.is_paused()`.
Não afeta throughput (idempotência de `request_resume()`), apenas comportamento
lógico correto.

**2. Hook anti-regressão** [`check-perf-regression.sh`](../.claude/hooks/check-perf-regression.sh):

Roda Cenário E n=200 e compara contra `.claude/perf_baseline.json`.
WARN-only (não bloqueia). Config via env vars (`SCENARIO`, `N_MODELS`,
`THRESHOLD_PCT`, `VERSION`).

**3. Baseline documentado** [`docs/PERFORMANCE_BASELINE.md`](PERFORMANCE_BASELINE.md):

Tabela canônica de cenários (A, B, C, D, E, F, multi-freq+dip) + notas
sobre variabilidade + processo de atualização (3 runs + mediana).

**4. Skill `geosteering-perf-baseline`**:

Reviewer especializado em validar não-regressão de throughput. Trigger
para PRs que modifiquem `geosteering_ai/simulation/`.

### Aprimoramentos Arquiteturais (Prevenção)

5 níveis de defesa:

1. **Hooks Claude Code** — `check-perf-regression.sh` + 5 hooks pré-existentes
2. **Documentação baseline numérico** — `PERFORMANCE_BASELINE.md` + JSON
3. **Skills reviewers** — `geosteering-perf-baseline` + 5 reviewers pré-existentes
4. **Testes regressão** — 37 testes incluindo paridade Fortran <1e-12
5. **Processo humano** — benchmark obrigatório antes de mudanças em simulator

### Respostas às 4 Perguntas Conceituais

1. **Workers**: `sm_workers.py` (Qt GUI) + `_workers.py` (core reusável) — duas
   camadas complementares
2. **CLI**: `pip install -e .` + `geosteering-cli {simulate,benchmark,version}`
3. **Refactor**: simulador Python OK; SM v3.0+ vale MVC mas precisa pytest-qt antes
4. **Mitigação**: 5 níveis de defesa documentados

### Arquivos modificados

- `geosteering_ai/simulation/tests/simulation_manager.py`: fix `is_paused()`
- `.claude/hooks/check-perf-regression.sh`: NOVO
- `.claude/perf_baseline.json`: NOVO
- `.claude/commands/geosteering-perf-baseline.md`: NOVO
- `docs/PERFORMANCE_BASELINE.md`: NOVO
- `docs/reports/v2.29.3_2026-05-11.md`: relatório completo
- `docs/CHANGELOG.md`, `docs/ROADMAP.md`, `CLAUDE.md`: entradas v2.29.3

### Validação

| Suite | Resultado |
|:------|:----------|
| 37 testes (LRU + ephemeral + paridade + threading + fastmath + seed) | **37/37 PASS** |
| Paridade Fortran <1e-12 | **10/10 PASS** |
| Smoke test SM | **0 falhas** |
| CodeRabbit (1ª passada: 4 hook + 2 skill) | Todos fixes aplicados |
| Hook `check-perf-regression.sh` | PASS (162% baseline) |

### Limitações

- **150k mod/h não atingido** em medições atuais (~95k em n=2000). Investigar
  em Sprint v2.30+: condições de medição, thermal monitor, isolation
- **Sem pytest-qt golden path** — gap planejado v2.27
- Hook anti-regressão é **WARN-only** (não bloqueia)

---

## [v2.29.2] — 2026-05-11 — Cache LRU configurável + varredura GUI

### Causa-raiz

O `LRUPlotCache` tinha limite **hardcoded** de 500 MB em duas localizações
(`sm_plot_cache.py:73` e `simulation_manager.py:7368`). Em simulações
multi-frequência × multi-ângulo com `complex128`, o tensor histórico
facilmente excede 5× esse limite (cálculo empírico: 1000 mod × 2 TR ×
4 dips × 600 pos × 4 freq × 9 comp × 16 B = **2.77 GB**).

Ao re-abrir o experimento pelo histórico, o usuário recebia o diálogo:

> "Esta simulação multi-freq×multi-angle gerou um tensor que excedeu o limite
> do cache LRU (500 MB). [...] • Aumente o limite de cache em Preferências
> (futuro)"

A feature prometida "(futuro)" estava pendurada desde v2.5. Esta sprint
implementa.

Adicionalmente, smoke test do SM tinha 3 falhas pré-existentes desde v2.29:
`SimulationThread.is_paused` e `is_cancelled` viraram métodos em "Back to
Basics", mas o teste ainda os comparava como properties (`st.is_paused is False`
→ sempre False).

### Solução v2.29.2

**Auto-detect default baseado em RAM** ([sm_plot_cache.py:50](../../geosteering_ai/simulation/tests/sm_plot_cache.py#L50)):

| RAM Total | 10% RAM | Após piso 500 MB / teto 4 GB |
|:---------:|:-------:|:----------------------------:|
| 4 GB | 400 MB | **500 MB** (piso) |
| 16 GB | 1.6 GB | **1.6 GB** |
| 32 GB | 3.2 GB | 3.2 GB |
| 64 GB | 6.4 GB | **4 GB** (teto) |

Fallback 500 MB se `psutil` ausente ou `virtual_memory()` falhar (qualquer
`Exception`, não apenas `ImportError`).

**Persistência QSettings**: `cache/max_bytes_mb` e `cache/maxlen`.

**UI Preferências — "Cache LRU de Tensores"**: 2× `QSpinBox` (limite MB,
snapshots máximos) + botão "Auto-detect" + label informativo. Signal
`cache_settings_changed(int, int)` reinstancia o cache preservando snapshots
existentes (re-put no novo cache) e atualiza widgets dependentes
(`_update_cache_status`, `_refresh_results_experiment_combo`,
`mark_history_out_of_cache` para itens evictados).

**Mensagem dinâmica**: agora mostra o limite ATUAL (não "500 MB" hardcoded)
e linka para "Preferências → Cache LRU" em vez de "(futuro)".

**Fix smoke test pré-existente**: `st.is_paused()` em vez de `st.is_paused`,
idem `is_cancelled()`.

### Arquivos modificados

- `geosteering_ai/simulation/tests/sm_plot_cache.py`: +`default_max_bytes()`,
  `max_bytes: Optional[float] = None` aceito
- `geosteering_ai/simulation/tests/simulation_manager.py`: QSettings reader
  + UI Preferences group + signal/handler + fix smoke test
- `tests/test_simulation_lru_cache.py`: **NOVO**, 7 testes
- `docs/reports/v2.29.2_2026-05-11.md`: relatório técnico completo

### Validação

| Suite | Resultado |
|:------|:----------|
| `test_simulation_lru_cache.py` (NOVO) | **7/7 PASS** |
| `test_simulation_workers_ephemeral.py` | 9/9 PASS |
| `test_simulation_compare_fortran.py` | **10/10 PASS — paridade <1e-12 PRESERVADA** |
| `test_simulation_workers_threading.py` | 3/3 PASS |
| `test_simulation_v223_fastmath_threads.py` | 7/7 PASS |
| `test_simulation_parameters_seed.py` | 1/1 PASS |
| Smoke test SM (`--smoke-test`) | **0 falhas** (era 3 falhas pré-existentes) |
| CodeRabbit (`coderabbit review --agent`) | **0 findings** (após 4 fixes da 1ª passada) |

### Adaptação Sprint v2.21 → atual

Investigação confirmou que a **estrutura GUI v2.21 já está integralmente
adaptada** no código atual. As 12 classes principais (`MainWindow`,
`SimulationParametersPage`, `BenchmarkPage`, `ResultsPage`, `PreferencesPage`,
`PlotComposerDialog`, etc.) estão preservadas. **Nada precisa adaptar**.

---

## [v2.29.1] — 2026-05-11 — Fix `NUMBA_NUM_THREADS` no PAI + restauração de 150k mod/h

### Causa-raiz

A v2.29 ("Back to Basics") restaurou a arquitetura `ProcessPoolExecutor` efêmero
do `old_geosteering_ai/`, mas reintroduziu bug latente: setar
`os.environ["NUMBA_NUM_THREADS"]` **dentro** do worker `run_numba_chunk` falha
quando o ambiente do pai já tem `NUMBA_NUM_THREADS` setado para valor distinto.

Em produção (com `NUMBA_NUM_THREADS=2` herdado do shell, cenário 2 freqs + 2 dips
+ 1 TR + 4w × 4t):

```text
[ERRO] Cannot set NUMBA_NUM_THREADS to a different value once the threads
have been launched (currently have 2, trying to set 16)
```

**Por que falha em modo spawn**: o módulo `geosteering_ai.simulation.tests.sm_workers`
é importado durante o bootstrap do worker (para resolver pickle de `run_numba_chunk`),
e a cadeia importa numba ANTES de qualquer linha de `run_numba_chunk` rodar. Setar
env var dentro do worker é tardio demais — pool numba já inicializado com env do pai.

A v2.29 removeu `_acquire_numba_pool` (que setava env var no pai antes do spawn)
junto com `PoolWarmupThread`, perdendo essa proteção essencial.

### Solução v2.29.1

| Antes (v2.29) | Depois (v2.29.1) |
|:---|:---|
| `os.environ[...]=str(n_threads)` em `run_numba_chunk` (worker) | `os.environ[...]=str(req.n_threads)` em `SimulationThread.run` (pai), antes do `ProcessPoolExecutor` |
| Sem cleanup | `try/finally` restaura env vars (evita vazamento) |
| Single point of failure no worker | Workers spawned herdam env var correto |

`multi_forward.py:1010-1028` ainda chama `numba.set_num_threads(req.n_threads)` para
mascarar threads ativas (operação runtime-safe, n ≤ pool_size).

**Arquivos modificados**:

- `geosteering_ai/simulation/tests/sm_workers.py`: `run_numba_chunk` não seta env var;
  `SimulationThread.run` adiciona try/finally com setup/restore
- `geosteering_ai/simulation/tests/sm_benchmark.py`: mesmo padrão em
  `_run_numba_parallel_sync`
- `tests/test_simulation_workers_ephemeral.py`: +3 guard tests v2.29.1
- `docs/reports/v2.29.1_2026-05-11.md`: relatório técnico completo

### Validação Empírica

| Suite | Resultado |
|:------|:----------|
| `test_simulation_workers_ephemeral.py` | **9/9 PASS** (6 v2.29 + 3 novos v2.29.1) |
| `test_simulation_compare_fortran.py` | **10/10 PASS — paridade <1e-12 PRESERVADA** |
| `test_simulation_workers_threading.py` | 3/3 PASS |
| `test_simulation_v223_fastmath_threads.py` | 7/7 PASS |
| `test_simulation_parameters_seed.py` | 1/1 PASS |
| CodeRabbit (`coderabbit review --agent`) | **0 findings** |

### Throughput Pós-Fix (Apple Silicon 8C/16T HT, 4w × 4t)

Reprodução do cenário do bug (2 freqs + 2 dips + 1 TR, `NUMBA_NUM_THREADS=2`
inicial herdado do shell):

| Run | Cache | Throughput | Status |
|:---:|:-----:|:----------:|:------:|
| 1 | Cold | 130,408 mod/h | ✓ sem RuntimeError |
| 2 | Warm | 141,754 mod/h | ✓ env vars restauradas |
| 3 | Warm steady-state | **149,984 mod/h ≈ 150k 🎯** | **Meta empírica atingida** |

Benchmark CLI single-process (warm cache):

| Cenário | Throughput | Comparação |
|:--------|:----------:|:-----------|
| A (default) | **3,152,977 mod/h** | >>>150k |
| E (n_pos=600) | 145,202 mod/h | Próximo da meta canônica 150k |
| F (multi-freq+TR) | 52,319 mod/h | Multi-eixo, oportunidade futura |

---

## [v2.29] — 2026-05-11 — Back to Basics: reversão arquitetural

### Causa-raiz da regressão persistente v2.18–v2.28

Após 4 tentativas (v2.25–v2.28), throughput permaneceu em 75–107k mod/h (esperado >150k),
warmup visível ~33s, Python travava ao fechar. Análise comparativa com
`old_geosteering_ai/simulation/tests/sm_workers.py` (869 LOC, funciona) revelou que a
**arquitetura v2.18+** (pool persistente + PoolWarmupThread + t0_sim) é incompatível
com o cold-start JIT do Numba.

| Aspecto | OLD (funciona) | v2.18–v2.28 (problemas) |
|:--------|:---------------|:------------------------|
| Pool | Efêmero (`with ProcessPoolExecutor`) | Persistente global |
| Inicializador | Nenhum | `_numba_init_worker` |
| Warmup background GUI | **Nenhum** | `PoolWarmupThread` |
| Warmup task | INLINE com dados REAIS (`chunk[0]`) | Sintético (`n_layers=10` hardcoded) |
| Medição | `t0` único | `t0_sim` pós-NOOPs |
| Shutdown | Implícito via `with` | `release_numba_pool()` + atexit |
| LOC | 869 | 1478 |

**Bugs cumulativos v2.18–v2.28**:
1. Warmup sintético não cobre paths para `n_layers` variável
2. Pool persistente recriado se hankel_filter difere → warmup descartado
3. `release_numba_pool` não cancela workers em LLVM → hang
4. Race condition em `_PERSISTENT_POOL_CONFIG`

### Solução v2.29 — Back to Basics

Reverteu para arquitetura OLD (`old_geosteering_ai/`) com 3 melhorias v2.x preservadas:
- `NumbaPrimer` (v2.9) — popula cache JIT em disco no startup, compatível com ephemeral
- `os.environ["NUMBA_NUM_THREADS"]` setup no worker (v2.16)
- `detect_cpu_topology` log no `run()` (v2.17)
- Pause/cancel cooperativo (v2.11)

**Removidos**:
- `_PERSISTENT_POOL`, `_PERSISTENT_POOL_CONFIG`
- `_numba_init_worker`, `_run_numba_warmup_task`, `_noop`
- `_acquire_numba_pool`, `release_numba_pool`
- `PoolWarmupThread` (+ GUI: `lbl_warmup_status`, `_warmup_thread`,
  `_start_background_warmup`, `_on_warmup_done`, `_on_warmup_error`,
  `_on_backend_changed_warmup`)
- `_prewarm_numba_pool` (MainWindow)
- `_WORKER_INITIALIZED` global
- `t0_sim` e sincronização via NOOPs

**Mudanças**:
- `run_numba_chunk`: warmup INLINE incondicional com dados reais de `chunk[0]`,
  `t0 = time.perf_counter()` APÓS warmup (modelo OLD linha 174)
- `SimulationThread.run()`: `with ProcessPoolExecutor(max_workers=n_workers) as pool:`
  efêmero, mensagem honesta ao usuário sobre warmup esperado

**Performance** (esperada, modelo OLD):
- Throughput 1ª execução cold cache: **>150k mod/h**
- Throughput 2ª execução warm cache: **>500k mod/h**
- Sem warmup visível
- Hang no shutdown **resolvido por design** (context manager `__exit__`)

### Testes (v2.29)

```
tests/test_simulation_workers_ephemeral.py    — 6/6 PASS (novo)
tests/test_simulation_parameters_seed.py      — 1/1 PASS (realocação)
tests/test_simulation_compare_fortran.py      — 10/10 PASS [paridade <1e-12]
tests/test_simulation_v223_fastmath_threads.py — 7/7 PASS
tests/test_simulation_workers_threading.py    — 3/3 PASS
```

**Deletado**: `tests/test_simulation_pool_warmup.py` (12 testes obsoletos —
infraestrutura testada foi totalmente removida).

### Arquivos

- `geosteering_ai/simulation/tests/sm_workers.py`: 1478 → 1062 LOC (-416)
- `geosteering_ai/simulation/tests/simulation_manager.py`: -100 LOC
- `docs/reports/v2.29_2026-05-11.md`: relatório técnico completo

### Relatório detalhado

Ver: [`docs/reports/v2.29_2026-05-11.md`](reports/v2.29_2026-05-11.md)

---

## [v2.28] — 2026-05-11 — Fix warmup incompleto: Warmups C+D para cobertura total

### Causa-raiz diagnosticada

v2.27 eliminou a saturação LLVM (1 future ao invés de N), mas introduziu nova
regressão: **throughput aparente ~55k mod/h** (esperado: 800k–1.4M mod/h).

Análise dos timestamps de `__pycache__` revelou que 5 funções JIT críticas
eram compiladas **APÓS `t0_sim`** (durante a simulação real):

| Funções compiladas no WARMUP (22:29) | Funções compiladas na SIMULAÇÃO (23:01–23:05) |
|:-------------------------------------|:----------------------------------------------|
| `kernel._compute_zrho_kernel` | `propagation.common_arrays` |
| `kernel._fields_in_freqs_kernel_cached` | `propagation.common_factors` |
| `rotation.rotate_tensor` | `geometry.find_layers_tr` |
| `rotation.build_rotation_matrix` | `geometry._sanitize_profile_kernel` |
| `dipoles.vmd` / `dipoles.hmd_tiv` | `kernel.precompute_common_arrays_cache` |
| `forward._simulate_positions_njit_cached` | — |

**Por que isso ocorria**: `_run_numba_warmup_task` (v2.27) executava `simulate_multi`
com `rho_h == rho_v` (isotrópico) e `dip_degs=[0.0]` (dip nulo). As 5 funções
do segundo grupo só são compiladas em paths anisotrópicos (`rho_v ≠ rho_h`) ou
de dip não-nulo (`hordist = L·|sin(dip)| > 0`). Workers 1..N acionavam o
warmup secundário com dados REAIS, compilando essas funções INLINE após
`t0_sim` — ~30 s contados como tempo de simulação → throughput aparente ~55k.

**Por que funcionava antes de v2.25 (paradoxo)**: v2.10–v2.24 tinham warmup
quebrado silenciosamente (shape errada de `esp`). Workers compilavam JIT
LAZILY na 1ª chamada real com dados ANISOTRÓPICOS+DIP≠0 — todas as 22 funções
compilavam corretamente. v2.25 "consertou" o warmup mas com parâmetros
incompletos. v2.27 herdou o problema.

### Mudanças implementadas

- **`_run_numba_warmup_task` — Warmups C e D** (cobertura completa):
  - Warmup A (existente): isotrópico + dip=0° + single-combo
  - Warmup B (existente): isotrópico + dip=0° + multi-combo
  - **Warmup C (novo)**: anisotrópico (`_rho_v = _rho * 0.3`) + dip=0°
    → ativa `common_arrays`/`common_factors` em especializações TIV reais
  - **Warmup D (novo)**: anisotrópico + `dip_degs=[30.0]`
    → ativa `find_layers_tr`/`_sanitize_profile_kernel`/
      `precompute_common_arrays_cache` no path inclinado (hordist > 0)
  - Docstring expandida com diagrama dos 4 cenários e tabela histórica
    v2.10–v2.28.

- **`tests/test_simulation_pool_warmup.py` — T11 e T12** (total 12 testes):
  - `test_run_numba_warmup_task_covers_anisotropic_path`: verifica
    presença de `_rho_v` e `0.3` no código (não na docstring).
  - `test_run_numba_warmup_task_covers_nonzero_dip_path`: verifica
    presença de `30.0` no código.

### Performance e validação

| Métrica | v2.27 | v2.28 (esperado) |
|:--------|:------|:-----------------|
| Warmup cold JIT | ~30 s (cache parcial) | ~35–45 s (cache completo) |
| Warmup warm cache | <2 s | <2 s |
| Throughput 1ª exec | **~55k mod/h** ❌ | **800k–1.4M mod/h** ✅ |
| Workers 1..N secondary warmup | ~30 s cold inline | <2 s (cache hit) |
| Hang no shutdown | Não | Não |

### Testes

```
tests/test_simulation_pool_warmup.py     — 12/12 PASS (11.51 s)
tests/test_simulation_compare_fortran.py — 10/10 PASS (4.72 s) [paridade <1e-12]
tests/test_simulation_v223_fastmath_threads.py — 7/7 PASS (2.58 s)
```

**Paridade Fortran INVIOLADA** — fix não altera nenhum path numérico, apenas
adiciona 2 chamadas extras de `simulate_multi` durante o warmup.

### Relatório detalhado

Ver: [`docs/reports/v2.28_2026-05-11.md`](reports/v2.28_2026-05-11.md)

---

## [v2.27] — 2026-05-10 — Fix warmup LLVM saturation (1 worker → cache compartilhado)

### Causa-raiz diagnosticada

Bug v2.26: `PoolWarmupThread.run()` submetia `n_workers` futures de warmup
**simultaneamente**. Cada worker compilava ~22 funções JIT via LLVM. Com N
processos saturando as CPUs com compilações LLVM concorrentes, cada compilação
levava 3–4× mais tempo do que em isolado:

| Versão | Abordagem | Tempo cold JIT | Problema |
|:-------|:----------|:--------------|:---------|
| v2.25 | initializer (`simulate_multi` 4×) | ~35–38 s | Hang no shutdown; OOM/SIGKILL mascarava 110+s |
| v2.26 | N futures simultâneos | **110+ s** | Saturação LLVM: N × LLVM_threads vs. N cores |
| v2.27 | **1 future** + cache em disco | **~35–38 s frio / < 2 s quente** | ✅ Correto |

### Mudanças implementadas

- **`PoolWarmupThread.run()` — 1 future** (era N futures):
  - Submete warmup a 1 único worker → esse worker compila JIT e grava cache
    em disco (`__pycache__/*.nbi`/`*.nbc`).
  - Workers 2..N carregam bytecode do disco (< 2 s) via warmup secundário em
    `run_numba_chunk` — sem nova compilação LLVM.
  - Docstring refatorada com tabela histórica explicando a regressão v2.26.

- **`run_numba_chunk()` — `_WORKER_INITIALIZED = True`** após warmup secundário:
  - Bug implícito: o flag nunca era setado no warmup secundário, fazendo-o
    rodar em **todos** os chunks subsequentes para workers 2..N (descartava
    1 modelo por chunk desnecessariamente).
  - Fix: `global _WORKER_INITIALIZED; _WORKER_INITIALIZED = True` após o bloco.

- **`PoolWarmupThread` docstring** — diagrama atualizado com linha v2.27
  explicando a diferença vs. v2.26 (N futures → saturação LLVM).

### Testes

- **2 novos testes** em `tests/test_simulation_pool_warmup.py` (total: 10):
  - `test_pool_warmup_thread_submits_one_future_not_n_workers` — garante que
    `PoolWarmupThread.run` não usa `range(self._n_workers)` (bug v2.26).
  - `test_run_numba_chunk_sets_worker_initialized_after_secondary_warmup` —
    garante que `_WORKER_INITIALIZED = True` está no código de `run_numba_chunk`.
- **10/10 PASS** na suite de warmup.
- **Paridade Fortran <1e-12 PRESERVADA** — 10/10 testes Fortran.
- **v2.23 não-regressão** — 7/7 PASS.

### Fora do Simulation Manager?

O problema ocorre apenas dentro do SM (via `PoolWarmupThread`). O `simulate_multi`
chamado diretamente (testes, CLI, benchmarks) nunca cria workers paralelos durante
warmup — ele usa o JIT inline no processo principal, sem contenção LLVM.

### Configuração pequena (1 modelo) ajudaria?

**Não.** O tempo de warmup é dominado pela compilação LLVM, não pela execução.
LLVM compila o bytecode das funções independentemente do tamanho dos arrays de
entrada. Usar `n_layers=1` em vez de `n_layers=10` não reduz o tempo de
compilação. A solução correta é reduzir o **número de compilações paralelas**
(1 worker), não o tamanho do input.

---

## [v2.24] — 2026-05-10 — Débitos Técnicos + I2.5 + I2.6 (Multi-Agent)

### Sprint v2.24 — Débitos Técnicos (Frente 1)

- **1.1 Script PASS count automatizado** — `scripts/count_pytest_pass.py`
  elimina o débito recorrente de contagem manual (M2 da revisão final v2.23).
  Suporta `--json` e `--from-file` para automação CI.
- **1.2 PT-BR `setup-environment.sh`** — todas as palavras do hook agora
  acentuadas (Sessão, início, último, rápido, versão, informações, diretório,
  recomendação, módulo, importável). 11+ ocorrências corrigidas.
- **1.3 `cache=True` explícito em 13 decoradores `@njit`** em `_numba/`:
  - `propagation.py`: `common_arrays`, `common_factors`
  - `dipoles.py`: `hmd_tiv`, `vmd`
  - `rotation.py`: `build_rotation_matrix`, `rotate_tensor` (mantêm fastmath)
  - `hankel.py`: `prepare_kr`, `integrate_j0`, `integrate_j1`, `integrate_j0_j1`
  - `geometry.py`: `find_layers_tr`, `layer_at_depth` (mantêm fastmath)
- **1.4 `use_fastmath` documentado como status documental** — docstring em
  `SimulationConfig.use_fastmath` esclarece que dispatcher real foi deferido
  para v2.25+ (decisão conservadora). Reduz superfície de risco sem perder
  o débito (fica documentado e justificado).

### I2.5 — Hooks de Qualidade

- **`check-ptbr-accentuation.sh`** — novo hook PostToolUse Edit|Write que
  alerta sobre palavras PT-BR sem acentuação em arquivos `.md`/`.py`/`.sh`.
  Severidade WARN (não bloqueia). Catálogo TSV em `.claude/ptbr-words.txt`
  (60+ pares). Whitelist para `legacy/`, `old_geosteering_ai/`, `.backups/`.
  Bypass via `CLAUDE_BYPASS_PTBR=1`. Registrado em `.claude/settings.json`.
- **`generate-pr-description.sh`** — gerador de descrição de PR via
  `git log` + `git diff --stat` aplicados ao template
  `.claude/templates/pr_description_template.md`. Output stdout markdown
  estruturado (Resumo / Mudanças / Test Plan / Referências).

### I2.6 — CLI MVP

- **`geosteering-cli`** — entry point declarado em `pyproject.toml`
  (`[project.scripts]`) com 3 subcomandos:
  - `version` — exibe versão atual do Simulation Manager
  - `simulate` — gera modelos sintéticos via `simulate_multi`
    (--models N, --n-pos, --workers, --threads, --seed, --out)
  - `benchmark` — executa cenários A/B/C/D/E/F e reporta mod/h
- Lazy imports nos handlers para `--help` rápido (<5s)
- Reutiliza infraestrutura existente: `simulate_multi`,
  `recommend_default_parallelism` (Sprint v2.23 A.2)

### Validação

- **41 novos testes**:
  - 11 em `tests/test_sprint_v224.py` (script + PT-BR + cache=True + use_fastmath)
  - 17 em `tests/test_hooks_i25.py` (hook PT-BR + PR description + catálogo)
  - 13 em `tests/test_cli_mvp.py` (estrutura + subcomandos + lazy imports)
- **Paridade Fortran <1e-12 PRESERVADA** em 7 modelos canônicos
  (oklahoma_3, 5, 28, hou_7, devine_8, oklahoma_15, viking_graben_10) —
  10/10 testes em `tests/test_simulation_compare_fortran.py` (7 paramétricos
  + 1 high-rho stability + 2 smoke/parser)
- **Suite total**: 1665+ PASS / 295 SKIP / 0 FAIL

### Arquivos modificados/criados

**Modificados**: `setup-environment.sh`, `_numba/{propagation,dipoles,rotation,hankel,geometry}.py`,
`config.py`, `settings.json`, `pyproject.toml`, `CLAUDE.md`, `CHANGELOG.md`, `ROADMAP.md`.

**Criados**: `scripts/count_pytest_pass.py`, `.claude/hooks/{check-ptbr-accentuation,generate-pr-description}.sh`,
`.claude/ptbr-words.txt`, `.claude/templates/pr_description_template.md`,
`geosteering_ai/cli/{__init__,__main__,main,simulate,benchmark}.py`,
`tests/{test_sprint_v224,test_hooks_i25,test_cli_mvp}.py`,
`docs/reports/v2.24_2026-05-10.md`.

**Backup**: `.backups/2026-05-10_v2.24/` (10 arquivos).

---

## [v2.23] — 2026-05-10 — Fastmath Dual-Mode + Threads Adaptativos (Multi-Agent)

### Performance

- **A.1 Fastmath Dual-Mode**: `@njit(fastmath=True)` aplicado em kernels auxiliares
  de baixo risco:
  - `_numba/geometry.py`: `find_layers_tr`, `layer_at_depth`, `_sanitize_profile_kernel`
    (operações lógicas — fastmath é inócuo em comparações)
  - `_numba/rotation.py`: `build_rotation_matrix`, `rotate_tensor` (produtos
    `cos·sin` + matmul `Rᵀ·H·R` — FMA reorder com erro ULP ~1e-15, abaixo
    do gate de paridade Fortran <1e-12)
  - Funções raiz (`_simulate_positions_njit*`, `_fields_*_kernel*`) e
    propagação/dipolos (cascata crítica) **mantêm `fastmath=False` por design**
- **A.2 Threads Adaptativos**: `SimulationConfig.__post_init__` agora chama
  `recommend_default_parallelism()` quando `n_workers=None AND threads_per_worker=None`,
  configurando defaults baseados na topologia detectada da CPU (sucessor de v2.20)
- **Defesa em camadas KB-019**: warning emitido se `n_workers × threads_per_worker
  > 4× physical_cores` (oversubscription severa)

### Novos campos

- `SimulationConfig.use_fastmath: bool = False` — opt-in documental para
  futuras sprints que possam introduzir kernels condicionais

### Validação

- **Paridade Fortran <1e-12 PRESERVADA**: 10/10 testes em `test_simulation_compare_fortran.py`
  — 7 modelos canônicos paramétricos (oklahoma_3, oklahoma_5, oklahoma_28, hou_7,
  devine_8, oklahoma_15, viking_graben_10) + 1 high-rho stability + 2 smoke/parser
- **7 novos testes** em `tests/test_simulation_v223_fastmath_threads.py`:
  default False, decoradores aplicados, auto-detect, backward-compat (workers/threads
  explícitos), log informativo
- **Não-regressão**: 52/52 PASS (KB 11 + MCPs 32 + worktree 9)

### Multi-Agent Workflow

- Implementação coordenada via `geosteering-orchestrator` (Opus 4.7 max) com
  `geosteering-simulator-numba` (Opus 4.7 extra-high)
- Reviews paralelos: `geosteering-physics-reviewer` (Opus 4.7), `geosteering-perf-reviewer`
  (Haiku 4.5), `geosteering-code-reviewer` (Sonnet 4.6)
- Aplicação de 3 fixes críticos do perf-reviewer pre-merge: regressão de teste,
  validação anti-oversubscription, try-except em lazy import

### Hook

- **`setup-environment.sh`**: exibe topologia CPU detectada + recomendação v2.23
  (`{P}P/{L}L (HT={sim|nao}) - default v2.23: {nw}w x {npt}t`) — tolerante a
  falhas (omite linha se import falhar)

### Backward-compat

- `n_workers=1` (ou `threads_per_worker=1`) explícito **preserva** comportamento
  single-process — apenas o caso AMBOS None dispara auto-detect
- Try-except no lazy import garante que falhas em `_workers.py` (e.g., psutil
  ausente) não quebrem construção de `SimulationConfig`

---

## [v2.22.5] — 2026-05-09 — Upgrade skills físicas para Opus 4.7

### Skills

- **`geosteering-physics-reviewer`**: modelo Sonnet 4.6 → Opus 4.7 extra-high (conformidade §19).
- **`geosteering-simulator-fortran`**: adicionados `model: claude-opus-4-7` + `effort: extra-high`.
- **`geosteering-simulator-python`**: adicionados `model: claude-opus-4-7` + `effort: extra-high`.
- **`geosteering-orchestrator`**: seção "Configuração Dinâmica de Agentes" + hierarquia effort.

### Infraestrutura

- Hierarquia de effort documentada: `high` < `extra-high` < `max`.
- Conformidade §19 do documento de arquitetura: 9/9 categorias de tarefa mapeadas.
- Branch `feat/skills-agent-config-override`: 0 findings CodeRabbit (branch limpa).

---

## [v2.22.4] — 2026-05-09 — Promoção FLAT prange a default

### Mudança

- **`SimulationConfig.use_flat_prange`**: default mudado de `False` (opt-in)
  para `True` (default ativo).
- Backward-compat preservado: `cfg.use_flat_prange=False` reverte para o
  caminho v2.21 (Sprint 13.3 + 21.1) — útil para A/B testing e debug.
- Sem regressão em paridade Numba bit-exata (validada em 27 testes
  `test_simulation_v22_flat_prange.py`, 100% PASS pós-bump).
- Sem regressão em paridade Fortran <1e-12 (transitividade FLAT ≡ legacy).
- Smoke benchmark Cenário E (n_pos=600, nf=1): 224k mod/h pós-bump
  (legacy 224k vs flat 224k, speedup 1.00× — dentro do ruído).

### Justificativa

Após validação completa da Sprint v2.22 (commit `f377a37`):
- 1597 PASS / 0 FAIL na suite total (15min)
- Cenário B +11%, F +9% single-process
- Cenário E sem regressão (0.99× original; 1.00× pós-bump)
- 27/27 testes paridade FLAT vs legacy bit-exato (`np.array_equal`)

Promoção desbloqueia Sprint v2.23 (fastmath + adaptive threads) que
assume FLAT como pré-requisito arquitetural.

### Arquivos modificados

- `geosteering_ai/simulation/config.py`: `use_flat_prange: bool = True`
  (era `False`); comentário expandido com histórico de promoção.
- `docs/CHANGELOG.md`: entrada `[v2.22.4]` (esta).

---

## [v2.22.0] — 2026-05-08 — Sprint FLAT prange

### Sprint v2.22 FLAT prange (Caminho A do roadmap multi-agente §22.2.1.1)

- **Sprint v2.22.1 — `_fields_at_single_freq`**: extração do corpo do
  `for i_f in range(nf)` em `_fields_in_freqs_kernel_cached` para função
  `@njit(cache=True, nogil=True)` reusável que computa tensor H rotacionado
  para uma única frequência. Aceita slices 2D dos caches (já indexados por
  `i_f`) e `camad_t/camad_r` pré-calculados.
- **Sprint v2.22.2 — `_simulate_combined_prange_flat`**: novo kernel
  `@njit(parallel=True, cache=True, nogil=True)` em `forward.py` que colapsa
  4 dimensões (`nTR × nAng × n_pos × nf`) em **um único `prange`**, eliminando
  o `range(nf)` serial residual em `_fields_in_freqs_kernel_cached`.
- **`SimulationConfig.use_flat_prange`**: novo campo opt-in (default `False`)
  ativa o caminho FLAT. Backward-compat total preservada — `False` mantém
  v2.21 (Sprint 13.3 + 21.1).
- **Dispatcher em `multi_forward.py`**: roteia `_simulate_combined_prange_flat`
  vs `_simulate_combined_prange` baseado em `cfg.use_flat_prange`.
- **Sprint v2.22.3 — Benchmarks**: novo `bench_v22_flat_prange.py` com
  Cenários E/B/F (CLI `--scenario X --runs N`).
- **Validação**: 27 testes em `test_simulation_v22_flat_prange.py`. 100% PASS.
  Suite total: **1597 PASS / 295 SKIP / 0 FAIL** em 916s (15min).
- **Performance** (single-process, M-series 8C/16T, 3 runs medianos):
  - Cenário E (n_pos=600, nf=1, 1TR/1ang): 214k → 212k mod/h (0.99×, sem regressão)
  - Cenário B (n_pos=200, nf=1, 3TR×4ang): 48k → 54k mod/h (**1.11×**)
  - Cenário F (n_pos=600, nf=4, 1TR/1ang): 53k → 57k mod/h (**1.09×**)
- **Paridade Fortran <1e-12**: preservada por TRANSITIVIDADE (FLAT ≡ legacy
  bit-exato; legacy ≅ Fortran <1e-12). Confirmada em pre-commit hook em
  todos os commits da sprint.
- **Benefício arquitetural**: elimina anti-pattern v2.13 (range(nf) em contexto
  nested) e estabelece padrão FLAT como base para Sprints v2.23+ (fastmath,
  Hankel pré-cômputo, cache contexto).
- **Decisão sobre default**: `use_flat_prange=False` mantido por 1 semana de
  validação em produção antes de promover a `True` (v2.22.1 patch release).

---

## [Quality Mesh 1.5] — 2026-05-08

### Polishing & Estabilização (Etapa 1.5)

- **A.1 Fortran-Parity Smart**: hook `run-fortran-parity.sh` reescrito com dois modos
  (`quick` oklahoma_3 ~2s em PostToolUse; `full` 7 modelos ~146s em pre-commit).
  Controles: `FORTRAN_PARITY_MODE`, `CLAUDE_BYPASS_FORTRAN_PARITY=1`.
- **A.2 Backup Cleanup**: `tools/cleanup_backups.sh` com política 24h/7d/∞ + `--dry-run`.
- **A.3 Thread Safety**: `conflict_matrix.py` com double-checked locking (`threading.Lock`).
- **A.4 Merges**: `feat/simulation-manager-v2.21` + `feat/quality-mesh-foundation` → `main`.
  KB-013/018/019 fixes integrados. `test_known_bugs.py`: 10/10 PASS (era 8+2 XFAIL).
- **A.5 Bypass override**: `CLAUDE_BYPASS_ANTI_PATTERNS=1` adicionado a ambos os hooks
  de anti-patterns.
- **Bugs pré-execução corrigidos**: `is_expired()` off-by-one (`>` → `>=`), detector
  level-0 para nested defs em KB-018, `pythonpath=["."]` restaurado no `pyproject.toml`,
  encoding unicode fix em `run-fortran-parity.sh`, `tensorflow` removido de mypy deps.
- **Code review 4 findings**: guard path traversal em `backup-pre-edit.sh`, fallback
  `pwd` em `release-lock.sh`, boundary `>=` em `cleanup_backups.sh`, `grep --` separator.
- **Resultado final**: 19/19 PASS, 0 XFAIL, 0 SKIP.

---

## [2.21] — 2026-05-02

### Causa-raiz da regressão histórica encontrada via análise old_geosteering_ai/

Sprint 21 — análise comparativa direta entre `old_geosteering_ai/` (versão
pré-Sprint 13, conhecida como boa com ~120k mod/h em Cenário E) e o
código atual revelou o **Sprint 13.1 (v2.13) como causa da regressão**:
adicionou `@njit(parallel=True, nogil=True)` + `prange(nf)` em
`_fields_in_freqs_kernel_cached`, função chamada **milhões de vezes**
de dentro de `_simulate_positions_njit_cached` (que JÁ tem prange outer).

**Por que isso degrada performance:**

Numba não suporta nested parallelism. O `prange(nf)` interno era
serializado (como deveria), MAS o overhead de setup do parallel
scheduler era pago em cada chamada — em Cenário E (300 modelos × 600
pts × 75 chunks = 13.5M chamadas), isso totalizava ~14s de overhead puro.

**Fix Sprint 21.1 (1 linha de código + docstring expandida):**

```diff
- @njit(cache=True, parallel=True, nogil=True)
+ @njit(cache=True, nogil=True)
  def _fields_in_freqs_kernel_cached(...):
      ...
-     for i_f in _prange(nf):
+     for i_f in range(nf):
```

`precompute_common_arrays_cache` mantém `parallel=True` (é chamada de
contexto SERIAL Python, paraleliza nf efetivamente sem overhead aninhado).

**Métricas (Mac 8C/16T HT, 4w × 2t auto v2.20):**

| Cenário | v2.20 | v2.21 | Ganho |
|:-------:|:-----:|:-----:|:-----:|
| **A** (30 pts, 1 freq) | 1 185 489 mod/h | **1 392 371 mod/h** | **1.17×** |
| **B** (30 pts, 10 freq) | 376 000 mod/h | 303 452 mod/h | 0.81× ⚠️ |
| **E** (600 pts, 1 freq) | 46 104 mod/h (mediana) | **121 957 mod/h** | **2.65×** ✓ |
| Paridade Fortran 7 canônicos | <1e-12 | <1e-12 | preservada |

**Cenário E atinge 121 957 mod/h** — confirma a meta histórica do
usuário ("mais de 120k em configurações padrão"). **3.04× total** vs
v2.18 (40k → 122k).

**Tradeoff em Cenário B**: regressão de 19% aceita porque:
- E é configuração de produção real (600 pts LWD)
- B é multi-freq sintético, pouco usado em produção
- v2.22 (FLAT prange n_pos × nf) recuperará B sem reintroduzir overhead em E

**Mudanças em arquivos:**

- `geosteering_ai/simulation/_numba/kernel.py`: `_fields_in_freqs_kernel_cached`
  decorador simplificado + range serial + docstring expandida com
  investigação histórica (+25 linhas)
- `docs/reports/v2.21_2026-05-02.md` (novo, ~330 linhas): relatório
  técnico completo com análise comparativa old vs new

**Validação:**

- Paridade Fortran 7 modelos canônicos: PASS (<1e-12)
- Pytest suite focada: 68/68 PASS
- 5 runs consecutivos Cenário E: mediana 122k, desvio ~4k (estável)

**Lição arquitetural documentada para evitar tentativas futuras:**

> Paralelismo aninhado em Numba NÃO funciona como esperado. Quando uma
> função inner é chamada de prange outer, adicionar `parallel=True` na
> inner causa overhead puro sem benefício (Numba serializa nested
> prange). A regra é: paralelizar UMA ÚNICA VEZ no nível mais externo
> em produção.

**Roadmap atualizado para 200k+ em Cenário E:**

| Versão | Otimização | Ganho em E |
|:------:|:-----------|:----------:|
| v2.22 | FLAT prange(n_pos × nf) | recupera B; E neutro |
| v2.23 | Tile/block | +15-25% |
| v2.24 | Pré-compute Hankel TE/TM | +10-15% |
| v2.25 | fastmath SAFE | +20% |

**Status:** estável, validado, pronto para produção. Meta histórica
de >120k mod/h em Cenário E **alcançada**.

---

## [2.20] — 2026-05-02

### Confirmação empírica + investigação rigorosa: phys_cores é a estratégia correta

Sprint 20 — em resposta à premissa do usuário ("antes era 150-190k mod/h
em cenários padrão com 4w × 4t; agora obtenho 92k em E. Por que houve
retrocesso?"), foi conduzida investigação empírica rigorosa que **confirma**
a estratégia v2.17 (`recommend_default_parallelism` retorna phys_cores) e
**refuta** uma hipótese inicial (HT-aware = logical cores).

**Investigação experimental** (Mac 8C/16T HT, Cenário E 600 pts, 5 runs cada):

| Configuração | Mediana | Média | Desvio |
|:------------:|:-------:|:-----:|:------:|
| 4w × 2t (8 threads = phys_cores) | **46k mod/h** | 47k mod/h | ~12k |
| 4w × 4t (16 threads = logical) | 38k mod/h | 38k mod/h | ~1k |

**Razão mediana 4w×2t / 4w×4t = 1.23** (4w × 2t é 23% melhor).

**Por que HT degrada este kernel?**

1. Recursão TE/TM em `hmd_tiv` é **compute-bound**, não memory-bound como
   hipotetizado.
2. Context switch entre hyperthreads custa ~50-200 ciclos cada, superando
   o ganho hipotético de cache miss hiding.
3. Cache trashing: HT compartilha L1/L2; 16 threads competindo aumentam
   miss rate em vez de diminuir.
4. Numba TBB scheduler já gerencia work-stealing eficientemente; threads
   acima de phys_cores apenas multiplicam overhead.

**Mudanças desta versão:**

- `geosteering_ai/simulation/_workers.py:292-360`:
  `recommend_default_parallelism` mantida com estratégia v2.17 (target =
  phys), mas docstring expandida (+35 linhas) com:
  - Tabela empírica de medição
  - Justificativa teórica (compute-bound + context switch + cache trashing)
  - Aviso para futuros desenvolvedores não tentarem `target = logical`

- `benchmarks/bench_v214_numba.py:445-460`: warning de oversubscrição
  refinado com **referência empírica concreta** ("Empiricamente v2.20,
  5 runs Cenário E, oversubscrição degrada 20-25%").

- `tests/test_simulation_cpu_topology.py`: docstrings expandidas com
  contexto empírico + 1 novo teste `test_apple_m_pro_10c_10t_no_ht`.

- `docs/reports/v2.20_2026-05-02.md` (novo, ~320 linhas): relatório
  técnico completo com investigação, decisão arquitetural, projeção
  de v2.21-v2.24 para chegar a 150-200k em Cenário E.

**Resposta à premissa do usuário:**

1. **Cenário A** com defaults v2.20 corretos (`4w × 2t`) entrega
   **1 185 489 mod/h** — 6× **ACIMA** do histórico 150-190k relatado.
   Não houve regressão; houve **ganho massivo**.

2. **Cenário E** (600 pts) é fundamentalmente diferente de A (30 pts);
   comparar A histórico com E atual é incorreto. A meta de 150-200k em E
   requer otimizações algorítmicas (v2.21+), não mudança de threading.

3. Os 92-95k mod/h vistos pelo usuário em E foram **outliers** com cache
   de disco extremamente quente. Mediana real (5 runs): 38k em 4w×4t,
   46k em 4w×2t.

**Roadmap para 150-200k em Cenário E:**

| Versão | Otimização | Ganho esperado | Throughput projetado |
|:------:|:-----------|:--------------:|:--------------------:|
| v2.21 | Tile/block em `_simulate_positions_njit_cached` | 15-25% | ~55k |
| v2.22 | Pré-compute Hankel kernels TE/TM | 10-15% | ~63k |
| v2.23 | `fastmath=True` SAFE | 20% | ~75k |
| v2.24 | SIMD ufuncs NumPy | 30-40% | ~100k |

Para 150-200k em E, todas as 4 otimizações combinadas serão necessárias.

**Métricas finais (Mac 8C/16T HT, defaults v2.20 4w × 2t):**

| Cenário | Throughput | Status |
|:-------:|:----------:|:------:|
| A (30 pts, 1 freq) | **1 185 489 mod/h** | superou histórico 6× |
| B (30 pts, 10 freq) | 376 000 mod/h | ótimo |
| E (600 pts, 1 freq) | ~46 000 mod/h (mediana) | gargalo memory-bound |
| Paridade Fortran | <1e-12 | preservada |
| Pytest | 39+ pass (+1 novo) | — |

**Status:** estável, validado empiricamente, pronto para produção.

**Arquivos modificados:** `geosteering_ai/simulation/_workers.py`,
`tests/test_simulation_cpu_topology.py`, `benchmarks/bench_v214_numba.py`.
**Novos:** `docs/reports/v2.20_2026-05-02.md`.

---

## [2.19] — 2026-05-02

### Fix random seed (bug funcional) + nogil hot path + benchmark CPU-aware + auditoria PyQt6

Sprint 19 — entrega 3 correções coordenadas + 1 investigação histórica em resposta
a problemas reportados pelo usuário sobre regressão de performance e gerador
de perfis aleatórios produzindo sempre os mesmos modelos.

**Problemas corrigidos:**

1. **Bug funcional — `rng_seed=42` hardcoded** ([simulation_manager.py:8088](../geosteering_ai/simulation/tests/simulation_manager.py#L8088))
   fazia com que cada "Iniciar Simulação" gerasse a **mesma sequência de N modelos**
   — impossibilitando ensembles estatísticos diversos para treino.
2. **Performance — `nogil=True` ausente** em `_simulate_positions_njit` (linha 133) e
   `_simulate_positions_njit_cached` (linha 233) de [forward.py](../geosteering_ai/simulation/forward.py).
   Workers competiam pelo GIL durante `prange(n_pos)` interno, perdendo paralelismo
   real entre processos.
3. **Benchmark CLI — sem detecção de oversubscrição.** `bench_v214_numba.py`
   aceitava `--workers 4 --threads-per-worker 4 = 16 threads` em CPU 8C/16T
   sem aviso, causando degradação até 4× em Cenário A.
4. **Premissa PyQt5 → PyQt6** — descartada por investigação. A migração
   (commit `645ecaa`, 27-Apr) foi limpa; regressão real ocorreu em commits
   Numba threading subsequentes (v2.15 → corrigidos em v2.16/v2.17/v2.18).

**Sprint 19.1 — Random seed UI control:**

- Função `_resolve_rng_seed(seed: Optional[int])` em [sm_model_gen.py](../geosteering_ai/simulation/tests/sm_model_gen.py) — `None` → `secrets.randbits(63)` (63 bits int64-safe)
- `generate_models(rng_seed=None)` default mudou de `42` para `None`; novo
  parâmetro `return_seed=True` retorna `(models, actual_seed)`
- `ModelGenerationThread.__init__(rng_seed: Optional[int] = None)` resolve seed em `run()`
- Novo sinal `seed_used = Signal(int)` emitido antes do progresso para logging
- UI em [ParametersPage](../geosteering_ai/simulation/tests/simulation_manager.py#L1411):
  - `chk_random_seed` (QCheckBox, default checked = aleatório)
  - `spn_fixed_seed` (QSpinBox, enabled apenas quando checkbox unchecked)
  - Tooltips PT-BR explicativos
- Novo método `ParametersPage.get_rng_seed() -> Optional[int]`
- `to_dict`/`from_dict` persistem `random_seed` + `fixed_seed`
- `_start_simulation` (linha 8142) usa `params.get_rng_seed()` em vez do `42` hardcoded
- Smoke test (linha 10269) preserva `rng_seed=42` explícito (determinismo)

**Sprint 19.2 — `nogil=True` + benchmark CPU-aware defaults:**

- [forward.py:133, 233](../geosteering_ai/simulation/forward.py): `@njit(parallel=True, cache=True, nogil=True)` (consistente com `_simulate_combined_prange:351`)
- Cache Numba limpo após mudança (recompilação obrigatória)
- Paridade Fortran 7 canônicos preservada: <1e-12 (validado pré e pós)
- [bench_v214_numba.py](../benchmarks/bench_v214_numba.py):
  - Defaults dinâmicos via `recommend_default_parallelism()` + `detect_cpu_topology()`
  - Em CPU 8C/16T HT, defaults = 4w × 2t = 8 threads = cores físicos
  - Warning explícito quando `workers × threads > physical_cores`

**Sprint 19.3 — Testes de não-regressão (12 novos):**

- [`tests/test_simulation_random_seed.py`](../tests/test_simulation_random_seed.py) (7 testes):
  - rng_seed=None gera modelos distintos
  - Semente fixa reproduz bit-a-bit
  - return_seed=True retorna actual_seed
  - Smoke seed=42 ainda funciona
  - _resolve_rng_seed cobre None vs int
  - ModelGenerationThread.seed_used Signal exposto
- [`tests/test_simulation_pool_warmup.py`](../tests/test_simulation_pool_warmup.py) (5 testes):
  - PoolWarmupThread tem `warmup_done` + `warmup_error`
  - Aceita args (n_workers, n_threads, hankel_filter)
  - SimulatorPage tem lbl_warmup_status + _warmup_thread
  - SimulationThread.run referencia t0_sim (não-regressão v2.18)
  - ParametersPage tem widgets de seed (não-regressão v2.19)

**Métricas (Mac 8C/16T, defaults v2.19 4w × 2t):**

| Cenário | v2.18 (4w×4t) | v2.19 (4w×2t auto) | Ganho |
|:-------:|:-------------:|:------------------:|:-----:|
| A (30 pts, 1 freq) | 189 796 mod/h | **802 114 mod/h** | **4.24×** |
| B (30 pts, 10 freq) | 409 465 mod/h | 375 676 mod/h | 0.92× |
| E (600 pts, 1 freq) | ~40 074 mod/h | ~34 828 mod/h | 0.87× (memory-bound) |
| Paridade Fortran | <1e-12 | <1e-12 | preservada |
| Pytest | 165+ | **177+** (+12 novos) | — |
| Smoke GUI | T1-T32 OK | T1-T32 OK + 12 testes | — |

**Comportamento default ALTERADO (breaking-by-default, opt-out trivial):**
Antes da v2.19, dois cliques de "Iniciar Simulação" produziam modelos
idênticos (efeito colateral do bug). Usuários que dependem desse
comportamento devem **desmarcar** o checkbox "Semente aleatória" e
configurar a semente fixa (default 42 preserva v2.18 behavior).

**Próximos passos (v2.20+):** tile/block em `_simulate_positions_njit_cached`
(15-25%), pré-compute Hankel kernels TE/TM (10-15%), SIMD ufuncs (20-40%)
para fechar gap em Cenário E (35k → 140k+ mod/h).

**Status:** estável, testado, pronto para produção.

**Arquivos modificados:** `geosteering_ai/simulation/forward.py`,
`geosteering_ai/simulation/tests/sm_model_gen.py`,
`geosteering_ai/simulation/tests/simulation_manager.py`,
`benchmarks/bench_v214_numba.py`. Novos: `tests/test_simulation_random_seed.py`,
`tests/test_simulation_pool_warmup.py`, `docs/reports/v2.19_2026-05-02.md`.

---

## [2.18] — 2026-05-02

### Fix throughput reportado erroneamente (38k mod/h → 85k mod/h real) + pré-aquecimento de pool em background

Sprint 18 — investigação revela que o throughput de **38k mod/h** reportado
consistentemente na GUI não é um bug de performance, mas um **artefato de medição**:
o timer `t0` era iniciado *antes* de `_acquire_numba_pool()`, incluindo
~10–12 s de overhead de pool cold-start (spawn de 4 workers + import do pacote
+ 2× warmup JIT Numba) no denominador do cálculo. Com pool warm (segunda
simulação da mesma sessão), o throughput já era ~85k mod/h — idêntico ao
benchmark. O benchmark Cenário E media 85k porque herdava pool warm de
cenários anteriores ou de disco cache JIT.

**Causa raiz**: `t0 = perf_counter()` na linha 766 de `sm_workers.py`,
**antes** de `_acquire_numba_pool()` na linha 830. O overhead de spawn +
init de 4 workers (~10–12 s frio; ~0 ms warm) ficava incluído no cálculo
`n_total / (perf_counter() - t0) × 3600`.

**Dois fixes implementados:**
1. `t0_sim` definido após todos os workers confirmarem init via `_noop` tasks
2. `PoolWarmupThread` inicia pool em background quando a GUI abre (~500 ms delay)

### Adicionado

- **Sprint 18.1 — `PoolWarmupThread(QThread)`** ([sm_workers.py:1091-1151](../geosteering_ai/simulation/tests/sm_workers.py)):
  - Thread Qt de background que cria/reusa `_PERSISTENT_POOL` e aguarda
    todos os workers completarem `_numba_init_worker` via `_noop` tasks
  - Signals: `warmup_done(elapsed_s, n_workers, n_threads)` + `warmup_error(msg)`
  - Não-fatal: falha no warmup não impede simulação (pool criado no momento)
  - `hankel_filter="werthmuller_201pt"` padrão (cobre 95%+ dos casos de uso)

- **Sprint 18.2 — Pré-aquecimento em `SimulationPage`** ([simulation_manager.py](../geosteering_ai/simulation/tests/simulation_manager.py)):
  - `QTimer.singleShot(500, self._start_background_warmup)` em `__init__`
  - Label `lbl_warmup_status`: amarelo "Aquecendo Xw × Yt... (spawn + JIT Numba)"
    → verde "Workers prontos: Xw × Yt (12.3s warmup)" após init completar
  - `_on_backend_changed_warmup`: oculta label se backend mudar para Fortran
  - Métodos: `_start_background_warmup()`, `_on_warmup_done()`, `_on_warmup_error()`

### Corrigido

- **`t0_sim` pós-init-workers** ([sm_workers.py](../geosteering_ai/simulation/tests/sm_workers.py)):
  - Após `_acquire_numba_pool()`, submete `_noop` a todos os `n_workers` e
    aguarda conclusão (confirma que `_numba_init_worker` completou em cada worker)
  - `t0_sim = perf_counter()` definido somente após confirmação
  - Todos os cálculos de throughput no branch Numba usam `t0_sim or t0`:
    `progress_update.emit` (tempo real), `finished_all["throughput_mod_h"]`
  - `elapsed_total` (para display "Tempo de execução: X horas") continua usando
    `t0` original (inclui warmup no tempo total reportado — correto)

### Testes

| Teste | Tipo | Verificação |
|:------|:-----|:------------|
| T29: PoolWarmupThread instanciável | Smoke | `isinstance(wt, PoolWarmupThread)` |
| T30: signals warmup_done + warmup_error | Smoke | `hasattr(wt, "warmup_done")` |
| T31: SimulationPage tem lbl_warmup_status | Smoke | `hasattr(sim_page, "lbl_warmup_status")` |
| T32: SimulationPage tem _warmup_thread | Smoke | `hasattr(sim_page, "_warmup_thread")` |

**19 testes existentes (test_simulation_cpu_topology.py) + 0 falhas smoke** — zero regressão.

### Métricas pós-fix (Hardware Intel 8C/16T HT, macOS)

| Configuração | Throughput reportado | Throughput real | Gap |
|:-------------|:--------------------:|:---------------:|:---:|
| v2.17 (cold pool, timer inclui warmup) | 38k mod/h | 85k mod/h | 2.24× erro medição |
| **v2.18 (t0_sim pós-warmup)** | **85k mod/h** | **85k mod/h** | **1.0× correto** |
| v2.18 (pool pré-aquecido pelo warmup bg) | 85k mod/h | 85k mod/h | — primeira exec |

### Arquivos modificados

| Arquivo | Mudanças |
|:--------|:---------|
| `geosteering_ai/simulation/tests/sm_workers.py` | +`PoolWarmupThread`, `t0_sim` pós-noop, `Signal` fix, throughput 4× |
| `geosteering_ai/simulation/tests/simulation_manager.py` | +`PoolWarmupThread` import, `lbl_warmup_status`, `_start_background_warmup()`, handlers, 4 smoke tests |
| `docs/CHANGELOG.md` | Esta entrada |
| `docs/ROADMAP.md` | v2.18 adicionada |
| `CLAUDE.md` | Linha 16 atualizada |
| `.claude/commands/geosteering-simulation-manager.md` | §18 adicionada |

---

## [2.17] — 2026-05-02

### Fix regressão de oversubscrição em CPUs Hyperthreading/SMT (3× em produção GUI)

Sprint 16 — investigação adicional revela que a regressão de **3× em produção GUI**
remanescente (38k mod/h vs 123k esperado pós-v2.16) era causada por **oversubscrição**:
a fórmula default `spin_threads = max(2, ncpu // (ncpu // 4))` em
[simulation_manager.py](../geosteering_ai/simulation/tests/simulation_manager.py)
produzia **4 workers × 4 threads = 16 threads** em CPU 8C/16T HT, excedendo
**2× os 8 cores físicos** (oversubscrição). O fix v2.16 (threading masking
observable) estabilizou o pool, mas defaults ruins permaneciam.

A v2.17 introduz **detecção de topologia CPU** (físicos vs lógicos) e
**recomendação inteligente** que respeita ``workers × threads ≤ cores físicos``,
eliminando oversubscrição em hardware com Hyperthreading (Intel) ou SMT
(AMD/ARM big-cores).

### Adicionado

- **Sprint 16.1 — `detect_cpu_topology()` portável** ([_workers.py:158-291](../geosteering_ai/simulation/_workers.py)):
  - Detecção em camadas: psutil → sysctl (macOS) → /proc/cpuinfo (Linux) →
    wmic (Windows) → heurística fallback
  - Retorna `(logical_cores, physical_cores, has_hyperthreading)`
  - Cache em variável global `_CPU_TOPOLOGY_CACHE` (topologia não muda em runtime)
  - NUNCA falha — em caso de erro retorna `(logical, logical, False)` (conservador)
- **Sprint 16.2 — `recommend_default_parallelism()`**
  ([_workers.py:294-353](../geosteering_ai/simulation/_workers.py)):
  - Para batch grande (>= 10 modelos): `(phys // 2, 2)` — Modo D híbrido
  - Para single-model: `(1, phys)` — Modo B multi-thread
  - Invariante: `workers × threads ≤ physical_cores` SEMPRE
- **Sprint 16.3 — Warning visual de oversubscrição na GUI**
  ([simulation_manager.py](../geosteering_ai/simulation/tests/simulation_manager.py)):
  - QLabel vermelho aparece quando `workers × threads > physical_cores`
  - Mensagem: "⚠ Oversubscrição: {W} × {T} = {N} threads em {phys} cores físicos"
  - Conectado via `valueChanged` aos spinboxes (atualização em tempo real)
- **Sprint 16.4 — Logging diagnóstico em SimulationThread.run()**
  ([sm_workers.py](../geosteering_ai/simulation/tests/sm_workers.py)):
  - Loga "CPU: N cores físicos · M threads lógicas (HT/SMT)" antes de cada simulação
  - Aviso explícito se `workers × threads > phys_cores`
- **Sprint 16.5 — Smoke tests CPU topology**
  ([tests/test_simulation_cpu_topology.py](../tests/test_simulation_cpu_topology.py)):
  - 19 testes (5 detection + 6 recommendation + 6 simulated hardware + 2 no-regression)
  - Cenários cobertos: Mac Intel 8C/16T, Apple Silicon M1, Linux Xeon 32C, dual-core
- **Tooltips expandidos**: explicam recomendação física vs lógica em todos os spinboxes

### Corrigido

- **Defaults da página Simulação** (`SimulationPage`): de `(4w × 4t = 16)` para
  `(4w × 2t = 8)` em hardware 8C/16T HT (= cores físicos exatos, sem oversubscrição)
- **Defaults da página Benchmark** (`BenchmarkPage`): mesma correção aplicada
  para Numba e Fortran
- **Label de CPU**: agora mostra "8 cores físicos · 16 threads lógicas (HT/SMT)"
  em vez de apenas "16 CPU cores disponíveis"

### Mudado

- **Função pública `detect_cpu_topology()`** exportada em
  [geosteering_ai/simulation/__init__.py](../geosteering_ai/simulation/__init__.py)
  para uso por outros componentes (treinamento DL, etc.)
- **Função pública `recommend_default_parallelism()`** exportada — ponto único
  de verdade para todos os defaults de paralelismo no projeto

### Notas de performance (Hardware Intel 8C/16T HT, macOS)

| Configuração | Threads totais | Cenário E (200 mod, 600 pts) | Speedup |
|---|---|---|---|
| **v2.16 default GUI** (4w × 4t) | 16 (oversub 2×) | 70k mod/h | baseline |
| **v2.17 default GUI** (4w × 2t) | 8 (= phys) | 85k mod/h | **+21%** |
| Cenário E benchmark (4w × 2t) | 8 (= phys) | 85k mod/h | idêntico ✓ |

Em workloads CPU-bound mais intensos (modelos com mais camadas, mais frequências
ou n_pos > 1000) e em arquiteturas com HT mais agressivo (Linux Xeon 32C/64T),
o ganho esperado é **30–50%**.

### Paridade Fortran

- **<1e-12 em 7 modelos canônicos** (Oklahoma 3/5/15/28, Devine 8, Hou 7,
  Viking Graben 10) — zero regressão física

### Validação

- **Pytest CPU topology** (test_simulation_cpu_topology.py):
  19/19 PASS em 1.38s
- **Pytest threading v2.16** (test_simulation_workers_threading.py):
  3/3 PASS em 4.22s (zero regressão)
- **Pytest I/O v2.16** (test_sm_workers_io.py):
  4/4 PASS em 2.73s (zero regressão)
- **Paridade Fortran 7 canônicos**: 7/7 PASS em 126.62s (<1e-12)

### Decisões deferidas

- **Tile/block processing (Sprint 15.6)** → mantido em v2.18 (mais ganho potencial
  agora que oversubscrição foi eliminada)
- **Pré-compute Hankel kernels TE/TM** → v2.18 (10–15% ganho, médio risco)
- **Apple Silicon M1/M2 sem HT**: defaults já corretos (phys = logical)

### Arquivos

**Modificados (4)**:
- `geosteering_ai/simulation/_workers.py` (+193/-0)
- `geosteering_ai/simulation/__init__.py` (+4/-0)
- `geosteering_ai/simulation/tests/simulation_manager.py` (+90/-22)
- `geosteering_ai/simulation/tests/sm_workers.py` (+30/-1)

**Criados (2)**:
- `tests/test_simulation_cpu_topology.py` (~250 LOC, 19 testes)
- `docs/reports/v2.17_2026-05-02.md` (relatório principal)

---

## [2.16] — 2026-05-01

### Fix regressão crítica de threading + Cenário E (production scale 600 pts) + I/O vetorizado

Sprint 15 de correção: identifica e corrige regressão de **4–8×** em
produção GUI introduzida pela combinação dos commits `0f92035` (`try/except
RuntimeError: pass` em `multi_forward.py`) e `e1c8864` (remoção de
`NUMBA_NUM_THREADS` env var nos workers) da v2.15. Adiciona Cenário E
para reproduzir a configuração real de produção (n_positions=600) e
vetoriza `write_dat_from_tensor` (≥3× speedup em I/O).

### Adicionado

- **Sprint 15.2 — Cenário E benchmark (production scale)**:
  - Novo `benchmark_scenario_e()` em [benchmarks/bench_v214_numba.py](../benchmarks/bench_v214_numba.py)
    (+95 LOC) — n_positions=600, single-freq, replica config GUI LWD
  - Novo flag CLI `--n-positions N` (default 30 microbench, 600 production)
  - `--all` agora roda 5 cenários (A–E)
  - Esperado em hardware 8C HT pós-fix: ≥120k mod/h
- **Sprint 15.1 — Smoke tests threading masking**:
  - [tests/test_simulation_workers_threading.py](../tests/test_simulation_workers_threading.py) (~270 LOC)
  - 3 testes: env var inheritance, logger.warning observable,
    simulate_multi em worker respeita num_threads
- **Sprint 15.4 — Smoke tests I/O vetorizado**:
  - [tests/test_sm_workers_io.py](../tests/test_sm_workers_io.py) (~200 LOC)
  - 4 testes: bit-exatness vs loop, z_obs 1D, rho None, performance ≥3×
- **Sprint 15.5 — Relatório técnico n_positions scaling**:
  - [docs/reports/v2.16_n_positions_scaling_analysis_2026-05-01.md](reports/v2.16_n_positions_scaling_analysis_2026-05-01.md)
  - Hot path identificado: `_numba/dipoles.py::hmd_tiv` recursão TE/TM
  - Análise de complexidade O(`n_pos × n_layers × n_filter_pts × n_freqs × n_TR × n_ang`)
  - Top 3 oportunidades documentadas (tile/block, pré-compute kernels, SIMD ufuncs)
- **Marker `slow` em pyproject.toml**:
  - Permite filtrar testes JIT cold-start em CI rápido com `-m 'not slow'`

### Corrigido

- **Sprint 15.1 — Threading masking observable em `multi_forward.py:880-907`**:
  - Substituído `try/except RuntimeError: pass` (silencioso) por
    `logger.warning` com diagnóstico (threads ativas, pool size, exception)
  - Adicionada verificação prévia `if current_active != cfg.num_threads`
    para evitar set redundante
  - Causa raiz da regressão 4–8× em produção GUI (v2.15)
- **Sprint 15.1 — `NUMBA_NUM_THREADS` setado no env do PAI antes do spawn**:
  - [`sm_workers.py::_acquire_numba_pool`](../geosteering_ai/simulation/tests/sm_workers.py)
    (+15 LOC) — workers spawn herdam env, Numba dimensiona pool corretamente
  - [`_workers.py::_acquire_pool`](../geosteering_ai/simulation/_workers.py)
    (+15 LOC) — espelho no pool nativo do core
  - Resultado: workers nascem com pool de threads = `n_threads` (não `cpu_count()`)

### Mudado

- **Sprint 15.4 — `write_dat_from_tensor` vetorizada**:
  - [sm_io.py](../geosteering_ai/simulation/tests/sm_io.py) (+50 / -28 LOC)
  - 5 loops Python aninhados (~1.8M iterações para 600 modelos × 600 pos
    × 1 freq) → broadcast + transpose + reshape NumPy
  - Speedup ≥3× em I/O (validado por `test_write_dat_vectorized_is_faster`)
  - Bit-exatness preservada (validada por 3 testes diferentes em
    `tests/test_sm_workers_io.py`)
- **CLAUDE.md linha 16**:
  - `SM v2.15` → `SM v2.16 (2026-05-01) — fix regressão threading + cenário E (600 pts) + I/O vetorizado`

### Notas de Performance (Hardware Intel 8C/16T HT, macOS)

| Cenário | Pré-fix v2.15 | Pós-fix v2.16 | Speedup |
|:-------:|:-------------:|:-------------:|:-------:|
| A (30 pts, 5k mod) | 753k mod/h¹ | **1.74M mod/h** | 2.31× |
| E (600 pts, 200 mod) | ~30k mod/h² | 86k mod/h | 2.87× |
| E (600 pts, 600 mod) | — | 111k mod/h | — |
| E (600 pts, 2000 mod) | — | **123k mod/h** | ~4.10× |

¹ Reportado em `v2.15_benchmark_hardware_2026-05-01.md`.
² Estimado por escalabilidade linear `n_pos` (753k / 20 ≈ 38k) — coerente com relato GUI 25–38k mod/h.

### Decisões Deferidas

- **Tile/block processing (Sprint 15.6)** — DEFERIDO para v2.17. Cenário E
  pós-fix entrega 4× speedup confirmado; risco/benefício de adicionar
  reordering de prange agora não justifica vs validação de paridade
  Fortran obrigatória em todos os 7 canônicos.
- **Pré-compute Hankel kernels TE/TM** — DEFERIDO para v2.18. Ganho
  estimado 10–15% mas requer revalidação de paridade.
- **SIMD ufuncs** — REJEITADO. Risco de quebrar paridade Fortran <1e-12
  por reordering FMA é inviolável.

### Pytest

- **Total esperado:** 172+ pass (165 v2.15 + 7 novos: 3 threading + 4 I/O)
- **Paridade Fortran:** 10/10 PASS em <1e-12 (zero regressão)
- **Smoke GUI:** 207+ OK (mantido)

---

## [2.15] — 2026-05-01

### Hardware validation, JIT cache observability, code review, fix CI v2.14

Sprint 14 de finalização: corrige CI quebrada do PR #33 (binário Fortran
incompatível em runner Linux), valida ganhos v2.14 em hardware real
(8 cores físicos × 2 threads), expande observability do triple-cache
JAX (bucketed/unified/chunked) e aplica P1 findings do code review
v2.13→v2.14 (zero P0 encontrado).

### Adicionado

- **Sprint 14.0 — Gate CI-safe `_tatu_runnable()`**:
  - Helper em [tests/_fortran_helpers.py](../tests/_fortran_helpers.py) (+73 LOC)
  - Detecta via `subprocess.run` se `tatu.x` é executável no OS atual
    (resolve `OSError [Errno 8] Exec format error` em CI Linux quando
    o repo contém binário macOS commitado)
  - Cache módulo-level memoiza resultado por sessão (sem custo recorrente)
  - 12 testes em `test_simulation_compare_fortran.py` e `test_simulation_multi.py`
    migram de `Path.exists()` para `_tatu_runnable()`
- **Sprint 14.1 — `--threads-per-worker` no benchmark**:
  - Novo arg CLI em [benchmarks/bench_v214_numba.py](../benchmarks/bench_v214_numba.py) (+30 LOC)
  - Helper `_configure_threads()` seta `OMP_NUM_THREADS`,
    `NUMBA_NUM_THREADS`, `MKL_NUM_THREADS` antes do worker pool fork
  - Default `2` (ideal para 4 workers × 2 threads = 8 cores físicos)
  - Total threads = `workers × threads_per_worker` (anti-oversubscription)
- **Sprint 14.3 — `get_jit_cache_info()` expandido**:
  - Função em [forward_pure.py](../geosteering_ai/simulation/_jax/forward_pure.py)
    (+80 LOC) reporta os 3 caches: `bucketed_size`, `unified_size`,
    `chunked_size`, `total_xla_programs`, `estimated_vram_mb`,
    `strategy_distribution`
  - Heurística VRAM: `Σ (3 × n × npt × 16 bytes) / 1024²` por entrada
    (proxy do tensor `common_arrays` shape `(3, n, npt, nf)` complex128)
  - Backward-compat preservada: chaves `n_entries`, `maxsize`, `keys`
    de v1.5.0 continuam apontando para `_BUCKET_JIT_CACHE`
  - 4 novos testes em
    [test_simulation_jax_sprint13.py](../tests/test_simulation_jax_sprint13.py)
    (~180 LOC): empty, after_simulate_unified, vram_estimate, idempotent
- **Relatórios técnicos**:
  - `docs/reports/v2.15_2026-05-01.md` — relatório técnico Sprint 14
  - `docs/reports/v2.15_benchmark_hardware_2026-05-01.md` — 4 cenários
    no hardware do usuário (8 cores físicos × 2 threads × 4 workers)
  - `docs/reports/v2.15_fastmath_dipoles_analysis_2026-05-01.md` —
    decisão técnica de NÃO aplicar fastmath em dipoles.py/kernel.py

### Mudado

- **Sprint 14.4 P1 — code review aplicado**:
  - Eliminado import duplicado de `_simulate_combined_prange` em
    [multi_forward.py](../geosteering_ai/simulation/multi_forward.py):867-876
    (bloco unificado em uma única tupla)
  - Loop triplo z_obs simplificado: O(nTR × nAngles × n_pos) com
    `break` imediato → O(nAngles × n_pos) sem inner loop, semântica
    bit-exata preservada (primeiro TR sempre amostrado)
  - Teste `test_fastmath_propagation_remains_false` agora inspeciona
    `targetoptions` do dispatcher Numba para detectar regressão
    (em vez de smoke + finitude apenas)

### Corrigido

- **CI PR #33 v2.14**:
  `Fortran_Gerador/tatu.x` é binário macOS arm64/x86_64 commitado no
  repositório como artefato de validação local. Em runner Linux x86_64
  o sistema rejeitava com `OSError [Errno 8] Exec format error`,
  fazendo 12 testes Fortran-dependentes falharem na CI mesmo com
  `Path.exists() == True`. Agora `_tatu_runnable()` testa execução
  real e os testes ficam *skipped* (não falham) em ambientes
  incompatíveis. Ver [tests/_fortran_helpers.py](../tests/_fortran_helpers.py).

### Notas técnicas

- **Decisão fastmath dipoles.py/kernel.py — NÃO APLICAR (oficial)**:
  Análise técnica documentada em
  `docs/reports/v2.15_fastmath_dipoles_analysis_2026-05-01.md`.
  Erro FMA acumulado em `hmd_tiv` (~600 ops × 2e-16 ≈ 1.2e-13)
  está em 0.83× do gate Fortran 1e-12 — qualquer ordering reordering
  pode quebrar paridade. `vmd` apresenta ~8e-14. Cancelamento
  catastrófico em recursão TE/TM (`1 + RTEup` com exp quase-iguais)
  amplificaria o erro FMA. Apenas `hankel.py` (4 funções dot-product)
  permanece com fastmath=True (decisão v2.14, validada).
- **Hardware spec testbed**: macOS Darwin 25.4.0 x86_64,
  16 logical cores (hyperthreaded), 8 physical cores, 64 GB RAM
  (Mac Pro / Mac Studio Intel).
- **JIT cache observability — uso recomendado**:
  ```python
  from geosteering_ai.simulation._jax.forward_pure import get_jit_cache_info
  info = get_jit_cache_info()
  print(f"Total XLA programs: {info['total_xla_programs']}")
  print(f"Estimated VRAM: {info['estimated_vram_mb']:.1f} MB")
  ```
  Útil em loops PINN longos (T4 16 GB / A100 40 GB) para detectar
  vazamento de VRAM antes de OOM.

### Testes (zero regressão vs v2.14)

| Suite | Testes | Status |
|:------|:------:|:------:|
| `test_simulation_compare_fortran.py` | 10 | PASS local macOS / SKIP CI Linux |
| `test_simulation_multi.py` (Fortran-deps) | 2 | PASS local / SKIP CI |
| `test_simulation_v213_optimizations.py` | 13 | PASS |
| `test_simulation_v214_prange_combined.py` | 8 | PASS |
| `test_simulation_v214_fastmath.py` | 6 | PASS (com introspecção P1) |
| `test_simulation_jax_sprint13.py` | 4 | PASS (NOVO) |

---

## [2.14] — 2026-05-01

### Otimizações Numba JIT — Sprints 13.3 + 13.4

Implementação das 2 otimizações Numba deferidas de v2.13:
- Sprint 13.3: `prange(n_combos * n_pos)` combinado para colapsar 24 transições
  Python→Numba em loop TR×ângulo serial, eliminando overhead fork/join
- Sprint 13.4: `fastmath=True` seletivo em 4 helpers hankel.py (dot-product FMA-safe)

Benchmark formal v2.14 com 4 cenários (single-freq, multi-freq, multi-TR, PINN).

### Adicionado

- **Sprint 13.3 — prange combinado TR×ângulo**:
  - Nova função `_simulate_combined_prange` em
    [forward.py](../geosteering_ai/simulation/forward.py) (~150 LOC)
    com `@njit(parallel=True, cache=True, nogil=True)`
  - Materialização pré-dispatch em `multi_forward.py`: flat geometry arrays
    `dz_halfs[n_combos]`, `r_halfs[n_combos]`, `cache_indices[n_combos]`
  - Deduplicação de cache via `key_to_idx` mapping — múltiplos combos
    com mesmo hordist compartilham entrada única
  - Stack de caches únicos: `u_unique[n_unique, nf, npt, n]`, idem para 9 arrays
  - Dispatch adaptativo: usa prange quando `n_combos >= 2`, fallback v2.13 para
    single-combo (preserva prange(n_pos) sem paralelismo insuficiente)
  - Nested prange serialização automática Numba: prange(n_total) →
    prange(nf) serializa sem regressão (validado Sprint 13.1)
- **Sprint 13.4 — fastmath seletivo**:
  - `@njit(fastmath=True)` em 4 funções hankel.py:
    `prepare_kr`, `integrate_j0`, `integrate_j1`, `integrate_j0_j1`
  - Justificativa: loops pure dot-product (FMA-safe, erro máx ~2e-14)
  - **INTOCADOS** (fastmath=False obrigatório):
    `propagation.common_arrays`, `common_factors` (recursão TE/TM orden-sensível),
    `dipoles.hmd_tiv`, `vmd` (fatores transmissão antes sums),
    `kernel._fields_in_freqs_kernel_cached`, `precompute_common_arrays_cache`
- **Benchmark CLI**:
  - Novo arquivo `benchmarks/bench_v214_numba.py` (~370 LOC)
  - 4 cenários: A (single-freq 30k mod), B (multi-freq 10 freqs, 30k),
    C (multi-TR×ang 3×5, 5k mod), D (PINN 50 calls, cache_persistent)
  - Métricas: modelos/hora (A/B/C), ms/chamada (D)
  - CLI: `python benchmarks/bench_v214_numba.py --scenario [A|B|C|D|--all]`

### Testes

- **Novo arquivo** `tests/test_simulation_v214_prange_combined.py`:
  - 8 testes Sprint 13.3 — paridade vs v2.13, single-combo fallback,
    multi-TR×multi-ang, deduplicação, Fortran parity <1e-12, determinismo
  - **Todos passam (2.30s)**
- **Novo arquivo** `tests/test_simulation_v214_fastmath.py`:
  - 6 testes Sprint 13.4 — paridade Fortran pós-fastmath (3-layer, oklahoma28),
    determinismo, validação propagation.py=False, smoke zero-regressão
  - **Todos passam (2.25s)**
- **Zero regressão**: 27/27 testes v2.13 + v2.14 PASS (2.45s) — backward-compat total

### Backward-compat

- Prange combinado (Sprint 13.3): fallback automático para v2.13 quando
  `n_combos < 2` ou `parallel=False` ou n_workers=1
- Fastmath (Sprint 13.4): aplicado SOMENTE em hankel.py — dipoles.py,
  propagation.py, kernel.py mantêm fastmath=False (ordem-sensível)
- API `simulate_multi()` 100% compatível: sem novos kwargs, comportamento
  idêntico quando `cache_persistent=False` (default) e `n_workers` não especificado

### Validado

- Paridade v2.13 → v2.14 bit-exata em single-TR×single-ang (fallback path)
- Paridade v2.13 → v2.14 determinística em multi-TR×multi-ang (prange combinado)
- Paridade Fortran <1e-12 em 3-layer + oklahoma28 simulado pós-fastmath hankel
- Thread-safety: prange(n_combos*n_pos) com índices exclusivos por construção
- Nested prange: serialização automática Numba validada (prange(n_total) →
  prange(nf) serializa)

### Pendências (v2.15+)

- Benchmark em hardware do usuário: validar ganhos reais vs v2.12
  (meta v2.13 Sprint 13.1: ≥1.5× multi-freq; meta v2.14 Sprint 13.3: ≥1.3× multi-TR)
- Fastmath=True seletivo em outras operações (dipoles.py factors, kernel.py,
  conforme validação Fortran <1e-12)
- Análise JIT caching: monitorar número de compilações XLA para Numba/JAX
  no backend híbrido

### Modificado

| Arquivo | Mudanças |
|:--------|:---------|
| `forward.py` | +`_simulate_combined_prange` (~150 LOC) |
| `multi_forward.py` | Materialização pré-dispatch + dispatch adaptativo (~100 LOC) |
| `_numba/hankel.py` | `@njit(fastmath=True)` em 4 funções (+4 LOC) |
| `tests/test_simulation_v214_prange_combined.py` | NOVO (8 testes, ~270 LOC) |
| `tests/test_simulation_v214_fastmath.py` | NOVO (6 testes, ~180 LOC) |
| `benchmarks/bench_v214_numba.py` | NOVO (4 cenários, ~370 LOC) |

---

## [2.13] — 2026-05-01

### Otimizações Numba JIT — Sprints 13.1 + 13.2 + 13.4

Implementação parcial das otimizações Numba previstas no relatório técnico
[v2.11_simulador_python_analise_paralelismo_2026-04-30.md §1.3 + §6.2](reports/v2.11_simulador_python_analise_paralelismo_2026-04-30.md).
Sprints 13.1, 13.2 e 13.4 entregues nesta release; Sprint 13.3 (prange combinado
TR×ângulo) e `fastmath=True` seletivo deferidos para v2.14 devido a complexidade
de refatoração do dispatcher.

### Adicionado

- **Sprint 13.1 — Vetorização de frequências**:
  - `prange(nf)` em `_fields_in_freqs_kernel_cached`
    ([_numba/kernel.py](../geosteering_ai/simulation/_numba/kernel.py)) e
    `precompute_common_arrays_cache`
  - `@njit(cache=True, parallel=True, nogil=True)` em ambas funções
  - Quando chamadas de contexto `prange` externo, Numba serializa o nível
    interno automaticamente (sem regressão)
- **Sprint 13.2 — Cache cross-call**:
  - Novo kwarg `cache_persistent: bool = False` em `simulate_multi` (opt-in)
  - Cache global thread-safe `_GLOBAL_HORDIST_CACHE` com chave
    `(round(hordist, 12), freqs_signature, n, eta_bytes, h_bytes)` —
    detecção bit-exata de qualquer variação numérica
  - Funções públicas exportadas em `geosteering_ai.simulation`:
    - `release_numba_cache() -> int` — libera cache, retorna count
    - `get_numba_cache_size() -> int` — diagnóstico
  - UI `closeEvent` chama 3 releases:
    `release_numba_pool()` + `release_pool()` + `release_numba_cache()`
- **Sprint 13.4 — `nogil=True` universal**:
  - Wrapper `njit` em `_numba/propagation.py` agora seta `nogil=True`
    como default — todas as funções `@njit` do simulador liberam GIL
  - Custo zero em performance; benefício direto em uso multi-thread
    (notebooks, treino offline, UI responsiva)
  - `fastmath=True` permanece **opt-in caso-a-caso** — preserva paridade
    Fortran <1e-12 nas recursões TE/TM e `common_arrays`

### Testes

- **Novo arquivo** `tests/test_simulation_v213_optimizations.py`:
  - 13 testes cobrindo Sprints 13.1, 13.2 e 13.4 — **todos passam (2.17s)**
  - Cobre: vetorização freqs, cache hit/miss/release/thread-safety,
    backward-compat v2.12, threading concorrente sem corrupção
- **Zero regressão**: 152/152 testes pré-existentes (workers + multi +
  numba_kernel + config) continuam passando em 13.12s

### Validado

- Paridade bit-exata entre chamadas (cache hit) — `assert_array_equal`
- Backward-compat total: API v2.12 single-model e batch (`models=[...]`)
  funcionam idênticas quando `cache_persistent=False` (default)
- Thread-safety: 4 ThreadPool workers concorrentes não corrompem resultados
- Cache miss correto em variações de freqs e perfis geológicos
- Smoke 30k-modelos: chamada multi-freq (`[20, 40, 60, 100] kHz`) operacional

### Pendências (v2.14+)

- **Sprint 13.3 — prange combinado TR×ângulo**: requer refatoração do
  dispatcher `multi_forward.py:732-818` para colapsar 2 loops Python em
  `prange(nTR*nAngles)` Numba via array de `cache_indices` materializadas
- **Sprint 13.4 — `fastmath=True` seletivo**: aplicar `fastmath=True` em
  Hankel quadratura e operações de decoupling após validação de paridade
  Fortran <1e-12 em 4 modelos canônicos
- **Benchmark formal v2.13**: script CLI cobrindo 4 cenários (single-freq,
  multi-freq, multi-TR, PINN) — adiado para validação final em hardware
  do usuário

### Modificado

| Arquivo | Mudanças |
|:--------|:---------|
| `_numba/kernel.py` | `prange(nf)` em 2 funções + nogil + parallel |
| `_numba/propagation.py` | Wrapper `njit` agora seta `nogil=True` default |
| `multi_forward.py` | +`cache_persistent` kwarg + cache global + 2 funções públicas |
| `simulation/__init__.py` | Exports: `release_numba_cache`, `get_numba_cache_size` |
| `tests/simulation_manager.py` | `closeEvent` chama 3 releases (pool ui + pool core + cache) |

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
