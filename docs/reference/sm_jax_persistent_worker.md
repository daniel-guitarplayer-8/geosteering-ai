# Simulador JAX GPU no SM MVVM — worker persistente + como obter throughput > Numba

> **Resposta ao "por que o JAX é mais lento no SM que no CLI" + guia de uso.**
> Investigação 2026-06-17 (RTX A6000). Ver `docs/reports/v2.58_sm_jax_coldstart_*`
> (cold-start XLA) e o report do worker persistente.

---

## 1. Por que o CLI era mais rápido que o SM (causa-raiz)

| | CLI | SM (antes) |
|:--|:--|:--|
| Processo JAX | **1 processo de longa duração** | **subprocesso novo a CADA run** |
| `CUDA/cuDNN init` | pago **1×** | re-pago **todo run** (~10-15 s) |
| Cache JIT em-processo | quente entre `--repeat` | descartado todo run |
| Reload do cache XLA de disco | 1× | **todo run** |

O SM roda o JAX num **subprocesso isolado** (TLS-safe: a GUI nunca importa JAX, senão o
init de CUDA numa `QThread` estoura o TLS / `_dl_allocate_tls_init`). Até a v2.58 esse
subprocesso era **efêmero** (criado e destruído por run em `gui/services/base.py::_pool_run`),
então o SM re-pagava **~17 s** de overhead por execução. O CLI roda **in-process** → paga
uma vez. **Medido (1000 modelos):** SM efêmero ~29 s/run vs CLI in-process ~12 s/run.

> ⚠️ O cache XLA de disco (`~/.cache/geosteering/jax_compilation_cache`) só evita o
> **compile** XLA único (~145 s na 1ª vez de cada forma); ele **não** evita o init-CUDA +
> reload por-processo. Só um **worker persistente** evita.

## 2. O worker persistente (fix)

A partir da v2.58 o subprocesso JAX é **persistente**: um único filho `spawn` fica vivo e
é reutilizado entre runs (espelha o pool persistente do Numba em `simulation/_workers.py`).

- **1ª sim** (jax/auto) cria o worker (init CUDA + compile da forma, se inédita no cache).
- **2ª+ sim** reusa o worker quente → **~12 s** (medido), **~3× mais rápido que o Numba (36.8 s)**.
- **Teardown** automático (`release_jax_pool`) ao fechar a janela (`closeEvent`), no
  `aboutToQuit` e no `atexit` → **sem processo GPU órfão** (contexto CUDA + VRAM liberados).
- **Self-heal**: se o filho morre (OOM/crash CUDA/segfault), o pool é descartado e a
  próxima sim recria um filho fresco (1 retry automático; depois, erro acionável).
- **TLS-safe preservado**: só o filho importa JAX; o processo da GUI nunca.

## 3. Como obter throughput > Numba JIT no SM (guia prático)

1. **Backend** = `jax` ou `auto` (em "Backend de paralelismo").
2. Aceite o **cold-start** da 1ª sim de cada *forma* nova (nº de modelos × geometrias ×
   nº de camadas × nº de posições): ela **compila** os kernels XLA (~min, custo ÚNICO,
   cacheado em disco). As seguintes reusam o cache (~12-30 s) e batem o Numba.
3. **Sature a GPU** (a GPU só ganha do Numba com trabalho suficiente):
   - **N grande** (ex.: ≥ 256-1000 modelos) — abaixo de 32 modelos/grupo o dispatcher cai
     p/ Numba de propósito (GPU subocupada).
   - **`n_layers fixo`** marcado → 1 grupo único (máxima ocupação, 1 compile).
   - **`Geometrias (K)` pequeno** (knob da v2.58) → menos compilações no cold-start
     (K=1 ⇒ 1 compile). ρₕ/ρᵥ/λ continuam variando por modelo.
4. **Mantenha o app aberto** entre simulações: o worker persistente fica quente → as
   sims seguintes são as mais rápidas. Fechar/reabrir paga o init de novo.
5. **Pré-aqueça** (opcional) para tirar o cold-start da 1ª sim:
   - **Boot warmup**: ligue `jax_boot_warmup` em **Preferências** → ao abrir o SM, o worker
     é aquecido em background (status "Aquecendo JAX GPU…"). Opt-in (desligado por padrão
     p/ não gastar GPU de quem só usa Numba).
   - **CLI**: `geosteering-warmup --jax` popula o cache de disco 1× — compartilhado com
     o SM. Para aquecer a GPU use **`export JAX_PLATFORMS=cuda,cpu`** (cuda p/ o compute
     **+ cpu** p/ o `jax.pure_callback` do kernel; só `cuda` QUEBRA o callback —
     "failed to find a local CPU device"). O SM em si NÃO seta `JAX_PLATFORMS` (deixa o
     JAX auto-detectar cuda+cpu), então o worker persistente já roda na GPU corretamente.

> **Caveat de cobertura (honestidade):** o warmup usa **uma** geometria homogênea, então
> pré-compila **1** das K geometrias-template. O ganho do init-CUDA do worker persistente
> vale p/ **todas** as formas; já o **compile** XLA por-forma só é pré-aquecido p/ a forma
> canônica — geometrias estocásticas não-canônicas ainda compilam na 1ª sim.

## 4. Regra de bolso

| Cenário | Backend recomendado |
|:--|:--|
| 1 sim avulsa, poucos modelos (< ~256) | **Numba** (sem cold-start; GPU subocupada) |
| Muitas sims na sessão, N grande | **JAX** (worker persistente: 2ª+ sim ~12 s) |
| Lote único enorme, máxima ocupação | **JAX** + `n_layers fixo` + K pequeno + boot warmup |

Ver também: `geosteering_ai/gui/services/base.py` (pool persistente),
`docs/reports/v2.58_sm_jax_coldstart_investigacao_2026-06-17.md`.
