# JAX GPU vs Numba — papel estratégico no Geosteering AI

| Campo | Valor |
|:------|:------|
| **Tipo** | Análise estratégica/arquitetural (reference) |
| **Data** | 2026-06-25 |
| **Autor** | Daniel Leal (análise: Claude Opus 4.8 1M, investigação read-only de 3 agentes Explore) |
| **Pergunta** | Dado que o JAX GPU é mais rápido apenas para geometria FIXA, qual é sua principal vantagem sobre o Numba no projeto? PINNs? treino offline? geosteering realtime? |
| **Método** | Grep/leitura exaustiva de `geosteering_ai/{simulation/_jax,_jacobian,losses,models,training,inference,data}/` + `docs/{ARCHITECTURE_v2,ROADMAP}.md` + ROADMAP |

---

## TL;DR

A vantagem principal do JAX **não é velocidade de forward** (fora do ponto-doce — **warm + geometria determinística + K pequeno/lotes grandes** — o Numba ganha; ver qualificador abaixo) — são **duas capacidades distintas**, e **nenhuma é o realtime**:

| # | Vantagem do JAX | Numba consegue? | Status |
|:-:|:--|:--|:--|
| **A** | **Diferenciabilidade (autodiff)** → Jacobiano **exato** ∂H/∂ρ via `jax.jacfwd` | ❌ Não (só diferenças finitas) | ✅ Construído/testado · ⚠️ **sem consumidor ainda** |
| **B** | **Throughput batched em GPU** p/ **geração offline de datasets** de treino | △ Em tese (muitos cores) | ✅ **EM PRODUÇÃO** |

Hipóteses do usuário: **PINNs → não** (são TensorFlow, não JAX); **treino offline → SIM, indiretamente** (JAX gera os *datasets*, não treina); **realtime → não** (só a rede Keras, sem simulador).

A vantagem **mais concreta hoje** é a **(B)**; a **mais estratégica para o futuro** é a **(A)**.

> **⚠️ Qualificador (errata 2026-06-26) — "geometria fixa" é NECESSÁRIA, não SUFICIENTE.**
> A premissa "o JAX é rápido para geometria FIXA" foi refinada empiricamente: no SM com
> `n_layers_fixed=15` o JAX mediu **349 s vs 73.8 s do Numba** — *mais lento*, mesmo com
> geometria fixa. Causa-raiz: `rng_seed=None` → esp (espessuras) **aleatório a cada run**
> → o cache XLA de disco **erra** → recompila ~216 programas (K templates × buckets × dims
> de chunk). A condição real para o JAX GPU vencer no forward é:
> **(1) warm** (cache de disco quente) · **(2) geometria DETERMINÍSTICA entre runs** ·
> **(3) K pequeno e/ou lotes grandes** que amortizem as compilações a frio. A **geração de
> dataset** (§2) satisfaz tudo (seed fixo → determinística; reexecuções quentes; batches
> enormes amortizam K≈32) — por isso paga. O **fix #1+#2** (PR #68: geometria
> determinística no colapso de templates + chunk balanceado) **faz o SM de geometria fixa
> satisfazer a condição também** — o 2º run cai de 349 s para **~30 s**. Onde a condição
> falha (ragged a frio, K grande sem amortização), o **Numba ganha**.

---

## 1. Vantagem A — Diferenciabilidade (o "moat" que o Numba não tem)

O Numba/Fortran computam H(ρ); para ∂H/∂ρ só conseguem **diferenças finitas** (4·n_layers forwards, erro O(δ²)~1e-4, lento). O JAX diferencia o forward **automaticamente e exato**.

**Evidência:**

- `geosteering_ai/simulation/_jacobian.py:597` — única chamada de autodiff "viva": `J_h, J_v = jax.jacfwd(_fwd, argnums=(0,1))(rho_h, rho_v)` → ∂H/∂ρₕ e ∂H/∂ρᵥ, shape `(n_pos, nf, 9, n_layers)` complex128.
- `geosteering_ai/simulation/_jax/forward_pure.py::forward_pure_jax` — forward **100% puro-JAX** (sem `pure_callback`) → traçável por `jacfwd`/`grad`. Módulos `_jax/{hankel,rotation,dipoles_native,propagation}.py` escritos diferenciáveis.
- **Acurácia:** jacfwd ≈ 1e-10–1e-14 (autodiff) vs FD ≈ 1e-4 (truncamento); vs TIV analítico `rtol < 1e-4`.

**Custo (CPU, do `sprint_5_1b`):**

| Backend (Jacobiano) | oklahoma_3 | oklahoma_28 (28 cam.) | Método |
|:--|--:|--:|:--|
| Numba FD | 3.6 ms | 7.2 ms (112 forwards) | diferenças finitas |
| JAX `jacfwd` | 17.1 ms | 17.4 ms | autodiff exato |

→ Na **CPU** o Numba FD é mais rápido (4.7×); o JAX **só ganha na GPU** (fusão XLA) **e** em **acurácia** (exato). O ponto não é velocidade — é que o gradiente exato habilita uma classe de métodos: **inversão determinística** (Gauss-Newton/Occam/LM), **quantificação de incerteza** (JᵀJ), **análise de sensibilidade**.

> ⚠️ **Honestidade:** hoje o Jacobiano é **infra construída, testada e pronta — SEM consumidor em produção** (não há módulo de inversão; `grad`-de-loss = 0). É o trilho esperando o trem (backlog P3 `C-inversion-alt`). Logo a vantagem A é **potencial/estratégica, não realizada ainda**.

---

## 2. Vantagem B — Throughput GPU para geração OFFLINE de datasets (a vantagem REAL hoje)

É **aqui que o JAX GPU efetivamente paga, em produção, hoje**. O pipeline DL precisa de **datasets grandes** (dezenas de milhares de modelos) p/ treinar a rede de inversão (TF/Keras). É o JAX que os gera rápido.

**Evidência:**

- `data/synthetic_generator.py:386` → `simulate_batch(backend="auto")` → `dispatch.py` roteia p/ **JAX bucketed** quando GPU + `n_models ≥ 32` + geometria **agrupável**.
- **Medido (Sprint C, A6000/A100):** **30.000 modelos em 491 s** via JAX batched+grouped (`benchmarks/_sprintc_phase_a_gen.py`); **~1.89× Numba** @ n=64.
- ROADMAP: **`A-jax-gpu-data-gen` (DONE v2.47)** + **`C-surrogate-train` (DONE v2.48)**.

**Reconciliação com a regressão JAX-lento (Turn 8):** a geração de dataset usa **n_layers FIXO** (tipicamente 5) + geometria **`templates`** + **seed FIXO** → esp **determinística** → o cache XLA acerta entre reexecuções, e os lotes enormes amortizam as K≈32 compilações → bucketed JAX = o ponto-doce (GPU saturada). É o caso "geometria fixa" **com a condição completa satisfeita** (warm + determinística + lote grande) onde o JAX vence. O SM interativo a frio com esp **aleatória** (`rng_seed=None`) era o oposto — por isso lá o Numba ganhava e o JAX cold-compilava a CADA run. O **fix #1 (PR #68)** fixa a geometria do colapso por uma semente estável da config → o SM de geometria fixa passa a satisfazer a condição (2º run ~30 s). O **ragged** (3–31, muitos grupos) permanece caro a frio → Numba segue o default certo lá. **Mesmo simulador, regimes opostos — o que separa não é só "fixa", é warm + determinística + K amortizável.**

---

## 3. As hipóteses do usuário, uma a uma

### "Seria nas PINNs?" → **Não (as PINNs são TensorFlow, não JAX)**

- `losses/pinns.py` (106 KB): **8 cenários PINN** (oracle, surrogate, maxwell, smoothness, skin_depth, continuity, variational, self_adaptive) — **todos TensorFlow**.
- O cenário "surrogate" usa forward **diferenciável** na loss — mas é **forward analítico TF** (`_analytical_forward_1d_complex`) **ou** rede **SurrogateNet (TCN, Keras/TF)** — **não** o simulador JAX.
- **Por quê não JAX:** os autodiffs de TF e JAX **não se conversam**; o forward na loss precisa do mesmo framework da rede (Keras→TF).
- 🔭 **Oportunidade futura (não-explorada):** trazer a **física JAX exata** p/ a loss PINN (via `jax2tf`/custom op), no lugar do forward analítico-TF — uso de alto valor do JAX.

### "No treino de inferência offline?" → **SIM, indiretamente (gera os DATASETS, não treina)**

- O **treino** da rede é **TensorFlow/Keras exclusivo** (`training/loop.py`: "PyTorch PROIBIDO"; 26 losses TF; zero JAX no treino).
- O JAX entra **a montante**: gera os **dados sintéticos** que alimentam o treino (Vantagem B). É o elo onde o JAX serve o "treino offline".
- Ruído é **on-the-fly** no treino (TF, `data/pipeline.py`); o forward **limpo** vem do `SyntheticDataGenerator` (JAX/auto).

### "Geosteering realtime?" → **Não (realtime = só a rede neural)**

- `inference/realtime.py` + `inference/pipeline.py`: roda **só o modelo Keras** (`raw 22-col → FV → GS → scale → model.predict()`), janela deslizante (`deque`, seq ≤ 600). **O simulador NÃO é invocado.**
- Realtime exige **latência determinística baixa** — oposto do perfil do JAX (cold-compile na 1ª forma). JAX é inadequado lá e corretamente não é usado.

---

## 4. Mapa completo — onde o JAX (não) ajuda

| Subsistema | Backend | JAX GPU paga? | Motivo |
|:--|:--|:--|:--|
| **Geração offline de dataset** (n≥32, agrupável) | JAX bucketed/auto | ✅ **SIM** (produção, 1.89×) | geometria fixa **+ warm + determinística + lote grande** → cache acerta, GPU saturada |
| **Jacobiano / sensibilidade** | JAX `jacfwd` | ✅ **SIM (qualitativo)** | autodiff exato (Numba não consegue); sem consumidor ainda |
| **Treino da rede (PINN/inversão)** | TensorFlow/Keras | ❌ Não | autodiff TF≠JAX; rede é Keras |
| **Geosteering realtime** | só Keras NN | ❌ Não | sem simulador; latência baixa |
| **SM interativo / CLI ad-hoc (ragged)** | Numba (default) | ❌ Não | ragged → 29 grupos → cold-compile |
| **Referência de paridade** | Fortran `tatu.x` | — | ground-truth <1e-12 |

**Divisão de trabalho (a "grande arquitetura"):**

- **TensorFlow/Keras** = cérebro (treino, losses, PINNs, inferência).
- **JAX** = física **diferenciável + batched-GPU** (geração de dataset + Jacobiano).
- **Numba** = física **rápida na CPU, indiferente à geometria** (SM interativo, ragged, CLI, batches pequenos/médios).
- **Fortran** = referência de paridade sagrada (<1e-12).

---

## 5. Conclusão estratégica

A vantagem principal do JAX sobre o Numba no Geosteering AI é, em ordem de valor:

1. **(Realizada, hoje) Throughput batched em GPU para geração offline de datasets de treino** — geometria fixa **determinística (seed fixo) + warm + grandes lotes** = o ponto-doce; ~1.89× Numba, 30k/491 s. Sustenta o treino da rede de inversão.
2. **(Estratégica, futura) Diferenciabilidade exata (Jacobiano ∂H/∂ρ via autodiff)** — a única coisa que o Numba/Fortran **fundamentalmente não fazem**. Trilho para **inversão determinística, UQ e física-na-loss (PINN com forward JAX real)**. Construída/testada, **aguardando o consumidor**.

**Não é:** PINNs (são TF), nem realtime (é só a rede). A confusão natural ("JAX = GPU = rápido sempre") se desfaz: o JAX rende **onde há geometria fixa determinística + warm + lotes grandes (datasets; e o SM fixo após o fix #1+#2)** ou **onde se precisa de gradiente exato (Jacobiano)** — não no forward interativo ragged-a-frio nem no realtime.

> **Implicação prática:** o JAX está corretamente posicionado. O maior **ganho futuro não-explorado** seria **conectar o Jacobiano/forward-JAX a um módulo de inversão determinística e/ou à loss PINN** (substituindo o forward analítico-TF pela física exata) — aí o JAX deixaria de ser "infra pronta" e viraria diferencial competitivo realizado.

---

## Apêndice — referências de código (file:line)

| Item | Local |
|:--|:--|
| autodiff `jax.jacfwd` (∂H/∂ρ) | `geosteering_ai/simulation/_jacobian.py:597` |
| forward diferenciável puro-JAX | `geosteering_ai/simulation/_jax/forward_pure.py::forward_pure_jax` |
| FD Numba (4·n_layers forwards) | `geosteering_ai/simulation/_jacobian.py:236` (`compute_jacobian_fd_numba`) |
| dispatcher de backend | `geosteering_ai/simulation/dispatch.py:94-180` (`simulate_batch`) |
| geração de dataset (JAX/auto) | `geosteering_ai/data/synthetic_generator.py:386` |
| 8 cenários PINN (TF) | `geosteering_ai/losses/pinns.py` |
| forward surrogate (TF analítico/neural) | `losses/pinns.py::_analytical_forward_1d_complex` · `models/surrogate.py` |
| treino TF/Keras exclusivo | `geosteering_ai/training/loop.py` |
| inferência realtime (só Keras) | `geosteering_ai/inference/{realtime,pipeline}.py` |
| geração 30k via JAX (491 s A6000) | `benchmarks/_sprintc_phase_a_gen.py` |
| ROADMAP JAX | `A-jax-gpu-data-gen` (v2.47) · `C-surrogate-train` (v2.48) · `C-inversion-alt` (P3) |

*Relacionado: a regressão JAX-lento do SM e o fix #1+#2 (geometria determinística no colapso de templates + chunk balanceado; warmup config-derivado adiado) — ver PR #68. Errata do qualificador (warm + determinística + K amortizável) em 2026-06-26.*
