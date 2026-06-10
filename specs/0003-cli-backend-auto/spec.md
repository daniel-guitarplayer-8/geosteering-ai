---
Spec: 0003-cli-backend-auto
Titulo: CLI --backend auto (expõe o dispatcher de backend já existente)
Backlog-Code: (novo) E-cli-auto
Trilha-Dominante: E
Produtos: [CLI]
Converge-Em: n/a   # não toca cálculo EM; só seleção de backend
Status: implementado
Released-As: v2.53-v2.56
Constituicao: 1.0
Autor: Daniel Leal
Data: 2026-06-05
---

# Spec 0003 — CLI `--backend auto`

## 1. Contexto e Problema

A biblioteca expõe um **dispatcher de backend medido** — `simulate_batch(backend="auto")` em
[geosteering_ai/simulation/dispatch.py](../../geosteering_ai/simulation/dispatch.py) — cuja função
`_resolve_backend` (`dispatch.py:94`) decide entre JAX GPU e Numba CPU por uma árvore de decisão
real (n_models, GPU disponível, agrupabilidade por geometria) **com guard anti-OOM 80 GB**
(`dispatch.py:313`). Essa capacidade já é **testada e segura** (cai para Numba sem GPU).

Porém a **CLI não a expõe**: [geosteering_ai/cli/_main.py:228](../../geosteering_ai/cli/_main.py)
trava em `choices=["numba","jax"]` com `default="numba"`. O usuário precisa escolher o backend
manualmente, contrariando o objetivo do produto (a CLI deve, por padrão, estar associada aos
**dois** simuladores Numba JIT **e** JAX GPU, deixando a escolha para a árvore medida).

| Estado | Onde | Evidência |
|:--|:--|:--|
| ausente | `cli/_main.py:228` | `choices=["numba","jax"]` — sem `auto` |
| pronto (não exposto) | `simulation/dispatch.py:94` | `_resolve_backend(...)` com árvore medida |
| pronto (não exposto) | `simulation/dispatch.py:313` | guard anti-OOM 80 GB |
| pré-voo de segurança | `cli/_exec.py` | `resolve_backend_preflight` (anti-crash TLS, v2.55) |

**Risco a respeitar (correção C6 da revisão crítica):** mudar o **default** de `numba` para `auto`
altera o comportamento observável de scripts/CI existentes que assumiam Numba. Por isso esta spec
adiciona `auto` como **opção** agora, mas **mantém `default="numba"` por 1 minor** com
`DeprecationWarning` anunciando a futura mudança de default (PEP 387, Princípio IX).

## 2. User Stories

| ID | Como… | Quero… | Para… | Prioridade |
|:--|:--|:--|:--|:--:|
| US-1 | Pesquisador DL | rodar `geosteering-cli simulate --backend auto` | deixar a CLI escolher o backend ótimo (JAX se GPU+agrupável, senão Numba) sem decidir hardware | Must |
| US-2 | Operador de CI | rodar a CLI sem `--backend` e obter comportamento estável | não quebrar pipelines que assumem Numba hoje | Must |
| US-3 | Pesquisador DL | ser avisado de que `auto` será o futuro default | migrar scripts a tempo | Should |

## 3. Requisitos Funcionais (RF)

| ID | Requisito | MoSCoW | Cobertura atual |
|:--|:--|:--:|:--|
| RF-1 | `--backend` aceita `auto` em `simulate` e `benchmark` | Must | NOVO |
| RF-2 | `auto` resolve via `dispatch._resolve_backend` (reusa árvore + guard OOM + pré-voo TLS) | Must | parcial (existe na lib) |
| RF-3 | `default` permanece `numba`; usar `default` sem `--backend` emite `DeprecationWarning` anunciando mudança futura para `auto` | Must | NOVO |
| RF-4 | A tabela/saída JSON reporta o backend **efetivamente resolvido** quando `auto` (não a string `auto`) | Should | NOVO |

### RF-1 — Critérios de Aceite
- [ ] **AC-1.1**: `geosteering-cli simulate --models 4 --backend auto` retorna exit code 0.
- [ ] **AC-1.2**: `geosteering-cli benchmark --scenario A --n 8 --backend auto` retorna exit code 0.
- [ ] **AC-1.3**: `geosteering-cli simulate --backend invalido` retorna exit code 2 (argparse rejeita).

### RF-2 — Critérios de Aceite
- [ ] **AC-2.1**: Em ambiente **sem GPU** (CI/CPU), `--backend auto` resolve para `numba` e a saída
  reporta `backend=numba` (não `auto`).
- [ ] **AC-2.2** (revisado pós-T00): A resolução `auto` produz a **mesma decisão** de backend que
  `dispatch._resolve_backend` para entradas equivalentes (teste de consistência mockando
  `_jax_gpu_available`), reusando as constantes/primitivas do dispatcher (`_GROUPABLE_RATIO_MAX`,
  `_N_MODELS_GPU_THRESHOLD`, `_jax_gpu_available`). **Não** chama o wrapper `_resolve_backend` — ele
  importa `_jax.multi_forward` e sonda a GPU antes da agrupabilidade (**risco de crash TLS**, v2.55).
  A CLI aplica a árvore em **ordem TLS-safe**: disqualificadores jax-free primeiro
  (`_count_geometry_groups` NumPy + limiar de modelos), sonda de GPU **por último**.

### RF-3 — Critérios de Aceite
- [ ] **AC-3.1**: `geosteering-cli simulate --models 4` (sem `--backend`) emite um `DeprecationWarning`
  cuja mensagem contém "auto" e "default" (capturável via `pytest.warns` ou stderr).
- [ ] **AC-3.2**: `geosteering-cli simulate --models 4 --backend numba` **não** emite o warning
  (escolha explícita silencia o aviso).

### RF-4 — Critérios de Aceite
- [ ] **AC-4.1**: `geosteering-cli simulate --models 4 --backend auto --json` produz JSON cujo campo
  de backend é `numba` ou `jax` (nunca `auto`).

## 4. Requisitos Não-Funcionais (RNF)

| ID | Categoria | Requisito | Métrica/Limite |
|:--|:--|:--|:--|
| RNF-1 | Paridade | não toca cálculo EM | **N/A** (Princípio I) — declarado explicitamente |
| RNF-2 | Compatibilidade | comportamento default inalterado (Numba) | scripts/CI existentes passam sem mudança |
| RNF-3 | Doc | flag e help atualizados | D5/D7; help descreve `auto` e o aviso de deprecação |
| RNF-4 | Plataforma | Python | 3.13 |
| RNF-5 | Cobertura | `cli/` | sem regressão; novos AC cobertos por teste |
| RNF-6 | UX | mensagem do `DeprecationWarning` | clara, acionável, com versão-alvo da mudança |

## 5. Escopo

### IN
- Adicionar `"auto"` a `choices` de `--backend` (compartilhado em `_add_common_backend_args`,
  `cli/_main.py:220`).
- Rota de resolução: quando `backend == "auto"`, delegar a `dispatch._resolve_backend` (via o
  caminho de execução em `cli/_exec.py`, reusando `resolve_backend_preflight`).
- `DeprecationWarning` quando o usuário **não** passa `--backend` (default implícito).
- Saída (tabela + JSON) reporta o backend **resolvido**.
- Testes cobrindo AC-1..AC-4; atualização do `--help`.

### OUT (fora desta spec)
- Mudar o **default** para `auto` (fica para o minor seguinte, após o período de `DeprecationWarning`) → backlog.
- Comandos `train`/`infer` na CLI → spec **0010**.
- Subcomando `generate-dataset` persistente → backlog.

## 6. [NEEDS CLARIFICATION] — RESOLVIDO (2026-06-05)
- [x] ~~Versão-alvo no `DeprecationWarning`~~ → **RESOLVIDO: `v2.57.0`** (próxima minor após o bump
  `2.56.0` da spec 0002). A mensagem cita "o default de `--backend` mudará de `numba` para `auto`
  em v2.57.0". Se a 0002 ainda não tiver landado, o número é mantido (é o alvo planejado).
- [x] ~~`auto` em `geosteering-warmup`?~~ → **RESOLVIDO: NÃO** (fora de escopo). O `geosteering-warmup`
  **já possui** seu próprio `--auto` (usado no CI: `geosteering-warmup --verbose --auto`), que é um
  mecanismo distinto (decide aquecer JAX/Numba). Não há conflito e não há mudança no warmup nesta spec.

**GATE-S: PASSOU** — 0 marcadores abertos.

## 7. Dependências e Riscos de Escopo

| Tipo | Item | Impacto |
|:--|:--|:--|
| Dep | spec 0001 (SDD bootstrap) | processo/Constituição vigente |
| Dep (soft) | spec 0002 (SSoT versão) | para citar a versão-alvo correta no warning |
| Risco | mudança de comportamento default | mitigado: default permanece `numba`; só `auto` é adicionado |
| Risco | crash TLS do JAX em geometria não-agrupável | mitigado: reusa `resolve_backend_preflight` (v2.55) |

## 8. Critério de Pronto da Spec (GATE-S checklist)
- [ ] 0 marcadores `[NEEDS CLARIFICATION]` abertos (2 pendentes → resolver com Daniel)
- [x] Todo RF tem ≥1 AC testável
- [x] Escopo IN/OUT explícito
- [x] `Produtos` e `Converge-Em` declarados
- [x] `Backlog-Code` proposto (`E-cli-auto`) — adicionar a `docs/ROADMAP.md §0` na promoção
- [x] Nenhum princípio da CONSTITUTION violado (não toca física; mantém compat; logging/print = exceção CLI)
