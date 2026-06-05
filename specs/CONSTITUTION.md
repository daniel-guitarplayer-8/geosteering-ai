# Constituição do Geosteering AI — Princípios Invioláveis do SDD

| Campo | Valor |
|:--|:--|
| **Versão da Constituição** | 1.0 |
| **Status** | Ativa |
| **Data** | 2026-06-05 |
| **Autoridade** | Daniel Leal |
| **Substitui** | — |
| **Mudança requer** | ADR formal em `docs/decisions/` + bump desta versão |

> Estes princípios têm precedência sobre QUALQUER spec, plan, task ou código.
> Uma spec que viole um princípio é REJEITADA no **GATE-S**. Um plan que viole é
> REJEITADO no **GATE-P**. Código que viole é BLOQUEADO por hook no **GATE-I**.

---

## Princípio I — Paridade Física Sagrada e os TRÊS Regimes de Tolerância

Qualquer alteração no caminho de cálculo EM (`simulation/_numba/`, `simulation/_jax/`,
`multi_forward.py`, `forward.py`) DEVE preservar a paridade no regime aplicável. Existem
**três regimes de tolerância distintos** — confundi-los é causa comum de rejeição indevida:

| Regime | Escopo | dtype | Tolerância | Gate |
|:--|:--|:--|:--:|:--|
| **Física Fortran** | forward EM vs oráculo `tatu.x` | `complex128` | **`max\|diff\| < 1e-12`** | `run-fortran-parity.sh` (pre-commit) + CI 10/10 |
| **JAX vs Numba** | backend GPU vs CPU | `complex64`/`complex128` | **`< 1e-10`** | `test_simulation_jax_*` |
| **Inferência treinada** | golden-test de modelo (saída) | `float32` | **`rtol < 1e-6`** | `test_golden_models.py` (G1) |

- **NUNCA** rejeite um golden-test de **inferência** por "não atingir 1e-12": ele opera em
  `float32`/cuDNN; o regime correto é `rtol < 1e-6`. A errata `<1e-12` é **só** da física.
- **C1 (CI):** o gate de paridade Fortran DEVE usar um binário compilado **sem** `-ffast-math`
  (alvo de Makefile `validation_portable` = `-O0 -march=x86-64-v2 -fno-fast-math -fsignaling-nans`).
  O alvo `portable` atual mantém `-ffast-math` e **não** serve como oráculo. Registrar em ADR.
- **Não-negociável:** nenhuma otimização de performance justifica quebrar paridade física.

## Princípio II — Errata Física Imutável

Os valores validados por `PipelineConfig.__post_init__()` são CONSTANTES:
- `FREQUENCY_HZ` default 20000.0 (range 100–1e6); `SPACING_METERS` default 1.0 (0.1–10.0);
  `SEQUENCE_LENGTH` default 600 (10–100000).
- `TARGET_SCALING == "log10"` (NUNCA "log"); `INPUT_FEATURES == [1,4,5,20,21]`;
  `OUTPUT_TARGETS == [2,3]`; `eps = 1e-12` (float32).
- Decoupling L=1.0: `ACp ≈ -0.079577` (planar); `ACx ≈ +0.159155` (axial); `ACx = -2·ACp`.
- **Correção registrada:** multi-frequência JÁ É suportada via `config.frequencies_hz` (array).
  Specs NÃO devem listar isto como gap (o gap é só UI + default escalar `frequency_hz`).
- **Gate:** hooks `reinject-errata.sh` + `validate-physics.sh`.

## Princípio III — TensorFlow/Keras Exclusivo em Produção

PyTorch é PROIBIDO em `geosteering_ai/{models,losses,training,inference,evaluation,data,
simulation,visualization,utils}/`. Permitido APENAS via `adapters/pytorch_adapter.py` isolado
(acesso por `get_adapter("pytorch")`). Runtime de produção congelado: **ONNX Runtime** (batch)
+ **TFLite/LiteRT INT8** (realtime). **Gate:** `validate-no-pytorch.sh`.

## Princípio IV — Python 3.13 Exclusivo

Todo código roda em Python 3.13 (conda env `Geosteering_AI` p/ GPU; venv `~/Geosteering_AI_venv`
p/ CPU). Python 3.14+ PROIBIDO (sem wheels PyQt6/JAX/SciPy). GPU dev é local RTX A6000
(Colab descontinuado v2.44).

## Princípio V — Config como Parâmetro (nunca globals)

Toda função/classe recebe `config: PipelineConfig` (ou `SimulationConfig`). PROIBIDO
`globals().get(...)`. Componentes via Factory/Registry (`ModelRegistry`, `LossFactory`,
`build_callbacks`).

## Princípio VI — Logging, nunca print

PROIBIDO `print()` em `geosteering_ai/` — usar `logging` (logger de `utils/`). EXCEÇÃO ÚNICA:
stdout intencional de comandos da CLI (contrato observável). **Gate:** `check-anti-patterns-precommit.sh`.

## Princípio VII — Padrões de Documentação D1–D14

Todo módulo novo/editado segue D1–D14 (mega-header D1, docstrings Google D5/D6, diagramas ASCII
D3, logging D9, `__all__` semântico D8, etc.) na profundidade do legado C28. O código é um
documento de referência executável, lido por geofísicos e engenheiros de poço.

## Princípio VIII — PT-BR com Acentuação Correta

Todo `.md`, docstring e comentário em PT-BR DEVE ter acentuação correta ("implementação", não
"implementacao"). Variáveis em inglês. **Gate:** `check-ptbr-accentuation.sh`.

## Princípio IX — SSoT do Planejamento (ADR-0001)

O SDD NÃO cria fonte concorrente de verdade. Backlog futuro vive SÓ em `docs/ROADMAP.md §0`.
Specs REFERENCIAM o `Backlog-Code`; a versão `vX.Y` é atribuída só no commit da sprint (R2).
Sem sub-letras (R3). A numeração `NNNN-slug` das specs é própria e desacoplada de `vX.Y`.

## Princípio X — MVVM para toda GUI

Studio e Simulation Manager usam **MVVM** sobre a fundação `geosteering_ai/gui/`. O monólito
`simulation_manager.py` NÃO é o esqueleto — a infra modular `sm_*.py` é. View (Qt) ⊥ ViewModel
(estado/comandos, **sem import Qt**, testável com pytest puro) ⊥ Model (biblioteca). Threading:
worker-object + `moveToThread` (I/O); ProcessPool efêmero (Numba); `deepcopy` em sinais mutáveis.

## Princípio XI — 4 Produtos sobre Fundação Compartilhada; Física Não-Duplicada

P1 Lib/API, P2 CLI, P3 Studio, P4 SM são produtos distintos que compartilham `simulation/`
(backend) + `gui/` (infra Qt) + biblioteca. **SM = `gui/`+`simulation/` apenas** (Numba+JAX+Fortran).
**Studio = `gui/` + biblioteca inteira** (MVVM próprio). Toda spec que toque cálculo EM declara
`Converge-Em: multi_forward.py`; o GATE-P bloqueia se a física for reimplementada/duplicada.

## Princípio XII — Spec antes de Código; Gates antes de Merge

Nenhuma linha de produção sem `spec.md` (GATE-S) → `plan.md` (GATE-P) → `tasks.md` (GATE-T).
Nenhum merge sem: pytest verde (fast + GPU quando aplicável), paridade física no regime correto,
cobertura não-regredida no módulo tocado (cov-diff ≥80% em código novo de GUI), reviewers
especializados acionados (physics-reviewer p/ Trilha A; perf-baseline p/ Trilha B; code-reviewer +
security-auditor para PRs sensíveis). Toda reprodutibilidade exige `ScalerRegistry` versionado
junto ao modelo (o scaler fit-on-clean é parte inseparável do contrato dataset→modelo→inferência).

---

*Mudar qualquer princípio exige ADR em `docs/decisions/` + bump da versão desta Constituição.*
