# `specs/` — Spec-Driven Development do Geosteering AI

**SDD = especificar ANTES de codar.** Fluxo: **Constitution → Specify → Plan → Tasks → Implement → Verify**, com um gate de verificação entre cada fase.

```
 ① Constitution  ──►  ② Specify  ──►  ③ Plan  ──►  ④ Tasks  ──►  ⑤ Implement  ──►  ⑥ Verify
 (uma vez,            (WHAT/WHY)      (HOW/tech)    (decompõe)    (código+commit)   (gates+merge)
  versionada)            │               │             │              │                 │
                      GATE-S          GATE-P        GATE-T        GATE-I            GATE-V
                   (0 NEEDS_         (constituição  (cada task    (hooks pre-commit: (CI verde +
                    CLARIFICATION,    respeitada;    atômica,      paridade, errata,  paridade +
                    AC testáveis)     ADR p/ design  testável,     no-pytorch, PT-BR, cobertura +
                                      irreversível)  ordenada)     anti-patterns)     reviewers)
```

## Onde o SDD se encaixa (ADR-0001)

O SDD **não substitui** a hierarquia de planejamento (ADR-0001). Ele encaixa **entre** o backlog
(`docs/ROADMAP.md §0`, o SSoT do *quê*) e o snapshot de sprint (`docs/sprints/v2.X.md`, o *que
aconteceu*). A spec **referencia** o `Backlog-Code`; **nunca** cunha versão futura (versão = 1º
commit da sprint, R2). A pasta `specs/NNNN-slug/` usa numeração própria, desacoplada de `vX.Y`.

## Quando criar uma spec

- Item de `docs/ROADMAP.md §0` com esforço **≥ 1 dia** OU que toque **≥ 3 arquivos** OU que cruze
  **≥ 2 produtos** (LIB/CLI/STU/SM).
- **NÃO** criar spec para: typo, bump de versão, fix de 1 linha → fluxo trivial direto.

## Passo a passo (por feature)

1. `git switch -c feat/<slug>` a partir de `main`.
2. `cp templates/spec-template.md specs/NNNN-slug/spec.md`. Preencher WHAT/WHY. Resolver **TODOS**
   os `[NEEDS CLARIFICATION]`. → **GATE-S**.
3. `cp templates/plan-template.md specs/NNNN-slug/plan.md`. Definir HOW. Promover ADR se a decisão
   for irreversível. → **GATE-P** (tabela de constituição sem violação).
4. `cp templates/tasks-template.md specs/NNNN-slug/tasks.md`. Decompor em tarefas atômicas
   testáveis, com ordem e dependências. → **GATE-T**.
5. Implementar tarefa-a-tarefa, **1 commit atômico por tarefa**. → **GATE-I** (hooks pre-commit).
6. Verificar: CI verde + reviewers + cobertura. → **GATE-V** → merge.
7. No merge: preencher `Released-As: vX.Y` na spec, criar `docs/sprints/v2.X.md`, atualizar
   `CHANGELOG`, mover o item no backlog (skill `geosteering-documentation`).

## Regra de fundação (inviolável)

Specs que tocam `geosteering_ai/gui/` ou `simulation/multi_forward.py` adquirem **lock**
(seção crítica) via `.claude/hooks/acquire-lock.sh`. Specs puramente de CLI/API/Lib **paralelizam
livremente**. Toda spec que toca cálculo EM declara `Converge-Em: multi_forward.py` (Princípio XI).

## Reviewers por trilha (GATE-V)

| Trilha | Reviewer obrigatório |
|:--|:--|
| A (simulador) | `geosteering-physics-reviewer` (Opus) — bloqueia se paridade quebrar |
| B (performance) | `geosteering-perf-baseline` + `geosteering-perf-reviewer` |
| C/D/E/F | `geosteering-code-reviewer` + `geosteering-security-auditor` (se PR sensível) |

## Estrutura do diretório

```
specs/
├── CONSTITUTION.md      princípios invioláveis (12)
├── README.md            este arquivo (o processo)
├── ROADMAP.md           etapas em ondas O0..O7, paralelização, capacidade
├── INDEX.md             tabela: spec ↔ Backlog-Code ↔ produto ↔ status
├── templates/           spec-template · plan-template · tasks-template
└── NNNN-slug/           spec.md · plan.md · tasks.md  (uma pasta por feature)
```

Próximo passo no fluxo guiado: ver `ROADMAP.md` (a fila ordenada) e a spec semente
`0003-cli-backend-auto/spec.md` (exemplo trabalhado, menor risco).
