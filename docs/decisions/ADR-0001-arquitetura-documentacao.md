# ADR-0001 — Arquitetura de Documentação de Roadmap (4 documentos disjuntos)

| Campo | Valor |
|:--|:--|
| **Status** | Aceita |
| **Data** | 2026-05-18 |
| **Decisor(es)** | Daniel Leal + Claude Opus 4.7 (1M context) |
| **Sprint executora** | v2.40.2 |
| **Substitui** | — (primeira ADR formal do projeto) |
| **Substituída por** | — |

---

## Contexto

Durante a Sprint v2.40.1, o usuário identificou que a definição da versão
futura **v2.41** aparecia em **5 lugares diferentes** com escopos divergentes:

1. `docs/CHANGELOG.md:137` — "SurrogateNet Training (Colab A100 mp16)" — 4-6h
2. `docs/reports/mvc_simulation_manager_studio_analysis_2026-05-18.md` (9× menções) — "Fase 0 split simulation_manager.py em package" — Jun 2026
3. `~/.claude/plans/muito-bem-como-base-memoized-twilight.md:332-335` — "v2.41a SurrogateNet + v2.41b POST /simulate"
4. `docs/reports/v2.39_api_rest_dockerfile_2026-05-18.md:323` — "Catálogo de Ruído 35 tipos"
5. `docs/reports/arquitetura_multiagente_aprofundamento_2026-05-02.md:9943` — "Sprint 41-43: Diffusion priors"

**Causas-raiz identificadas** (relatório consolidado em sessão Claude
2026-05-18):

1. **Ausência de Single Source of Truth (SSoT)** — `ROADMAP.md` existe mas
   cobre apenas v2.0-v2.40 + v2.25-v2.31 como `PLANEJADO` (especulativos
   que foram pulados).
2. **Reports inventam roadmaps locais** — cada `docs/reports/v2.X_*.md` tem
   uma seção "próximos passos" escrita sem consultar/atualizar ROADMAP.md.
3. **Planos em `~/.claude/plans/` viram pseudo-canônicos** — embora
   efêmeros por design, sobrevivem entre sessões e são citados como
   referência.
4. **Múltiplas trilhas paralelas competindo pelo mesmo slot numérico** —
   v2.41 da trilha C (DL) ≠ v2.41 da trilha B (perf), mas o número de
   sprint é GLOBAL.
5. **Sub-versões improvisadas** — `v2.41a` e `v2.41b` violam SemVer e
   proliferam sem governança.
6. **Ausência de INDEX/TOC** — não existe `docs/INDEX.md` declarando
   a hierarquia de leitura.
7. **Versões PLANEJADAS órfãs** — v2.25-v2.31 marcadas `PLANEJADO` no
   ROADMAP nunca foram executadas (foram puladas em favor de v2.32-v2.40),
   criando ambiguidade.
8. **CHANGELOG.md misturava passado e futuro** — entradas `⏳ v2.41+`
   violam o padrão Keep-a-Changelog.

---

## Decisão

Adotar a **arquitetura de 4 documentos com responsabilidades disjuntas**:

### Os 4 Documentos

| Documento | Responsabilidade Única | Editor |
|:--|:--|:--|
| **`docs/ROADMAP.md`** | Backlog priorizado (épicos, sem números de versão fixos) | Humano + skill `geosteering-documentation` |
| **`docs/sprints/CURRENT.md`** | Plano detalhado da sprint ATIVA | Plan agent (auto via `/plan` ou `/geosteering-orchestrator`) |
| **`docs/sprints/v2.X.md`** | Snapshot imutável de cada sprint completada | Plan agent ao merge |
| **`docs/CHANGELOG.md`** | Histórico cronológico reverso de releases (Keep-a-Changelog) | Skill `geosteering-documentation` |

### Documentos de Apoio

| Documento | Responsabilidade |
|:--|:--|
| `docs/INDEX.md` | Porta de entrada — declara a hierarquia + fluxo de leitura |
| `docs/decisions/ADR-XXXX.md` | Decisões irreversíveis ou de alto impacto |
| `docs/reports/v2.X_*.md` | Análise técnica long-form pós-sprint (não-canônica) |

### Regras Duras

**R1 (SSoT)**: Nenhum outro arquivo do projeto pode definir o que é uma
versão futura. Reports, skills, hooks, code comments — todos devem
**referenciar** `ROADMAP.md` em vez de cunhar definições próprias.

**R2 (Versão tardia)**: Versões `vX.Y` são atribuídas no **primeiro commit
da sprint**, não antes. Antes disso, itens vivem no backlog identificados
apenas por **code** (e.g., `C-noise-35`).

**R3 (Sem sub-letras)**: Não usar `vX.Ya`, `vX.Yb`. Se uma sprint precisa
ser dividida, criar duas sprints sequenciais (`vX.Y` + `vX.Y+1`).

**R4 (Trilhas como metadata)**: Cada item de backlog tem campo `Trilha`
(A/B/C/D/E/F). Sprints com itens de múltiplas trilhas devem ter trilha
**dominante** declarada.

**R5 (Hook de validação)**: Pre-commit hook `check-version-references.sh`
falha se um arquivo `.md`/`.py` contém `vX.Y` (com `X.Y > current + 1`)
sem referência explícita a `docs/ROADMAP.md`.

**R6 (CHANGELOG histórico puro)**: Não usar `⏳ vX.Y+` ou `(PLANEJADO)` em
CHANGELOG. Apenas releases passadas, cronológico reverso.

**R7 (ADRs para conflitos)**: Quando há discordância sobre escopo de sprint
(e.g., SurrogateNet vs MVC para próxima sprint), criar ADR que documenta
a decisão e a justificativa.

---

## Consequências

### Positivas

- **Eliminação de ambiguidade**: leitor sabe exatamente onde olhar para
  cada pergunta ("qual a próxima sprint?" → ROADMAP.md, sempre).
- **Atribuição just-in-time**: versões só existem quando há compromisso
  real (commit), eliminando "wishful thinking versions".
- **Histórico imutável**: sprints completadas têm snapshot em
  `sprints/v2.X.md` que nunca muda, facilitando auditoria.
- **Onboarding mais rápido**: novo colaborador lê INDEX.md → ROADMAP.md
  → CURRENT.md e tem visão completa em <10 minutos.
- **Drift documental detectado automaticamente**: hook bloqueia commits
  que inventam versões fora do ROADMAP.

### Negativas / Trade-offs

- **Refator inicial pesado**: 1 sprint dedicada (v2.40.2, ~6h) para
  reorganizar tudo, sem ganho funcional de produto.
- **Disciplina contínua necessária**: skill `geosteering-documentation`
  precisa ser atualizada para sempre tocar ROADMAP.md ao criar reports;
  agentes humanos precisam respeitar a regra R1.
- **Hook adiciona fricção em commits**: contribuidores que mencionam
  versões futuras em comentários de código terão commits bloqueados até
  adicionar referência a ROADMAP.md.
- **Reports históricos preservados como-estão**: relatórios passados que
  contêm definições de v2.41 não serão reescritos (custo > benefício);
  apenas ADR-0001 e INDEX.md explicam que reports não são canônicos.

### Riscos

- **R-1**: Hook muito agressivo bloqueia commits legítimos (e.g., comentário
  histórico em changelog). Mitigação: hook só falha para `X.Y > current + 1`
  (pula 1 versão); referências a passado são livres.
- **R-2**: Skill `geosteering-documentation` esquece de atualizar ROADMAP.md.
  Mitigação: skill atualizada com instrução explícita + checklist visível.
- **R-3**: Desenvolvedores resistem à disciplina e voltam a inventar
  definições locais. Mitigação: ADR-0001 visível no INDEX.md; hook
  enforça automaticamente.

---

## Implementação

Executada em **Sprint v2.40.2** (branch `refactor/v2.40.2-doc-architecture`),
em 4 fases:

### Fase 1 — Limpeza
- Remover entradas `PLANEJADO` v2.25-v2.31 obsoletas de ROADMAP.md
- Limpar entradas `⏳ v2.41+` de CHANGELOG.md
- Arquivar planos de `~/.claude/plans/` em `docs/sprints/archive/`

### Fase 2 — Refatoração
- Criar `docs/INDEX.md` como porta de entrada
- Criar `docs/sprints/` com `CURRENT.md`, `v2.40.md`, `v2.40.1.md`
- Reescrever `ROADMAP.md` como backlog priorizado por trilhas A-F
- Criar `docs/decisions/` com este ADR-0001

### Fase 3 — Governança
- Adicionar hook `check-version-references.sh` (PreCommit)
- Atualizar CLAUDE.md com seção "Hierarquia de Planejamento"
- Atualizar skill `geosteering-documentation` para tocar ROADMAP.md

### Fase 4 — Validação
- Smoke test do hook
- Verificação que ROADMAP.md sozinho responde "qual a próxima sprint?"

---

## Alternativas Consideradas (e rejeitadas)

### A. Status quo + documentar limitação em CLAUDE.md
- **Por que rejeitada**: não resolve, só explica. Discrepâncias continuam
  proliferando a cada sprint.

### B. Apenas criar INDEX.md + ADR-0001 sem refatorar (1h)
- **Por que rejeitada**: mitigação parcial; entradas órfãs em ROADMAP
  (v2.25-v2.31 PLANEJADO) e CHANGELOG (`⏳ v2.41+`) continuam causando
  confusão na próxima sprint.

### C. Hot-fix: editar ROADMAP.md declarando v2.41 = SurrogateNet (15min)
- **Por que rejeitada**: resolve só o caso v2.41 atual; problema reaparece
  em v2.42 sem a mudança estrutural.

### D. Versionamento por trilha (v2.41-trilhaC, v2.41-trilhaB)
- **Por que rejeitada**: complexidade extra desnecessária; trilhas como
  metadata + versionamento monotônico global é mais simples e suficiente.

---

## Referências

- Relatório de investigação consolidado: sessão Claude Code 2026-05-18
- Plano executor: `docs/sprints/v2.40.2.md` (a ser criado ao merge)
- Hook: `.claude/hooks/check-version-references.sh`
- INDEX: `docs/INDEX.md`

---

*Status: Aceita. Imutável após merge.*
