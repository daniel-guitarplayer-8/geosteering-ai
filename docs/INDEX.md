# Geosteering AI — Índice de Documentação

**Versão**: 1.0 (criado em 2026-05-18, refactor v2.40.2 — ver
[ADR-0001](decisions/ADR-0001-arquitetura-documentacao.md))

Esta página é a **porta de entrada** para toda a documentação do projeto.
Use-a para descobrir onde está cada tipo de informação.

---

## 🎯 Para PLANEJAR trabalho futuro

| # | Documento | Conteúdo | Quem edita |
|:-:|:--|:--|:--|
| 1 | **[ROADMAP.md](ROADMAP.md)** | Backlog priorizado (épicos por trilha + sprints históricas). **Única fonte de verdade do futuro.** | Humano + skill `geosteering-documentation` |
| 2 | **[sprints/CURRENT.md](sprints/CURRENT.md)** | Plano detalhado da sprint EM EXECUÇÃO (escopo, commits, gates). | Plan agent (`/plan` ou `/geosteering-orchestrator`) |

**Regra-dura (ADR-0001)**: Nenhum outro arquivo do projeto pode definir o que
é uma versão futura. Reports, skills, hooks, code comments — todos devem
**referenciar** `ROADMAP.md` em vez de cunhar definições próprias.

---

## 📜 Para CONSULTAR HISTÓRICO

| # | Documento | Conteúdo |
|:-:|:--|:--|
| 3 | **[CHANGELOG.md](CHANGELOG.md)** | Lista cronológica reversa de releases (Keep-a-Changelog). **Apenas passado.** |
| 4 | **`sprints/v2.X.md`** | Plano arquivado de cada sprint completada (imutável após merge). |
| 5 | **[sprints/archive/](sprints/archive/)** | Planos antigos importados de `~/.claude/plans/` (histórico de sessões). |
| 6 | **`reports/v2.X_*_YYYY-MM-DD.md`** | Análises técnicas pós-sprint (relatórios long-form). |
| 7 | **`reports/fase{N}_*.md`** | Relatórios de fases arquiteturais (Fase 1, 2, etc.) |
| 8 | **[known_bugs.md](known_bugs.md)** | KB-XXX: bugs conhecidos catalogados + workarounds. |
| 9 | **[PERFORMANCE_BASELINE.md](PERFORMANCE_BASELINE.md)** | Baselines de throughput (Cenários A-H + CI warm-cache + TF Training). |

---

## 🏛️ Para ENTENDER ARQUITETURA

| # | Documento | Conteúdo |
|:-:|:--|:--|
| 10 | **[ARCHITECTURE_v2.md](ARCHITECTURE_v2.md)** | Arquitetura técnica completa (§1-§75). Doc canônico de design. |
| 11 | **[../CLAUDE.md](../CLAUDE.md)** | Convenções, proibições absolutas, errata física, padrões D1-D14. |
| 12 | **[physics/](physics/)** | Contexto físico (tensor EM, GS, FV, Archie, decoupling). |
| 13 | **[reference/](reference/)** | Catálogos técnicos (arquiteturas, losses, noise types, simulator analysis). |

---

## ⚖️ Para REGISTRAR DECISÕES MAJORES

| # | Documento | Conteúdo |
|:-:|:--|:--|
| 14 | **[decisions/](decisions/)** | ADRs (Architecture Decision Records) — decisões irreversíveis ou de alto impacto. |
| 15 | **[decisions/ADR-0001-arquitetura-documentacao.md](decisions/ADR-0001-arquitetura-documentacao.md)** | Refatoração doc v2.40.2 (esta arquitetura de 4 documentos). |

**Quando criar um ADR**:
- Decisão de design difícil de reverter (escolha de framework, padrão arquitetural)
- Mudança de governança documental ou processo
- Conflito entre 2+ propostas que precisa ser resolvido formalmente
- Trade-off físico vs performance que estabelece precedente

---

## 🎓 Hierarquia de Leitura

```text
┌─────────────────────────────────────────────────────────────────┐
│  Recém-chegado ao projeto?                                       │
│      1. Leia CLAUDE.md (regras + proibições)                     │
│      2. Leia este INDEX.md (mapa)                                │
│      3. Leia ROADMAP.md (o que está sendo construído)            │
│      4. Leia ARCHITECTURE_v2.md §1-§5 (sumário arquitetural)     │
│                                                                  │
│  Trabalhando em uma sprint?                                      │
│      1. Leia sprints/CURRENT.md (escopo da sprint)               │
│      2. Leia ADRs relevantes (decisões já tomadas)               │
│      3. Leia sprints/v2.X.md de sprints similares passadas       │
│                                                                  │
│  Auditando uma feature passada?                                  │
│      1. Leia CHANGELOG.md (qual release introduziu?)             │
│      2. Leia reports/v2.X_*.md (análise técnica detalhada)       │
│      3. Leia sprints/v2.X.md (plano original vs executado)       │
│                                                                  │
│  Tomando uma decisão arquitetural?                               │
│      1. Verifique decisions/ (ADR já existe?)                    │
│      2. Verifique known_bugs.md (KB-XXX impacta a decisão?)      │
│      3. Crie novo ADR se for decisão difícil de reverter         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Fluxo de Vida de uma Sprint

```text
┌────────────────────────────────────────────────────────────────┐
│  Estado: BACKLOG (item em ROADMAP.md)                          │
│      │                                                          │
│      ▼ priorização (humano + dependências resolvidas)          │
│  Estado: CANDIDATE (próximo na fila)                           │
│      │                                                          │
│      ▼ commit do primeiro arquivo da sprint                    │
│  Estado: IN_PROGRESS                                           │
│      │   - Versão vX.Y atribuída AGORA (não antes)             │
│      │   - sprints/CURRENT.md criado/atualizado                │
│      │   - Branch git criada                                   │
│      │                                                          │
│      ▼ merge na main                                           │
│  Estado: DONE                                                  │
│      │   - sprints/CURRENT.md → sprints/v2.X.md (arquivo)      │
│      │   - CHANGELOG.md atualizado (append)                    │
│      │   - ROADMAP.md: item movido de backlog para histórico   │
│      │   - reports/v2.X_*.md criado (análise técnica)          │
└────────────────────────────────────────────────────────────────┘
```

---

## 📚 Sub-projetos Relacionados

| Subprojeto | Localização | Documentação |
|:--|:--|:--|
| Simulador Fortran v10 | `Fortran_Gerador/` | `Fortran_Gerador/README.md` |
| Simulador Python (Numba+JAX) | `geosteering_ai/simulation/` | `docs/reference/analise_cenarios_otimizacao_simulador_numba.md` |
| API REST | `geosteering_ai/api/` | docstrings inline + `docs/reports/v2.39_api_rest_dockerfile_*.md` |
| CLI | `geosteering_ai/cli/` | `geosteering-cli --help` |
| MCP Colab Bridge | `.claude/commands/geosteering-colab-mcp.md` | skill própria |

---

*Última atualização: 2026-05-18 (refactor v2.40.2 — ADR-0001).*
