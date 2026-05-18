# Arquivo Histórico de Planos de Sprint

Esta pasta contém planos de sprint **arquivados** — snapshots imutáveis do que
foi planejado em sessões passadas de Claude Code (`~/.claude/plans/`).

## Status

Estes arquivos são **histórico**, não **fonte de verdade**. Após a refatoração
documental v2.40.2 (ADR-0001), planos de sprints futuras vivem em:

- **Backlog vivo**: [docs/ROADMAP.md](../../ROADMAP.md)
- **Sprint em execução**: [docs/sprints/CURRENT.md](../CURRENT.md)
- **Sprints completadas**: `docs/sprints/v2.X.md`

## Convenção de Nomenclatura

Arquivos seguem o padrão `YYYY-MM-DD__<topic>.md`, onde:

- `YYYY-MM-DD` é a data de criação do plano original (modtime do arquivo
  em `~/.claude/plans/`)
- `<topic>` é o primeiro H1 do plano (ou nome do arquivo se ausente),
  truncado a 50 caracteres

## Inventário

| Arquivo | Sessão | Tópico |
|:--|:--|:--|
| `2026-04-26__Plano-de-Pesquisa-Abstração-Multi-Backend-de-Plota.md` | Plot abstração | Pesquisa multi-backend |
| `2026-04-28__Plano--Simulation-Manager-v28--Correções-de-Bugs-C.md` | SM v2.8 | Bug fixes |
| `2026-05-09__Pesquisa-PyTorch-como-Backend-do-Keras-3x-no-Geost.md` | Backend research | PyTorch via Keras 3.x |
| `2026-05-09__Plano--Análise-de-Viabilidade-de-Datasets-Públicos.md` | Datasets | SDAR/Volve/Teapot viability |
| `2026-05-15__Plano-Sprints-v233--v234--v235-Bundle-de-Estabiliz.md` | v2.33-v2.35 | Bundle de estabilização |
| `2026-05-15__Sprint-v236--Plano-Executivo-Geosteering-AI-Numba-.md` | v2.36 | Plano executivo Numba |
| `2026-05-18__Plano-Sprint-v2401--Patch-dos-5-Findings-dos-Revis.md` | v2.40.1 | Patch reviewer findings |

## Por que não deletar?

Planos arquivados preservam **contexto histórico** sobre como decisões foram
tomadas. Útil para:

- Auditar mudanças de escopo (e.g., "por que v2.25 PLANEJADO virou outra coisa?")
- Onboarding de novos colaboradores (entender a evolução do roadmap)
- Análise de drift (comparar planejado vs executado)

**Não** referencie estes arquivos como autoridade — eles são snapshots
congelados no tempo. Para o estado atual do roadmap, sempre consulte
[docs/ROADMAP.md](../../ROADMAP.md).
