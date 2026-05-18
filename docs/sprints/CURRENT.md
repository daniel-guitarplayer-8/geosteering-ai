# Sprint Atual

> Este arquivo contém o plano detalhado da sprint **em execução**.
> Após o merge, deve ser renomeado para `v2.X.md` (snapshot imutável)
> e este arquivo fica vazio até a próxima sprint.

---

## Estado Atual

**Nenhuma sprint em execução.** Última sprint completada: **v2.40.2**
(refator documental) — ver [v2.40.2.md](v2.40.2.md).

Próximos candidatos no backlog: ver [../ROADMAP.md](../ROADMAP.md) §Backlog.

---

## Template para Sprint em Execução

Quando uma nova sprint começa, popule esta seção seguindo o template abaixo:

```markdown
## Sprint vX.Y — <Título Curto>

| Campo | Valor |
|:--|:--|
| **Versão alvo** | vX.Y (atribuída no primeiro commit, não antes) |
| **Branch** | `feat/vX.Y-<slug>` ou `refactor/vX.Y-<slug>` ou `patch/vX.Y.Z-<slug>` |
| **Trilha** | A (sim) / B (perf) / C (DL) / D (geo) / E (infra) / F (gov) |
| **Item de backlog** | `<code>` em ROADMAP.md (e.g., `C-surrogate-train`) |
| **Esforço estimado** | Xh em N sessões |
| **Data início** | YYYY-MM-DD |
| **Modelo Claude** | Sonnet 4.6 / Opus 4.7 / etc |

### Contexto

(Por que esta sprint agora, o que ela desbloqueia, dependências satisfeitas)

### Escopo

(Lista de arquivos a criar/modificar, decisões D1-Dn confirmadas)

### Commits Planejados

| # | Commit | Esforço | Dep |
|--:|:--|--:|:--|
| C01 | `<type>(<scope>): <message>` | Xmin | — |
| ... | | | |

### Critérios de Aceitação

- [ ] Suite pytest preserva N PASS / 0 FAIL
- [ ] Paridade Fortran <1e-12 inalterada (se tocar `simulation/`)
- [ ] mypy 0 erros novos
- [ ] Pre-commit hooks ALL PASS
- [ ] Reviewers fan-out: 0 findings BLOQUEANTES

### Logs em Andamento

(Atualizar à medida que commits são feitos — facilita pick-up entre sessões)
```

---

*Template alinhado com ADR-0001 (arquitetura de 4 documentos).*
