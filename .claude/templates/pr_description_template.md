## Resumo

{SUMMARY}

## Mudanças

{CHANGES}

## Test Plan

- [ ] `pytest tests/ -v --tb=short` passa sem regressões
- [ ] `pytest tests/test_simulation_compare_fortran.py -v` mantém paridade <1e-12 (13/13 modelos canônicos)
- [ ] Smoke tests novos (se aplicável): `pytest tests/test_<feature>.py -v`
- [ ] `python scripts/count_pytest_pass.py` reflete contagem atualizada

## Referências

- Branch: `{BRANCH}`
- Base: `{BASE}`
- Commits: {COMMIT_COUNT}
- Arquivos modificados: {FILE_COUNT}

{COMMITS_LIST}

---

🤖 Gerado com [Claude Code](https://claude.com/claude-code) — hook `generate-pr-description.sh`

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
