# Relatório — {PROJECT_NAME} {VERSION}

> **Gerado automaticamente** via `.claude/templates/report_template.md`  
> **Data**: {DATE}  
> **Branch**: {BRANCH}  
> **Commits nesta versão**: {COMMIT_COUNT}  
> **Smoke test baseline**: {SMOKE_PREV} → {SMOKE_NEW} OK / 0 FAIL

---

## Resumo Executivo

{EXECUTIVE_SUMMARY}

<!-- Instrução: 2–3 frases cobrindo: (a) o que mudou, (b) por que, (c) impacto. -->

---

## 1. Bugs Corrigidos ({BUG_COUNT})

| # | Bug | Root Cause | Arquivo(s) | Linha(s) | Fix Aplicado |
|:-:|:----|:-----------|:-----------|:--------:|:-------------|
{BUG_TABLE_ROWS}

<!-- Formato de linha:
| 1 | Descrição curta | Causa raiz técnica | arquivo.py | L123–145 | Descrição do fix |
-->

---

## 2. Features / Melhorias ({FEATURE_COUNT})

| # | Feature | Arquivo(s) | Linhas | Impacto |
|:-:|:--------|:-----------|:------:|:--------|
{FEATURE_TABLE_ROWS}

---

## 3. Remoções / Deprecações ({REMOVAL_COUNT})

| Item Removido | Motivo | Substituto |
|:--------------|:-------|:-----------|
{REMOVAL_TABLE_ROWS}

---

## 4. Revisão de Código (CodeRabbit)

**Findings**: {CR_TOTAL} ({CR_MAJOR} major, {CR_MINOR} minor)

| # | Severidade | Arquivo | Descrição | Ação |
|:-:|:----------:|:--------|:----------|:-----|
{CR_TABLE_ROWS}

<!-- Ação: Corrigido | False positive (motivo) | Adiado -->

---

## 5. Testes

### Smoke Tests

| Versão | OK | FAIL | Delta |
|:-------|:--:|:----:|:-----:|
| Anterior ({VERSION_PREV}) | {SMOKE_PREV} | 0 | — |
| **{VERSION}** | **{SMOKE_NEW}** | **0** | **+{SMOKE_DELTA}** |

### Novos Testes Adicionados

| ID | Descrição | Checks |
|:---|:----------|:------:|
{NEW_TESTS_ROWS}

---

## 6. Arquivos Modificados

| Arquivo | Tipo | LOC Δ | Descrição |
|:--------|:----:|:-----:|:----------|
{FILES_TABLE_ROWS}

<!-- Tipo: NEW | MOD | DEL -->

---

## 7. Integridade dos Simuladores

```bash
# Verificação: simuladores JAX/Numba/Fortran intocados
git diff --name-only | grep -E "_jax|_numba|forward|Fortran"
# Saída esperada: (vazia)
```

**Resultado**: {SIMULATOR_INTEGRITY}

---

## 8. Notas de Migração / Breaking Changes

{MIGRATION_NOTES}

<!-- Se não houver: "Nenhuma breaking change nesta versão." -->

---

## 9. Próximos Passos

{NEXT_STEPS}

<!-- Lista priorizada do que vem na próxima versão -->

---

*Relatório gerado por Claude Code via template `.claude/templates/report_template.md`*  
*Versão do template: 1.0.0 (2026-04-28)*
