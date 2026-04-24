# Diagnóstico — 4 Falhas Pré-existentes em `TestSourceSynchronization`

> **Data**: 2026-04-16
> **Versão do simulador**: v1.6.0 (PR #25)
> **Suite afetada**: `tests/test_simulation_filters.py::TestSourceSynchronization`
> **Status operacional do simulador**: ✅ Não afetado — filtros Hankel funcionam normalmente

---

## 1. Resumo Executivo

Quatro testes de `TestSourceSynchronization` falham com `FileNotFoundError` porque o
arquivo Fortran fonte `Fortran_Gerador/filtersv2.f08` **nunca foi commitado** ao
repositório git. Esses testes **não representam bugs** na lógica de simulação: os
filtros Hankel em `.npz` são válidos e todos os 1.459+ testes de simulação passam
normalmente. A falha é uma **inconsistência estrutural do repositório** — o arquivo
de proveniência dos filtros existe apenas localmente na máquina do desenvolvedor
original.

---

## 2. Os 4 Testes Afetados

| # | Nome do Teste | Tipo de Falha | Mensagem |
|---|:--------------|:--------------|:---------|
| 1 | `test_source_hash_synchronized[werthmuller_201pt]` | `FileNotFoundError` | `No such file or directory: '.../Fortran_Gerador/filtersv2.f08'` |
| 2 | `test_source_hash_synchronized[kong_61pt]` | `FileNotFoundError` | `No such file or directory: '.../Fortran_Gerador/filtersv2.f08'` |
| 3 | `test_source_hash_synchronized[anderson_801pt]` | `FileNotFoundError` | `No such file or directory: '.../Fortran_Gerador/filtersv2.f08'` |
| 4 | `test_verify_mode_returns_zero` | `AssertionError` (returncode=1) | `[ERROR] Arquivo fonte não encontrado: .../Fortran_Gerador/filtersv2.f08` |

Todos os 4 falham no mesmo ponto: qualquer tentativa de abrir
`Fortran_Gerador/filtersv2.f08` lança `FileNotFoundError` porque o arquivo não existe.

---

## 3. Design dos Testes — Intenção Original

A classe `TestSourceSynchronization` implementa um mecanismo de **cadeia de
custódia** (*chain-of-trust*) para os filtros Hankel. A arquitetura é:

```
filtersv2.f08  ──── scripts/extract_hankel_weights.py ────▶  *.npz
(fonte Fortran)         (extrator + parser regex)           (filtros prontos)
     │                                                            │
     └── SHA-256 do arquivo fonte ──── gravado em metadata ──────┘
                                       (campo source_sha256)
```

O fluxo correto de trabalho seria:

1. Desenvolvedor modifica `filtersv2.f08` (ex: corrige coeficientes de um filtro)
2. Roda `python scripts/extract_hankel_weights.py` → re-gera os `.npz` com o novo SHA-256
3. `pytest` valida que o SHA-256 dos `.npz` bate com o SHA-256 atual do `.f08`
4. Se bater → ✅ "os filtros estão sincronizados com o Fortran"
5. Se não bater → ❌ "alguém alterou o Fortran sem re-extrair"

**O propósito é detectar drift acidental** entre o código Fortran de geração dos
filtros e os arquivos `.npz` em uso. É um guard de integridade, não um teste
funcional.

---

## 4. Causa Raiz

### 4.1 O arquivo nunca foi commitado

```bash
$ git log --all --oneline -- "Fortran_Gerador/filtersv2.f08"
# (saída vazia — sem histórico, nunca commitado)

$ git ls-files Fortran_Gerador/
Fortran_Gerador/Makefile
Fortran_Gerador/PerfilaAnisoOmp.f08
Fortran_Gerador/RunAnisoOmp.f08
Fortran_Gerador/magneticdipoles.f08
Fortran_Gerador/tatu_f2py_wrapper.f08
# filtersv2.f08 ausente da lista
```

O arquivo `filtersv2.f08` foi usado em `2026-04-11T15:31:20` (data de extração nos
`.npz`) mas **nunca adicionado ao git**. Provável causa: esqueceu-se de incluí-lo no
commit que adicionou os filtros `.npz` e o extrator.

### 4.2 Apenas um SHA-256 em disco, sem fonte

Os três `.npz` armazenam o mesmo hash do arquivo ausente:

```
werthmuller_201pt.npz → source_sha256: 94f58d00353be103... (extraído 2026-04-11)
kong_61pt.npz         → source_sha256: 94f58d00353be103... (extraído 2026-04-11)
anderson_801pt.npz    → source_sha256: 94f58d00353be103... (extraído 2026-04-11)
```

O hash `94f58d00...` é evidência de que o arquivo existiu localmente — mas ele
nunca entrou no repositório.

### 4.3 O Fortran simulator também é afetado

O arquivo `filtersv2.f08` **não é apenas** um arquivo de proveniência — ele contém
as definições das sub-rotinas `J0J1Wer`, `J0J1Kong` e `J0J1And`, que são chamadas
diretamente por `PerfilaAnisoOmp.f08`:

```fortran
! PerfilaAnisoOmp.f08, linhas 284–292
call J0J1Kong(npt_active, krJ0J1, wJ0, wJ1)   ! Kong 61-pt
call J0J1And(krJ0J1, wJ0, wJ1)                 ! Anderson 801-pt
call J0J1Wer(npt_active, krJ0J1, wJ0, wJ1)    ! Werthmuller 201-pt
```

```bash
$ grep "subroutine J0J1" Fortran_Gerador/*.f08
# (saída vazia — nenhum arquivo trackeado contém as definições)
```

Isso significa que o simulador Fortran **não pode ser compilado** a partir de um
clone limpo do repositório. O binário `tatu.x` (que está trackeado) funciona porque
foi compilado quando `filtersv2.f08` ainda existia localmente.

---

## 5. Impacto — Matriz de Severidade

| Área | Impacto | Severidade |
|:-----|:--------|:----------:|
| **Simulação Python (Numba + JAX)** | Zero — `.npz` são válidos e carregados corretamente | ✅ Nenhum |
| **Testes funcionais (1.459+ testes)** | Zero — passam normalmente | ✅ Nenhum |
| **Testes de sincronia (4 testes)** | 4 FAIL — não podem verificar proveniência | 🟡 Médio |
| **Compilação Fortran (make)** | Falha — `J0J1*` indefinidos sem `filtersv2.f08` | 🔴 Alto |
| **Clone limpo do repo** | Fortran inoperante; Python funcional | 🔴 Alto |
| **`scripts/extract_hankel_weights.py`** | Falha no modo `--verify` e re-extração | 🟡 Médio |
| **Auditoria de filtros** | Impossível confirmar proveniência dos coeficientes | 🟡 Médio |

---

## 6. Opções de Correção

### Opção A — Adicionar `filtersv2.f08` ao repositório (Recomendada) ⭐

```bash
# Se o arquivo ainda existe localmente:
git add Fortran_Gerador/filtersv2.f08
git commit -m "fix(filters): adicionar filtersv2.f08 ausente do repositório"
```

**Vantagens**:
- Restaura a cadeia de custódia completa (intenção original dos testes)
- Permite compilar o Fortran em clone limpo
- Os 4 testes passarão imediatamente (o SHA-256 no `.npz` já corresponde ao arquivo)
- Custo zero de manutenção

**Desvantagens**:
- Requer que o arquivo ainda exista localmente (se perdido, ver Opção C)

**Verificação pós-commit**:
```bash
python -m pytest tests/test_simulation_filters.py::TestSourceSynchronization -v
# Esperado: 4 PASS
```

---

### Opção B — Marcar testes com `pytest.mark.skipif` quando arquivo ausente

```python
# tests/test_simulation_filters.py
_FILTERSV2 = Path(__file__).resolve().parent.parent / "Fortran_Gerador" / "filtersv2.f08"

@pytest.mark.skipif(not _FILTERSV2.exists(), reason="filtersv2.f08 não presente")
class TestSourceSynchronization:
    ...
```

**Vantagens**:
- Elimina os 4 FAIL em ambientes sem o arquivo (CI, colaboradores)
- Mantém os testes ativos quando o arquivo está presente
- Sem alteração na lógica dos testes

**Desvantagens**:
- Não corrige a ausência do arquivo (problema Fortran persiste)
- Cadeia de custódia silenciosa em ambientes sem o arquivo
- SKIP silencioso pode mascarar problemas futuros

---

### Opção C — Reconstruir `filtersv2.f08` a partir dos coeficientes dos `.npz`

Se o arquivo original foi perdido, os coeficientes dos filtros estão preservados
nos `.npz`. Seria necessário escrever um script reverso que reconstrói o Fortran
a partir dos arrays NumPy. Trabalhoso e arriscado (formato Fortran exato pode
divergir).

**Viabilidade**: Baixa sem o arquivo original.

---

### Opção D — Incorporar definições `J0J1*` em arquivo Fortran já trackeado

Mover as sub-rotinas `J0J1Wer`, `J0J1Kong`, `J0J1And` para dentro de
`magneticdipoles.f08` ou criar `filters.f08` novo e commitado.

**Vantagens**: Restaura compilabilidade sem depender do arquivo original
**Desvantagens**: Requer reconstrução das definições; os SHA-256 nos `.npz`
ficariam obsoletos (arquivo renomeado → hash diferente → testes fariam FAIL
até nova extração)

---

## 7. Recomendação

**Opção A** — se `filtersv2.f08` ainda existir localmente. É a solução de menor
custo e único path que restaura completamente a intenção do design sem modificar
os testes ou os `.npz`.

Para verificar se o arquivo ainda existe localmente:

```bash
ls -la Fortran_Gerador/filtersv2.f08 2>/dev/null && echo "EXISTE" || echo "PERDIDO"
```

Se o arquivo **não existir mais** localmente, a Opção B (skipif) é a alternativa
pragmática enquanto a reconstrução não é possível.

---

## 8. Estado Atual dos Filtros (Confirmado Funcional)

Os filtros em si são **corretos e funcionais**. A tabela abaixo confirma que os
`.npz` estão carregando sem erro e contêm os dados esperados:

| Filtro | Pontos | SHA armazenado | Extraído em |
|:-------|:------:|:---------------|:------------|
| `werthmuller_201pt` | 201 | `94f58d00...` | 2026-04-11T15:31:20 |
| `kong_61pt` | 61 | `94f58d00...` | 2026-04-11T15:31:20 |
| `anderson_801pt` | 801 | `94f58d00...` | 2026-04-11T15:31:20 |

Os três compartilham o mesmo SHA-256 do arquivo fonte, indicando extração num
único `python scripts/extract_hankel_weights.py` em 2026-04-11.

---

## 9. Conclusão

| Pergunta | Resposta |
|:---------|:---------|
| São bugs na lógica de simulação? | **Não** |
| São falhas nos coeficientes dos filtros? | **Não** |
| São inconsistências de código? | **Não** |
| São falhas de proveniência/repositório? | **Sim** |
| Afetam a simulação Python (Numba/JAX)? | **Não** |
| Afetam a compilação do Fortran? | **Sim** (em clone limpo) |
| Correção principal? | Adicionar `filtersv2.f08` ao git |

Os 4 FAIL são um **debt de repositório**: um arquivo essencial (fonte Fortran dos
filtros Hankel) foi usado para gerar os `.npz` mas nunca commitado. A correção
é simples se o arquivo ainda existe localmente — `git add` + commit.
