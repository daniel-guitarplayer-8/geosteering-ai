---
Spec: NNNN-slug
Titulo: <título curto em PT-BR>
Backlog-Code: <ex.: C-noise-35 | (novo) E-cli-auto>     # referência ao ROADMAP §0
Trilha-Dominante: <A|B|C|D|E|F>                          # R4 ADR-0001
Produtos: [LIB|CLI|STU|SM]                                # quais dos 4 produtos afeta
Converge-Em: <ex.: multi_forward.py | n/a>                # ponto físico único (Princípio XI)
Status: planejado                                         # planejado|parcial|implementado
Released-As: <vazio até o merge>                          # vX.Y só no merge (R2)
Constituicao: 1.0                                         # versão da CONSTITUTION respeitada
Autor: <nome>
Data: YYYY-MM-DD
---

# Spec NNNN — <Título> (WHAT / WHY)

## 1. Contexto e Problema
<2–4 parágrafos. Que dor real existe? Citar arquivos como `caminho:linha`.
Referenciar relatórios/ADRs/componentes de `docs/architecture/`.>

| Estado | Onde | Evidência |
|:--|:--|:--|
| ausente | `geosteering_ai/ingest/` | diretório não existe |
| parcial | `geosteering_ai/cli/_main.py:NN` | `--backend` sem opção `auto` |

## 2. User Stories
> "Como **\<persona\>**, quero **\<ação\>**, para **\<benefício\>**."
> Personas: Geofísico · Engenheiro de Poço · Pesquisador DL · Operador SM · Dev da Biblioteca/API.

| ID | Como… | Quero… | Para… | Prioridade |
|:--|:--|:--|:--|:--:|
| US-1 | Geofísico | importar um LAS de campo | inverter dados reais no Studio | Must |

## 3. Requisitos Funcionais (RF) — MoSCoW + Critérios de Aceite Testáveis
> Cada RF tem ID, prioridade MoSCoW e **≥1 critério de aceite VERIFICÁVEL**
> (Given/When/Then ou comando + saída esperada). **Sem AC testável = RF inválido.**

| ID | Requisito | MoSCoW | Cobertura atual |
|:--|:--|:--:|:--|
| RF-1 | Parser LAS produz array 22-col compatível com `data/loading.py` | Must | NOVO |

### RF-1 — Critérios de Aceite
- [ ] **AC-1.1**: Dado um `.las` com curvas {GR, RES}, quando `LASReader.read(path)`, então
  retorna `np.ndarray` shape `(N, 22)` dtype `float64`.
- [ ] **AC-1.2**: `pytest tests/test_ingest_las.py::test_22col_shape` PASSA.
- [ ] **AC-1.3**: Paridade no regime aplicável (Princípio I) — declarar `N/A` se não toca cálculo EM.

## 4. Requisitos Não-Funcionais (RNF)
| ID | Categoria | Requisito | Métrica/Limite |
|:--|:--|:--|:--|
| RNF-1 | Performance | inversão realtime por sample | < 100 ms no perfil **HW-FIELD-MIN** |
| RNF-2 | Paridade | EM se tocado | física <1e-12 / JAX-Numba <1e-10 / inferência rtol<1e-6 |
| RNF-3 | Doc | módulos novos | D1–D14 conformes |
| RNF-4 | Plataforma | Python | 3.13 exclusivo |
| RNF-5 | Cobertura | módulo tocado | sem regressão; GUI nova ≥80% (cov-diff) |
| RNF-6 | Segurança | arquivos externos (`.gsproj`/LAS) | sem pickle; `SecureArchiveReader`; errata fail-fast |

## 5. Escopo
### IN (esta spec entrega)
- <bullet concreto e verificável>
### OUT (explicitamente fora — evita scope creep)
- <bullet — e para onde vai, se for backlog futuro>

## 6. [NEEDS CLARIFICATION]
> Toda ambiguidade material vira um marcador. **GATE-S exige ZERO marcadores abertos.**
- [ ] [NEEDS CLARIFICATION] <pergunta> — proposta default: <…> — decisor: <…>

## 7. Dependências e Riscos de Escopo
| Tipo | Item | Impacto |
|:--|:--|:--|
| Dep | spec 0004-gui-foundation | bloqueante p/ Studio/SM |
| Risco | dataset de campo indisponível | mitigação: sintético + holdout; TECH PREVIEW |

## 8. Critério de Pronto da Spec (GATE-S checklist)
- [ ] 0 marcadores `[NEEDS CLARIFICATION]` abertos
- [ ] Todo RF tem ≥1 AC testável (comando ou Given/When/Then)
- [ ] Escopo IN/OUT explícito
- [ ] `Produtos` e `Converge-Em` declarados no front-matter
- [ ] `Backlog-Code` existe em ROADMAP §0 (ou item novo proposto)
- [ ] Nenhum princípio da CONSTITUTION violado
