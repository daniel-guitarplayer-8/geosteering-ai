---
Spec: NNNN-slug
Plano-de: spec.md
Status: planejado
Data: YYYY-MM-DD
---

# Plano NNNN — <Título> (HOW / tech)

> Pré-requisito: `spec.md` passou GATE-S (0 `[NEEDS CLARIFICATION]`).

## 1. Gate de Constituição (preencher ANTES da arquitetura)
> **GATE-P falha** se qualquer "Viola?" = Sim sem ADR aprovado.

| Princípio | Aplicável? | Viola? | Como o plano cumpre |
|:--|:--:|:--:|:--|
| I Paridade (3 regimes) | sim | não | não toca `_numba/`/`_jax/`; ou re-roda parity no regime correto |
| II Errata imutável | sim | não | usa defaults de `PipelineConfig` |
| III TF/Keras exclusivo | sim | não | sem torch fora de adapter |
| IV Python 3.13 | sim | não | — |
| V Config-parâmetro | sim | não | `config: PipelineConfig` em toda fn |
| VI Logging | sim | não | logger de `utils/` |
| VII D1–D14 | sim | não | mega-header + docstrings |
| VIII PT-BR | sim | não | — |
| IX SSoT | sim | não | referencia `Backlog-Code` |
| X MVVM (GUI) | <se GUI> | não | View ⊥ ViewModel ⊥ Model |
| XI Fundação / física | sim | não | `Converge-Em: multi_forward.py` |
| XII Gates/Scaler | sim | não | reviewers + ScalerRegistry se há treino/inferência |

## 2. Arquitetura Técnica
```
<diagrama ASCII (bordas Unicode): componentes, fluxo de dados, camadas MVVM se GUI,
 ponto de convergência física multi_forward.py>
```
- Padrões obrigatórios: Factory/Registry, DataPipeline, Config-parâmetro.
- Backend: TF/Keras (prod) · Numba/JAX (simulação) · Qt6 + MVVM (GUI).

## 3. Contratos / APIs
> Assinaturas **EXATAS** (a implementação é feita contra estas). Tipos, retornos, exceções.

```python
# geosteering_ai/<modulo>/<arquivo>.py
def funcao(config: PipelineConfig, x: np.ndarray) -> Resultado:
    """<docstring Google D5: Args, Returns, Raises, Note, Example>"""
```

| Endpoint/Função | Entrada | Saída | Erros |
|:--|:--|:--|:--|
| `POST /v1/simulate` | `SimRequest` (JSON) | `SimResponse` (22-col) | 401/422/500 |

## 4. Estrutura de Arquivos (novos + modificados)
| Arquivo | Ação | Conteúdo |
|:--|:--|:--|
| `geosteering_ai/ingest/__init__.py` | criar | exports `__all__` (D8) |
| `geosteering_ai/ingest/las.py` | criar | `LASReader` |
| `tests/test_ingest_las.py` | criar | AC-1.1..1.3 |
| `geosteering_ai/cli/_main.py` | modificar | flag `--backend auto` |

## 5. Decisões de Design / ADRs
| Decisão | Opções | Escolha | Justificativa | Vira ADR? |
|:--|:--|:--|:--|:--:|
| lib LAS | lasio vs parser próprio | lasio | maduro, MIT | não |
| <irreversível/cross-produto> | … | … | … | **sim → ADR-XXXX** |

> Decisão irreversível ou de alto impacto cross-produto → criar `docs/decisions/ADR-XXXX.md`
> ANTES de implementar (R7 ADR-0001).

## 6. Riscos Técnicos e Mitigações
| Risco | Prob. | Impacto | Mitigação |
|:--|:--:|:--:|:--|
| regressão de paridade | baixa | crítico | parity gate + physics-reviewer |
| OOM GPU high-config | média | alto | auto-chunk (`dispatch.py` guard 80GB) |

## 7. Estratégia de Teste
| Camada | O quê | Onde |
|:--|:--|:--|
| unit | contratos §3 | `tests/test_*.py` |
| paridade | EM se tocado (regime correto) | `run-fortran-parity.sh` |
| golden | inferência se há modelo | `tests/test_golden_models.py` (rtol<1e-6) |
| perf | se Trilha B/A | `.claude/perf_baseline.json` |
| GUI | se MVVM | ViewModel pytest puro + pytest-qt (xvfb-run) |

## 8. Critério de Pronto do Plano (GATE-P checklist)
- [ ] Tabela de constituição sem violação sem-ADR
- [ ] Contratos com assinaturas exatas
- [ ] ADRs irreversíveis criados
- [ ] Estratégia de teste cobre todos os AC da spec
