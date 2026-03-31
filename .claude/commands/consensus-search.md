# Consensus Search — Pesquisa Científica para Geosteering AI

## Identidade

| Atributo | Valor |
|:---------|:------|
| **Skill** | consensus-search |
| **Fase** | A (imediata) — via WebFetch/WebSearch |
| **Propósito** | Buscar artigos científicos peer-reviewed para validação e expansão do pipeline |
| **Projeto** | Geosteering AI v2.0 |
| **Fontes** | Consensus API, Semantic Scholar, ArXiv |

---

## Descrição

Esta skill permite buscar artigos científicos diretamente do Claude Code para:

1. **Validação de constantes físicas** — confirmar frequency_hz, spacing_meters, skin depth
2. **Expansão do catálogo de losses** — descobrir novas formulações PINN/geofísicas
3. **Descoberta de arquiteturas** — identificar redes seq2seq para inversão EM
4. **Validação de PINNs** — convergência, lambda schedules, Helmholtz
5. **Verificação do estado-da-arte** — posicionar o pipeline vs. literatura recente
6. **Alinhamento científico** — fundamentar decisões de design com evidências

---

## Uso

Invoke esta skill com uma query de pesquisa científica:

```
/consensus-search <query>
```

### Exemplos de Queries Relevantes ao Projeto

```bash
# Validação de constantes físicas
/consensus-search LWD electromagnetic inversion 20kHz skin depth formation resistivity

# Expansão de losses
/consensus-search physics-informed neural network geophysics loss function

# Descoberta de arquiteturas
/consensus-search sequence-to-sequence deep learning resistivity inversion 1D

# Validação de PINNs
/consensus-search Helmholtz equation PINN electromagnetic inversion convergence

# Estado-da-arte geosteering
/consensus-search deep learning geosteering real-time resistivity inversion LWD

# Anisotropia TIV
/consensus-search transversely isotropic resistivity inversion neural network
```

---

## Instrução ao Agente

Ao receber uma query via `/consensus-search <query>`:

### Passo 1 — Busca Multi-Fonte

Execute buscas em paralelo nas seguintes fontes, usando as ferramentas WebSearch e WebFetch:

**Fonte 1: Semantic Scholar API (acesso aberto, sem API key)**
```
URL: https://api.semanticscholar.org/graph/v1/paper/search
Params: query=<query>&limit=10&fields=title,abstract,year,citationCount,authors,externalIds,url
```

**Fonte 2: ArXiv API (acesso aberto, sem API key)**
```
URL: http://export.arxiv.org/api/query
Params: search_query=all:<query>&start=0&max_results=10&sortBy=relevance
```

**Fonte 3: WebSearch genérica**
```
Query: "<query>" site:scholar.google.com OR site:arxiv.org OR site:doi.org
```

### Passo 2 — Filtragem por Relevância

Filtrar resultados usando os seguintes critérios de relevância para o projeto:

| Critério | Peso | Descrição |
|:---------|:----:|:----------|
| **Domínio EM/LWD** | Alto | Artigos sobre inversão eletromagnética, LWD, geosteering |
| **Deep Learning** | Alto | Redes neurais para geofísica, seq2seq, PINNs |
| **Resistividade** | Alto | Inversão de resistividade, anisotropia TIV |
| **Citações** | Médio | Priorizar artigos com > 10 citações |
| **Recência** | Médio | Priorizar artigos de 2020-2026 |
| **Reprodutibilidade** | Baixo | Código disponível, dados abertos |

### Passo 3 — Relatório Estruturado

Gerar relatório no seguinte formato:

```markdown
## Resultados da Pesquisa: "<query>"

### Top 5 Artigos Relevantes

| # | Título | Autores | Ano | Citações | Fonte | DOI |
|:-:|:-------|:--------|:---:|:--------:|:-----:|:----|
| 1 | ... | ... | ... | ... | ... | ... |

### Insights para o Pipeline

1. **Validação**: [O que os artigos confirmam sobre o pipeline atual]
2. **Expansão**: [Novas técnicas/arquiteturas identificadas]
3. **Gaps**: [O que a literatura sugere que falta no pipeline]

### Recomendações de Implementação

- [Ação 1]: [Descrição] — Prioridade: [Alta/Média/Baixa]
- [Ação 2]: [Descrição] — Prioridade: [Alta/Média/Baixa]

### Referências Bibliográficas (BibTeX)

@article{...}
```

### Passo 4 — Cache Local (Opcional)

Se resultados relevantes forem encontrados, sugerir ao usuário salvar em:
```
docs/reference/papers/<query_slug>.md
```

---

## Queries Pré-Configuradas para o Projeto

As seguintes queries são recomendadas para validação periódica do pipeline:

### Q1: Validação da Errata Física
```
LWD electromagnetic tool frequency 20kHz spacing 1 meter skin depth anisotropic
```

### Q2: Estado-da-Arte em Inversão EM com DL
```
deep learning electromagnetic inversion resistivity well logging 2024 2025
```

### Q3: PINNs para Geofísica
```
physics-informed neural network PINN geophysics electromagnetic inversion
```

### Q4: Arquiteturas seq2seq para Séries Temporais Geofísicas
```
sequence to sequence neural network geophysical time series inversion
```

### Q5: Geosteering em Tempo Real
```
real-time geosteering deep learning resistivity look-ahead LWD
```

### Q6: Anisotropia TIV e Constraint Layers
```
transverse isotropy resistivity constraint neural network output layer
```

### Q7: Loss Functions para Inversão Geofísica
```
loss function geophysical inversion physics regularization neural network
```

### Q8: Noise Injection e Robustez
```
noise injection curriculum learning deep learning robustness geophysics
```

---

## Integração com o Pipeline

Quando artigos relevantes forem encontrados, mapear para componentes do v2.0:

| Componente do Pipeline | Tipo de Artigo Relevante |
|:-----------------------|:------------------------|
| `losses/catalog.py` | Novas formulações de loss para inversão |
| `losses/pinns.py` | PINNs, Helmholtz, TIV constraints |
| `models/registry.py` | Novas arquiteturas seq2seq |
| `noise/functions.py` | Técnicas de noise injection |
| `data/feature_views.py` | Transformações de componentes EM |
| `data/geosignals.py` | Novos geosinais compensados |
| `training/callbacks.py` | Schedules, curriculum, adaptação |
| `config.py` | Constantes físicas, hiperparâmetros |
| `inference/realtime.py` | Latência, causalidade, sliding window |

---

## Limitações

| Aspecto | Limitação | Mitigação |
|:--------|:----------|:----------|
| **Rate limits** | Semantic Scholar: 100 req/5 min | Cache local em docs/reference/papers/ |
| **Paywall** | Muitos papers requerem acesso institucional | Usar abstract + ArXiv preprints |
| **Qualidade** | Nem todo resultado é relevante ao domínio EM/LWD | Filtragem manual pelo usuário |
| **Latência** | WebFetch pode ser lento (~5-10s por fonte) | Busca paralela em múltiplas fontes |

---

## Referência

- Semantic Scholar API: https://api.semanticscholar.org/
- ArXiv API: https://info.arxiv.org/help/api/
- Consensus: https://consensus.app (requer API key para automação)
- Projeto: docs/reference/consensus_integration.md