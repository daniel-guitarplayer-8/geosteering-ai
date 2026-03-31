# ArXiv / Semantic Scholar Search — Pesquisa em Repositórios Abertos

## Identidade

| Atributo | Valor |
|:---------|:------|
| **Skill** | arxiv-search |
| **Opção** | B — context7-plugin + ArXiv/Semantic Scholar (acesso aberto) |
| **Propósito** | Buscar preprints e artigos em repositórios abertos para expansão científica |
| **Projeto** | Geosteering AI v2.0 |
| **Fontes** | ArXiv API, Semantic Scholar API (ambas sem API key) |

---

## Descrição

Skill complementar à `consensus-search`, focada exclusivamente em fontes abertas
(ArXiv e Semantic Scholar). Ideal para:

- Preprints recentes ainda não indexados pelo Consensus
- Artigos com texto completo disponível gratuitamente
- Pesquisa rápida sem dependência de API key externa
- Busca em categorias específicas do ArXiv (cs.LG, physics.geo-ph, eess.SP)

---

## Uso

```
/arxiv-search <query> [--category <cat>] [--year <min_year>]
```

### Exemplos

```bash
# Busca geral
/arxiv-search physics-informed neural network electromagnetic inversion

# Busca filtrada por categoria ArXiv
/arxiv-search PINN resistivity inversion --category physics.geo-ph

# Busca filtrada por ano
/arxiv-search geosteering deep learning --year 2023
```

---

## Instrução ao Agente

Ao receber uma query via `/arxiv-search`:

### Passo 1 — Determinar Categorias ArXiv Relevantes

Mapear a query para categorias ArXiv do projeto:

| Categoria | Descrição | Relevância |
|:----------|:----------|:-----------|
| `physics.geo-ph` | Geofísica | Direta — inversão EM, resistividade |
| `cs.LG` | Machine Learning | Direta — arquiteturas, PINNs |
| `eess.SP` | Processamento de Sinais | Alta — sinais EM, denoising |
| `physics.comp-ph` | Física Computacional | Alta — simulação EM, PDE |
| `cs.AI` | Inteligência Artificial | Média — métodos gerais |
| `stat.ML` | Estatística/ML | Média — incerteza, Bayesian |

### Passo 2 — Busca em Paralelo

**ArXiv API:**
```
URL: http://export.arxiv.org/api/query
Params:
  search_query: cat:<category> AND all:<query>
  start: 0
  max_results: 15
  sortBy: relevance
  sortOrder: descending
```

**Semantic Scholar API:**
```
URL: https://api.semanticscholar.org/graph/v1/paper/search
Params:
  query: <query>
  limit: 15
  fields: title,abstract,year,citationCount,authors,externalIds,url,openAccessPdf
  fieldsOfStudy: Geology,Computer Science,Physics,Engineering
```

### Passo 3 — Classificação de Relevância

Classificar cada resultado por relevância ao Geosteering AI v2.0:

| Score | Critério |
|:-----:|:---------|
| **5** | Inversão EM 1D + Deep Learning (match exato) |
| **4** | PINNs para geofísica OU geosteering com DL |
| **3** | Inversão geofísica genérica OU DL para séries temporais |
| **2** | Geofísica de poço OU arquiteturas seq2seq |
| **1** | ML genérico OU geofísica sem DL |

### Passo 4 — Relatório com Links Diretos

```markdown
## Pesquisa ArXiv/Semantic Scholar: "<query>"

### Artigos Encontrados (ordenados por relevância)

| # | Score | Título | Autores | Ano | Cit. | ArXiv | PDF |
|:-:|:-----:|:-------|:--------|:---:|:----:|:-----:|:---:|
| 1 | 5 | ... | ... | ... | ... | [link] | [pdf] |

### Artigos com Texto Completo Disponível
[Lista de artigos com openAccessPdf disponível]

### Mapeamento para Componentes do Pipeline
| Artigo | Componente v2.0 | Potencial Impacto |
|:-------|:----------------|:-------------------|
| #1 | losses/pinns.py | Nova formulação de resíduo Maxwell |
```

---

## Categorias ArXiv Prioritárias para o Projeto

```
physics.geo-ph    → Geofísica (inversão, resistividade, anisotropia)
cs.LG             → Machine Learning (PINNs, transformers, seq2seq)
eess.SP           → Sinais (denoising, feature extraction EM)
physics.comp-ph   → Computacional (simulação EM, PDE solvers)
cs.NE             → Redes Neurais (arquiteturas, optimização)
```

---

## Diferenças em Relação ao consensus-search

| Aspecto | consensus-search | arxiv-search |
|:--------|:-----------------|:-------------|
| **Fontes** | Consensus + Scholar + ArXiv | Apenas ArXiv + Scholar |
| **API key** | Consensus requer key | Nenhuma necessária |
| **Cobertura** | Papers publicados (peer-reviewed) | Preprints + publicados |
| **Velocidade** | Média (3 fontes) | Rápida (2 fontes abertas) |
| **Texto completo** | Raramente disponível | Frequentemente disponível |
| **Recomendação** | Validação e estado-da-arte | Exploração e descoberta |

---

## Referência

- ArXiv API: https://info.arxiv.org/help/api/
- Semantic Scholar API: https://api.semanticscholar.org/
- Categorias ArXiv: https://arxiv.org/category_taxonomy
- Projeto: docs/reference/consensus_integration.md