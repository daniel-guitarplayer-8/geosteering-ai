# Integração Consensus — Pesquisa Científica para Geosteering AI v2.0

**Documento:** Guia completo da integração de pesquisa científica ao Claude Code
**Data:** 2026-03-31
**Autor:** Daniel Leal
**Versão:** 1.0.0
**Referência:** CLAUDE.md seção "Plugins e Agentes Especializados"

---

## 1. Visão Geral

O Geosteering AI v2.0 integra pesquisa científica diretamente ao fluxo de
desenvolvimento via Claude Code. A integração permite validar constantes físicas,
descobrir novas arquiteturas e losses, e alinhar o pipeline com o estado-da-arte
da literatura publicada.

### Arquitetura da Integração

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Claude Code                                                            │
│                                                                         │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐      │
│  │  /consensus-search│  │  /arxiv-search   │  │  MCP Server      │      │
│  │  (Skill — Fase A) │  │  (Skill — Op. B) │  │  (Fase B — Op. A)│      │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘      │
│           │                      │                      │                │
│           ▼                      ▼                      ▼                │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Fontes de Dados                                                 │   │
│  │                                                                   │   │
│  │  ┌─────────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐   │   │
│  │  │ Semantic     │  │ ArXiv    │  │WebSearch │  │ Consensus  │   │   │
│  │  │ Scholar API  │  │ API      │  │ (Op. C)  │  │ API (Op. A)│   │   │
│  │  │ (sem key)    │  │ (sem key)│  │          │  │ (com key)  │   │   │
│  │  └─────────────┘  └──────────┘  └──────────┘  └────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│           │                                                              │
│           ▼                                                              │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Cache Local: docs/reference/papers/*.json                       │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Hook F: validate-scientific-refs.sh (Fase C — Op. D)            │   │
│  │  Evento: PostToolUse (Edit|Write) — informativo                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Fases de Implementação

### Fase A — Skill via WebFetch (Imediata)

| Atributo | Valor |
|:---------|:------|
| **Arquivo** | `.claude/commands/consensus-search.md` |
| **Tipo** | Skill Claude Code (invocável via `/consensus-search`) |
| **Fontes** | Semantic Scholar API + ArXiv API + WebSearch |
| **API Key** | Nenhuma necessária (fontes abertas) |
| **Status** | ✅ Implementado |

**Uso:**
```bash
/consensus-search PINN electromagnetic inversion resistivity
```

A skill instrui o Claude Code a:
1. Buscar em paralelo no Semantic Scholar e ArXiv
2. Filtrar por relevância ao Geosteering AI (scoring 1-5)
3. Gerar relatório Markdown estruturado
4. Sugerir cache local

### Fase B — MCP Server Dedicado (Opção A — Padrão)

| Atributo | Valor |
|:---------|:------|
| **Diretório** | `tools/consensus-mcp-server/` |
| **Arquivo principal** | `tools/consensus-mcp-server/server.py` |
| **Dependências** | `mcp>=1.0.0`, `httpx>=0.25.0` |
| **API Keys** | `S2_API_KEY` (opcional), `CONSENSUS_API_KEY` (opcional) |
| **Status** | ✅ Ativo (registrado em `.mcp.json` + `enableAllProjectMcpServers`) |

**Ferramentas MCP expostas:**

| Ferramenta | Descrição | Parâmetros |
|:-----------|:----------|:-----------|
| `search_papers` | Busca multi-fonte | query, limit, year_min, year_max, save_cache |
| `get_paper_details` | Detalhes por DOI/ArXiv ID | paper_id |
| `search_arxiv_papers` | Busca exclusiva ArXiv | query, category, limit |
| `list_cached_papers` | Inventário do cache | (nenhum) |

**Ativação:** Registrar em `.mcp.json` (raiz do projeto):
```json
{
  "mcpServers": {
    "consensus": {
      "command": "python",
      "args": ["tools/consensus-mcp-server/server.py"],
      "env": {
        "S2_API_KEY": "${S2_API_KEY}"
      }
    }
  }
}
```

E ativar aprovação automática em `.claude/settings.json`:
```json
{
  "enableAllProjectMcpServers": true
}
```

### Fase C — Automação Inteligente (Opção D — Hooks)

| Atributo | Valor |
|:---------|:------|
| **Arquivo** | `.claude/hooks/validate-scientific-refs.sh` |
| **Evento** | PostToolUse (Edit\|Write) |
| **Tipo** | Informativo (exit 0 — não bloqueia) |
| **Status** | ✅ Ativo (registrado em PostToolUse do settings.json) |

**Regras de detecção:**

| Regra | Detecta | Ação |
|:------|:--------|:-----|
| R1 | Constantes físicas novas/modificadas | Sugere `/consensus-search` |
| R2 | Referências sem formatação padrão | Alerta formatação D12 |
| R3 | Novas factories de loss/modelo | Sugere busca bibliográfica |
| R4 | Referências em MD sem links | Sugere DOI/ArXiv URL |

---

## 3. Opções de Busca

### Opção A — MCP Server (Padrão, Recomendado)

```
┌────────────────────────────────────────────────────────────────┐
│  MCP Server: consensus-scientific-search                        │
│                                                                  │
│  Vantagens:                                                      │
│    ✓ Ferramentas nativas no Claude Code (sem skill manual)       │
│    ✓ Busca paralela (Semantic Scholar + ArXiv)                   │
│    ✓ Cache automático em docs/reference/papers/                  │
│    ✓ Scoring de relevância integrado                             │
│    ✓ Extensível para novas fontes (Consensus API, CrossRef)      │
│                                                                  │
│  Requisitos:                                                     │
│    - pip install mcp httpx                                       │
│    - Configuração em .claude/settings.json                       │
│    - S2_API_KEY (opcional, melhora rate limit)                   │
│                                                                  │
│  Quando usar:                                                    │
│    - Desenvolvimento contínuo com consultas frequentes            │
│    - Integração profunda com o fluxo de trabalho                 │
│    - Necessidade de cache e histórico de buscas                  │
└────────────────────────────────────────────────────────────────┘
```

### Opção B — ArXiv/Semantic Scholar via Skill

```
┌────────────────────────────────────────────────────────────────┐
│  Skill: /arxiv-search                                           │
│                                                                  │
│  Vantagens:                                                      │
│    ✓ Sem dependências (usa WebFetch nativo)                      │
│    ✓ Sem API key necessária                                      │
│    ✓ Acesso direto a preprints com PDF                           │
│    ✓ Filtro por categorias ArXiv                                 │
│                                                                  │
│  Limitações:                                                     │
│    - Sem cache automático                                        │
│    - Sem scoring de relevância automático                        │
│    - Latência de WebFetch (~5-10s por fonte)                     │
│                                                                  │
│  Quando usar:                                                    │
│    - Exploração rápida de preprints recentes                     │
│    - Quando MCP Server não está configurado                      │
│    - Busca por texto completo gratuito                           │
└────────────────────────────────────────────────────────────────┘
```

### Opção C — WebSearch/WebFetch Customizados

```
┌────────────────────────────────────────────────────────────────┐
│  Via WebSearch/WebFetch direto (sem skill)                       │
│                                                                  │
│  Uso: Pedir ao Claude Code para buscar diretamente              │
│    "Busque artigos sobre PINN electromagnetic inversion"         │
│                                                                  │
│  Vantagens:                                                      │
│    ✓ Zero configuração                                           │
│    ✓ Funciona imediatamente                                      │
│                                                                  │
│  Limitações:                                                     │
│    - Resultados não estruturados                                 │
│    - Sem scoring de relevância                                   │
│    - Sem cache                                                   │
│    - Dependente do motor de busca web                            │
│                                                                  │
│  Quando usar:                                                    │
│    - Buscas pontuais e ad-hoc                                    │
│    - Quando nenhuma outra opção está disponível                  │
└────────────────────────────────────────────────────────────────┘
```

### Opção D — Hooks de Automação Científica

```
┌────────────────────────────────────────────────────────────────┐
│  Hook: validate-scientific-refs.sh                               │
│                                                                  │
│  Vantagens:                                                      │
│    ✓ Automático (detecta sem intervenção)                        │
│    ✓ Informativo (não bloqueia o fluxo)                          │
│    ✓ Padronização de referências bibliográficas                  │
│                                                                  │
│  Limitações:                                                     │
│    - Apenas detecta, não busca automaticamente                   │
│    - Regex-based (pode ter falsos positivos)                     │
│                                                                  │
│  Quando usar:                                                    │
│    - Sempre ativo como lembrete passivo                          │
│    - Complementar às outras opções                               │
└────────────────────────────────────────────────────────────────┘
```

---

## 4. Score de Relevância

O sistema classifica artigos de 1 a 5 por relevância ao Geosteering AI:

| Score | Critério | Palavras-chave |
|:-----:|:---------|:---------------|
| **5** | Match direto ao projeto | electromagnetic inversion, LWD, geosteering, resistivity inversion |
| **4** | PINNs para geofísica | PINN geophysics, Helmholtz, anisotropic resistivity, TIV |
| **3** | Inversão geofísica + DL | geophysical inversion, deep learning inversion, well logging |
| **2** | ML para geociências | geoscience, subsurface, petrophysics, borehole |
| **1** | Genérico | ML sem contexto geofísico específico |

---

## 5. Queries Pré-Configuradas

### Para Validação do Pipeline Atual

| ID | Query | Componente Validado |
|:--:|:------|:--------------------|
| Q1 | `LWD electromagnetic tool frequency 20kHz spacing 1 meter` | config.py (errata) |
| Q2 | `skin depth electromagnetic formation resistivity anisotropic` | losses/pinns.py (Maxwell) |
| Q3 | `physics-informed neural network PINN electromagnetic inversion` | losses/pinns.py (todos) |
| Q4 | `deep learning resistivity inversion sequence to sequence 1D` | models/ (44 arquiteturas) |

### Para Expansão do Pipeline

| ID | Query | Componente a Expandir |
|:--:|:------|:----------------------|
| Q5 | `loss function geophysical inversion physics regularization` | losses/catalog.py |
| Q6 | `noise injection curriculum learning geophysics robustness` | noise/functions.py |
| Q7 | `real-time geosteering deep learning look-ahead LWD` | inference/realtime.py |
| Q8 | `uncertainty quantification MC dropout ensemble geophysics` | inference/uncertainty.py |

---

## 6. Mapeamento Fontes → Componentes do Pipeline

| Fonte Científica | Componentes Beneficiados |
|:-----------------|:-------------------------|
| Artigos sobre inversão EM | config.py, losses/, models/ |
| Papers de PINNs | losses/pinns.py, training/loop.py |
| Novas arquiteturas DL | models/registry.py, models/*.py |
| Técnicas de noise injection | noise/functions.py, noise/curriculum.py |
| Geosteering em tempo real | inference/realtime.py, models/geosteering.py |
| Quantificação de incerteza | inference/uncertainty.py, evaluation/ |
| Transformações de features EM | data/feature_views.py, data/geosignals.py |

---

## 7. Configuração Completa

### Arquivo `.claude/settings.json` — Seção MCP

```json
{
  "mcpServers": {
    "consensus": {
      "command": "python",
      "args": ["tools/consensus-mcp-server/server.py"],
      "env": {
        "S2_API_KEY": "${S2_API_KEY}",
        "CONSENSUS_API_KEY": "${CONSENSUS_API_KEY}"
      }
    }
  }
}
```

### Variáveis de Ambiente

| Variável | Obrigatória | Descrição |
|:---------|:-----------:|:----------|
| `S2_API_KEY` | Não | Semantic Scholar API key (melhora rate limit) |
| `CONSENSUS_API_KEY` | Não | Consensus API key (habilita fonte adicional) |

---

## 8. Histórico

| Versão | Data | Descrição |
|:-------|:-----|:----------|
| 1.0.0 | 2026-03-31 | Implementação inicial — Fases A/B/C + Opções A/B/C/D |
