# Arquitetura de Desenvolvimento Multi-Agente — Geosteering AI 2.0
## Claude Code + Google Antigravity + Orquestração Avançada de IAs

<!-- Metadados -->
| Campo | Valor |
|:------|:------|
| **Data** | 2026-05-02 |
| **Versão do projeto** | Geosteering AI v2.0 |
| **Modelos utilizáveis** | Claude Opus 4.7 (1M ctx), Sonnet 4.6, Haiku 4.5 |
| **Plano** | Claude Max 5× |
| **Ambiente** | Google Antigravity + extensão Claude Code |
| **Propósito** | Relatório base para aprofundamento com Opus 4.7 1M |

---

> **⚠️ Recomendação de Modelo**: Este relatório foi gerado com Claude Sonnet 4.6.
> Para o aprofundamento arquitetural completo, **recomendo fortemente Claude Opus 4.7 (1M contexto)**.
> Motivo: o projeto Geosteering AI tem ~46.000 LOC em 73 arquivos Python, mais Fortran, benchmarks,
> documentação e arquivos de configuração. O Opus 4.7 com 1M de contexto pode carregar o projeto
> inteiro + o histórico desta conversa + este relatório em um único contexto coerente, permitindo
> raciocínio arquitetural de nível superior que o Sonnet 4.6 não consegue sustentar com a mesma
> profundidade. Este documento serve como briefing para essa sessão com Opus.

---

## Sumário

1. [O Ecossistema de Desenvolvimento](#1-o-ecossistema-de-desenvolvimento)
2. [Claude Code — Inventário Completo de Recursos](#2-claude-code--inventário-completo-de-recursos)
3. [Google Antigravity e Google AI Pro — Integração](#3-google-antigravity-e-google-ai-pro--integração)
4. [Arquitetura Multi-Agente para Geosteering AI 2.0](#4-arquitetura-multi-agente-para-geosteering-ai-20)
5. [Workflows Detalhados por Domínio](#5-workflows-detalhados-por-domínio)
6. [Estratégia de Git Worktrees](#6-estratégia-de-git-worktrees)
7. [MCP Servers — Extensões Avançadas](#7-mcp-servers--extensões-avançadas)
8. [Hooks — Automação e Qualidade Contínua](#8-hooks--automação-e-qualidade-contínua)
9. [Seleção de Modelos por Tarefa](#9-seleção-de-modelos-por-tarefa)
10. [Orquestração de IAs em Diferentes Camadas](#10-orquestração-de-ias-em-diferentes-camadas)
11. [Roadmap de Implementação do Ambiente](#11-roadmap-de-implementação-do-ambiente)
12. [Estrutura de Arquivos de Configuração](#12-estrutura-de-arquivos-de-configuração)
13. [Análise de Riscos e Mitigações](#13-análise-de-riscos-e-mitigações)
14. [Recomendações para Sessão com Opus 4.7 1M](#14-recomendações-para-sessão-com-opus-47-1m)

---

## 1. O Ecossistema de Desenvolvimento

### 1.1 Visão Geral do Ambiente

O desenvolvimento do Geosteering AI 2.0 ocorre em um ecossistema único que combina três camadas de inteligência artificial:

```
╔═══════════════════════════════════════════════════════════════════════════╗
║  CAMADA 1 — AMBIENTE IDE: Google Antigravity                              ║
║  (IDE baseado em nuvem/local com IA integrada + Google AI Pro)            ║
║                                                                           ║
║  ┌─────────────────────────────────────────────────────────────────────┐  ║
║  │  Extensão Claude Code (Claude Max 5×)                              │  ║
║  │    └─ Acesso a Claude Opus 4.7 / Sonnet 4.6 / Haiku 4.5           │  ║
║  │    └─ Multi-agent orchestration nativo                             │  ║
║  │    └─ MCP Servers configuráveis                                    │  ║
║  │    └─ Hooks, Skills, Worktrees                                     │  ║
║  └─────────────────────────────────────────────────────────────────────┘  ║
║                                                                           ║
║  ┌─────────────────────────────────────────────────────────────────────┐  ║
║  │  Google AI integrado (Gemini 2.5 Pro / Gemini Flash)               │  ║
║  │    └─ Completions no editor                                        │  ║
║  │    └─ Busca de código                                              │  ║
║  │    └─ Review automático de PR                                      │  ║
║  └─────────────────────────────────────────────────────────────────────┘  ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  CAMADA 2 — EXECUÇÃO: GitHub Actions + Local                              ║
║  (CI/CD, testes automáticos, validação)                                   ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  CAMADA 3 — GPU: Google Colab Pro+                                        ║
║  (Treinamento de redes neurais: T4/A100)                                  ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

### 1.2 Complexidade do Projeto Geosteering AI 2.0

Para justificar a necessidade de uma arquitetura multi-agente robusta, o projeto tem:

| Domínio | LOC | Arquivos | Complexidade |
|:--------|:---:|:--------:|:------------:|
| Configuração + Pipeline | ~5.000 | 8 | Alta (dataclass, validação) |
| Dados (loading, splitting, FV, GS) | ~8.000 | 11 | Alta (física geofísica) |
| Modelos (48 arqs, 9 famílias) | ~12.000 | 13 | Muito Alta |
| Losses (26 funções) | ~3.000 | 3 | Média |
| Treinamento + Callbacks | ~4.000 | 6 | Alta |
| Simulador Python (Numba JIT) | ~6.000 | 12+ | Extremamente Alta |
| Simulador Fortran (`tatu.x`) | ~15.000 | 8 (F08) | Extremamente Alta |
| Inferência + Exportação | ~2.500 | 4 | Média |
| Avaliação + Visualização | ~5.000 | 22 | Média |
| Utilitários + Testes | ~3.000 | 6 | Baixa-Média |
| **Total** | **~63.500** | **~93** | — |

**Obs.**: Fortran incluído para contexto. Python direto: ~46.000 LOC / 73 arquivos.

---

## 2. Claude Code — Inventário Completo de Recursos

### 2.1 Tipos de Agentes Disponíveis

Claude Code suporta quatro paradigmas de agência, cada um com trade-offs claros:

#### 2.1.1 Subagentes (Principal mecanismo)

```
Invocação: tool Agent(subagent_type=..., prompt=..., isolation=..., model=...)

Tipos especializados disponíveis:
  Explore              → Busca somente leitura no codebase
  Plan                 → Planejamento arquitetural
  feature-dev:code-architect    → Design de features
  feature-dev:code-explorer     → Análise de features existentes
  feature-dev:code-reviewer     → Revisão de código
  code-simplifier               → Refatoração e limpeza
  coderabbit:code-reviewer      → Análise profunda de PRs
  context7-plugin:docs-researcher → Busca de documentação
  claude-code-guide             → Dúvidas sobre Claude Code
  general-purpose               → Tarefas multi-step autônomas
  plugin-dev:agent-creator      → Criação de novos agentes
  plugin-dev:plugin-validator   → Validação de plugins
  statusline-setup              → Configuração de linha de status
  superpowers:code-reviewer     → Review pós-implementação de sprints
```

**Características dos subagentes**:
- Isolamento de contexto (não veem a conversa principal)
- Retornam resultado único ao agente orquestrador
- Podem ser executados em paralelo (envio em único bloco de tool calls)
- Podem receber parâmetro `isolation: "worktree"` para isolamento total
- Podem rodar em background com `run_in_background: true`

#### 2.1.2 Agent Teams (Experimental, Alto Poder)

```bash
# Ativar
export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1
```

```
Modelo:
  Orquestrador ──→ Cria teammates com tarefas específicas
                   ├── Teammate A: Especialista Numba JIT
                   ├── Teammate B: Especialista TensorFlow/Keras
                   └── Teammate C: Especialista Fortran/F08

  Teammates podem:
    - Comunicar entre si (SendMessage to: teammate_name)
    - Trabalhar em paralelo na mesma base de código
    - Reportar ao orquestrador quando concluem

  Armazenamento: ~/.claude/teams/{name}/config.json + task list
  Custo: Muito alto (cada teammate = janela de contexto completa)
  Uso ideal: Debugging complexo, análise multi-perspectiva, code review
```

#### 2.1.3 Parallel Agent Execution (Múltiplos subagentes simultâneos)

```python
# Exemplo: Executar 3 agentes em paralelo em uma única resposta
# → Claude envia 3 tool calls Agent() no mesmo bloco de resposta

Agent(research_Numba_optimization)     # roda simultaneamente
Agent(research_JAX_vmap)               # roda simultaneamente
Agent(research_TF_XLA_optimization)    # roda simultaneamente

# Todos retornam antes de prosseguir
```

**Regra**: Agentes independentes (sem dependência de resultado entre si) devem ser enviados em um único bloco. Claude Code executa todos em paralelo automaticamente.

#### 2.1.4 Background Agents

```
Agent(prompt=..., run_in_background=true)

Comportamento:
  - Inicia sem bloquear o orquestrador
  - Notificação automática quando conclui
  - Ideal para: rodar benchmarks longos, gerar relatórios, compilar Fortran
  - Não esperar: continuar trabalho independente enquanto o agent roda
```

### 2.2 Git Worktrees — Desenvolvimento Paralelo Isolado

```
claude --worktree feat/simulator-v2.22   # Cria worktree para sprint 22
claude --worktree feat/ml-pipeline-v3.0  # Cria worktree para pipeline ML
claude --worktree hotfix/fortran-parity  # Cria worktree para hotfix

Estrutura criada:
  .git/worktrees/feat-simulator-v2.22/   (checkout isolado, branch própria)
  .git/worktrees/feat-ml-pipeline-v3.0/
  .git/worktrees/hotfix-fortran-parity/

  Cada worktree:
    ✓ Branch própria
    ✓ Staging area independente
    ✓ Sem conflito de arquivo com outras worktrees
    ✗ Compartilha .git/ (histórico, objetos)

.worktreeinclude:
  Copia arquivos gitignored para cada worktree:
  .env → tokens, API keys
  Geosteering_AI_venv/ → não recria venv em cada worktree
```

**Uso com subagentes**:
```python
# Subagente com isolamento total em worktree
Agent(
    subagent_type="feature-dev:code-architect",
    prompt="Implemente FLAT prange em kernel.py",
    isolation="worktree",   # ← cria worktree dedicada, limpa após
    model="opus"
)
```

### 2.3 Skills e Slash Commands

```
Localização: .claude/commands/*.md
             ~/.claude/commands/*.md (global)

Estrutura de uma skill:
  ---
  name: geosteering-simulator
  description: |
    Simulador Python Numba JIT — domínio, padrões, otimizações.
    Usar para qualquer tarefa relacionada a _numba/, forward.py,
    multi_forward.py, propagation.py, dipoles.py, kernel.py.
  tools: [Read, Edit, Bash, Agent]
  model: claude-opus-4-7   # Modelo específico para esta skill
  ---

  [conteúdo da skill: regras, padrões, exemplos]

Invocação:
  /geosteering-simulator    ← pelo usuário no chat
  Skill tool                ← pelo próprio Claude Code (auto-detecção)
```

**Skills do projeto Geosteering AI existentes**:
- `geosteering-v2`: Domínio físico + padrões código v2.0
- `geosteering-simulator-python`: Simulador Python otimizado
- `geosteering-simulator-fortran`: Simulador Fortran v10.0
- `geosteering-v5015`: Código legado (referência)
- `consensus-search`: Pesquisa científica
- `arxiv-search`: Busca de preprints

### 2.4 TodoWrite — Rastreamento de Tarefas

```
Funcionalidade: Mantém lista de tarefas persistente por sessão
Uso ideal: Sprints com múltiplas etapas

Fluxo:
  1. Agent cria lista inicial de TODOs
  2. Cada tarefa completada → marked done imediatamente
  3. Lista visível para o orquestrador
  4. Subagentes podem atualizar TODOs
```

### 2.5 Hooks — Automação Contínua

Os hooks executam em resposta a eventos do ciclo de vida do Claude Code:

```
EVENTOS DE HOOK DISPONÍVEIS:
  PreToolUse        → Antes de qualquer tool call (pode bloquear)
  PostToolUse       → Após tool call (pode processar resultado)
  PostToolBatch     → Após lote de tool calls
  SubagentStart     → Antes de subagente iniciar
  SubagentStop      → Após subagente terminar
  TaskCreated       → Nova tarefa em agent teams
  TaskCompleted     → Tarefa concluída em agent teams
  TeammateIdle      → Teammate sem trabalho (pode atribuir nova tarefa)
  Stop              → Antes de Claude finalizar resposta
  SessionStart      → Início de sessão
  SessionEnd        → Fim de sessão

TIPOS DE HOOK:
  1. Command    → Shell script (mais rápido, determinístico)
  2. Prompt     → LLM single-turn (julgamento baseado em IA)
  3. Agent      → LLM multi-turn (pode usar tools, ler arquivos)
  4. HTTP       → Webhook externo (logs, monitoramento)
  5. MCP Tool   → Ferramenta MCP pré-conectada
```

**Escopo e herança de hooks**:
```
~/.claude/settings.json          (usuário — todos os projetos)
    .claude/settings.json        (projeto — equipe compartilha)
    .claude/settings.local.json  (projeto — gitignored, segredos)
        plugin/hooks.json        (plugin — quando habilitado)
```

### 2.6 CLAUDE.md — Instruções Hierárquicas

```
Hierarquia de carregamento (da mais específica para mais geral):
  CLAUDE.md                              (raiz do projeto)
  src/.claude/CLAUDE.md                  (diretório src/)
  geosteering_ai/.claude/CLAUDE.md       (pacote principal)
  geosteering_ai/simulation/CLAUDE.md    (módulo simulador)
  tests/.claude/CLAUDE.md               (testes)
  ~/.claude/CLAUDE.md                    (global do usuário)

Uso avançado — regras por caminho:
  .claude/rules/simulation.md → aplicado a **/simulation/**
  .claude/rules/fortran.md    → aplicado a **/*.f08, **/*.f90
  .claude/rules/tests.md      → aplicado a **/tests/**
```

### 2.7 MCP Servers — Protocolo de Extensão

```
Model Context Protocol (MCP): padrão aberto para extensão de IAs

TIPOS DE INTEGRAÇÃO:
  1. Stdio Server (local):
     → Processo Python/Node local
     → Comunicação via stdin/stdout
     → Zero latência de rede
     → Ideal para ferramentas de domínio específico

  2. HTTP/SSE Server (remoto):
     → Microserviço acessível via rede
     → Pode ter estado persistente
     → Ideal para integrações com infraestrutura existente

  3. Claude.ai Connectors:
     → Pré-integrados (Figma, Google Drive, Consensus, bioRxiv)
     → Ativados via settings.json

CONFIGURAÇÃO:
  .claude/mcp-servers.json:
  {
    "geosteering-physics": {
      "type": "stdio",
      "command": "python",
      "args": [".claude/mcp/physics_validator.py"],
      "tools": ["validate_em_tensor", "check_fortran_parity"]
    },
    "jupyter-runtime": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "jupyter_mcp_server"]
    }
  }
```

### 2.8 Scheduled Tasks e Automação

```
/loop [intervalo] [prompt]

Variantes:
  /loop 5m run-benchmarks          → intervalo fixo de 5 minutos
  /loop check-colab-training       → intervalo dinâmico (Claude escolhe)
  /loop 15m                        → manutenção geral automática

CronCreate (via ToolSearch):
  → Agendamento com expressão cron (5 campos)
  → Persiste entre sessões
  → Ideal para: relatórios periódicos, sync de resultados

ScheduleWakeup:
  → Sleep dinâmico dentro de /loop
  → Respeita TTL de cache (< 270s = cache warm)
  → Ideal para: monitoramento de build longo, aguardar GPU
```

---

## 3. Google Antigravity e Google AI Pro — Integração

### 3.1 Contexto: Google Antigravity como Ambiente de Desenvolvimento

> **Nota sobre conhecimento**: Meu treinamento vai até agosto de 2025. "Google Antigravity" pode ser um produto lançado após essa data ou uma evolução de produtos existentes. A análise abaixo integra o que sei sobre o ecossistema Google AI para desenvolvimento e as capacidades que qualquer ambiente de desenvolvimento moderno do Google ofereceria.

O Google Antigravity, no contexto desta arquitetura, funciona como:
```
┌──────────────────────────────────────────────────────────────────────────┐
│  GOOGLE ANTIGRAVITY (IDE + Runtime + Cloud)                              │
│                                                                          │
│  Componentes esperados:                                                  │
│    1. Editor/IDE com IA nativa (successor ao Project IDX)               │
│       → Suporte a extensões VS Code (onde Claude Code roda)             │
│       → Integração com Google Gemini como IA nativa                     │
│       → Terminal integrado com acesso a VMs                             │
│                                                                          │
│    2. Google AI Pro (assinatura):                                        │
│       → Gemini 2.5 Pro / Gemini Ultra para completions                  │
│       → Gemini Code Assist avançado                                      │
│       → Acesso a ferramentas avançadas de análise de código             │
│       → Integração com Google Cloud (BigQuery, Vertex AI, Colab)        │
│                                                                          │
│    3. Claude Code (extensão Max 5×):                                    │
│       → Roda DENTRO do Antigravity como extensão                        │
│       → Acessa o filesystem do projeto                                  │
│       → Executa subagentes e workflows                                  │
│       → MCP Servers locais no ambiente                                  │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Modelo de Interação IA-IA (Claude + Gemini)

```
CENÁRIO: Dois modelos de IA distintos, complementares

  Google Gemini (via Antigravity):
    Pontos fortes:
      ✓ Conhecimento do ecossistema Google (TF, JAX, GCP, Colab)
      ✓ 2M de contexto (Gemini 1.5 Pro) ou equivalente
      ✓ Integração nativa com Google Drive, BigQuery, Colab
      ✓ Code suggestions em tempo real no editor (inline)
      ✓ Busca semântica em código
      ✓ Google Search integrado
    Uso ideal no projeto:
      • Completions inline ao escrever código TF/JAX/Keras
      • Consultas à documentação do Google Cloud
      • Busca de exemplos de código no Google
      • Análise de Jupyter notebooks (.ipynb) no Colab
      • Sugestões de otimização XLA/TF

  Claude Code (via extensão Max 5×):
    Pontos fortes:
      ✓ Raciocínio arquitetural profundo
      ✓ Execução de multi-agent workflows
      ✓ Ferramenta de software completa (Edit, Bash, Read, Write)
      ✓ Memória persistente entre sessões
      ✓ Skills customizáveis para domínio específico
      ✓ Hooks para automação
      ✓ Opus 4.7 para tarefas de máxima complexidade
    Uso ideal no projeto:
      • Implementação de sprints complexos
      • Refatoração arquitetural
      • Análise de bugs físicos/geofísicos
      • Orquestração de subagentes especializados
      • Geração de relatórios técnicos
      • Code review profundo
```

### 3.3 Estratégia de Orquestração IA Dual

```
CAMINHO DE DECISÃO: Quando usar qual IA?

┌─────────────────────────────────────────────────────────────────────┐
│  Tarefa requer raciocínio arquitetural profundo?                    │
│    SIM → Claude Code (Opus 4.7 se complexidade máxima)              │
│    NÃO → continua                                                   │
│                                                                     │
│  Tarefa é completion inline ao editar um arquivo específico?        │
│    SIM → Gemini (mais rápido, contexto local, inline)               │
│    NÃO → continua                                                   │
│                                                                     │
│  Tarefa envolve ecossistema Google (JAX, TF, Colab, GCP)?          │
│    SIM → Gemini como primeira consulta, Claude para implementar     │
│    NÃO → continua                                                   │
│                                                                     │
│  Tarefa é multi-arquivo, cross-module, sprint completo?             │
│    SIM → Claude Code com Agent workflow                             │
│    NÃO → Claude Code one-shot                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.4 Google Colab Pro+ como Camada de GPU

```
Papel na arquitetura:
  Geosteering AI não treina localmente — usa Google Colab Pro+ para:
    - Treinamento de redes neurais (T4/A100/TPU)
    - Validação GPU do simulador JAX
    - Benchmarks de alta escala
    - Geração de datasets sintéticos (30k-100k modelos)

Integração com Claude Code:
  1. Claude Code edita código localmente
  2. git push → GitHub
  3. Colab: pip install git+https://github.com/...@tag
  4. Colab executa treinamento
  5. Claude Code monitora via /loop polling do GitHub Actions

MCP Server "colab" (a construir):
  .claude/mcp/colab_monitor.py:
    - connect_to_colab_runtime()
    - get_training_metrics()
    - trigger_colab_cell()
    - get_gpu_memory_usage()
```

---

## 4. Arquitetura Multi-Agente para Geosteering AI 2.0

### 4.1 Hierarquia de Agentes

```
╔════════════════════════════════════════════════════════════════════════╗
║  NÍVEL 0 — ORQUESTRADOR (Daniel + Claude Code principal)              ║
║  Modelo: Claude Opus 4.7 (sessões de sprint) / Sonnet 4.6 (rotina)   ║
║  Contexto: CLAUDE.md + memory/ + plano ativo                         ║
╠════════════════════════════════════════════════════════════════════════╣
║  NÍVEL 1 — AGENTES DE DOMÍNIO (subagentes especializados)             ║
║                                                                        ║
║  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐    ║
║  │ Agente Simulador │  │ Agente ML/DL     │  │ Agente Fortran   │    ║
║  │ Python Numba JIT │  │ TF/Keras         │  │ tatu.x           │    ║
║  │ Modelo: Opus     │  │ Modelo: Sonnet   │  │ Modelo: Sonnet   │    ║
║  │ Skill: sim-py    │  │ Skill: geo-v2    │  │ Skill: sim-fort  │    ║
║  └──────────────────┘  └──────────────────┘  └──────────────────┘    ║
║                                                                        ║
║  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐    ║
║  │ Agente JAX/GPU   │  │ Agente Dados     │  │ Agente Testes    │    ║
║  │ vmap, XLA, TPU   │  │ Pipeline, GS, FV │  │ pytest, CI       │    ║
║  │ Modelo: Opus     │  │ Modelo: Sonnet   │  │ Modelo: Haiku    │    ║
║  └──────────────────┘  └──────────────────┘  └──────────────────┘    ║
╠════════════════════════════════════════════════════════════════════════╣
║  NÍVEL 2 — AGENTES DE QUALIDADE (validação cruzada)                   ║
║                                                                        ║
║  ┌──────────────────────────────────────────────────────────────────┐  ║
║  │ Agente Review Físico  │ Agente Perf  │ Agente Doc PT-BR         │  ║
║  │ paridade Fortran      │ benchmarks  │ acentuação, D1-D14        │  ║
║  │ Modelo: Sonnet        │ Modelo: Haiku│ Modelo: Haiku            │  ║
║  └──────────────────────────────────────────────────────────────────┘  ║
╠════════════════════════════════════════════════════════════════════════╣
║  NÍVEL 3 — INFRAESTRUTURA (hooks automáticos, sem interação)          ║
║  Hooks: PreToolUse, PostToolUse, Stop                                  ║
║  Scripts: pytest, mypy, ruff, paridade Fortran automática             ║
╚════════════════════════════════════════════════════════════════════════╝
```

### 4.2 Agentes Especializados — Definição Detalhada

#### Agente Simulador Python (Critical Path)

```yaml
# .claude/commands/geosteering-simulator-python.md (já existe, expandir)
---
name: geosteering-simulator-python
description: |
  Simulador Python Numba JIT para EM 1D TIV. Domínio completo:
  _numba/ (kernel, propagation, dipoles, hankel, geometry, rotation),
  forward.py, multi_forward.py, _workers.py.
  Padrões: @njit sem nested prange, nogil=True, cache=True.
  Paridade Fortran <1e-12 é inviolável.
  Anti-pattern: parallel=True em funções chamadas de prange outer.
tools: [Read, Edit, Bash, Agent(Explore), Agent(feature-dev:code-reviewer)]
model: claude-opus-4-7
isolation: worktree  # Toda sprint nova = worktree isolada
---
```

#### Agente ML/DL Pipeline

```yaml
# .claude/commands/geosteering-ml-pipeline.md
---
name: geosteering-ml-pipeline
description: |
  Pipeline TF/Keras para inversão 1D de resistividade. Domínio:
  models/ (48 arqs), losses/ (26 funções), training/, inference/.
  Framework exclusivo: TensorFlow 2.x / Keras (PyTorch PROIBIDO).
  Config sempre via PipelineConfig. Factory pattern obrigatório.
tools: [Read, Edit, Bash, WebFetch]
model: claude-sonnet-4-6
---
```

#### Agente de Testes (Haiku — econômico)

```yaml
# .claude/commands/geosteering-tests.md
---
name: geosteering-tests
description: |
  Geração e manutenção de testes para o projeto Geosteering AI.
  pytest, paridade Fortran, smoke tests, validação física.
  Sempre rodar pytest antes de reportar sucesso.
tools: [Read, Bash(pytest*), Write]
model: claude-haiku-4-5-20251001
---
```

### 4.3 Topologia de Comunicação entre Agentes

```
PADRÃO 1: PIPELINE (dependência sequencial)
  Orquestrador
    → Agente Arquitetura [Explore + Plan]
        → Agente Implementação [Edit + Write]
            → Agente Review [Read + análise]
                → Agente Testes [Bash + pytest]
                    → Orquestrador (resultado final)

PADRÃO 2: PARALELO (independência)
  Orquestrador
    ├─→ Agente Simulador Numba [sprint 22.1] ←paralelos→
    ├─→ Agente ML Pipeline [refactor] ←paralelos→
    └─→ Agente Docs [update CHANGELOG] ←paralelos→
    (todos retornam para orquestrador)

PADRÃO 3: FAN-OUT + MERGE (análise multi-perspectiva)
  Orquestrador
    ├─→ Review Agente A [segurança]
    ├─→ Review Agente B [performance]
    └─→ Review Agente C [física/geofísica]
    (merge de resultados pelo orquestrador)

PADRÃO 4: BACKGROUND + MONITOR (jobs longos)
  Orquestrador
    └─→ Agente Benchmark [run_in_background=true]
  (continua trabalhando em outra coisa)
  → Notificação automática quando benchmark termina
```

---

## 5. Workflows Detalhados por Domínio

### 5.1 Workflow de Sprint do Simulador Numba JIT

```
Sprint (ex: v2.22 — FLAT prange):

  FASE 1 — PLANEJAMENTO (15 min)
    Orquestrador + Opus 4.7:
      → /geosteering-simulator-python
      → Análise do código atual (forward.py, kernel.py)
      → Plano de implementação: arquivos a modificar, riscos, testes

  FASE 2 — EXPLORAÇÃO PARALELA (5 min)
    Subagente Explore A: "Como _fields_in_freqs_kernel_cached é chamado?"
    Subagente Explore B: "Quais testes de paridade Fortran existem?"
    (executados em paralelo)

  FASE 3 — IMPLEMENTAÇÃO EM WORKTREE (60-90 min)
    claude --worktree feat/simulator-v2.22
      → Subagente code-architect: design do FLAT kernel
      → Implementação em forward.py
      → Adaptação em multi_forward.py
      → Novos testes em tests/test_simulation_flat_prange.py

  FASE 4 — VALIDAÇÃO CRUZADA (20 min)
    Subagente físico: "Verificar paridade Fortran nos 7 modelos canônicos"
    Subagente performance: "Rodar bench_v214_numba.py Cenários E, F, G"
    (paralelos)

  FASE 5 — REVIEW E MERGE (15 min)
    Subagente code-reviewer: review do diff completo
    Commit + Push + PR

  FASE 6 — DOCUMENTAÇÃO (20 min)
    Subagente Haiku: update CHANGELOG + ROADMAP + CLAUDE.md
    Subagente Haiku: update memory/project_simulation_manager_v222.md
```

### 5.2 Workflow de Treinamento de Rede Neural

```
Ciclo de Treinamento:

  FASE 1 — GERAÇÃO DE DADOS (background)
    Subagente Simulador [run_in_background=true]:
      → Gera 30.000 modelos via ProcessPoolExecutor
      → Salva em data/raw/synthetic_v3/
      → Notifica ao concluir

    Enquanto aguarda:
    Subagente ML:
      → Prepara config YAML de treinamento
      → Atualiza PipelineConfig para novo dataset

  FASE 2 — PUSH PARA COLAB
    Claude Code:
      → git push tag v{version}
      → Gera notebook Colab com pip install @tag
      → Instrução para Daniel executar no Colab Pro+

  FASE 3 — MONITORAMENTO
    /loop 10m check-colab-training
      → Verifica GitHub Actions para logs
      → Alerta se loss divergiu
      → Reporta métricas parciais

  FASE 4 — AVALIAÇÃO
    Subagente Avaliação:
      → Compara métricas com modelo anterior
      → Verifica overfitting por modelo geológico
      → Gera relatório de validação
```

### 5.3 Workflow de Code Review Multi-Perspectiva

```
Para PRs de alta criticidade (simulador, física):

  Agent Teams (3 teammates):
    Teammate A — Especialista Físico:
      "Verificar se tensores EM preservam simetria de Maxwell,
       paridade Fortran <1e-12, indistinguibilidade de unidades"

    Teammate B — Especialista Performance:
      "Verificar nested prange, nogil=True, cache=True,
       oversubscription, benchmark Cenário E e A"

    Teammate C — Especialista Código:
      "PEP8, tipagem, docstrings D1-D14, PT-BR acentuado,
       logging vs print, PipelineConfig vs globals"

    Orquestrador:
      → Coleta resultados dos 3 teammates
      → Prioriza findings: P0 > P1 > P2
      → Gera relatório de review consolidado
```

### 5.4 Workflow de Alta Resistividade (Novo)

```
Validação para ρ > 1000 Ω·m:

  Subagente Físico (Opus 4.7):
    → Verifica estabilidade numérica em propagation.py
    → Analisa overflow/underflow para exp(kr × Δz) com kr alto
    → Valida contra modelos canônicos propostos:
       carbonato_5c (ρ_max=5000), evaporita_3c (ρ_max=100000)

  Subagente Testes:
    → Cria test_simulation_high_resistivity.py
    → 5 novos modelos canônicos
    → Gate: <1e-12 vs Fortran em todos

  Loop de validação:
    while not all_pass:
      → Rodar paridade Fortran nos novos modelos
      → Se falhar: Subagente Físico analisa e corrige
      → Se passar: commit
```

### 5.5 Workflow de Documentação Automática

```
Hook PostToolUse(Edit):
  → Se arquivo editado em geosteering_ai/:
    Subagente Haiku [model=haiku]:
      → Verificar se docstring tem PT-BR acentuado
      → Verificar se comentários seguem padrão D1-D14
      → Sugerir correções se necessário

Hook Stop:
  → Se ≥5 commits desde último relatório em docs/reports/:
    Subagente Haiku:
      → Gerar relatório MD seguindo template
      → Atualizar CHANGELOG
      → Salvar em docs/reports/v{version}_{date}.md
```

---

## 6. Estratégia de Git Worktrees

### 6.1 Topologia de Branches Proposta

```
main ──────────────────────────────────────────────────────────→ produção

feat/simulator-v2.22 ──→ merge em main (FLAT prange)
feat/simulator-v2.23 ──→ merge em main (fastmath + adaptive threads)
feat/ml-pipeline-v3.0 ─→ merge em main (SurrogateNet v3)
feat/jax-v1.7 ──────────→ merge em main (vmap real flip default)
hotfix/* ──────────────→ merge em main (urgente)

Worktrees mapeadas 1:1 com features ativas:
  ~/Geosteering_AI/          (main)
  ~/Geosteering_AI_sim22/    (feat/simulator-v2.22)
  ~/Geosteering_AI_ml3/      (feat/ml-pipeline-v3.0)
```

### 6.2 Configuração do `.worktreeinclude`

```
# .worktreeinclude
.env                        # variáveis de ambiente e tokens
.claude/settings.local.json # hooks locais (não commitados)
Geosteering_AI_venv/        # NÃO recriar venv em cada worktree
__pycache__/                # cache Python (não recriar)
.numba_cache/               # cache Numba compilado
```

### 6.3 Isolamento por Subagente

```python
# Fluxo: Subagente com isolation="worktree"

# Orquestrador invoca:
Agent(
    subagent_type="feature-dev:code-architect",
    prompt="""
    Implemente Sprint 22.1: FLAT prange em forward.py.

    Contexto: kernel.py:670 tem _fields_in_freqs_kernel_cached com
    range(nf) serial. Criar novo kernel _simulate_combined_prange_flat
    que use prange(n_combos × n_pos × nf) com decomposição de índice flat.

    Ver: geosteering_ai/simulation/forward.py linhas 351-468
    Meta: paridade Fortran <1e-12 preservada + speedup ≥1.3× em Cenário F
    """,
    isolation="worktree"  # ← branch dedicada criada automaticamente
)

# Resultado: se o agente fizer mudanças → retorna path + branch
#            se não fizer mudanças → worktree limpa automaticamente
```

---

## 7. MCP Servers — Extensões Avançadas

### 7.1 MCP Servers a Construir para Geosteering AI

#### 7.1.1 `physics-validator` — Validação EM em tempo real

```python
# .claude/mcp/physics_validator.py

"""
MCP Server para validação física do simulador Geosteering AI.
Tools expostas ao Claude Code:

  validate_em_tensor(H_tensor, n_layers, rho_h, rho_v)
    → Verifica simetria de Maxwell: H_xy ≈ H_yx (tolerância 1e-10)
    → Verifica unidades: |H| em A/m, faixa física razoável

  run_fortran_parity(model_name, n_positions=30)
    → Executa comparação Python vs Fortran para modelo canônico
    → Retorna max|H_py - H_fort| / max|H_fort|

  check_high_resistivity(rho_max, n_layers)
    → Verifica risco de overflow em propagation.py
    → Retorna: safe/warning/danger com threshold estimado

  get_canonical_models()
    → Lista os 7 modelos canônicos existentes
    → Retorna paths, n_layers, rho_max de cada
"""
```

#### 7.1.2 `numba-profiler` — Profiling JIT em tempo real

```python
# .claude/mcp/numba_profiler.py

"""
Tools:
  get_jit_cache_info()
    → Retorna status do cache Numba para cada função @njit
    → Indica se recompilação é necessária

  benchmark_scenario(scenario_name, n_runs=5)
    → Executa bench_v214_numba.py para cenário específico
    → Retorna mediana de throughput em mod/h

  analyze_prange_efficiency(function_name)
    → Mede tarefas/thread, overhead de fork/join
    → Estima eficiência de paralelismo atual
"""
```

#### 7.1.3 `colab-bridge` — Integração com Google Colab Pro+

```python
# .claude/mcp/colab_bridge.py

"""
Tools:
  get_training_status(notebook_path)
    → Consulta GitHub Actions / Drive para logs de treinamento
    → Retorna: epoch atual, loss, val_loss, ETA

  get_gpu_metrics()
    → Consulta utilização de GPU no Colab
    → Retorna: VRAM usada, temperatura, throughput

  trigger_colab_cell(notebook_path, cell_index)
    → Aciona execução remota via API do Colab
    → Requer: COLAB_API_TOKEN em .env
"""
```

#### 7.1.4 `consensus-mcp` (já disponível, otimizar)

```json
{
  "claude.ai Consensus": {
    "type": "http",
    "url": "https://consensus.anthropic.com/mcp",
    "tools": ["search_preprints", "get_preprint", "get_categories"]
  }
}
```

**Uso no projeto**:
- Busca automática de papers sobre otimização de simuladores EM
- Validação de algoritmos de inversão com literatura recente
- Pesquisa de arquiteturas DL para dados sísmicos/geofísicos

### 7.2 Configuração do MCP em `.claude/mcp-servers.json`

```json
{
  "physics-validator": {
    "type": "stdio",
    "command": "python",
    "args": [".claude/mcp/physics_validator.py"],
    "env": {
      "PYTHONPATH": ".",
      "NUMBA_CACHE_DIR": ".numba_cache"
    }
  },
  "numba-profiler": {
    "type": "stdio",
    "command": "python",
    "args": [".claude/mcp/numba_profiler.py"]
  },
  "jupyter-runtime": {
    "type": "stdio",
    "command": "python",
    "args": ["-m", "jupyter_mcp_server", "--runtime-dir", ".jupyter/runtime"]
  },
  "consensus-search": {
    "type": "http",
    "url": "https://consensus.anthropic.com/mcp"
  },
  "google-drive-colab": {
    "type": "http",
    "url": "https://drive.googleapis.com/mcp",
    "headers": {
      "Authorization": "Bearer ${GOOGLE_DRIVE_TOKEN}"
    }
  }
}
```

---

## 8. Hooks — Automação e Qualidade Contínua

### 8.1 Hooks para Geosteering AI

```json
// .claude/settings.json — Hooks de projeto

{
  "hooks": {
    "PostToolUse": [
      {
        "name": "pytest-on-simulation-edit",
        "description": "Rodar testes do simulador quando kernels são editados",
        "type": "command",
        "matcher": {
          "tool": "Edit|Write",
          "file_pattern": "**/simulation/**/*.py"
        },
        "command": [
          "bash", "-c",
          "cd /Users/daniel/Geosteering_AI && source ~/Geosteering_AI_venv/bin/activate && pytest geosteering_ai/simulation/tests/ -x -q --tb=short 2>&1 | tail -20"
        ],
        "timeout_ms": 120000
      },
      {
        "name": "check-ptbr-accent",
        "description": "Verificar acentuação PT-BR em arquivos Python editados",
        "type": "prompt",
        "matcher": {
          "tool": "Edit|Write",
          "file_pattern": "geosteering_ai/**/*.py"
        },
        "prompt": "O arquivo editado contém comentários ou docstrings em português? Se sim, verificar se há palavras sem acentuação (ex: 'implementacao' em vez de 'implementação'). Reportar apenas se encontrar problemas reais."
      },
      {
        "name": "validate-physics-constants",
        "description": "Verificar valores físicos críticos (errata imutável)",
        "type": "command",
        "matcher": {
          "tool": "Edit",
          "file_pattern": "geosteering_ai/config.py"
        },
        "command": [
          "bash", "-c",
          "python -c \"import sys; sys.path.insert(0, '.'); from geosteering_ai.config import PipelineConfig; PipelineConfig(); print('Errata OK')\" 2>&1"
        ]
      }
    ],
    "PreToolUse": [
      {
        "name": "block-pytorch",
        "description": "Bloquear importação de PyTorch no projeto",
        "type": "command",
        "matcher": {
          "tool": "Edit|Write",
          "file_pattern": "geosteering_ai/**/*.py"
        },
        "command": [
          "bash", "-c",
          "if echo \"$CLAUDE_TOOL_INPUT\" | grep -q 'import torch'; then echo 'BLOCK: PyTorch é proibido neste projeto. Use TensorFlow/Keras.'; exit 1; fi"
        ]
      }
    ],
    "Stop": [
      {
        "name": "run-full-test-suite",
        "description": "Validação final antes de encerrar sessão com modificações",
        "type": "command",
        "command": [
          "bash", "-c",
          "cd /Users/daniel/Geosteering_AI && source ~/Geosteering_AI_venv/bin/activate && pytest tests/ -q --tb=line 2>&1 | tail -5"
        ],
        "timeout_ms": 300000
      }
    ],
    "SessionStart": [
      {
        "name": "load-context",
        "description": "Carregar contexto do projeto ao iniciar sessão",
        "type": "command",
        "command": [
          "bash", "-c",
          "echo '=== GEOSTEERING AI v2.0 — CONTEXTO CARREGADO ===' && echo 'Versão: v2.21 (cba27dd)' && echo 'Testes: 68/68 PASS' && echo 'Cenário E: 122k mod/h'"
        ]
      }
    ]
  }
}
```

### 8.2 Hook de Paridade Fortran Automática

```bash
#!/bin/bash
# .claude/hooks/check_fortran_parity.sh
# Executado após qualquer modificação em arquivos Numba críticos

CRITICAL_FILES=(
    "geosteering_ai/simulation/_numba/kernel.py"
    "geosteering_ai/simulation/_numba/propagation.py"
    "geosteering_ai/simulation/_numba/dipoles.py"
    "geosteering_ai/simulation/forward.py"
)

MODIFIED="$1"
IS_CRITICAL=false

for f in "${CRITICAL_FILES[@]}"; do
    if [[ "$MODIFIED" == *"$f"* ]]; then
        IS_CRITICAL=true
        break
    fi
done

if $IS_CRITICAL; then
    echo "⚡ Arquivo crítico modificado. Verificando paridade Fortran..."
    source ~/Geosteering_AI_venv/bin/activate
    cd /Users/daniel/Geosteering_AI
    pytest tests/test_simulation_compare_fortran.py -v -k fortran_python_numba \
           --tb=short 2>&1 | tail -20
    if [ $? -ne 0 ]; then
        echo "❌ PARIDADE FORTRAN FALHOU — Reverter mudanças ou corrigir!"
        exit 1
    fi
    echo "✅ Paridade Fortran preservada (<1e-12)"
fi
```

---

## 9. Seleção de Modelos por Tarefa

### 9.1 Guia de Seleção Claude

```
╔══════════════════════════════════════════════════════════════════════════╗
║  CLAUDE OPUS 4.7 (1M contexto)                                          ║
║  Quando usar:                                                            ║
║    • Análise arquitetural completa (carrega o projeto inteiro)           ║
║    • Debugging de física EM complexa                                     ║
║    • Design de novos algoritmos (propagação, recursão estável)          ║
║    • Sessões de sprint de alta complexidade (>5 arquivos)               ║
║    • Raciocínio sobre trade-offs físicos vs performance                 ║
║    • Análise de regressão misteriosa em múltiplos commits               ║
║  Custo: Alto (Max 5×)                                                    ║
╠══════════════════════════════════════════════════════════════════════════╣
║  CLAUDE SONNET 4.6 (200k contexto)                                      ║
║  Quando usar:                                                            ║
║    • Implementação de sprints bem planejados (<5 arquivos)              ║
║    • Geração de código TF/Keras (pipeline ML)                           ║
║    • Análise de benchmarks e interpretação de resultados                ║
║    • Code review de PRs (features)                                      ║
║    • Geração de testes (quando o contexto é suficiente)                 ║
║    • Conversas de desenvolvimento do dia-a-dia                          ║
║  Custo: Médio (padrão do Max 5×)                                        ║
╠══════════════════════════════════════════════════════════════════════════╣
║  CLAUDE HAIKU 4.5 (200k contexto)                                       ║
║  Quando usar:                                                            ║
║    • Verificação de acentuação PT-BR em docstrings                      ║
║    • Update de CHANGELOG, ROADMAP, CLAUDE.md                           ║
║    • Geração de tabelas e listas estruturadas                           ║
║    • Busca de padrões simples no código                                 ║
║    • Testes simples e smoke tests                                       ║
║    • Hooks de verificação rápida                                        ║
║  Custo: Baixo (ideal para automação em lote)                            ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### 9.2 Seleção por Subagente no Código

```python
# Exemplos de seleção de modelo por tipo de tarefa

# Tarefa de alta complexidade física → Opus
Agent(
    subagent_type="feature-dev:code-architect",
    model="opus",  # ← explícito
    prompt="Design do kernel FLAT prange com análise completa de correctness"
)

# Tarefa de implementação rotineira → Sonnet (default)
Agent(
    subagent_type="general-purpose",
    prompt="Atualizar CHANGELOG.md com entrada para v2.22"
    # model não especificado → herda do agente pai
)

# Tarefa trivial → Haiku
Agent(
    subagent_type="Explore",
    model="haiku",  # ← econômico para busca simples
    prompt="Encontrar todos os arquivos com @njit no projeto"
)
```

---

## 10. Orquestração de IAs em Diferentes Camadas

### 10.1 Mapa Completo de Inteligência no Projeto

```
┌─────────────────────────────────────────────────────────────────────────┐
│  CAMADA DE DESIGN E RACIOCÍNIO                                          │
│                                                                         │
│  Claude Opus 4.7 (1M ctx):                                             │
│    • Arquitetura do simulador Numba JIT                                │
│    • Decisões de trade-off performance vs física                       │
│    • Análise de regressões históricas                                  │
│    • Design de workflows multi-agente                                  │
│                                                                         │
│  Google Gemini 2.5 Pro (via Antigravity):                              │
│    • Completions inline baseadas em contexto local do arquivo          │
│    • Sugestões de código JAX/TF em tempo real                         │
│    • Consultas à documentação do Google                                │
├─────────────────────────────────────────────────────────────────────────┤
│  CAMADA DE IMPLEMENTAÇÃO                                                │
│                                                                         │
│  Claude Sonnet 4.6:                                                    │
│    • Implementação de sprints planejados                               │
│    • Subagentes especializados (ML, dados, testes)                     │
│    • Code review de features                                           │
│                                                                         │
│  Claude Haiku 4.5:                                                     │
│    • Automação em lote via hooks                                       │
│    • Geração de documentação de baixa complexidade                     │
│    • Testes boilerplate                                                │
├─────────────────────────────────────────────────────────────────────────┤
│  CAMADA DE EXECUÇÃO (não-IA, determinístico)                           │
│                                                                         │
│  Numba JIT (@njit, prange, nogil):                                     │
│    • Simulação física: hmd_tiv, vmd, propagation                      │
│    • Paralelismo de CPU                                                │
│                                                                         │
│  TensorFlow/Keras + XLA:                                               │
│    • Treinamento e inferência de redes neurais                        │
│    • Compilação JIT para GPU (T4/A100 no Colab)                       │
│                                                                         │
│  JAX + vmap:                                                           │
│    • Backend alternativo do simulador para GPU                        │
│    • Diferenciação automática (gradiente do forward model)            │
│                                                                         │
│  Fortran (`tatu.x`, OpenMP):                                           │
│    • Ground truth físico (paridade <1e-12 com Python)                 │
│    • Benchmark de performance                                          │
│                                                                         │
│  pytest + GitHub Actions:                                              │
│    • CI/CD: validação contínua                                        │
│    • Regressão automática                                              │
└─────────────────────────────────────────────────────────────────────────┘
```

### 10.2 Prompt Engineering para Domínio Geofísico

Ao usar múltiplos agentes em paralelo, é fundamental que cada agente receba o contexto físico correto:

```markdown
# Template de prompt para agentes do simulador:

"""
Contexto físico obrigatório:
- Simulador EM 1D em meio TIV (transversalmente isotrópico)
- Propagação de ondas EM: HMD (dipolo magnético horizontal) + VMD (vertical)
- Integração de Hankel sobre kr com filtro Werthmuller 201pt (npt=201)
- Tensor completo de 9 componentes: H = [Hxx, Hxy, Hxz, Hyx, Hyy, Hyz, Hzx, Hzy, Hzz]
- Paridade Fortran <1e-12 é INVIOLÁVEL — qualquer modificação deve preservar

Constraints de implementação:
- @njit com parallel=True: NUNCA aninhado (nested prange overhead)
- nogil=True: obrigatório em todas as funções @njit do hot path
- cache=True: obrigatório em funções de alto custo
- fastmath: APENAS após validação de paridade com modelos de alta ρ
"""
```

---

## 11. Roadmap de Implementação do Ambiente

### 11.1 Fase 1 — Fundação (Esta semana)

```
TAREFA 1.1: Configurar hooks de automação
  Criar: .claude/settings.json com hooks definidos em §8.1
  Testar: modificar um arquivo em simulation/ e verificar pytest automático
  Tempo: 2h

TAREFA 1.2: Criar MCP Server physics-validator
  Criar: .claude/mcp/physics_validator.py (§7.1.1)
  Configurar: .claude/mcp-servers.json
  Testar: Claude Code chama validate_em_tensor() diretamente
  Tempo: 4h

TAREFA 1.3: Criar Skills especializadas
  Expandir: .claude/commands/geosteering-simulator-python.md (já existe)
  Criar: .claude/commands/geosteering-ml-pipeline.md
  Criar: .claude/commands/geosteering-tests.md
  Tempo: 2h

TAREFA 1.4: Configurar .worktreeinclude
  Criar: .worktreeinclude com paths corretos para venv e caches
  Testar: claude --worktree test-worktree → verificar venv disponível
  Tempo: 30min
```

### 11.2 Fase 2 — Workflows Ativos (Próximas 2 semanas)

```
TAREFA 2.1: Primeiro sprint com worktree isolada
  Objetivo: Sprint 22.1 (FLAT prange) em worktree dedicada
  Agentes: Plan → code-architect (isolation=worktree) → code-reviewer
  Validação: benchmark + paridade Fortran automáticos via hooks

TAREFA 2.2: Workflow de monitoramento de treinamento
  Criar: .claude/mcp/colab_bridge.py
  Implementar: /loop 10m check-colab-training
  Testar: Com próximo treinamento de rede neural no Colab

TAREFA 2.3: Agent Teams para code review
  Habilitar: CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1
  Criar: Team de 3 reviewers para PR de simulador
  Documentar: Resultados e custo vs subagentes individuais
```

### 11.3 Fase 3 — Maturidade (Próximo mês)

```
TAREFA 3.1: MCP numba-profiler completo
  Integração com bench_v214_numba.py
  Claude pode pedir benchmark antes de sugerir otimização

TAREFA 3.2: CronCreate para relatórios periódicos
  CronCreate: "0 9 * * 1" = toda segunda de manhã
  Ação: Subagente Haiku gera resumo da semana + próximas prioridades

TAREFA 3.3: Hook de validação científica
  Integração com Consensus MCP para referências de papers
  Valida automaticamente novos algoritmos contra literatura

TAREFA 3.4: CLAUDE.md hierárquico completo
  CLAUDE.md (raiz, já existe)
  geosteering_ai/simulation/CLAUDE.md (regras específicas do simulador)
  geosteering_ai/models/CLAUDE.md (regras de arquiteturas)
  tests/CLAUDE.md (regras de testes)
```

---

## 12. Estrutura de Arquivos de Configuração

### 12.1 Estrutura Proposta Completa

```
Geosteering_AI/
├── CLAUDE.md                          ← Instruções globais (já existe)
├── .claude/
│   ├── settings.json                  ← Hooks de projeto
│   ├── settings.local.json            ← Hooks locais (gitignored)
│   ├── mcp-servers.json               ← Configuração MCP
│   ├── commands/                      ← Skills / slash commands
│   │   ├── geosteering-v2.md          ← Skill principal (existe)
│   │   ├── geosteering-simulator-python.md ← (expandir)
│   │   ├── geosteering-simulator-fortran.md ← (existe)
│   │   ├── geosteering-ml-pipeline.md ← (criar)
│   │   ├── geosteering-tests.md       ← (criar)
│   │   └── consensus-search.md        ← (existe)
│   ├── mcp/                           ← MCP Servers locais
│   │   ├── physics_validator.py       ← (criar)
│   │   ├── numba_profiler.py          ← (criar)
│   │   └── colab_bridge.py            ← (criar)
│   ├── hooks/                         ← Scripts de hooks
│   │   ├── check_fortran_parity.sh    ← (criar)
│   │   ├── validate_physics.py        ← (criar)
│   │   └── update_memory.py           ← (criar)
│   ├── templates/
│   │   └── report_template.md         ← (já existe)
│   └── plans/                         ← Planos de sprint
│       └── *.md
│
├── geosteering_ai/
│   └── simulation/
│       └── CLAUDE.md                  ← (criar) regras específicas simulador
│
└── .worktreeinclude                   ← (criar) arquivos para copiar em worktrees
```

---

## 13. Análise de Riscos e Mitigações

| Risco | Probabilidade | Impacto | Mitigação |
|:------|:-------------:|:-------:|:----------|
| Subagente introduz bug físico | Média | Alto | Hook de paridade Fortran automático após edições críticas |
| Contexto insuficiente em Sonnet | Média | Médio | Usar Opus para tarefas multi-arquivo, Sonnet para mono-arquivo |
| Oversubscription de contexto em Agent Teams | Alta | Médio | Limitar a 3 teammates máximo; preferir subagentes isolados |
| PyTorch importado por acidente | Baixa | Alto | Hook PreToolUse bloqueia `import torch` em geosteering_ai/ |
| fastmath quebra paridade Fortran | Baixa | Alto | Gate: testes em 7+3 modelos canônicos antes de habilitar |
| Custo excessivo de tokens com Opus | Média | Médio | Opus apenas para sprints complexos; Sonnet para rotina |
| Worktree não limpa após subagente | Baixa | Baixo | Auto-cleanup se sem mudanças; manual review se com mudanças |
| MCP Server falha silenciosamente | Baixa | Médio | Logging estruturado em cada MCP server; fallback graceful |

---

## 14. Recomendações para Sessão com Opus 4.7 1M

### 14.1 Por Que Opus 4.7 1M para o Aprofundamento

1. **Contexto completo do projeto**: Com 1M tokens, o Opus pode carregar:
   - Todo o código Python (~46k LOC)
   - Documentação técnica (~200 páginas)
   - Histórico de decisões (memory/, reports/)
   - Este relatório + documentação de otimização de cenários
   - Código Fortran (`tatu.x`) para comparação

2. **Raciocínio arquitetural superior**: O Opus 4.7 tem capacidade de raciocinar sobre interações sutis entre módulos que o Sonnet pode não perceber.

3. **Domínio físico-matemático**: A análise de trade-offs entre paridade Fortran e otimizações Numba requer raciocínio matemático de alto nível que o Opus trata melhor.

### 14.2 Como Preparar a Sessão com Opus

```bash
# 1. Verificar que está no ambiente correto
cd ~/Geosteering_AI
git status  # verificar branch

# 2. Carregar memória relevante
# Claude Opus lerá automaticamente:
#   - CLAUDE.md (projeto)
#   - MEMORY.md (índice de memórias)
#   - Arquivos de memória relevantes

# 3. Preparar contexto adicional para a sessão
cat << 'EOF' > /tmp/opus_context.md
# Contexto para Sessão Opus 4.7 1M — Arquitetura Multi-Agente

## O que foi feito até aqui:
1. Análise de todos os 8 cenários de simulação (nf, nAng, nTR, n_pos)
2. Relatório técnico completo em docs/reference/analise_cenarios_otimizacao_simulador_numba.md
3. Rascunho de arquitetura multi-agente em docs/reports/arquitetura_multiagente_geosteering_ai_2026-05-02.md

## O que queremos aprofundar com Opus:
1. Validar e refinar a arquitetura de multi-agentes proposta
2. Definir EXATAMENTE quais hooks, skills e MCP servers implementar primeiro
3. Projetar o workflow de sprint para v2.22 (FLAT prange) com isolamento total
4. Definir estratégia de integração Claude Code + Google Antigravity
5. Planejar a estrutura de CLAUDE.md hierárquicos por módulo

## Constraints que Opus deve conhecer:
- TensorFlow/Keras EXCLUSIVO (PyTorch PROIBIDO)
- Paridade Fortran <1e-12 INVIOLÁVEL
- Anti-pattern documentado: nested prange em Numba
- Hardware: Mac 8C/16T, Colab T4/A100 para GPU
- Budget: Claude Max 5×, Google AI Pro
EOF
```

### 14.3 Perguntas para o Opus Aprofundar

1. **"Dado o código atual do simulador (kernel.py, forward.py, propagation.py), qual é a sequência ótima de implementação das 6 otimizações propostas (O1-O6), considerando dependências técnicas entre elas?"**

2. **"Como projetar um CLAUDE.md hierárquico que instrua subagentes de forma diferente dependendo do módulo em que estão trabalhando (simulador vs ML vs Fortran)?"**

3. **"Dado que o projeto tem 73 arquivos Python e ~46k LOC, qual é a estratégia de worktree e branching que melhor suporta desenvolvimento paralelo de simulador + ML pipeline sem conflitos?"**

4. **"Como integrar o fluxo de desenvolvimento local (Claude Code + Antigravity) com o fluxo de treinamento GPU (Colab Pro+) de forma que Claude Code possa monitorar e reagir ao treinamento em andamento?"**

5. **"Quais são os riscos de regressão física (paridade Fortran, valores físicos) mais prováveis ao usar subagentes para implementar otimizações de performance no simulador Numba?"**

---

## Nota Final: Recomendação de Modelo

**Recomendo fortemente usar Claude Opus 4.7 com 1M de contexto para o aprofundamento desta análise.**

Justificativas objetivas:

| Critério | Sonnet 4.6 (este relatório) | Opus 4.7 1M |
|:---------|:---------------------------:|:-----------:|
| Contexto máximo | 200k tokens | 1M tokens |
| Codebase completa em contexto | Não (46k LOC > 200k tokens) | **Sim** |
| Raciocínio arquitetural multi-sistema | Bom | **Excelente** |
| Trade-offs físicos vs computacionais | Adequado | **Superior** |
| Custo por sessão | Max 5× (padrão) | Max 5× (mais caro) |
| Recomendado para | Sprints rotineiros | **Sessões arquiteturais** |

Este relatório é o **briefing** para a sessão com Opus. Quando abrir a sessão com Opus 4.7 1M, forneça:
1. Este arquivo (`docs/reports/arquitetura_multiagente_geosteering_ai_2026-05-02.md`)
2. O arquivo de análise de cenários (`docs/reference/analise_cenarios_otimizacao_simulador_numba.md`)
3. Os arquivos de código críticos (`forward.py`, `kernel.py`, `propagation.py`)
4. CLAUDE.md + MEMORY.md

O Opus terá contexto suficiente para aprofundar e validar cada aspecto desta arquitetura com precisão superior.

---

*Relatório gerado em 2026-05-02 com Claude Sonnet 4.6. Preparado como briefing para aprofundamento com Claude Opus 4.7 (1M contexto) no desenvolvimento profissional do Geosteering AI 2.0.*
