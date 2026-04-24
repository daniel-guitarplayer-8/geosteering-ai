# Colab MCP Server — Análise e Proposta para o Projeto Geosteering AI

> **Data**: 2026-04-16
> **Contexto**: Sprint 12 / v1.6.0 — validação GPU T4 ainda manual (E11 do plano)
> **Autor**: Daniel Leal

---

## 1. O que é o Protocolo MCP

O **Model Context Protocol** (MCP) é um protocolo aberto lançado pela Anthropic em
2024 que padroniza como agentes de IA se comunicam com ferramentas externas. A
arquitetura é cliente-servidor:

```
┌────────────────────────────────────────────────────────────────┐
│  Claude Code (cliente MCP)                                     │
│    ↓  JSON-RPC over stdio/SSE                                  │
│  MCP Server (processo local ou remoto)                         │
│    ↓  protocolo nativo da ferramenta                           │
│  Ferramenta (Google Drive, Figma, Colab, Jupyter, ...)         │
└────────────────────────────────────────────────────────────────┘
```

Este projeto já usa três servidores MCP ativos: **context7** (docs de bibliotecas),
**Figma** e **Google Drive**. O Colab MCP Server se encaixaria exatamente no mesmo
padrão de configuração via `settings.json`.

---

## 2. O Colab MCP Server — O que É

### 2.1 Servidor Oficial Google (`googlecolab/colab-mcp`)

Em **março de 2026**, o Google lançou o servidor MCP oficial para o Google Colab,
disponível em `github.com/googlecolab/colab-mcp`. É open-source, mantido
diretamente pela equipe do Colab.

**O que ele permite:**

| Capacidade | Descrição |
|:-----------|:----------|
| **Criar células** | Claude cria células de código/texto no notebook aberto |
| **Executar células** | Claude dispara execução e lê o output (stdout, stderr, plots) |
| **Gerenciar dependências** | `!pip install` via Claude sem interação manual |
| **Ler variáveis** | Inspecionar estado do kernel (arrays, DataFrames, métricas) |
| **Ciclo completo de notebook** | Criar, organizar, executar, exportar resultados |
| **GPU disponível** | Acessa o runtime configurado na sessão (T4, A100, TPU) |

### 2.2 Arquitetura de Funcionamento

```
┌─────────────────────────────────┐     ┌──────────────────────────────┐
│  MÁQUINA LOCAL                  │     │  GOOGLE COLAB (navegador)    │
│                                 │     │                              │
│  Claude Code CLI                │     │  Runtime T4 / A100           │
│       ↓ MCP protocol            │◄───►│  Kernel Python 3.10          │
│  colab-mcp server               │     │  JAX + CUDA disponíveis      │
│  (processo local, uvx)          │     │  geosteering_ai instalado    │
│       ↓ WebSocket/HTTP          │     │       ↑                      │
│  Extensão Colab no browser ─────┼─────┘  Extensão MCP no Colab      │
└─────────────────────────────────┘
```

**Pré-requisito importante**: a sessão Colab precisa estar aberta no navegador com
a extensão MCP instalada. O servidor não opera de forma completamente headless —
requer a aba do Colab ativa para mediar as execuções.

### 2.3 Configuração (Claude Code `settings.json`)

```json
{
  "mcpServers": {
    "colab-mcp": {
      "command": "uvx",
      "args": ["git+https://github.com/googlecolab/colab-mcp"],
      "timeout": 30000
    }
  }
}
```

---

## 3. Comparativo de Soluções Disponíveis

| Solução | Mantenedor | Maturidade | Suporte Colab | GPU | Estado do Kernel |
|:--------|:-----------|:----------:|:-------------:|:---:|:----------------:|
| **colab-mcp** (googlecolab) | Google | ★★★★☆ | Nativo | ✅ T4/A100/TPU | Por sessão |
| **jupyter-mcp-server** (Datalayer) | Empresa | ★★★★★ | Experimental | ✅ via Jupyter | Persistente |
| **ClaudeJupy** | Comunidade | ★★★☆☆ | Não | ✅ via Jupyter local | Persistente entre turns |
| **jupyter-notebook-mcp** | Comunidade | ★★☆☆☆ | Não | ✅ via Jupyter | Por sessão |

**Para este projeto**: `colab-mcp` é a recomendação principal por suportar T4/A100
nativamente e ser mantido pelo Google.

---

## 4. Vantagens Específicas para o Projeto Geosteering AI

### 4.1 Eliminar o Passo Manual E11 do Plano Sprint 12

O plano atual (cosmic-riding-garden.md) define E11 como:

> *"Colab T4 manual — re-rodar `notebooks/validate_jax_unified_gpu.ipynb` +
> `bench_sprint12_regression.py` — confirmar produção 600×3 ≥ 2× speedup"*

Com o Colab MCP Server, Claude poderia **executar E11 autonomamente**:

```
Claude Code                              Colab T4
─────────────                            ────────
1. Abre notebook via MCP              →  Runtime inicia com T4
2. Executa célula de setup            ←  "JAX GPU detectado: Tesla T4"
3. Roda bench_sprint12_regression.py  ←  Resultados 192 pontos
4. Lê throughput_mod_h × strategy     ←  "unified 600×3: 2.3× vs bucketed"
5. Gate: ≥ 2× ?  → GO/NOGO           
6. Commita relatório automaticamente  
```

### 4.2 Ciclo de Validação GPU Acelerado

| Etapa atual | Tempo (manual) | Com MCP (estimado) |
|:------------|:--------------:|:------------------:|
| Abrir Colab, aguardar runtime T4 | ~3 min | ~3 min (mesmo) |
| Copiar código, colar célula, rodar | ~5 min | ~30 s (automático) |
| Ler e interpretar resultados | ~10 min | ~2 min (Claude lê) |
| Gerar relatório e commitar | ~15 min | ~5 min (automático) |
| **Total por validação** | **~33 min** | **~11 min** |

### 4.3 Benchmarks A/B Sistemáticos

O plano Sprint 12 prevê otimizações E4-E6 (chunking, donate_argnums, lax.scan)
que requerem comparações A/B em GPU. Manualmente, isso é lento e propenso a erro.
Com MCP:

```python
# Claude poderia executar automaticamente:
for chunk_size in [32, 64, 128, 256, None]:
    result = execute_in_colab(f"""
        cfg = SimulationConfig(jax_strategy="unified",
                               jax_position_chunk_size={chunk_size})
        t = bench(cfg, n_pos=600, nf=3, repeats=10)
        print(f"chunk={chunk_size}: {{t:.1f}} ms")
    """)
    # Lê output e constrói tabela comparativa
```

### 4.4 Validação Contínua em GPU (CI-like para GPU)

Atualmente, a validação GPU é **100% manual** e ocorre apenas antes de merges
importantes. Com MCP integrado ao workflow:

```
Commit local → pytest CPU (automático) → Gate CPU PASS
                                               ↓
                               Claude abre Colab MCP, executa validação GPU
                               (semi-automático: Claude opera, humano aprova merge)
```

### 4.5 Integração com Google Drive MCP (já instalado)

O projeto já tem o **Google Drive MCP** configurado. Esses dois servidores se
complementam:

```
┌──────────────────────────────────────────────────────────────┐
│  Google Drive MCP (já ativo)  +  Colab MCP (a instalar)     │
│                                                              │
│  Claude escreve script para Drive → Colab lê do Drive →     │
│  executa com GPU → resultados no Drive → Claude lê e        │
│  gera relatório → commit automático                         │
└──────────────────────────────────────────────────────────────┘
```

Esse pipeline já é viável com ferramentas existentes no projeto.

---

## 5. Automação de Benchmarks GPU — Viabilidade Técnica

### 5.1 O que é Totalmente Automático

Com `colab-mcp` configurado e sessão Colab aberta com T4:

- ✅ Instalação de dependências (`pip install git+...@v1.6.0`)
- ✅ Execução de `bench_sprint12_regression.py --matrix full --backend gpu`
- ✅ Leitura de resultados (throughput, VRAM, XLA count, parity)
- ✅ Execução de `pytest tests/test_simulation_jax_sprint12.py` com GPU
- ✅ Captura de output do `jax.profiler.trace()` e export para TensorBoard
- ✅ Verificação de gates (≥ 2× speedup, paridade < 1e-10)
- ✅ Geração de relatório e commit ao repositório

### 5.2 O que Ainda Requer Interação Humana

- ⚠️ **Abrir a sessão Colab** com runtime T4 selecionado (1 clique)
- ⚠️ **Manter a aba do navegador aberta** durante a execução
- ⚠️ **Reconexão após timeout de sessão** (~12h gratuito, ~24h Pro+)
- ⚠️ **Cotas de GPU**: Colab Pro+ tem cotas mensais; execuções longas consomem CUs
- ⚠️ **Aprovação de merge**: Claude executa, humano aprova o PR

### 5.3 Limitações Técnicas

| Limitação | Impacto | Mitigação |
|:----------|:--------|:----------|
| Sessão precisa estar aberta | Claude não pode iniciar runtime headless | Abrir antes de chamar Claude |
| Timeout de inatividade (~30 min) | Benchmarks longos podem ser interrompidos | keepalive cell + timeout=600s |
| Sem acesso a arquivos locais diretamente | Código precisa ser instalado via pip ou Drive | `pip install git+...` |
| Output de células é texto | Plots precisam ser salvos como arquivo | `plt.savefig('result.png')` + Drive MCP |
| GPU não garantida (fila Colab) | Runtime pode ser CPU em horários de pico | Agendar para madrugada |

### 5.4 Pipeline Completo de Validação GPU Proposto

```
┌─────────────────────────────────────────────────────────────────────┐
│  PIPELINE AUTOMÁTICO COM COLAB MCP                                  │
│                                                                     │
│  Pré-condição: Colab aberto com T4, colab-mcp configurado           │
│                                                                     │
│  1. [Claude] pip install geosteering_ai @ branch atual              │
│  2. [Claude] Executa: python -c "import jax; print(jax.devices())"  │
│     [Colab]  ← "[GpuDevice(id=0, process_index=0), dtype=float32]"  │
│  3. [Claude] bench_sprint12_regression.py --matrix full --backend gpu│
│     [Colab]  ← CSV com 192 linhas (throughput, VRAM, XLA, parity)   │
│  4. [Claude] Avalia gates: 600×3 unified ≥ 2× bucketed?             │
│     → GO: gera docs/reference/sprint12_gpu_results.md               │
│     → NOGO: abre issue, descreve regressão, sugere fix              │
│  5. [Claude] Re-roda validate_jax_unified_gpu.ipynb (G1-G5)         │
│  6. [Claude] Commit resultado + fecha E11 no plano                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. Como Instalar e Configurar

### 6.1 Pré-requisitos

```bash
# 1. Instalar uv (se não tiver)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Verificar se uvx está disponível
uvx --version
```

### 6.2 Configurar em Claude Code (`~/.claude/settings.json`)

```json
{
  "mcpServers": {
    "colab-mcp": {
      "command": "uvx",
      "args": ["git+https://github.com/googlecolab/colab-mcp"],
      "timeout": 60000,
      "env": {}
    }
  }
}
```

> **Nota**: `timeout: 60000` (60s) é recomendado para células que executam
> benchmarks longos. O padrão de 30s pode causar timeout falso em runs pesados.

### 6.3 Configurar a Sessão Colab

No Colab, ao abrir o notebook:

1. `Runtime → Change runtime type → T4 GPU`
2. Instalar extensão MCP (primeira vez): célula de setup do `colab-mcp`
3. A extensão fica ativa enquanto a aba estiver aberta

### 6.4 Célula de Setup Padrão para Este Projeto

```python
# Célula 1 — Setup Geosteering AI em Colab (executada via MCP)
!pip install -q "git+https://github.com/daniel-guitarplayer-8/geosteering-ai.git@main"
!pip install -q jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

import jax
print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")

from geosteering_ai.simulation import __version__
print(f"geosteering_ai.simulation version: {__version__}")
```

---

## 7. Casos de Uso Prioritários para Geosteering AI

### Caso 1 — E11 Sprint 12 (Imediato)

```python
# Claude executa via MCP:
exec_cell("""
from benchmarks.bench_sprint12_regression import run_matrix
results = run_matrix(backend='gpu', matrix='full')
results.to_csv('/content/sprint12_gpu_results.csv')
print(results[['n_pos','nf','strategy','throughput_mod_h','ratio_unified_bucketed']].to_string())
""")
```

### Caso 2 — Validação de Paridade após cada PR

```python
# Teste rápido de paridade JAX-GPU vs CPU (gate pré-merge):
exec_cell("""
import pytest, sys
result = pytest.main([
    'tests/test_simulation_jax_sprint12.py',
    '-v', '--tb=short', '-q'
], plugins=[])
sys.exit(result)
""")
```

### Caso 3 — Treino SurrogateNet (F5 do Roadmap)

O Roadmap prevê treinar `SurrogateNet TCN (127M params)` e `ModernTCN (204M params)`
no Colab Pro+ com GPU. Com MCP, Claude poderia:

1. Iniciar o treino com hiperparâmetros configurados
2. Monitorar loss a cada época (lendo output da célula)
3. Detectar divergência e ajustar `learning_rate` automaticamente
4. Salvar checkpoints no Google Drive via Drive MCP
5. Gerar relatório de treino ao final

### Caso 4 — Profiling XLA com jax.profiler

```python
# Sprint 12 E2 — profiler automático nas 5 configs críticas:
exec_cell("""
import jax
with jax.profiler.trace('/content/profile_600x3'):
    result = simulate_multi_jax(rho_h, rho_v, esp, z, freqs,
                                tr_spacings, dip_degs, cfg=cfg_unified)
# Salva trace para TensorBoard
!cp -r /content/profile_600x3 /content/drive/MyDrive/geosteering_profiles/
""")
```

---

## 8. Riscos e Limitações

| # | Risco | Probabilidade | Mitigação |
|---|:------|:-------------:|:----------|
| R1 | Sessão Colab expira durante benchmark longo | 🟡 Média | Dividir em jobs curtos (<20 min); keepalive cell |
| R2 | Cotas de GPU esgotadas (Colab Pro+) | 🟡 Média | Reservar validações GPU para pré-merge críticos |
| R3 | `colab-mcp` ainda em versão alpha (março 2026) | 🟡 Média | Ter fallback manual documentado; monitorar issues no GitHub |
| R4 | Output de benchmarks truncado por limite de texto | 🟢 Baixa | Salvar CSV no Drive, ler via Drive MCP |
| R5 | Versão do pacote instalada no Colab diverge do local | 🟡 Média | Fixar hash do commit no `pip install` |
| R6 | Custo adicional se usar Colab Enterprise API | 🟢 Baixa | `colab-mcp` free é suficiente para este caso de uso |

---

## 9. Comparação com o Workflow Atual

| Etapa | Workflow Atual (Manual) | Com Colab MCP |
|:------|:------------------------|:--------------|
| Abrir Colab + selecionar T4 | Humano | Humano (1 clique) |
| Instalar dependências | Humano (colar célula) | Claude (automático) |
| Executar benchmark 192 pts | Humano (30-40 min espera) | Claude (mesma espera, sem atenção) |
| Ler e interpretar resultados | Humano (10-15 min) | Claude (2-3 min) |
| Gerar relatório de validação | Humano (15-30 min) | Claude (automático) |
| Commitar resultado | Humano | Claude |
| **Total de atenção humana** | **~60-90 min** | **~5-10 min** |

---

## 10. Recomendação

### Curto Prazo (Sprint 12 — imediato)

**Instalar `colab-mcp` e usar para E11**: automatizar a validação GPU T4 que está
pendente no plano. O retorno imediato é alto (valida regressão 600×3 e paridade
vmap_real) com custo de configuração baixo (~15 min de setup).

### Médio Prazo (v1.6.1 — flip default unified)

Antes de flipar o default `jax_strategy` de `"bucketed"` para `"unified"` (PR #26),
executar uma bateria de validação GPU mais extensa. Com MCP, isso pode ser automatizado
como pré-condição do PR.

### Longo Prazo (F5 — Treino SurrogateNet)

O caso de uso mais impactante do projeto é o **treino supervisionado de SurrogateNet**
no Colab Pro+ com A100. Com MCP, Claude pode monitorar epochs, ajustar hiperparâmetros
e salvar checkpoints sem intervenção humana contínua — liberando o desenvolvedor para
outras tarefas enquanto o treino ocorre.

---

## 11. Próximos Passos Concretos

```bash
# Passo 1 — Instalar uv (se não tiver)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Passo 2 — Testar colab-mcp localmente
uvx git+https://github.com/googlecolab/colab-mcp --help

# Passo 3 — Adicionar ao settings.json do Claude Code
# (editar ~/.claude/settings.json conforme seção 6.2)

# Passo 4 — Abrir Colab com T4, instalar extensão MCP
# (seguir README do googlecolab/colab-mcp)

# Passo 5 — Pedir a Claude: "Execute E11 do Sprint 12 no Colab"
```

---

## Referências

- [Google Developers Blog — Announcing the Colab MCP Server (2026-03)](https://developers.googleblog.com/announcing-the-colab-mcp-server-connect-any-ai-agent-to-google-colab/)
- [GitHub — googlecolab/colab-mcp](https://github.com/googlecolab/colab-mcp)
- [Anthropic — Model Context Protocol (MCP) Documentation](https://docs.anthropic.com/mcp)
- [Datalayer — jupyter-mcp-server](https://github.com/datalayer/jupyter-mcp-server)
- [InfoQ — Google Brings MCP Support to Colab (2026-04)](https://www.infoq.com/news/2026/04/colab-mcp-server/)
- Plano Sprint 12: `.claude/plans/cosmic-riding-garden.md` (E11 — validação GPU pendente)
