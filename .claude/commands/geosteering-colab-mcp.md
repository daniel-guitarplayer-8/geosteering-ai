---
name: geosteering-colab-mcp
description: |
  Bridge entre Claude Code (local) e Google Colab Pro+ (remoto, GPU T4/A100)
  via MCP oficial `googlecolab/colab-mcp`. Automatiza upload, execução e
  download de notebooks de templates pré-definidos em
  `notebooks/colab_templates/`: treinamento mp16+XLA, validação JAX GPU,
  benchmark tf.data. Implementa Tier B (browser MCP) da arquitetura 4-tier
  Colab (Parte V do doc de aprofundamento). Trigger: "executar colab",
  "rodar em GPU", "validar JAX em Colab", "treinar remoto", "benchmark
  mp16", "MCP colab". Sprint v2.40 (I2.2).
tools:
  - Read
  - Bash
  - WebFetch
  - TodoWrite
model: claude-sonnet-4-6
effort: medium
constraints:
  - "F1 PT-BR acentuação inviolável em mensagens ao usuário"
  - "F2 nunca executar treinamento sem confirmação do usuário (custo Colab)"
  - "F3 sempre verificar token GCP via hook colab-token-refresh antes de invocar MCP"
  - "F4 cap de timeout MCP em 60min (notebooks longos > usar Tier C documentado)"
  - "F5 não automatizar setup ngrok (segurança: token não vive em hook bash)"
  - "F6 sempre exportar resultados em JSON para docs/perf_baselines/ ou docs/colab_runs/"
---

# Bridge Colab MCP — Geosteering AI v2.40

## Identidade

| Atributo | Valor |
|:---------|:------|
| **Skill** | geosteering-colab-mcp |
| **Modelo** | Claude Sonnet 4.6 |
| **Effort** | `medium` |
| **Origem da spec** | §22 (Parte V) doc aprofundamento + roadmap v2.39 §2.3 |
| **Status** | Implementado em Sprint v2.40 (I2.2) |
| **MCP backend** | `googlecolab/colab-mcp` (oficial, configurado em `.mcp.json:10-14`) |
| **Tier** | B (browser-attached) — Tier C documentado como fallback |

---

## Quando Invocar

### INVOCAR PARA

- **"Rode `validate_jax_gpu_v240` em Colab T4"** — automação validação JAX
- **"Treine ResNet com mp16 em Colab Pro+"** — automação treinamento
- **"Benchmark tf.data + mp16 em A100"** — automação medição performance
- **"Valide paridade JAX GPU vs Numba CPU"** — checagem cruzada de inversão
- **"Upload notebook X para Colab e execute"** — fluxo genérico
- **"MCP colab"** — invocação explícita
- **Hook detectou token GCP expirando** — sugerir refresh proativo

### NÃO INVOCAR PARA

- Inferência via REST (já temos `POST /predict` local) — desnecessário Colab
- Simulação Numba CPU (`geosteering-cli simulate`) — roda local sem GPU
- Debugging local de código TF — usar `pytest -m "not gpu"` no venv
- Edição de notebooks Jupyter local (`notebooks/main.ipynb`) — usar VS Code

---

## Tier de Automação Adotado (D2)

### Tier B — Browser MCP (DEFAULT v2.40)

```text
┌─────────────────────────────────────────────────────────────┐
│  Claude Code (local)                                         │
│     │                                                        │
│     ▼                                                        │
│  mcp__colab-mcp__open_colab_browser_connection              │
│     │                                                        │
│     ▼                                                        │
│  Chrome com extensão Colab MCP (RODANDO no Mac)             │
│     │                                                        │
│     ▼                                                        │
│  Colab Pro+ runtime (T4/A100) na Google Cloud               │
│     │                                                        │
│     ▼                                                        │
│  Notebook executa: clone repo → install → run → export JSON │
└─────────────────────────────────────────────────────────────┘
```

**Pré-requisitos**:
1. Chrome instalado e logado em conta com Colab Pro+
2. Extensão Colab MCP instalada (ver §Setup)
3. Token GCP válido (`gcloud auth print-access-token` retorna sem erro)
4. ngrok HTTP tunnel ativo (se notebook precisa chamar API REST local)

### Tier C — Headless MCP (FALLBACK documentado, NÃO implementado v2.40)

`pdwi2020/colab-exec` via OAuth — runtime headless sem navegador. Adiar
para Sprint v2.45+ quando houver demanda de execução noturna automática
(>1h, sem usuário presente).

### Tier D — Custom MCP (POSTERGADO p/ v2.50+)

MCP interno do projeto. Vantagem: profiling fino. Custo: manter código MCP
próprio. Não há demanda imediata.

---

## Templates Disponíveis (v2.40)

| Template | Cells | Runtime alvo | Tempo médio | Uso |
|:---|:---:|:---|:---:|:---|
| `train_v240_mp16.ipynb` | 10 | T4 ou A100 | 30–60 min | Treinar ResNet 18 mp16+XLA |
| `validate_jax_gpu_v240.ipynb` | 8 | T4 | 5–10 min | Validar 109 testes JAX em GPU |
| `benchmark_tfdata_mp16.ipynb` | 12 | T4 + A100 | 10–20 min | Medir samples/sec 4 configs |

**Caminho absoluto**: `notebooks/colab_templates/`

---

## Setup (Pré-Sprint v2.40)

### 1. Instalar extensão Colab MCP no Chrome

```bash
# Via UV (gerenciado pelo .mcp.json)
uvx git+https://github.com/googlecolab/colab-mcp --help
```

Em seguida, instalar a extensão Chrome conforme instruções do repositório
oficial. Reiniciar Claude Code para o MCP detectar.

### 2. Validar tokens GCP

```bash
gcloud auth print-access-token  # deve retornar token válido
gcloud auth list                # deve mostrar conta com Colab Pro+
```

Se expirado:

```bash
gcloud auth login  # abre navegador para refresh
```

O hook `colab-token-refresh.sh` (PreToolUse) verifica timestamp
do `~/.config/gcloud/access_tokens.db` antes de qualquer Bash com
`colab` ou `gcloud` no comando.

### 3. Expor API REST local via ngrok (apenas para `train_v240_mp16`)

> ⚠️ **AVISO DE SEGURANÇA — ngrok expõe sua API publicamente**
>
> URLs `*.ngrok.app` são acessíveis a **QUALQUER** pessoa que descubra ou
> receba a URL. **Não há autenticação por padrão**. Implicações operacionais:
>
> 1. **Não use conta ngrok pessoal para produção** — use uma conta dedicada
>    (corporativa, segregada por projeto). Conta pessoal vincula seu nome
>    a tunnels que ficam logados nos painéis do ngrok.
> 2. **Use ngrok authtoken (FREE) para identificar sua conta** — `ngrok
>    config add-authtoken <token>` é gratuito e habilita logs no dashboard.
>    Para restrição efetiva de acesso (OAuth Google/GitHub, IP allowlist,
>    mTLS) você precisa de **Edge Policies (planos pagos $20+/mês)**.
>    O `--basic-auth user:pass` foi DEPRECATED no agent v3 — use
>    `--basic-auth-user user --basic-auth-pass pass` ou edge policies.
> 3. **Nunca exponha endpoints com dados sensíveis** sem `X-API-Key` (header
>    de autenticação planejado para Sprint v2.45+) ou rate limiting.
> 4. **Mate o tunnel** (`Ctrl+C` em ngrok) ao terminar — URLs ativas
>    indefinidamente são alvo de scanners automáticos.
> 5. **Monitore o painel ngrok** (`http://localhost:4040`) durante a sessão
>    para detectar requests de origens inesperadas (IPs, User-Agents).
> 6. **Trate a URL como segredo de curta duração** — não cole em commits,
>    prints públicos, screenshots compartilhados ou notebooks versionados.
>    URLs `*.ngrok.app` ficam expostas em: (a) histórico do Chrome, (b)
>    logs do Colab quando pasted em código, (c) clipboard managers, (d)
>    relatórios automáticos. Considere já expirada após sessão pública.

```bash
# Terminal 1: API REST local
source ~/Geosteering_AI_venv/bin/activate
export GEOSTEERING_MODEL_PATH=/path/to/pipeline_salvo
geosteering-api --host 127.0.0.1 --port 8000 &

# Terminal 2: ngrok tunnel
pip install pyngrok
ngrok http 8000  # copiar URL https://*.ngrok.app
```

Free tier ngrok: 40 req/min — suficiente para validação leve, insuficiente
para treinamento que dispara muitas chamadas. Upgrade $8/mês desbloqueia
unlimited + domínio fixo.

---

## Workflows

### Workflow A — Validação JAX GPU (resposta à pergunta D3)

**Trigger**: "Valide JAX em Colab GPU" / "Rode os 109 testes JAX em T4"

**Passos**:
1. Hook `colab-token-refresh` valida token (auto)
2. Abrir Chrome em Colab Pro+ via MCP
   ```text
   mcp__colab-mcp__open_colab_browser_connection
     → URL: https://colab.research.google.com/
   ```
3. Upload `notebooks/colab_templates/validate_jax_gpu_v240.ipynb`
4. Selecionar Runtime → GPU T4
5. Run all cells (notebook é auto-contido):
   - Cell 1: `!pip install -e git+https://github.com/daniel-guitarplayer-8/geosteering-ai.git@v2.40#egg=geosteering-ai[all]`
   - Cell 2: `import jax; assert jax.devices()[0].platform == 'gpu'`
   - Cell 3: `!pytest tests/test_simulation_jax_*.py -m gpu -v --json-report --json-report-file=/tmp/jax_gpu.json`
   - Cell 4: Compara cada teste contra baseline Numba CPU (paridade <1e-10)
   - Cell 5: Exporta `/tmp/paridade_jax_gpu_T4.json` para Drive
6. Download JSON via MCP
7. Commit em `docs/colab_runs/v2.40_jax_gpu_T4_{date}.json`

**Critério de sucesso**: 109/109 PASS + max_paridade < 1e-10 em todos.

### Workflow B — Treinamento Remoto mp16

**Trigger**: "Treine ResNet com mp16 em Colab" / "Run train_v240_mp16"

**Passos**:
1. Confirmar com usuário (custo ~$1-5 em Colab Pro+)
2. Garantir API REST local com ngrok ativo
3. Upload `train_v240_mp16.ipynb`
4. Configurar variável de notebook `colab_api_base_url` = URL ngrok
5. Selecionar Runtime → GPU A100 (ideal) ou T4 (fallback)
6. Run all:
   - Clone repo + install `.[all]`
   - Load dataset de Drive (path em config notebook)
   - `setup_mixed_precision_policy(config)` ANTES de `build_model`
   - `TrainingLoop(config).run()` por N épocas
   - Push `model.keras` + `scalers.joblib` + `config.yaml` para Drive
   - Smoke test: POST `{ngrok_url}/predict` com 1 amostra de val
7. Download artefatos via MCP
8. Salvar resultados em `docs/colab_runs/v2.40_train_{date}/`

**Critério de sucesso**: val_loss converge, smoke test 200 OK.

### Workflow C — Benchmark v2.40 (tf.data + mp16)

**Trigger**: "Benchmark mp16 em Colab" / "Medir samples/sec v2.40"

**Passos**:
1. Upload `benchmark_tfdata_mp16.ipynb`
2. Runtime → T4 primeiro, depois A100 (2 runs separados)
3. Run all — notebook mede 4 configurações × 5 runs cada:
   - C1: fp32 baseline (use_mixed_precision=False, use_xla=False)
   - C2: +mp16 (use_mixed_precision=True)
   - C3: +mp16+XLA (use_xla=True)
   - C4: +mp16+XLA+tf.data tuned (shuffle=N/2, num_parallel_calls=8)
4. Calcula mediana e stdev. Reject runs com stdev>5%
5. Exporta JSON `docs/perf_baselines/v2.40_tf_training_{device}.json`
6. Atualiza `docs/PERFORMANCE_BASELINE.md` seção "TF Training Throughput"
   (commit dedicado pelo usuário, não automático)

**Critério de sucesso**: C2 > C1 × 1.15 (ganho mp16 ≥15%) em T4.

---

## Princípios

1. **Confirmação humana antes de treinar**: notebooks de treinamento custam
   tempo de GPU; sempre perguntar antes de invocar MCP open_browser
2. **Token health-check proativo**: hook colab-token-refresh deve rodar
   antes; se token <50min de validade, avisar usuário
3. **Templates auto-contidos**: cada `.ipynb` clona repo + instala — não
   depender de estado prévio da sessão Colab
4. **Output sempre em JSON**: facilita versionamento + diff entre runs
5. **Caminho do JSON é convenção**: `docs/perf_baselines/v2.40_*_{device}.json`
   ou `docs/colab_runs/v2.40_*_{date}.json`
6. **MCP timeout 60min**: notebooks mais longos → orientar Tier C v2.45+

---

## Troubleshooting

### "MCP colab-mcp não responde"

1. Verificar se Chrome está aberto: `ps aux | grep -i chrome`
2. Verificar extensão Colab MCP instalada e ativa
3. Verificar `.mcp.json:10-14` aponta para `uvx git+https://github.com/googlecolab/colab-mcp`
4. Restart Claude Code (`/restart` ou kill e reabrir)

### "Token GCP expirado"

```bash
gcloud auth login    # abre navegador, refaz OAuth
gcloud auth print-access-token  # confirma novo token
```

### "ngrok rate-limit (40 req/min) atingido"

- Reduzir frequência de POST /predict no notebook (batch maior)
- Upgrade ngrok $8/mês para unlimited
- Alternativa: Cloudflare Tunnel (free, exige domínio próprio)

### "Notebook falha em `pip install -e git+...`"

- Verificar URL repo: `https://github.com/daniel-guitarplayer-8/geosteering-ai.git`
- Confirmar tag `v2.40` existe no remote (`git ls-remote --tags origin`)
- Em Colab, usar `!pip install --upgrade pip` antes

### "Paridade JAX GPU vs Numba CPU > 1e-10"

- Esperado em ~5% dos testes (precisão fp32 GPU vs fp64 CPU)
- Notebook marca como XFAIL com TODO
- Não bloqueia sprint — investigar caso a caso v2.41+

---

## Limitações

- **NÃO substituo** validação Fortran (paridade <1e-12 mantida via simulator Python local)
- **NÃO controlo** Colab runtime selection (usuário escolhe T4/A100 na UI)
- **NÃO automatizo** setup ngrok (segurança: token bash inseguro)
- **NÃO faço** debugging interativo em Colab (apenas dispara + lê output)
- **NÃO substituo** GitHub Actions GPU runner (futuro v2.42+, orçamento dedicado)

---

## Referências

- `docs/reports/v2.39_proximos_passos_roadmap_2026-05-17.md` §2.3 — spec I2.2
- `docs/ARCHITECTURE_v2.md` §22 (Parte V) — 4-tier A/B/C/D
- `.mcp.json:10-14` — config MCP oficial
- `.claude/hooks/colab-token-refresh.sh` — hook companheiro
- `notebooks/colab_templates/__README.md` — doc dos 3 templates
