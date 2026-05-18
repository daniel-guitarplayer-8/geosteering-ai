# Templates Colab — Geosteering AI v2.40 (I2.2 MCP colab-bridge)

Templates de notebooks Jupyter prontos para execução remota em **Google Colab Pro+**
via Claude Code MCP (`mcp__colab-mcp__open_colab_browser_connection`).

Cada template é **auto-contido**: clona o repositório, instala dependências, executa
a operação alvo, exporta resultados em JSON e os salva no Google Drive.

---

## Templates Disponíveis

| Arquivo | Cells | Runtime alvo | Tempo médio | Saída |
|:---|:---:|:---|:---:|:---|
| `train_v240_mp16.ipynb` | 10 | A100 (ideal) ou T4 | 30–60 min | `model.keras` + `scalers.joblib` + history JSON |
| `validate_jax_gpu_v240.ipynb` | 8 | T4 | 5–10 min | `paridade_jax_gpu_T4.json` (109 testes) |
| `benchmark_tfdata_mp16.ipynb` | 12 | T4 + A100 | 10–20 min | `v2.40_tf_training_{device}.json` |

---

## Convenções

### 1. Variáveis de notebook (configuráveis sem editar células)

Cada template tem uma célula de configuração no topo. Variáveis padronizadas:

```python
# Configuração obrigatória
GIT_TAG          = "v2.40"        # tag do repo a clonar
RUNTIME_DEVICE   = "T4"            # T4 | A100 | V100
DRIVE_PATH       = "/content/drive/MyDrive/Geosteering_AI/v2.40/"  # output dir

# Configuração opcional (com defaults seguros)
SEED             = 42
COLAB_API_BASE_URL = ""            # set para "https://*.ngrok.app" se notebook chama API REST local
```

### 2. Output JSON padronizado

Todo template exporta um JSON em `/content/drive/.../{nome_resultado}.json`:

```json
{
  "template": "validate_jax_gpu_v240",
  "git_tag": "v2.40",
  "runtime": "T4",
  "timestamp_utc": "2026-05-18T15:30:00Z",
  "results": { ... },
  "duration_sec": 312.5,
  "exit_code": 0
}
```

Esse JSON é o **único artefato que importa**. Claude Code MCP faz download e commita em:

- `docs/perf_baselines/v2.40_*.json` — para benchmarks (performance baseline)
- `docs/colab_runs/v2.40_*_{date}.json` — para runs ad-hoc (não-baseline)

### 3. Mensagens em PT-BR

Todas as mensagens `print()` em notebooks (e.g. progresso, status) usam PT-BR
acentuado para consistência com o resto do projeto. Comentários em código em PT-BR
quando explicativos, EN quando técnicos puros.

### 4. Idempotência

Notebooks podem ser re-executados sem efeitos colaterais:
- Clone do repo com `git pull` se já existe
- Output JSON sobrescreve com timestamp atualizado
- Cell de smoke test antes de cell de execução pesada

---

## Como Usar (Manual)

```text
1. Abrir Colab Pro+ no navegador (https://colab.research.google.com/)
2. Upload do template desejado
3. Runtime → Change runtime type → GPU → T4 ou A100
4. Run all (Ctrl+F9)
5. Aguardar conclusão (tempo médio na tabela acima)
6. Download do JSON de saída via UI ou MCP
```

## Como Usar (Automatizado via Claude Code MCP)

```text
Prompt no Claude Code:
  "Rode validate_jax_gpu_v240 em Colab T4"

Skill geosteering-colab-mcp dispara:
  1. Verifica token GCP via hook colab-token-refresh.sh
  2. mcp__colab-mcp__open_colab_browser_connection
  3. Upload template
  4. Configurar runtime via UI bridge
  5. Run all + aguardar
  6. Download JSON
  7. Commit em docs/colab_runs/
```

Detalhes completos em `.claude/commands/geosteering-colab-mcp.md`.

---

## Setup do Ambiente Colab

Cada template começa com **bloco de setup standard**:

```python
# Cell 1: clonar repo na tag específica
!git clone --depth 1 --branch v2.40 https://github.com/daniel-guitarplayer-8/geosteering-ai.git
%cd geosteering-ai

# Cell 2: instalar com extras [all]
!pip install -q -e ".[all]"

# Cell 3: validar imports e GPU
import jax, tensorflow as tf
assert jax.devices()[0].platform == "gpu", "GPU JAX não detectada!"
assert len(tf.config.list_physical_devices("GPU")) > 0, "GPU TF não detectada!"
print(f"✓ JAX device: {jax.devices()[0]}")
print(f"✓ TF GPUs: {tf.config.list_physical_devices('GPU')}")
```

---

## Templates Futuros (v2.41+)

Adiados desta sprint para focar nos 3 mínimos viáveis:

- `eda_dataset_v240.ipynb` — exploratory data analysis em Colab GPU
- `sm_multi_freq_dip_v240.ipynb` — Simulation Manager com GPU
- `geosignal_grids_v240.ipynb` — geradores de geosinais
- `picasso_dtb_v240.ipynb` — Picasso plots + DTB curves
- `train_surrogate_tcn_v240.ipynb` — treinar SurrogateNet TCN 127M
- `train_surrogate_modern_tcn_v240.ipynb` — treinar ModernTCN 204M

---

## Limitações Conhecidas (v2.40)

- **Free tier ngrok**: 40 req/min — pode estrangular `train_v240_mp16` se notebook
  chama `POST /predict` muitas vezes. Workaround: upgrade $8/mês.
- **MCP timeout 60min**: notebooks muito longos (>1h) precisam Tier C
  (`pdwi2020/colab-exec`) — adiado para v2.45+.
- **Chrome obrigatório**: Tier B exige extensão Colab MCP no navegador.
  Headless Linux precisa Tier C.
- **A100 fila**: Pro+ não garante A100 imediato; T4 é fallback.

---

*Templates criados em 2026-05-18 como parte da Sprint v2.40 (I2.2 MCP colab-bridge).*
