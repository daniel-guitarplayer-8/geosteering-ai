# Plano Sprint v2.40.1 — Patch dos 5 Findings dos Revisores

**Branch alvo**: `patch/v2.40.1-reviewers-findings`
**Base**: `main` @ tag `v2.40` (commit `0dab72b`)
**Data**: 2026-05-18
**Esforço estimado**: 1–2h em 1 sessão

---

## Context

A Sprint v2.40 foi mergeada em `main` e tagueada como `v2.40`. Durante o fan-out de
code-review/security-audit pós-merge, 5 findings de severidade BAIXO foram identificados
e adiados para este patch. Eles não bloqueiam funcionalmente o release v2.40 mas
representam débito técnico que vale a pena fechar antes da próxima sprint (v2.41) para:

1. Evitar poluição de logs INFO em treinos com Optuna (100+ trials × N épocas)
2. Adicionar guard regression test contra import circular `registry → training`
3. Aumentar segurança operacional ao expor API REST local via ngrok
4. Garantir rastreabilidade exata (commit hash, não só tag) de runs Colab
5. Clarificar comentário ambíguo sobre origem do cap `tf_shuffle_buffer_size=100000`

**Resultado esperado**: 5 fixes pequenos commitados, suite 1653+ PASS preservada,
patch tagueado como `v2.40.1`, sem mudanças em arquitetura ou produção crítica.

---

## Validações Pré-Plano (Já Confirmadas via Read-Only)

| Item | Estado Atual Verificado | Linha |
|:---|:---|:---|
| Code #2 | `setup_mixed_precision_policy` chama `set_global_policy` + `logger.info` SEMPRE, sem checar policy atual | `training/loop.py:259-273` |
| Code #5 | `TestBuildModelWithMpPolicyV240` (3 testes) JÁ existe — mas sem teste EXPLÍCITO para ordem de import | `tests/test_training.py:986-1019` |
| Security #5 | Seção ngrok existe mas SEM aviso de URL pública por padrão | `.claude/commands/geosteering-colab-mcp.md:147-163` |
| Security #6 | Templates têm `GIT_TAG = "v2.40"` + JSON com `"git_tag"`; SEM `git rev-parse HEAD` | 3 `.ipynb` em `notebooks/colab_templates/` |
| Code #8 | Comentário `(validado em __post_init__)` ambíguo — "validado" pode ser lido como empírico | `config.py:363` |

---

## Decisões (D1–D5) — Confirmadas via AskUserQuestion

| # | Decisão | Origem |
|:--:|:---|:---|
| **D1** | Idempotência via `tf.keras.mixed_precision.global_policy().name == target`: skip + log DEBUG se já correto | Resolução do Code #2 |
| **D2** | Adicionar `test_no_circular_import_registry_to_training` (redundância intencional como regression guard) | Usuário escolheu "Adicionar teste explícito" |
| **D3** | Warning explícito em ngrok §Setup item 3 — URL pública por padrão + recomendação de ngrok auth/private endpoints + não usar conta pessoal para produção | Resolução do Security #5 |
| **D4** | `!git rev-parse HEAD > /tmp/git_commit.txt` após clone → ler em Python → adicionar `"git_commit"` ao JSON | Usuário escolheu "git rev-parse após clone" |
| **D5** | Trocar `(validado em __post_init__)` por `(assertado em __post_init__; valor heurístico — não medido empiricamente)` | Usuário escolheu "Trocar 'validado' por 'assertado'" |

---

## Arquivos a Modificar (5)

| Arquivo | Mudança | LOC |
|---|---|---:|
| `geosteering_ai/training/loop.py` | Idempotência em `setup_mixed_precision_policy` (D1) | +12/-4 |
| `tests/test_training.py` | +1 teste explícito de ordem de import (D2) | +25 |
| `.claude/commands/geosteering-colab-mcp.md` | Aviso ngrok URL pública (D3) | +12 |
| `notebooks/colab_templates/train_v240_mp16.ipynb` | +cell git rev-parse + campo git_commit no JSON (D4) | +5 |
| `notebooks/colab_templates/validate_jax_gpu_v240.ipynb` | +cell git rev-parse + campo git_commit no JSON (D4) | +5 |
| `notebooks/colab_templates/benchmark_tfdata_mp16.ipynb` | +cell git rev-parse + campo git_commit no JSON (D4) | +5 |
| `geosteering_ai/config.py` | Trocar comentário linha 363 (D5) + também linha 1092 (consistência) | +2/-2 |

**Total**: ~66 LOC em 6 arquivos (1 patch sprint, sem novos arquivos).

---

## Detalhamento dos Fixes

### Fix #1 — Idempotência em `setup_mixed_precision_policy` (D1)

**Arquivo**: `geosteering_ai/training/loop.py:259-273`

**Estado atual**:
```python
if config.use_mixed_precision:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    logger.info("Mixed precision ATIVADO (módulo-level)...")
else:
    tf.keras.mixed_precision.set_global_policy("float32")
    logger.info("Mixed precision DESATIVADO (módulo-level)...")
```

**Após fix**:
```python
target = "mixed_float16" if config.use_mixed_precision else "float32"
current = tf.keras.mixed_precision.global_policy().name

if current == target:
    logger.debug(
        f"Mixed precision policy já está em '{target}' — chamada idempotente, "
        f"ignorada (evita poluição de logs em Optuna multi-trial)."
    )
    return

tf.keras.mixed_precision.set_global_policy(target)
if config.use_mixed_precision:
    logger.info("Mixed precision ATIVADO (módulo-level): policy='mixed_float16'. ...")
else:
    logger.info("Mixed precision DESATIVADO (módulo-level): policy='float32' (reset defensivo).")
```

### Fix #2 — Teste explícito de import order (D2)

**Arquivo**: `tests/test_training.py` (adicionar dentro de `TestBuildModelWithMpPolicyV240` ou novo classe).

**Novo teste**:
```python
def test_no_circular_import_registry_to_training(self):
    """Guard regression: importar registry NÃO deve disparar import top-level de training.

    Sprint v2.40 D5 — build_model_with_mp_policy faz lazy import de
    setup_mixed_precision_policy DENTRO da função para evitar ciclo:
        models.registry → training.loop → (volta para registry?)

    Este teste prova que importar registry sem touch em training funciona,
    E que importar registry seguido de training (e vice-versa) não levanta
    ImportError nem ModuleNotFoundError.
    """
    import importlib
    import sys

    # Limpar imports cached para forçar reimport limpo
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("geosteering_ai.models") or \
           mod_name.startswith("geosteering_ai.training"):
            del sys.modules[mod_name]

    # Importar registry PRIMEIRO (não deve disparar training.loop top-level)
    from geosteering_ai.models import registry
    assert "build_model_with_mp_policy" in registry.__all__

    # Agora importar training — não deve falhar
    from geosteering_ai.training import loop
    assert hasattr(loop, "setup_mixed_precision_policy")

    # Inverter ordem: limpar e importar training PRIMEIRO
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("geosteering_ai.models") or \
           mod_name.startswith("geosteering_ai.training"):
            del sys.modules[mod_name]

    from geosteering_ai.training import loop as loop2  # noqa: F811
    from geosteering_ai.models import registry as registry2  # noqa: F811
    assert callable(registry2.build_model_with_mp_policy)
```

### Fix #3 — Warning ngrok URL pública (D3)

**Arquivo**: `.claude/commands/geosteering-colab-mcp.md:147-163`

**Adicionar BLOCO de warning ANTES do bash de instalação ngrok**:
```markdown
> ⚠️ **AVISO DE SEGURANÇA — ngrok expõe sua API publicamente**
>
> URLs `*.ngrok.app` são acessíveis a QUALQUER pessoa que descubra/receba a URL.
> Não há autenticação por padrão. Implicações:
>
> 1. **Não use conta ngrok pessoal para produção** — use conta corporativa/dedicada.
> 2. **Considere ngrok auth tokens** (planos pagos) para restringir acesso por usuário.
> 3. **Nunca exponha endpoints com dados sensíveis** sem `X-API-Key` (planejado v2.45+).
> 4. **Mate o tunnel** (`Ctrl+C` em ngrok) ao terminar — URLs ativas indefinidamente
>    são atacáveis por scanners automáticos.
> 5. **Monitore o painel ngrok** (`http://localhost:4040`) para detectar requests
>    inesperados durante a sessão.
```

### Fix #4 — Commit hash em templates (D4)

**Arquivos**: 3 `.ipynb` em `notebooks/colab_templates/`

Após cada `!git clone --depth 1 --branch {GIT_TAG} ...`, adicionar:

```python
# Capturar commit hash exato (mais imutável que tag, que pode ser movida)
!cd geosteering-ai && git rev-parse HEAD > /tmp/git_commit.txt
with open("/tmp/git_commit.txt") as f:
    GIT_COMMIT = f.read().strip()
print(f"✓ Pinned commit: {GIT_COMMIT[:12]}")
```

E no JSON de saída de cada notebook, adicionar campo:
```python
"git_commit": GIT_COMMIT,  # hash imutável, complementa git_tag
```

### Fix #5 — Comentário config.py (D5)

**Arquivo**: `geosteering_ai/config.py`

Linha 363:
```python
# ANTES: Cap 100000 anti-OOM em GPU T4 (validado em __post_init__).
# DEPOIS: Cap 100000 anti-OOM em GPU T4 (assertado em __post_init__;
#         valor heurístico — não medido empiricamente).
```

Linha 1092 (consistência — mesmo comentário em outro lugar):
```python
# ANTES: tf_shuffle_buffer_size: 0 = desativar shuffle, ate 100000 (anti-OOM T4).
# DEPOIS: tf_shuffle_buffer_size: 0 = desativar shuffle, ate 100000
#         (cap heurístico anti-OOM T4 — não medido empiricamente).
```

---

## Plano de Execução (5 commits granulares)

| # | Commit | Esforço | Dep |
|--:|---|---:|---|
| C01 | `fix(training): idempotência setup_mixed_precision_policy (Code #2)` | 15 min | — |
| C02 | `test(training): guard regression import circular registry→training (Code #5)` | 15 min | — |
| C03 | `docs(colab-mcp): aviso ngrok URL pública (Security #5)` | 10 min | — |
| C04 | `feat(notebooks): pinning commit hash em 3 templates (Security #6)` | 20 min | — |
| C05 | `style(config): clarificar comentários cap 100000 tf.data (Code #8)` | 5 min | — |

**Sequência ótima**: Todos os 5 commits são independentes (sem dependências), podem ser
feitos em qualquer ordem ou em paralelo (1 sessão única, ~65 min total).

---

## Reviewers Pós-Implementação (Fan-Out Paralelo)

Após os 5 commits:
1. **`/code-review:code-review`** — revisão interna code-quality
2. **`/coderabbit:coderabbit-review`** — revisão externa (post-push)
3. **`/geosteering-code-reviewer`** — revisão domínio-específica do projeto

Em paralelo (single message, multiple tool calls). Findings BLOQUEANTES (CRÍTICO/ALTO) → fix imediato. BAIXO → adiar para v2.40.2 ou v2.41.

---

## Critérios de Aceitação

### Funcionais
- [ ] `setup_mixed_precision_policy(config)` chamada 2× consecutivas com mesmo config emite log INFO apenas 1× (segunda chamada → DEBUG)
- [ ] `pytest tests/test_training.py::TestBuildModelWithMpPolicyV240 -v` → 4/4 PASS (3 existentes + 1 novo)
- [ ] `pytest tests/test_training.py::TestBuildModelWithMpPolicyV240::test_no_circular_import_registry_to_training -v` → PASS isoladamente
- [ ] Templates Colab geram JSON com chave `"git_commit"` populada com hash de 40 chars
- [ ] Ngrok §Setup tem bloco `⚠️ AVISO DE SEGURANÇA`
- [ ] `config.py:363` e `config.py:1092` não contêm a palavra "validado"

### Suite total
- [ ] `pytest tests/ -q --tb=no --ignore=tests/test_simulation_manager_gui.py` → **1654 PASS** (1653 + 1 novo) / 0 FAIL
- [ ] Paridade Fortran <1e-12 preservada (não tocamos `simulation/`)
- [ ] mypy 0 erros novos
- [ ] Pre-commit hooks: PT-BR + check-anti-patterns + ruff + ruff-format → ALL OK

### Reviewers
- [ ] `/code-review:code-review` → 0 findings BLOQUEANTES
- [ ] `/coderabbit:coderabbit-review` → 0 findings críticos
- [ ] `/geosteering-code-reviewer` → 0 findings BLOQUEANTES

---

## Riscos & Mitigações

| # | Risco | Probab. | Mitigação |
|--:|---|---|---|
| R1 | Teste import-order falha por imports compartilhados em conftest.py | Média | Limpeza explícita de `sys.modules` no teste isola o cenário |
| R2 | Idempotência quebra teste que reseta policy entre testes | Baixa | Testes resetam via `set_global_policy("float32")` no `finally` — funcionará pois é mudança real de estado |
| R3 | Cell git rev-parse falha se cwd não estiver em geosteering-ai/ | Média | Usar `cd geosteering-ai && git rev-parse HEAD` (cwd explícito) |
| R4 | Tag v2.40.1 conflita com convenção (patch deveria ser v2.40.1 ou v2.40-patch?) | Baixa | Seguir SemVer: v2.40.1 (patch) |

---

## Verificação End-to-End

```bash
# 1. Setup branch (a partir de main @ v2.40)
cd ~/Geosteering_AI
git checkout main && git pull
git checkout -b patch/v2.40.1-reviewers-findings

# 2. Aplicar 5 fixes em commits independentes
# (Edits manuais conforme detalhamento acima)

# 3. Testes locais
pytest tests/test_training.py::TestBuildModelWithMpPolicyV240 -v
pytest tests/test_config.py -v
pytest tests/ -q --tb=no --ignore=tests/test_simulation_manager_gui.py

# 4. Smoke import (proves no circular)
python -c "from geosteering_ai.models.registry import build_model_with_mp_policy; print('OK')"
python -c "from geosteering_ai.training.loop import setup_mixed_precision_policy; print('OK')"

# 5. Smoke idempotência (call 2x, count INFO logs)
python -c "
import logging
logging.basicConfig(level=logging.INFO)
from geosteering_ai.config import PipelineConfig
from geosteering_ai.training.loop import setup_mixed_precision_policy
cfg = PipelineConfig(use_mixed_precision=True)
print('--- 1ª chamada ---')
setup_mixed_precision_policy(cfg)
print('--- 2ª chamada (deve ser silenciosa em INFO) ---')
setup_mixed_precision_policy(cfg)
"

# 6. Fan-out reviewers (paralelo)
# /code-review:code-review
# /coderabbit:coderabbit-review
# /geosteering-code-reviewer

# 7. Merge + tag
git checkout main
git merge --no-ff patch/v2.40.1-reviewers-findings
git tag -a v2.40.1 -m "Patch v2.40.1: 5 findings de revisores (Code #2/#5/#8 + Security #5/#6)"
git push origin main --tags
```

---

## Critical Files Reference

| Arquivo | Linha | Função |
|---|---|---|
| [geosteering_ai/training/loop.py:222-273](geosteering_ai/training/loop.py#L222-L273) | `setup_mixed_precision_policy` — alvo do Fix #1 |
| [tests/test_training.py:986-1019](tests/test_training.py#L986-L1019) | `TestBuildModelWithMpPolicyV240` — onde adicionar Fix #2 |
| [.claude/commands/geosteering-colab-mcp.md:147-163](.claude/commands/geosteering-colab-mcp.md#L147-L163) | Seção ngrok — alvo do Fix #3 |
| [notebooks/colab_templates/train_v240_mp16.ipynb](notebooks/colab_templates/train_v240_mp16.ipynb) | Template treinamento — alvo do Fix #4 |
| [notebooks/colab_templates/validate_jax_gpu_v240.ipynb](notebooks/colab_templates/validate_jax_gpu_v240.ipynb) | Template validação JAX — alvo do Fix #4 |
| [notebooks/colab_templates/benchmark_tfdata_mp16.ipynb](notebooks/colab_templates/benchmark_tfdata_mp16.ipynb) | Template benchmark — alvo do Fix #4 |
| [geosteering_ai/config.py:363](geosteering_ai/config.py#L363) | Comentário cap 100000 — alvo do Fix #5 |
| [geosteering_ai/config.py:1092](geosteering_ai/config.py#L1092) | Comentário consistente — alvo do Fix #5 |

---

## Próximos Passos Pós-v2.40.1

Após merge desta patch:
- **v2.41a — SurrogateNet Training** (4-8h GPU) — modelo TCN/ModernTCN treinado em Colab A100
- **v2.41b — POST /simulate** (6-8h) — expor simulador JAX via API REST
- **v2.42a — Catálogo de Ruído** (8-10h) — 35 tipos físicos restantes em `noise/functions.py`
- **v2.42b — use_flat_prange default** (1-2h) — +20-47% throughput após validação cache

---

*Plano elaborado 2026-05-18 via /plan + AskUserQuestion. 1 Explore agent + 3 AskUserQuestion para confirmar D2/D5/D6. Pronto para ExitPlanMode.*
