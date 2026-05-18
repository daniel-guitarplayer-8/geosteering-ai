# Plano — Simulation Manager v2.8 — Correções de Bugs (Crosshair, Modos EM, Correlação) + Redesign

> **Branch atual**: `feat/simulation-manager-v2.7c` (renomear para `feat/simulation-manager-v2.8`)
> **Smoke test baseline**: 124/124 PASS (v2.7c)
> **Invariante absoluto**: simuladores JAX/Numba/Fortran INTOCADOS; zero impacto em performance computacional
> **Pós-implementação**: rodar `/code-review` e corrigir issues críticos/warning

---

## Contexto

Três bugs confirmados pela investigação de código + análise da imagem fornecida pelo usuário:

### Bug 1 — Crosshair distorce dados em modo Re + Im (`sm_crosshair.py`)
**Root cause**: Race condition de timing em `_on_motion()`. Quando `_bg` (dict de backgrounds blitting) está vazio — situação que ocorre logo após `update_axes()` ser chamado — `_on_motion()` chama `_on_resize_or_draw(None)` para capturar o background. Se isso acontece ANTES de `draw_idle()` renderizar o novo plot (timing não garantido), os pixels capturados são do estado anterior (stale). Ao fazer `restore_region(bg_stale)` nos plots de Re, os dados do plot correto são sobrescritos pelo background antigo, causando a distorção visível na imagem.

O bug é específico para Re+Im (2 colunas) porque: (a) há mais axes por figura, (b) axes adjacentes com `sharey=True` têm interação de blit mais sensível ao timing.

**Localização**: `sm_crosshair.py:183-215` (`_on_motion`) + `simulation_manager.py:5931-5945` (`_refresh_crosshair_axes`)

### Bug 2 — Modos "Só Re" / "Só Im" não filtram + "Magnitude + Fase" exige escala manual (`sm_plots.py` + `simulation_manager.py`)

**Root cause 1** — `sm_plots.py:1154-1164`: O bloco `_scale_to_kind` sobrescreve o `kind` explícito do usuário. Quando o usuário seleciona `"Só Re"` em `combo_kind_mode` mas `scale_mode="re_im"` (padrão), o código mapeia `kind` de volta para `"Re + Im"`. Resultado: `"Só Re"` se comporta igual a `"Re + Im"`.

**Root cause 2** — `simulation_manager.py`: `combo_kind_mode.currentIndexChanged` não tem nenhum callback conectado. Mudar o modo não dispara replot automático — usuário precisa clicar "Plotar" manualmente.

**Localização**: `sm_plots.py:1149-1164` + `simulation_manager.py:~4453` (inicialização de `combo_kind_mode`)

### Bug 3 — Matriz de Correlação: Spearman/Kendall não exibe + dialog limitado (`sm_correlation.py`)

**Root cause**: Spearman e Kendall estão corretamente implementados em `compute_correlation_matrix()`, mas o dialog NÃO conecta `combo_method.currentIndexChanged` a `_on_recompute()`. Usuário precisa mudar o método E clicar "Recomputar" — UX não intuitivo, especialmente porque a computação inicial de Pearson ocorre automaticamente.

Adicionalmente, o dialog atual é minimal (heatmap + tabela simples) sem estatísticas derivadas, seleção de componentes, p-values ou comparação entre métodos.

**Localização**: `sm_correlation.py:241-400` (CorrelationAnalysisDialog)

---

## Arquivos Afetados

| Arquivo | Seção | Mudança |
|:--------|:------:|:--------|
| `sm_crosshair.py` | linha 183-215 | Fix `_on_motion()`: retornar cedo quando `_bg` vazio |
| `simulation_manager.py` | linha ~4453 | Conectar `combo_kind_mode` a callback de auto-replot |
| `simulation_manager.py` | linha ~5931 | `_refresh_crosshair_axes()`: usar `canvas.draw()` síncrono |
| `sm_plots.py` | linhas 1149-1164 | Remover override `_scale_to_kind` que sobrescreve kind explícito |
| `sm_correlation.py` | linhas 241-400 | Redesign do CorrelationAnalysisDialog com tabs + auto-recompute |
| `simulation_manager.py` | smoke tests | +5 testes T8–T12 |

**NÃO TOCAR**: simuladores (`_jax/`, `_numba/`, `forward*.py`, `Fortran_Gerador/`), `sm_workers.py`, `sm_io.py`, `sm_benchmark.py`, `sm_plot_backends/`.

---

## Fix 1 — Crosshair: eliminar race condition de background stale

### 1a — `sm_crosshair.py` (`_on_motion`, linha ~191-192)

```python
# ANTES (linha 191-192):
if not self._bg:
    self._on_resize_or_draw(None)  # ← captura BG potencialmente stale

# DEPOIS:
if not self._bg:
    return  # Aguarda draw_event capturar BG correto antes de processar motion
```

**Justificativa**: `draw_idle()` já foi chamado em `enable()`. O `draw_event` vai disparar na próxima iteração do event loop e capturar o BG correto. Retornar cedo evita que o primeiro movimento de mouse use um BG stale — o crosshair simplesmente não aparece no primeiro pixel, mas aparece corretamente a partir do segundo movimento (imperceptível para o usuário).

### 1b — `simulation_manager.py` (`_refresh_crosshair_axes`, linha ~5941-5943)

```python
# DEPOIS de cm.update_axes(list(fig.axes)), forçar draw síncrono
# para garantir que draw_event captura BG APÓS o novo plot ser renderizado:
try:
    self.canvas.canvas.draw()  # síncrono — garante render antes do próximo motion
except Exception:
    pass
```

**Justificativa**: `draw()` (síncrono) garante que o plot está renderizado antes de qualquer evento de mouse. Isso resolve o problema de timing para o caso de replot. O custo de performance é aceitável (ocorre apenas ao replottar, não em cada frame do crosshair).

---

## Fix 2 — Modos EM: kind explícito + auto-replot

### 2a — `sm_plots.py` (linhas 1149-1164): Remover override incorreto

```python
# REMOVER o bloco inteiro (linhas 1154-1164):
# if kind not in ("Magnitude + Fase", "Re + Im"):
#     _scale_to_kind = { "re_im": "Re + Im", ... }
#     if scale_mode in _scale_to_kind:
#         kind = _scale_to_kind[scale_mode]

# SUBSTITUIR por comentário explicativo:
# v2.8: kind vem diretamente do combo_kind_mode do usuário.
# scale_mode modula COMO magnitude/fase é computada (log10/dB/rad),
# nunca sobrescreve o tipo de layout (1-col vs 2-col).
```

**Impacto**: Quando usuário seleciona "Só Re" com `scale_mode="re_im"` (padrão), `kind` permanece `"Só Re"`. Plot correto: 1 coluna com apenas Re.

**Compatibilidade**: `plot_geosignals()` e `plot_tensor_full()` não usam o parâmetro `kind` de `plot_em_profile()` — zero impacto nessas funções.

### 2b — `simulation_manager.py` (próximo à linha 4633): Conectar combo_kind_mode

No mesmo bloco onde `combo_plot_kind.currentIndexChanged` é conectado, adicionar:

```python
# Linha existente:
self.combo_plot_kind.currentIndexChanged.connect(
    self._update_scale_combos_visibility
)
# ADICIONAR após:
self.combo_kind_mode.currentIndexChanged.connect(
    self._on_kind_mode_changed
)
```

Novo método `_on_kind_mode_changed()`:

```python
def _on_kind_mode_changed(self, *_args) -> None:
    """Auto-replot ao mudar Modo (EM) — v2.8."""
    if self._active_bundle is None and self._current_sim is None:
        return  # Sem dados — não replotar
    self._on_plot()
```

---

## Fix 3 — Correlation Dialog: auto-recompute + redesign com tabs

### Estrutura do novo CorrelationAnalysisDialog (sm_correlation.py)

**Layout Redesenhado:**
```
CorrelationAnalysisDialog (1000 × 760 px, não-modal)
├── Painel de Configurações (topo, QFrame)
│   ├── Método: [pearson ▾]   Dados: ○Re(H) ●|H|   Componentes: [Todos ▾]
│   └── Status + indicador de progresso (spinner ou label)
├── QTabWidget
│   ├── Tab 1: "Mapa de Calor"    ← heatmap imshow (atual, melhorado)
│   ├── Tab 2: "Tabela"           ← tabela numérica (atual, melhorado)
│   └── Tab 3: "Pares Principais" ← top-N correlações + p-values
└── Barra inferior: [Exportar CSV] [Exportar PNG]  [Fechar]
```

### Mudanças no `_on_recompute()`:
- Conectar `combo_method.currentIndexChanged.connect(self._on_recompute)` → auto-recompute
- Conectar `check_real.stateChanged.connect(self._on_recompute)` → auto-recompute
- Adicionar `btn_recompute` visível como fallback para computação manual

### Tab 3 — "Pares Principais":
- Calcula p-values via `scipy.stats.pearsonr/spearmanr/kendalltau` (segundo retorno)
- Mostra lista ordenada por |correlação| descrescente: par, r, p-value, interpretação verbal
- Interpretação: |r|≥0.9 = "Muito forte", 0.7–0.9 = "Forte", 0.4–0.7 = "Moderado", <0.4 = "Fraco"
- Exclui diagonal (auto-correlação = 1.0)

### Seleção de componentes:
- Combo "Componentes" com opções: "Todos (9)", "Diagonal (Hxx,Hyy,Hzz)", "Custom..."
- "Custom..." abre checkboxes para seleição de subset

### Export:
- "Exportar CSV": `pd.DataFrame(matrix, index=labels, columns=labels).to_csv()` — fallback se pandas indisponível: `np.savetxt()`
- "Exportar PNG": `canvas.save(path)` via `EMCanvas.save()`

---

## Novos Smoke Tests (+5, total ≥ 129)

```python
# T8: Crosshair _on_motion retorna cedo quando _bg vazio (sem stale capture)
# T9: combo_kind_mode="Só Re" → plot_em_profile recebe kind="Só Re" (não overridden)
# T10: combo_kind_mode="Magnitude + Fase" com scale_mode="re_im" → kind="Magnitude + Fase"
# T11: _on_kind_mode_changed() existe e é chamável (conexão existindo)
# T12: CorrelationAnalysisDialog: combo_method change dispara _on_recompute (auto-compute)
```

---

## Sequência de Implementação

1. **Fix 1** (`sm_crosshair.py` + `_refresh_crosshair_axes`) — isolado, baixo risco
2. **Fix 2a** (`sm_plots.py` remoção de override) — 1 bloco de 10 linhas
3. **Fix 2b** (`simulation_manager.py` conexão + `_on_kind_mode_changed`) — ~15 linhas
4. **Fix 3** (`sm_correlation.py` redesign) — maior escopo (~200 LOC novo dialog)
5. **Smoke tests T8–T12**
6. **`python -m geosteering_ai.simulation.tests.simulation_manager --smoke-test`** → ≥129 PASS, 0 FAIL
7. **`/code-review`** (CodeRabbit) → corrigir issues críticos/warning

---

## Verificação Final

```bash
cd ~/Geosteering_AI
source ~/Geosteering_AI_venv/bin/activate

# Smoke test
python -m geosteering_ai.simulation.tests.simulation_manager --smoke-test
# Esperado: ≥129 PASS, 0 FAIL

# Verificar simuladores intocados
git diff --name-only | grep -E "_jax|_numba|forward|Fortran"
# Esperado: nenhuma saída
```

---

## Relatório Final

Ao término da implementação, gerar relatório completo incluindo:
- Bugs corrigidos com root cause e linha exata
- Mudanças UX no dialog de correlação
- Novos testes e baseline de smoke tests
- Verificação de integridade dos simuladores
