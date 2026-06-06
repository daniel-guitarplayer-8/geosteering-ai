# Fluxo de design de GUI com Qt Designer (`.ui`) — Geosteering AI

| Atributo | Valor |
|:--|:--|
| **Aplica-se a** | Simulation Manager MVVM (`apps/sim_manager/`) e futuro Geosteering AI Studio |
| **Helper** | `geosteering_ai.gui.qt_compat.load_ui` |
| **Primeiro `.ui`** | [`apps/sim_manager/perspectives/simulation/simulator.ui`](../../apps/sim_manager/perspectives/simulation/simulator.ui) |
| **Teste-guarda** | `tests/test_gui_ui_files.py` |
| **Princípio** | MVVM (Constituição §X) — o `.ui` é **só a View**; o ViewModel permanece Python puro |

Este documento descreve como **desenhar a GUI visualmente** (em vez de só por prompt),
editar no Qt Designer/Qt Creator e devolver ao Claude Code para implementação.

---

## 1. Por que `.ui` (e por que é MVVM-safe)

Um arquivo `.ui` é **XML declarativo** que descreve a árvore de widgets e o layout.
Ele **não importa nada** e **não contém lógica** — logo, por construção, **não pode
ferir a pureza do ViewModel**. A separação fica:

```
simulator.ui (XML, só widgets)  ──load_ui──▶  View fina (Qt, à mão)
                                                  │  conecta sinais (Qt + VMSignal)
                                                  │  lê valores por objectName
                                                  ▼
                                       SimulationViewModel (Python PURO, INTOCADO)
```

**Regra de ouro:** nem o `.ui`, nem a View, podem conter regra de domínio
(validação da errata, `compute_n_pos`, OOM guard). Tudo isso vive no ViewModel.

---

## 2. Como editar a GUI no Qt Designer / Qt Creator

### Abrir
```bash
# Qt Designer standalone (vem com PyQt6/PySide6 ou pelo pacote pyqt6-tools):
designer apps/sim_manager/perspectives/simulation/simulator.ui
# OU no Qt Creator: File ▸ Open ▸ simulator.ui (abre na aba Design)
```

> Se não tiver o `designer` instalado: `pip install pyqt6-tools` (traz o `qt6-tools designer`)
> ou use o Qt Creator (instalação completa do Qt).

### Editar
- Arraste widgets, ajuste layouts, espaçamentos, labels, ranges, tooltips — tudo visual.
- **NÃO renomeie os `objectName`s do contrato** (ver §4) sem combinar — a View fina os usa
  para ler valores e conectar sinais. Renomear quebra o binding silenciosamente
  (o teste-guarda `tests/test_gui_ui_files.py` falha cedo nesse caso).
- Para **adicionar** um campo novo: dê um `objectName` semântico (ex.: `myNewSpin`) e
  avise o Claude Code para ligá-lo ao ViewModel.

### Devolver ao Claude Code
Basta **salvar o `.ui` e commitar** (ou anexar na conversa). O Claude Code lê o XML
diretamente e implementa a View fina + o binding ao ViewModel.

---

## 3. Como o código carrega o `.ui`

Use o helper único (binding-agnóstico PyQt6/PySide6):

```python
from geosteering_ai.gui.qt_compat import load_ui

ui = load_ui("apps/sim_manager/perspectives/simulation/simulator.ui")
ui.runButton.clicked.connect(on_run)        # acesso por objectName
freqs_text = ui.freqsEdit.text()            # ler valores
```

**Padrão de composição** (recomendado, funciona igual nos dois bindings): a View fina
embute a widget carregada e acessa os filhos por `objectName`:

```python
class SimulatorView(QtWidgets.QWidget):
    def __init__(self, vm, parent=None):
        super().__init__(parent)
        self._vm = vm
        self._ui = load_ui(_UI_PATH)                 # widget de topo do .ui
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._ui)
        # binding: View ↔ Qt signals ↔ VMSignal (VM puro, intocado)
        self._ui.runButton.clicked.connect(self._on_run)
        self._vm.changed.connect(self._on_vm_changed)
```

> **`pyuic6` vs `uic.loadUi`:** este projeto usa **`load_ui` (runtime)**, mantendo o
> `.ui` como **fonte única** (sem `.py` gerado no repo). Não usamos `pyuic6` — se um dia
> usar, lembre que o `.py` gerado **não se edita à mão** (é regenerado a cada save).

---

## 4. Contrato de `objectName`s (binding)

A View fina depende destes nomes. O teste `tests/test_gui_ui_files.py::test_simulator_ui_binding_contract_intact`
falha se algum sumir. Principais grupos do `simulator.ui`:

| Grupo | objectNames | Estado no MVVM |
|:--|:--|:--|
| Geometria/aquisição | `freqsEdit` `dipsEdit` `trsEdit` `h1Spin` `tjSpin` `pMedSpin` `nModelsSpin` `nPosLabel` | **Já ligado** |
| Backend | `backendCombo` (numba/jax/auto + fortran*) | **Já ligado** (fortran*) |
| Geologia | `geoModeCombo` `geoGeneratorCombo` `geoDistrCombo` `geoRhoMinSpin` `geoRhoMaxSpin` `geoAnisoCheck` `geoLambdaMinSpin` `geoLambdaMaxSpin` `geoNlfCheck` `geoNlfSpin` `geoNlMinSpin` `geoNlMaxSpin` `geoMinThickSpin` `geoSeedRandomCheck` `geoSeedSpin` | **Já ligado** |
| Sessão | `saveButton` `openButton` `statusLabel` | **Já ligado** |
| Execução* | `runButton` `pauseButton` `resumeButton` `cancelButton` `progressBar` `outputDirEdit` `browseOutButton` | *Design-alvo Fatia 6a |
| Log/Histórico* | `logEdit` `historyList` `historySearchEdit` | *Design-alvo Fatia 6a/6c |
| Perfis/Hankel/Paralelismo* | `canonicalCombo` `autoGeoCheck` `genRandomRadio` `genManualRadio` `editLayersButton` `hankelAutoRadio` `hankelManualRadio` `hankelPathEdit` `hankelOrderSpin` `workersSpin` `threadsSpin` | *Design-alvo Fatias 6b/6h |

\* **Design-alvo** = o widget já está no `.ui` (para você desenhar a tela completa), mas
ainda não tem binding no ViewModel — será ligado na fatia indicada (ver
`docs/reports/sm_mvvm_investigacao_paridade_design_sdd_premortem_2026-06-06.md`).

---

## 5. Loop de iteração visual (o agente "vê" a GUI)

O Claude Code consegue **renderizar a GUI offscreen e olhar o resultado** — sem display,
usando a infra de CI headless que já existe (`QT_QPA_PLATFORM=offscreen`):

```python
# render offscreen → PNG (o agente depois LÊ o PNG e ajusta)
import os; os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from geosteering_ai.gui.qt_compat import QtWidgets, load_ui
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
ui = load_ui("apps/sim_manager/perspectives/simulation/simulator.ui")
ui.resize(1000, 880)
ui.grab().save("/tmp/preview.png")   # QWidget.grab() é offscreen-friendly
```

Isso transforma "gerar UI às cegas" em **iteração visual fechada**: gera → screenshot →
o agente enxerga alinhamento/espaçamento/widgets → corrige. Foi assim que o
`simulator.ui` foi validado (incluindo a correção do `&` virando mnemônico nos títulos).

---

## 6. Pitfalls

- **`--` em comentário `.ui`** é XML inválido (comentário XML não pode conter `--`).
  Use `──` (box-drawing) ou outro caractere.
- **`&` em título/label** vira mnemônico do Qt (`&aquisição` ⇒ sublinha o "a"). Para um
  `&` literal use `&amp;&amp;`, ou evite `&` (use "e"/"·").
- **`objectName` duplicado** quebra `findChild`. Mantenha-os únicos no arquivo.
- **Editar o `.py` do `pyuic6` à mão** — não fazemos isso (usamos `load_ui` em runtime).
- **Lógica no `.ui`/na View** — proibido. ViewModel é o único lugar de regra de domínio.

---

## 7. Recomendação (resumo)

- **Telas grandes/estáticas** (futuro Studio, formulários densos) → desenhe no **Qt Designer (`.ui`)**.
- **Telas dinâmicas/compostas** (galeria, grupos parametrizados) → **Python à mão**.
- **ViewModels** → **sempre** Python puro, à mão, nunca tocados pelo Designer.
- **Iteração** → use o **loop de screenshot offscreen** (§5).

Ref. completa: `docs/reports/sm_mvvm_investigacao_paridade_design_sdd_premortem_2026-06-06.md` (§5).
