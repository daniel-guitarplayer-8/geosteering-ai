---
Spec: 0011b-sm-params
Tarefas-de: spec.md
Status: planejado
Data: 2026-06-06
---

# Tarefas 0011b — Params completos

## Constituição (gate)
Sem violação: física intocada (só `simulate_batch`); `positions_z` replica a fórmula Fortran do
monólito (paridade de comportamento); VM puro (Princípio X); `core` não importa `gui`/`apps`. Sem ADR.

## Contratos (assinaturas)
```python
# gui/services/sim_request.py
@dataclass(frozen=True)
class SimRequest:
    frequencies_hz: tuple[float,...] = (20000.0,)
    tr_spacings_m:   tuple[float,...] = (1.0,)
    dip_degs:        tuple[float,...] = (0.0,)
    h1: float = 1.0          # NOVO — altura do 1º ponto-médio (m)
    tj: float = 10.0         # NOVO — janela de investigação (m)
    p_med: float = 1.0       # NOVO — passo de medidas (m)
    n_models: int = 2
    backend: str = "numba"

def _compute_positions_z(req) -> np.ndarray:   # FÓRMULA FORTRAN EXATA
    cos_d = max(1e-6, math.cos(math.radians(abs(req.dip_degs[0]))))
    n_pos = max(1, int(math.ceil(req.tj / (req.p_med * cos_d))))
    return np.linspace(-req.h1, req.tj - req.h1, n_pos, dtype=np.float64)
# _build_batch usa _compute_positions_z (substitui o linspace genérico)

# apps/.../viewmodel.py  (PURO)
class SimulationViewModel(BaseViewModel):
    # estado: _frequencies/_dips/_tr_spacings (tuplas) + _h1/_tj/_p_med + _n_models + _status
    @property frequencies/dips/tr_spacings/h1/tj/p_med/n_models (+ setters via _set)
    @property n_pos -> int            # derivado (read-only): max(1, ceil(tj/(p_med·cos dip0)))
    def validate() -> list[str]       # errata POR ELEMENTO + h1/tj/p_med > 0
    def run()                          # monta SimRequest completo → service.run
```

## Lista
| # | Tarefa | Cobre | Arquivos | Dep |
|:--|:--|:--|:--|:--:|
| **T01** | `SimRequest` + `_compute_positions_z` (Fortran) + `_build_batch` usa | RF-1/RF-2/AC-1 | `gui/services/sim_request.py` | — |
| **T02** | `SimulationViewModel` multi-valor + validação completa + `n_pos` derivado | RF-3/AC-2/AC-4 | `apps/.../viewmodel.py` | T01 |
| **T03** | `SimulatorView` inputs (CSV + spinboxes) + label n_pos | RF-4 | `apps/.../view.py` | T02 |
| **T04** | Atualizar testes skeleton (escalar→multi) + novos (positions_z fidelidade, validação, e2e multi-config) | RF-5/AC-1..5 | `tests/test_sim_app_skeleton.py` | T03 |
| **T05** | Verify: ruff/mypy + suíte + regressão SM 16/16 + fronteira | AC-5 | — | T04 |

## Commit
Sem commitar até revisão; pré-aplicar auto-fixers; footer Co-Authored-By.

## GATE-T
- [x] Toda tarefa → ≥1 AC; última = verify (regressão + fronteira); fidelidade `positions_z` testada (AC-1).
