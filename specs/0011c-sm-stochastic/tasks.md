---
Spec: 0011c-sm-stochastic
Tarefas-de: spec.md
Status: planejado
Data: 2026-06-06
---

# Tarefas 0011c — Geração estocástica (paridade monólito; pool adiado)

## Constituição (gate)
Sem violação: física só `simulate_batch` (<1e-12); `positions_z` Fortran intacto; gerador PURO (Princípio X);
`core` não importa `gui`/`apps`; monólito só rewire de import (comportamento preservado, bit-paridade da seed).
DRY via extração (não duplicar). Sem ADR (extração Strangler já é padrão do projeto, specs 0004-0007).

## Contratos (assinaturas)
```python
# gui/services/stochastic_geology.py  (NOVO — core extraído de sm_model_gen, PURO sem Qt)
@dataclass class GenConfig: total_depth, n_layers_min, n_layers_max, n_layers_fixed, rho_h_min/max,
    rho_h_distribution, anisotropic, lambda_min/max, min_thickness, generator, normal_mu_log/sigma_log
def generate_models(cfg, n_models, rng_seed=None, return_seed=False) -> list[dict]   # MODEL_KEYS
MODEL_KEYS = ("n_layers","rho_h","rho_v","lambda","thicknesses")
GENERATORS_AVAILABLE = [sobol, halton, niederreiter, mersenne_twister, uniform, normal, box_muller]
# + _resolve_rng_seed, _generate_one_model, _build_n_layer_choices, _sample_vector, _samples_*,
#   _generate_thicknesses, _log_transform, _uniform_transform  (todos movidos)

# simulation/tests/sm_model_gen.py  (REWIRE — re-importa o core + mantém Qt)
from geosteering_ai.gui.services.stochastic_geology import (GenConfig, generate_models, MODEL_KEYS,
    GENERATORS_AVAILABLE, _resolve_rng_seed, _generate_one_model, _build_n_layer_choices)
DEFAULT_GEN_CHUNK_SIZE = 500
class ModelGenerationThread(QThread): ...   # FICA (Qt-coupled)
__all__ = [...]  # re-exporta tudo que os consumidores usam

# gui/services/sim_request.py
@dataclass(frozen=True) class SimRequest:  # += campos geologia
    geology_mode: str = "fixed"            # "fixed" | "stochastic"
    n_layers_min: int = 3; n_layers_max: int = 11; n_layers_fixed: Optional[int] = None
    rho_h_min: float = 1.0; rho_h_max: float = 1000.0; rho_h_distribution: str = "loguni"
    anisotropic: bool = True; lambda_min: float = 1.0; lambda_max: float = sqrt(2)
    min_thickness: float = 1.0; generator: str = "sobol"
    normal_mu_log: float = 2.0; normal_sigma_log: float = 1.0
    rng_seed: Optional[int] = None
def _genconfig_from_request(req) -> GenConfig          # total_depth=req.tj
def _simulate_grouped(models, positions_z, req) -> (H6, info)  # agrupa por n_layers; reassembla ordem
# _run_simulation: dispatch fixed/stochastic

# apps/.../viewmodel.py: properties de geologia + validate() (espelha GenConfig.validate) + run()
# apps/.../view.py: widgets de geologia
```

## Lista
| # | Tarefa | Cobre | Arquivos | Dep |
|:--|:--|:--|:--|:--:|
| **T01** | Extrair core puro → `stochastic_geology.py`; rewire `sm_model_gen.py` (re-export) | RF-1/RF-2/AC-1 | `gui/services/stochastic_geology.py` (novo), `simulation/tests/sm_model_gen.py` | — |
| **T02** | Verificar regressão monólito (paridade bit-a-bit + suites) | AC-2 | — | T01 |
| **T03** | `SimRequest` geology fields + `_genconfig_from_request` + `_simulate_grouped` + dispatch `_run_simulation` | RF-3/RF-4/AC-3/AC-4 | `gui/services/sim_request.py` | T01 |
| **T04** | VM properties geologia + validação; View widgets | RF-5/RF-6/AC-5 | `apps/.../viewmodel.py`, `apps/.../view.py` | T03 |
| **T05** | Testes (extração paridade, agrupamento ordem/shape, VM, e2e estocástico) + GATE-V + revisão adversarial + commit | RF-7/AC-1..6 | `tests/test_sm_stochastic.py` (novo) + `tests/test_sim_app_skeleton.py` | T04 |

## Commit
Sem commitar até revisão; pré-aplicar auto-fixers; footer Co-Authored-By. Pode separar em 2 commits:
(A) extração+rewire (refactor, regressão verde) e (B) integração no app (feature).

## GATE-T
- [x] Toda tarefa → ≥1 AC; T02 = regressão da extração (bit-paridade); última = verify+review;
  agrupamento (AC-4) e extração pura (AC-1) testados; pool adiado (não há tarefa de pool).
