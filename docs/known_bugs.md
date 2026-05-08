# Bugs Conhecidos do Geosteering AI

**Última atualização**: 2026-05-07
**Mantenedor**: Daniel Leal

Este documento cataloga bugs históricos do projeto com IDs estáveis (`KB-XXX`)
para suportar:

1. **Detecção automática** via `.claude/hooks/check-anti-patterns.sh`
   (regex em `.claude/anti-patterns.txt`).
2. **Testes de regressão** com nome `test_kbXXX_*` em `tests/`.
3. **Documentação institucional** — quando um bug é encontrado, ele recebe
   um KB-ID e fica catalogado para todas as iterações futuras.

> **Política**: ao corrigir um bug que pode ser reintroduzido, ANTES do fix
> adicione (1) entrada aqui, (2) regra em `anti-patterns.txt` e (3) teste
> de regressão. Esta ordem garante que o hook bloqueia a reintrodução.

---

## Sumário

| ID | Versão Intro | Versão Fix | Severidade | Sumário |
|:--:|:------------:|:----------:|:----------:|:--------|
| KB-001 | v1.0 | v1.0.1 | ALTA | Sinal trocado em decoupling Hxz |
| KB-002 | v2.0 | v2.0.5 | MÉDIA | Curriculum 3-fase off-by-one em epoch=0 |
| KB-013 | v2.13 | v2.21 | **CRÍTICA** | Nested `prange` em `_fields_in_freqs_kernel_cached` |
| KB-018 | v2.18 | v2.19 | ALTA | `rng_seed=42` hardcoded na GUI |
| KB-019 | v2.19 | v2.20 | MÉDIA | Defaults de threading com oversubscription em CPUs HT/SMT |

---

## KB-001 — Sinal trocado em decoupling Hxz

| Campo | Valor |
|:------|:------|
| **Severidade** | ALTA |
| **Versão introduzida** | v1.0 |
| **Versão corrigida** | v1.0.1 |
| **Arquivos afetados** | `geosteering_ai/simulation/_numba/dipoles.py` |

### Causa raiz

O fator de decoupling para a componente cruzada `Hxz` foi inicialmente
implementado com sinal positivo, contradizendo o limite analítico para
dipolos magnéticos horizontais (HMD) sob convenção `e^{-iωt}`.

### Sintoma

Comparações com Fortran indicavam diferença de ~2× no módulo de `Hxz`
para configurações com TX-RX desalinhados (dip ≠ 0°), enquanto componentes
diagonais (`Hxx`, `Hyy`, `Hzz`) permaneciam corretas (<1e-12).

### Hook de prevenção

Não há regex específico — KB-001 é prevenida por `validate-physics.sh`
que verifica os fatores `ACp = -1/(4πL³)` e `ACx = +1/(2πL³)` em
`config.py` e por testes de paridade Fortran.

### Teste de regressão

`tests/test_simulation_compare_fortran.py::test_decoupling_signs`

---

## KB-002 — Curriculum 3-fase off-by-one em epoch=0

| Campo | Valor |
|:------|:------|
| **Severidade** | MÉDIA |
| **Versão introduzida** | v2.0 |
| **Versão corrigida** | v2.0.5 |
| **Arquivos afetados** | `geosteering_ai/noise/curriculum.py` |

### Causa raiz

O scheduler de ruído curricular calculava a fração de progresso com
`epoch / total_epochs` em vez de `(epoch + 1) / total_epochs`,
resultando em ruído zero na epoch 0 de cada fase em vez do valor inicial
calibrado por fase (Phase 1: 0.001, Phase 2: 0.01, Phase 3: 0.05).

### Sintoma

Métricas de validação irregulares no início de cada transição de fase;
modelos treinados ficavam mais sensíveis a ruído do que o desejado.

### Hook de prevenção

`check-anti-patterns.sh` com regex `epoch\s*/\s*total_epochs` em
`*noise/curriculum.py`. O padrão correto exige `+1` no numerador.

### Teste de regressão

`tests/test_curriculum.py::test_kb002_phase_transition_no_off_by_one`

---

## KB-013 — Nested `prange` em Numba (CRÍTICO)

| Campo | Valor |
|:------|:------|
| **Severidade** | **CRÍTICA** |
| **Versão introduzida** | v2.13 (Sprint 13.1, commit `0f92035`) |
| **Versão corrigida** | v2.21 (Sprint 21.1, commit `cba27dd`) |
| **Arquivos afetados** | `geosteering_ai/simulation/_numba/kernel.py` |

### Causa raiz

A Sprint 13.1 (v2.13) adicionou `@njit(parallel=True, nogil=True)` com
`prange(nf)` na função `_fields_in_freqs_kernel_cached`. Esta função é
chamada **milhões de vezes** de dentro de `_simulate_combined_prange`
(em `forward.py`), que **já tem prange outer**.

Numba serializa o `prange` interno automaticamente (não há paralelismo
real aninhado), mas o overhead de setup do parallel scheduler é pago
em **cada chamada do kernel**. Em Cenário E (600 pts × 4 workers × 1
combo), isso acumula ~14s extras sem ganho funcional.

### Sintoma

Cenário E (600 pts, 4w × 2t): **122k mod/h → 46k mod/h** (-62%)
sem alteração visível no código. A regressão passou despercebida porque:
1. Cenário A (30 pts) continuou veloz (overhead absoluto pequeno).
2. Paridade Fortran <1e-12 foi preservada (não é bug numérico).
3. Os benchmarks só foram comparados intra-versão, não inter-versão.

A causa-raiz só foi identificada em v2.21 via análise comparativa com
`old_geosteering_ai/simulation/_numba/kernel.py` (versão pré-v2.13).

### Fix

Sprint 21.1: `@njit(parallel=True, nogil=True)` → `@njit(cache=True, nogil=True)`,
`prange(nf)` → `range(nf)` em `_fields_in_freqs_kernel_cached`. Mantém
paralelismo na camada externa (`_simulate_combined_prange`) onde o ganho
é real, removendo o overhead de scheduler aninhado.

### Hook de prevenção

`check-anti-patterns.sh` com regex `@njit\([^)]*parallel=True` em
`*_numba/kernel.py`. Qualquer tentativa de adicionar `parallel=True` a
funções "folha" do kernel é bloqueada.

### Teste de regressão

`tests/test_regression_simulator.py::test_kb013_no_nested_prange_in_kernel` —
inspeciona o decorador via `inspect.getsource` e falha se `parallel=True`
aparecer em funções que são chamadas de dentro de prange outer.

---

## KB-018 — `rng_seed=42` hardcoded na GUI

| Campo | Valor |
|:------|:------|
| **Severidade** | ALTA (bug funcional) |
| **Versão introduzida** | v2.18 |
| **Versão corrigida** | v2.19 |
| **Arquivos afetados** | `geosteering_ai/simulation/tests/simulation_manager.py:8088` |

### Causa raiz

Na refatoração da GUI Simulation Manager v2.18, o parâmetro `rng_seed`
da `ModelGenerationThread` foi hardcoded com valor `42` em vez de ler
da UI ou usar `secrets.randbits(63)`.

### Sintoma

Toda execução de "Iniciar Simulação" produzia a **mesma sequência de
modelos** (mesmas camadas, espessuras, ρh/ρv, dip), tornando o gerador
inútil para criação de ensembles estatísticos para treino de redes neurais.

### Fix (v2.19)

UI control com `chk_random_seed` (default checked) + `spn_fixed_seed`
(habilitado quando unchecked) + `get_rng_seed() -> Optional[int]`.
`rng_seed=None` → `secrets.randbits(63)` em `ModelGenerationThread.run()`.
Sinal `seed_used = Signal(int)` emite a semente real para logging.

### Hook de prevenção

`check-anti-patterns.sh` com regex `rng_seed\s*=\s*42` em
`*simulation_manager.py`. Smoke test (`_run_smoke_test`) é exceção
explícita — usa `rng_seed=42` para determinismo de teste.

### Teste de regressão

`tests/test_simulation_random_seed.py::test_kb018_random_seed_produces_different_models`

---

## KB-019 — Defaults de threading com oversubscription em CPUs HT/SMT

| Campo | Valor |
|:------|:------|
| **Severidade** | MÉDIA |
| **Versão introduzida** | v2.19 |
| **Versão corrigida** | v2.20 |
| **Arquivos afetados** | `geosteering_ai/simulation/tests/sm_workers.py` |

### Causa raiz

A função `recommend_default_parallelism()` em v2.19 usava `os.cpu_count()`
(cores lógicos) para definir `workers × threads_per_worker`, resultando
em `4w × 4t = 16` em CPUs 8C/16T (HT habilitado). Numba paraleliza em
threads de computação intensa onde HT/SMT degrada (ALU saturation).

### Sintoma

Cenário E em CPU 8C/16T: 4w × 2t (8 threads, fisicamente otimo) atingia
46k mod/h vs 4w × 4t (16 threads, oversubscribed) com 38k mod/h (-25%).

### Fix (v2.20)

`recommend_default_parallelism()` agora usa `physical_cores` (via
`psutil.cpu_count(logical=False)`) com fallback heurístico. Total de
threads ≤ `physical_cores`. Investigação empírica em v2.20 (5 runs ×
2 configs) confirmou a estratégia.

### Hook de prevenção

`check-anti-patterns.sh` com regex que detecta combinações 4w × 4t
hardcoded em `*sm_workers.py`. Decisões de paralelismo devem passar
por `recommend_default_parallelism()`.

### Teste de regressão

`tests/test_sm_workers.py::test_kb019_no_oversubscription_in_defaults`

---

## Como adicionar um novo KB

```markdown
## KB-XXX — Título curto

| Campo | Valor |
|:------|:------|
| **Severidade** | BAIXA / MÉDIA / ALTA / CRÍTICA |
| **Versão introduzida** | vX.Y (commit hash) |
| **Versão corrigida** | vX.Y (commit hash) |
| **Arquivos afetados** | path/file.py |

### Causa raiz
[Explicação técnica detalhada]

### Sintoma
[Comportamento observável]

### Fix
[O que foi alterado]

### Hook de prevenção
[Regex em anti-patterns.txt OU outro mecanismo]

### Teste de regressão
[Caminho do teste, nome da função]
```

Após adicionar:
1. Registrar regex em `.claude/anti-patterns.txt`
2. Criar teste `test_kbXXX_*` em `tests/`
3. Atualizar tabela de Sumário acima
