# Análise Técnica: Redução de Warmup no Simulador Python Numba JIT

**Documento:** `docs/reports/warmup_analysis_jit_2026-05-12.md`
**Versão do Simulador:** v2.30
**Data:** 2026-05-12
**Autor:** Daniel Leal
**Status:** Investigação completa — sem implementação

---

## Sumário Executivo

Este documento analisa cinco estratégias para reduzir ou eliminar o tempo de
warmup do simulador Python Numba JIT (`geosteering_ai/simulation/_numba/`).
A investigação combina leitura do código-fonte, inspeção dos artefatos de cache
em disco (`.nbc`/`.nbi`) e benchmarks empíricos de tempo de inicialização.

**Achado principal**: o cold-start de **111 segundos** em novo processo Python
não é resolvível completamente com as ferramentas Numba disponíveis na versão
0.61.2, pois a causa-raiz é o backend LLVM que recompila bitcode (`.nbc`) para
código nativo a cada novo processo. `cache=True` já está implementado em todas
as 18 funções `@njit` e provê o maior ganho possível dentro da arquitetura JIT.

| Estratégia | Status | Impacto Possível | Viabilidade |
|:-----------|:------:|:-----------------|:-----------:|
| 1. `cache=True` | ✅ Implementada | ~3-5× vs. sem cache | Máxima (já ativa) |
| 2. AOT `cc.compile()` | ❌ Não implementada | Zero warmup teórico | Baixa — incompatível com `parallel=True` |
| 3. Warmup Manual | ⚠️ Parcial | Resolve intra-sessão | Média |
| 4. Tipos Consistentes | ⚠️ Maioria ok | −20-40 s (consolidar 2 especializações) | Alta |
| 5. Pré-import de Bibliotecas | ✅ Implementada | Oportunidade em `NUMBA_CACHE_DIR` ramdisk | Alta |

---

## 1. Diagnóstico — Estado Atual

### 1.1 Medições Empíricas

Todas as medições foram realizadas em Apple Silicon (macOS 25.4.0, Python 3.13,
Numba 0.61.2) no ambiente `~/Geosteering_AI_venv`.

| Cenário | Tempo Medido | Observação |
|:--------|:------------:|:-----------|
| Import `simulate_multi` (mesmo processo, memória quente) | **1,864 s** | LLVM já carregado; `.nbc` já mapeados |
| 1ª chamada JIT (mesmo processo, carregando `.nbc` do disco) | **1,007 s** | LLVM backend compila bitcode → código nativo |
| 2ª chamada (função em memória RAM) | **0,001 s** | Dispatch direto sem compilação |
| Cold-start completo (novo subprocess Python, `.nbc` em disco) | **111 s** | 16 `.nbc` carregados + recompilados pelo LLVM |
| Razão de overhead warmup vs. steady-state | **697×** | |

### 1.2 Arquitetura de Cache do Numba

```
Caminho de compilação na PRIMEIRA execução absoluta (sem .nbc):
  Python fonte (.py)
    → Análise de bytecode (Numba type inference)    ~50-200s por função complexa
    → Geração de LLVM IR                            ~10-50s
    → Otimizações LLVM (passes: mem2reg, inlining)  ~20-100s
    → Backend LLVM → código nativo (x86-64)         ~10-50s
    → Salva .nbc (LLVM bitcode) + .nbi (metadados)
  Total estimado para 18 funções: ~300-600s

Caminho com cache=True, NOVO processo Python (.nbc em disco):
  .nbc (LLVM bitcode)
    → Otimizações LLVM (reduzidas, mas não eliminadas)   ~5-40s por função
    → Backend LLVM → código nativo (x86-64)              ← AINDA COMPILA
    → Função disponível em memória
  Total medido: ~111s

Caminho com cache=True, MESMO processo Python:
  Código nativo já em RAM (cache em memória do processo)
    → Dispatch direto via Numba dispatcher
  Total: 0,001s
```

**O equívoco de implementação**: `.nbc` armazena **LLVM bitcode** — uma
representação intermediária compilada, mas não código de máquina nativo.
Cada novo processo Python deve executar o backend LLVM para converter
o bitcode em instruções x86-64. Funções grandes como `hmd_tiv` (~700 linhas
de kernel Numba) consomem dezenas de segundos neste passo.

### 1.3 Inventário de Funções @njit e Cache em Disco

```
Localização: geosteering_ai/simulation/_numba/__pycache__/
```

| Função | Arquivo `.nbc` | Tamanho | Especializações | Decoradores |
|:-------|:--------------:|--------:|:---------------:|:------------|
| `_fields_in_freqs_kernel` | `kernel.*-302.*.nbc` | **1.842 KB** | 1 | `@njit(cache=True)` |
| `_fields_in_freqs_kernel_cached` | `kernel.*-671.*.nbc` | **1.533 KB** | 1 | `@njit(cache=True, nogil=True)` |
| `_fields_at_single_freq` | `kernel.*-886.*.nbc` | **1.512 KB** | 1 | `@njit(cache=True, nogil=True)` |
| `hmd_tiv` | `dipoles.*-180.*.nbc` | **1.337 KB × 2** | **2 ⚠️** | `@njit(cache=True)` |
| `vmd` | `dipoles.*-708.*.nbc` | **734 KB × 2** | **2 ⚠️** | `@njit(cache=True)` |
| `precompute_common_arrays_cache` | `kernel.*-582.*.nbc` | **697 KB** | 1 | `@njit(cache=True, parallel=True, nogil=True)` |
| `common_arrays` | `propagation.*-251.*.nbc` | **556 KB** | 1 | `@njit(cache=True)` |
| `_compute_zrho_kernel` | `kernel.*-538.*.nbc` | **166 KB** | 1 | `@njit(cache=True)` |
| `_sanitize_profile_kernel` | `geometry.*-189.*.nbc` | **161 KB** | 1 | `@njit(cache=True, fastmath=True)` |
| `common_factors` | `propagation.*-483.*.nbc` | **157 KB** | 1 | `@njit(cache=True)` |
| `rotate_tensor` | `rotation.*-169.*.nbc` | **95 KB** | 1 | `@njit(cache=True, fastmath=True)` |
| `build_rotation_matrix` | `rotation.*-108.*.nbc` | **41 KB** | 1 | `@njit(cache=True, fastmath=True)` |
| `find_layers_tr` | `geometry.*-219.*.nbc` | **31 KB** | 1 | `@njit(cache=True, fastmath=True)` |
| `layer_at_depth` | `geometry.*-306.*.nbc` | **21 KB** | 1 | `@njit(cache=True, fastmath=True)` |
| **Total** | | **~12,8 MB** | **16 arquivos** | |

```
forward.py — __pycache__/ (raiz do pacote simulation):
  _simulate_combined_prange_flat   .nbc  ~2 MB  @njit(parallel=True, cache=True, nogil=True)
  _simulate_positions_njit_cached  .nbc  ~1 MB  @njit(parallel=True, cache=True, nogil=True)
```

**Total geral**: 14 `.nbi` + 16 `.nbc` = 30 artefatos de cache em disco (~15 MB)

### 1.4 Anomalia Detectada: Duplas Especializações em `hmd_tiv` e `vmd`

```
dipoles.hmd_tiv-180.py313.1.nbc   ← especialização 1
dipoles.hmd_tiv-180.py313.2.nbc   ← especialização 2  ⚠️ duplica o custo de loading
dipoles.vmd-708.py313.1.nbc        ← especialização 1
dipoles.vmd-708.py313.2.nbc        ← especialização 2  ⚠️ duplica o custo de loading
```

Investigação com `hmd_tiv.signatures` revelou que chamadas dentro do mesmo
processo Python geram apenas **1 especialização** para os tipos padrão.
As 2 especializações em disco foram criadas em sessões anteriores onde a
função foi invocada com variações de **flag de mutabilidade** nos arrays
(`mutable=True` vs `mutable=False` no tipo Numba `Array`).

A diferença observada na assinatura:
```
Especialização 1: Array(float64, 1, 'C', True,  aligned=True)  ← array writeable
Especialização 2: Array(float64, 1, 'C', False, aligned=True)  ← array readonly
```

Ambas são funcionalmente idênticas, mas o Numba gera binários LLVM separados.
O custo adicional: ~20-40 s extras no cold-start (carregamento de 2 × 1,3 MB
para `hmd_tiv` + 2 × 734 KB para `vmd`).

---

## 2. Estratégia 1: `cache=True`

### 2.1 Status: ✅ IMPLEMENTADO — Todos os 18 `@njit`

O wrapper `njit()` definido em
[`_numba/propagation.py:173-203`](../../../geosteering_ai/simulation/_numba/propagation.py)
aplica `cache=True` como **default global**:

```python
def njit(*args, **kwargs):
    kwargs.setdefault("cache", True)
    kwargs.setdefault("fastmath", False)
    kwargs.setdefault("error_model", "numpy")
    kwargs.setdefault("nogil", True)
    return _numba_njit(*args, **kwargs)
```

Isso garante que qualquer nova `@njit` adicionada ao projeto herda cache
automaticamente sem necessitar de configuração explícita.

### 2.2 O Que `cache=True` Faz — e Não Faz

```
COM cache=True, PRIMEIRA execução absoluta (sem .nbc):
  Bytecode Python → LLVM IR → Otimizações → Código nativo → salva .nbc
  Tempo: ~300-600s (todas as funções)

COM cache=True, NOVO processo Python (.nbc em disco):
  .nbc (bitcode) → Otimizações → Código nativo   ← ainda recompila backend!
  Tempo: ~111s

COM cache=True, MESMA sessão Python (segunda chamada):
  Código nativo em RAM → dispatch direto
  Tempo: 0,001s (697× mais rápido que o primeiro carregamento)
```

### 2.3 Efetividade Medida

| Condição | Tempo Total | Redução |
|:---------|:-----------:|:-------:|
| Sem cache, execução absoluta (estimado) | ~300-600 s | — |
| Com cache=True, novo processo | **111 s** | ~3-5× |
| Com cache=True, mesma sessão (2ª chamada) | **0,001 s** | ~300.000× |

### 2.4 Limitações Estruturais Irresolúveis

O `.nbc` não é código de máquina — é LLVM bitcode. O motivo pelo qual
o Numba não armazena código nativo puro (ELF/Mach-O com símbolos
relocáveis) é que o código nativo é altamente dependente de:

- Versão do processador (AVX2 vs. AVX-512, AArch64 vs. x86)
- Versão do sistema operacional (ABI, libc)
- Versão do LLVM/Numba (pode mudar otimizações)
- Endereços de memória (ASLR — position-independent code necessário)

Por isso, o Numba optou por armazenar LLVM IR otimizado (mais portável
que código nativo, mas ainda requer o passo final de codegen LLVM).

### 2.5 Conclusão

Estratégia totalmente implementada. É a mais impactante disponível dentro
da arquitetura JIT. Não resolve o cold-start de 111 s — este é um limite
fundamental do modelo `cache=True` do Numba.

---

## 3. Estratégia 2: AOT — Compilação Ahead-of-Time com `cc.compile()`

### 3.1 Status: ❌ NÃO IMPLEMENTADO — Análise revela incompatibilidade crítica

A compilação AOT do Numba (`numba.pycc` / `numba.cc`) gera uma extensão
Python nativa (`.so` / `.pyd`) com **código de máquina pré-compilado**.
Em teoria, elimina completamente o warmup:

```python
# Conceito AOT — não implementado no projeto
from numba.pycc import CC

cc = CC('geosteering_kernels_aot')

@cc.export('common_arrays', 'complex128[:,:](int64, int64, float64, ...)')
def _common_arrays_impl(n, npt, hordist, ...):
    # implementação idêntica à atual
    ...

cc.compile()
# Gera: geosteering_kernels_aot.cpython-313-darwin.so (código nativo, ~zero warmup)

# Em runtime:
from geosteering_kernels_aot import common_arrays  # import instantâneo
```

### 3.2 Limitações Críticas para Este Projeto

| Limitação | Impacto | Funções Afetadas |
|:----------|:--------|:-----------------|
| **`parallel=True` + `prange` não suportado em AOT** | Bloqueia todas as funções de alto throughput | `_simulate_combined_prange_flat`, `_simulate_positions_njit_cached`, `precompute_common_arrays_cache` |
| **`numba.pycc` depreciado em Numba 0.52+** | API instável, sem substituto público estável | Todas |
| **Assinaturas de tipo fixas obrigatórias** | `tuple` de retorno de 9 arrays (`common_arrays`) é problemático | `common_arrays`, `common_factors` |
| **Recompilação por plataforma** | Incompatível com `pip install` cross-platform | Todas |
| **Sem suporte a `nogil=True` explícito** | Perde vantagem em ThreadPoolExecutor externo | Todas |

### 3.3 Análise Função-a-Função

```
CANDIDATAS a AOT (funções folha sem prange):
  ✓ common_arrays()                   — sem prange, tipos razoavelmente fixos
  ✓ common_factors()                  — sem prange; retorno: tupla de 6 arrays
  ✓ find_layers_tr()                  — sem prange, tipos simples
  ✓ layer_at_depth()                  — sem prange, tipos simples
  ✓ rotate_tensor()                   — sem prange, 3×3 matrix
  ✓ build_rotation_matrix()           — sem prange, simples
  ~ hmd_tiv()                         — sem prange internamente, mas
                                        recebe arrays de cache (mutabilidade variável)
  ~ vmd()                             — idem
  ~ _fields_in_freqs_kernel_cached()  — sem prange, mas chamado de prange outer

INCOMPATÍVEIS com AOT (usam prange):
  ✗ precompute_common_arrays_cache()     — parallel=True + prange(nf)
  ✗ _simulate_positions_njit()           — parallel=True + prange(n_positions)
  ✗ _simulate_positions_njit_cached()    — parallel=True + prange(n_positions)
  ✗ _simulate_combined_prange()          — parallel=True + prange(n_total)
  ✗ _simulate_combined_prange_flat()     — parallel=True + prange(n_total 4D)
```

### 3.4 Estratégia Híbrida AOT Parcial (Teórica)

Compilar apenas as funções folha como AOT e manter as funções paralelas
como JIT com `cache=True`:

```
Economia estimada com AOT parcial (6 funções leaf):
  common_arrays      → AOT  → economiza ~15-20s no cold-start
  common_factors     → AOT  → economiza ~5-8s
  find_layers_tr     → AOT  → economiza ~1-2s
  rotate_tensor      → AOT  → economiza ~3-5s
  hmd_tiv            → AOT  → economiza ~15-20s (maior economia)
  vmd                → AOT  → economiza ~8-12s
  Subtotal estimado:           ~47-67s de redução

Funções paralelas permanecem JIT:
  _simulate_combined_prange_flat   → JIT ~20-30s (dominante)
  precompute_common_arrays_cache   → JIT ~10-15s
  Subtotal restante:               ~30-45s
```

**Cold-start com AOT parcial**: 111s → ~45-65s (redução de ~40-50%)

### 3.5 Por Que AOT Parcial Não Vale o Esforço

1. A API `numba.pycc` está depreciada sem substituto estável
2. O build de `.so` requer setup de compilação C++ por plataforma
3. Cada update do código-fonte requer recompilação AOT manual
4. As funções paralelas (`_simulate_combined_prange_flat`) continuam
   dominando o cold-start mesmo sem as funções leaf
5. A economia de ~50% no cold-start (111s → 55s) tem menor impacto
   prático que o pool persistente (que elimina o cold-start em chamadas 2+)

### 3.6 Conclusão

AOT **não é viável** para as funções críticas de performance do projeto
(todas as `@njit(parallel=True)`). Uma implementação híbrida parcial seria
possível com `numba.pycc` depreciado, mas a relação custo/benefício é
desfavorável dado que o pool persistente já resolve o problema em uso contínuo.

---

## 4. Estratégia 3: Pré-aquecimento Manual (Warmup Function)

### 4.1 Status: ⚠️ PARCIALMENTE IMPLEMENTADO em 3 pontos

#### 4.1.1 CLI `benchmark` — Warmup Call Explícito

Implementado em [`cli/benchmark.py:262-269`](../../../geosteering_ai/cli/benchmark.py):

```python
# Warmup com shape COMPLETO (W2 code-review v2.30): pré-aquece todas as
# especializações JIT do cenário, incluindo multi-freq, multi-TR e multi-dip.
_ = simulate_multi(
    positions_z=positions_z,
    models=models[:1],          # 1 modelo descartado
    cfg=cfg,
    frequencies_hz=frequencies_hz,
    tr_spacings_m=tr_spacings_m,
    dip_degs=dip_degs,
)
# Run cronometrado começa APÓS este warmup
t0 = time.perf_counter()
```

#### 4.1.2 Warmup Inline em `run_numba_chunk` — Sprint v2.28

Implementado em `simulation/tests/sm_workers.py` (Warmup C + D):

```python
# Warmup C: isotrópico _rho_v = _rho*0.3 + dip = 0°
# Warmup D: anisotrópico + dip = 30°
# Garantia: todos os caminhos JIT (isotrópico, anisotrópico, tilted) acionados
```

#### 4.1.3 Pool Persistente — `_workers.py`

Implementado em [`_workers.py:137-139`](../../../geosteering_ai/simulation/_workers.py):

```python
_PERSISTENT_POOL: Optional[ProcessPoolExecutor] = None
_PERSISTENT_POOL_CONFIG: Optional[Tuple[int, int, str]] = None
_LOCK = threading.Lock()
```

O pool é **reutilizado entre chamadas** com mesma configuração `(n_workers,
n_threads, hankel_filter)`. Após o primeiro cold-start, todas as chamadas
subsequentes têm overhead ≈ 0 (código nativo já em memória nos workers).

### 4.2 Por Que Warmup Manual Não Resolve o Cold-start de 111 s

```
Diagnóstico da primeira chamada de warmup em processo novo:

  simulate_multi(models[:1], ...)   ← o próprio warmup PAGA o cold-start
    ├── Dispatch para _workers.py   → ProcessPool.submit()
    ├── Worker spawned              → novo processo Python (~0,5s)
    ├── Worker importa simulate_multi → carrega 14 @njit (não compila ainda)
    ├── _simulate_combined_prange_flat() primeira chamada no worker
    │     ├── Numba: .nbc em disco? → SIM → carrega .nbc (~10-40s LLVM)
    │     └── Função disponível em memória
    ├── hmd_tiv() primeira chamada
    │     ├── Numba: .nbc em disco? → SIM → carrega .nbc × 2 especializações
    │     └── Função disponível (~15-30s)
    └── ... (demais funções) ...
  Total do warmup call: ~107s

  O warmup manual EM SI dura 111s — é a causa do cold-start, não a solução.
```

### 4.3 O Que o Warmup Manual RESOLVE e NÃO RESOLVE

**Resolve:**
- **Intra-sessão**: garante que a primeira call "real" cronometrada já
  encontra todo o código nativo em memória RAM
- **Multi-especialização**: garante que variantes de tipo (multi-TR, multi-dip,
  multi-freq) são pré-compiladas antes do benchmark cronometrado (Sprint v2.28)
- **Pool quente**: após o warmup, o `_PERSISTENT_POOL` fica ativo e todas
  as chamadas subsequentes têm latência de sub-segundo

**Não resolve:**
- Cold-start de 111s na primeira sessão do dia (ou primeiro uso após boot)
- Não reduz o tempo de espera que o usuário experimenta na primeira invocação

### 4.4 Oportunidades Não Implementadas

#### 4.4.1 Script `geosteering-warmup` (entry point pós-instalação)

```python
# pyproject.toml:
# [project.scripts]
# geosteering-warmup = "geosteering_ai.cli.warmup:main"

def main():
    """Pré-aquece todos os kernels JIT. Executar uma vez após pip install."""
    print("Pré-compilando kernels Numba JIT (uma vez por instalação)...")
    from geosteering_ai.simulation import simulate_multi
    from geosteering_ai.simulation.config import SimulationConfig
    import numpy as np

    cfg = SimulationConfig(n_workers=1, threads_per_worker=1)
    simulate_multi(
        rho_h=np.array([1., 100., 1.]),
        rho_v=np.array([1., 100., 1.]),
        esp=np.array([5.]),
        positions_z=np.linspace(-5, 5, 10),
        frequencies_hz=[2000., 20000., 100000., 400000.],  # todos os tipos de freq
        tr_spacings_m=[0.5, 1.0, 1.5, 2.0],               # todos os TR
        dip_degs=[0., 15., 30., 45.],                      # todos os dips
        cfg=cfg,
    )
    print("Compilação concluída. Próximos usos serão instantâneos (intra-sessão).")
```

**Benefício**: usuário não é surpreendido pelo cold-start de 111s na primeira
simulação. A compilação ocorre com mensagem explicativa no momento da instalação.

#### 4.4.2 Background Warmup Thread no Startup do CLI

```python
# Em geosteering_ai/cli/main.py — adicionar antes do parse de args:
import threading

def _background_warmup():
    """Inicia warmup JIT em background enquanto CLI processa args."""
    try:
        from geosteering_ai.simulation.multi_forward import simulate_multi
        # trigger mínimo — apenas carrega os módulos
    except Exception:
        pass

_warmup_thread = threading.Thread(target=_background_warmup, daemon=True)
_warmup_thread.start()
# A partir daqui, parse de argumentos ocorre normalmente
# Se o usuário demora > 2s para confirmar o comando, o import já foi feito
```

**Benefício limitado**: reduz ~2s do import time, não os 109s de LLVM loading.
Útil em shells interativos onde o usuário lê o help antes de executar.

### 4.5 Conclusão

O warmup manual está **adequadamente implementado** para uso contínuo
(pool persistente elimina overhead nas chamadas 2+). A oportunidade
de maior impacto é o script de pós-instalação `geosteering-warmup`
que transforma o cold-start de "surpresa de 111s" em "informação ao usuário".

---

## 5. Estratégia 4: Tipos Consistentes — Evitar Recompilação

### 5.1 Status: ⚠️ MAIORIA IMPLEMENTADA — Anomalia em `hmd_tiv` e `vmd`

#### 5.1.1 Garantias de Tipo já Implementadas

O wrapper `fields_in_freqs` em
[`_numba/kernel.py:244-251`](../../../geosteering_ai/simulation/_numba/kernel.py)
normaliza todos os arrays antes de passar ao kernel:

```python
rho_h    = np.ascontiguousarray(rho_h,    dtype=np.float64)
rho_v    = np.ascontiguousarray(rho_v,    dtype=np.float64)
esp      = np.ascontiguousarray(esp,      dtype=np.float64)
freqs_hz = np.ascontiguousarray(freqs_hz, dtype=np.float64)
krJ0J1   = np.ascontiguousarray(krJ0J1,   dtype=np.float64)
wJ0      = np.ascontiguousarray(wJ0,      dtype=np.float64)
wJ1      = np.ascontiguousarray(wJ1,      dtype=np.float64)
```

Garantia: nenhum `float32` ou array não-contíguo chega aos kernels pela
API pública. Evita recompilação por tipo float — a causa mais comum de
proliferação de especializações em projetos Numba.

#### 5.1.2 Anomalia: Duas Especializações de `hmd_tiv` e `vmd`

A inspeção dos artefatos de cache revela `.nbc` duplos:

```
dipoles.hmd_tiv-180.py313.1.nbc   1.337 KB  ← especialização 1
dipoles.hmd_tiv-180.py313.2.nbc   1.337 KB  ← especialização 2  (quase idêntico)
```

**Investigação**: `hmd_tiv.signatures` retorna 1 especialização em uma
sessão limpa — a duplicação no disco veio de sessões anteriores com
variação no flag de **mutabilidade** dos arrays de entrada:

```
Especialização 1: Array(float64, 1, 'C', mutable=True,  aligned=True)
                  → arrays writeable (ex: retorno de np.zeros/np.empty)

Especialização 2: Array(float64, 1, 'C', mutable=False, aligned=True)
                  → arrays readonly (ex: resultado de operações intermediárias)
```

O Numba trata arrays writeable e readonly como tipos distintos, gerando
binários LLVM separados mesmo quando a função não escreve no array.

**Custo do problema**:
- Cold-start: ~20-40 s extras por ter que carregar 2 × (`hmd_tiv` + `vmd`)
- Espaço em disco: ~4,1 MB de `.nbc` duplicados (desnecessariamente)

#### 5.1.3 Como Diagnosticar

```python
# Diagnóstico completo das especializações ativas
from geosteering_ai.simulation._numba.dipoles import hmd_tiv, vmd

# Chamar a função primeiro para poplar signatures
# (as signatures são armazenadas per-processo, não persistidas em disco)

hmd_tiv.inspect_types()    # lista todas as especializações compiladas
# Saída mostra: quais tipos de argumento geraram cada .nbc
```

#### 5.1.4 Como Resolver

**Opção A — Limpar cache e forçar 1 especialização** (imediato, sem código):

```bash
# Apagar .nbc existentes para hmd_tiv e vmd
rm geosteering_ai/simulation/_numba/__pycache__/dipoles.*-180.*.nbc
rm geosteering_ai/simulation/_numba/__pycache__/dipoles.*-708.*.nbc
# Na próxima execução, apenas 1 especialização será compilada
```

**Opção B — Assinatura explícita em `@njit`** (robusto):

```python
# Forçar tipo único via eager compilation (assinatura explícita)
from numba import complex128, float64, int64
from numba.typed import List

@njit(
    # Declaração explícita da assinatura (sem polimorfismo)
    (float64, float64, float64,  # Tx, Ty, Tz
     int64, int64, int64, int64,  # n, camad_r, camad_t, npt
     float64[:], float64[:], float64[:],  # krJ0J1, wJ0, wJ1
     ...),
    cache=True
)
def hmd_tiv(...):
    ...
```

**Opção C — `np.ascontiguousarray` antes de toda chamada interna**:
Garantir que todos os arrays passados para `hmd_tiv` sejam sempre
`mutable=True` ao normalizar antes da chamada em `_fields_in_freqs_kernel`.

**Recomendação**: Opção A é a mais simples e imediata.
Opção B é mais robusta mas requer manutenção manual da assinatura.

#### 5.1.5 Consistência de Tipos Inteiros

Todos os callers passam `n`, `camad_t`, `camad_r` como `int64` (Python
`int` em 64-bit = `int64`; `np.int64` = `int64`). Não há risco de
recompilação por `int32` vs `int64` no pipeline atual.

### 5.2 Conclusão

Tipos float64 estão consistentes em toda a cadeia. A anomalia das 2
especializações de `hmd_tiv`/`vmd` é corrigível com limpeza do cache
(Opção A) sem nenhuma mudança no código-fonte. Redução esperada
de cold-start: **−20-40 s** (de 111s → ~75s).

---

## 6. Estratégia 5: Pré-importar Bibliotecas

### 6.1 Status: ✅ IMPLEMENTADO — Lazy imports nos handlers CLI

A arquitetura de lazy import está implementada desde Sprint v2.24 I2.6:

```python
# geosteering_ai/cli/simulate.py:167-168
def handle_simulate(args):
    # Lazy imports — Sprint v2.24 I2.6 — evita carregar numba em --help
    from geosteering_ai.simulation import simulate_multi
    from geosteering_ai.simulation.config import SimulationConfig
```

Resultado: `geosteering-cli --help` retorna em **< 0,1 s** sem tocar o Numba.
O import do Numba ocorre apenas quando o subcomando é invocado.

### 6.2 Decomposição do Tempo de Import (1,864 s)

```
Import simulate_multi — cadeia de importação:
  ├── numpy (provavelmente já no sys.modules)          ~0,05 s
  ├── import numba (LLVM infrastructure)              ~0,50 s
  ├── import _numba.propagation (define njit wrapper)  ~0,20 s
  ├── import _numba.geometry   (define 3 @njit)        ~0,25 s
  │     Note: @njit com cache=True NÃO compila no import
  │     Apenas registra o decorador — compilação adiada para 1ª chamada
  ├── import _numba.kernel     (define 5 @njit)        ~0,30 s
  ├── import _numba.dipoles    (define 2 @njit)        ~0,25 s
  ├── import _numba.rotation   (define 2 @njit)        ~0,15 s
  ├── import forward           (define 4 @njit)        ~0,20 s
  └── import multi_forward     (orquestrador)          ~0,15 s

  TOTAL: ~1,864 s (medido)
```

### 6.3 Detalhe Crítico: Import ≠ Compilação

```
@njit(cache=True)
def hmd_tiv(...):
    ...
# No momento do import: Numba registra o decorator, mas NÃO lê o .nbc
# O .nbc é lido apenas na PRIMEIRA CHAMADA da função
# Isso significa que o import de 1,864s é "barato" — a compilação
# (1,007s no mesmo processo) é adiada para o primeiro uso real
```

Esta arquitetura de lazy compilation (já implementada pelo Numba por padrão)
é correta — import rápido, compilação adiada.

### 6.4 Oportunidade: `NUMBA_CACHE_DIR` em Ramdisk

O acesso ao `.nbc` do disco pode ser acelerado com o cache em filesystem
de alta velocidade:

```bash
# Linux (tmpfs em /dev/shm):
export NUMBA_CACHE_DIR=/dev/shm/geosteering_numba_cache

# macOS (ramdisk via hdiutil):
# diskutil erasevolume HFS+ "NumbaCache" $(hdiutil attach -nomount ram://2048000)
export NUMBA_CACHE_DIR=/Volumes/NumbaCache/numba

# Ou simplesmente /tmp (geralmente tmpfs no macOS):
export NUMBA_CACHE_DIR=/tmp/geosteering_numba_cache
```

**Benefício estimado**: leitura de `.nbc` de disco SSD → ramdisk:
10-30% de redução no cold-start (a maior parte do tempo é LLVM backend
CPU-bound, não I/O-bound). Impacto: −10-30s em HDD; −5-15s em NVMe.

**Implementação no projeto**: adicionar ao `main.py` do CLI:

```python
import os, tempfile
if "NUMBA_CACHE_DIR" not in os.environ:
    # Default: /tmp para aproveitar tmpfs quando disponível
    _default_cache = os.path.join(tempfile.gettempdir(), "geosteering_numba_cache")
    os.makedirs(_default_cache, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = _default_cache
```

### 6.5 Conclusão

Lazy import está corretamente implementado. A oportunidade não explorada
de maior impacto é `NUMBA_CACHE_DIR` em tmpfs/ramdisk, que pode
reduzir 10-30% do cold-start com 3 linhas de código.

---

## 7. Estratégias Adicionais Descobertas na Análise

Além das 5 estratégias propostas, a investigação revelou oportunidades não
mencionadas no escopo original:

### 7.1 `NUMBA_DISABLE_JIT=1` para Testes (já funcional)

O projeto já tem suporte dual-mode (`HAS_NUMBA` flag em `propagation.py`).
Em CI/CD ou testes unitários que não precisam de performance, é possível
desativar o JIT completamente:

```bash
NUMBA_DISABLE_JIT=1 pytest tests/ -v  # executa em NumPy puro, sem warmup
```

Não reduz o cold-start em produção, mas elimina-o completamente em ambientes
de desenvolvimento/CI.

### 7.2 Persistência do Pool entre Invocações CLI (Daemon Pattern)

O `_PERSISTENT_POOL` em `_workers.py` é singleton **intra-processo**, mas
não sobrevive entre invocações CLI separadas:

```
Cenário atual (sem daemon):
  geosteering-cli simulate ...  → cold-start 111s
  geosteering-cli simulate ...  → cold-start 111s  (novo processo!)
  geosteering-cli simulate ...  → cold-start 111s  (novo processo!)

Cenário com daemon (não implementado):
  geosteering-daemon start      → cold-start 111s (uma vez)
  geosteering-cli simulate ...  → socket IPC → worker quente → 0,001s
  geosteering-cli simulate ...  → socket IPC → worker quente → 0,001s
```

Implementação: servidor Unix socket (`asyncio.start_unix_server`) com
protocolo simples de serialização de modelos (pickle/msgpack). Complexidade:
alta. Benefício: transformar 111s em 0,001s para uso CLI intensivo.

### 7.3 Consolidar as Especializações Desnecessárias

Limpar `.nbc` duplos de `hmd_tiv` e `vmd` e monitorar com:

```bash
# Adicionar ao pre-commit hook ou CI:
# Conta especializações por função e alerta se > 1
python -c "
from geosteering_ai.simulation._numba.dipoles import hmd_tiv, vmd
# Trigger compilation first
import ... (warm up)
assert len(hmd_tiv.signatures) == 1, f'hmd_tiv tem {len(hmd_tiv.signatures)} specs'
assert len(vmd.signatures) == 1, f'vmd tem {len(vmd.signatures)} specs'
"
```

### 7.4 `numba.experimental.jitclass` para Encapsular Estado

Para o filtro Hankel (carregado uma vez por sessão), um `jitclass` permite
encapsular estado JIT-compilado. Menos aplicável ao modelo atual sem estado
global de kernels, mas pode ser útil se o filtro for otimizado no futuro.

---

## 8. Comparativo Final

### 8.1 Resumo das Estratégias

| # | Estratégia | Status Atual | Cold-start Atual | Cold-start Após Fix | Dificuldade |
|:-:|:-----------|:------------:|:----------------:|:-------------------:|:-----------:|
| 1 | `cache=True` | ✅ Completo | 111 s | **111 s** (limite) | N/A |
| 2 | AOT `cc.compile()` | ❌ Inviável | 111 s | ~55 s (parcial) | Alta ⚠️ |
| 3a | Warmup intra-sessão | ✅ Completo | N/A | 0,001 s (intra) | N/A |
| 3b | Script pós-instalação | ❌ Faltando | Surpresa usuário | Info ao usuário | Baixa ✓ |
| 3c | Background thread | ❌ Faltando | 111 s | 109 s (melhora import) | Baixa ✓ |
| 4 | Tipos consistentes | ⚠️ Anomalia | 111 s | **~75 s** (−36 s) | Baixíssima ✓ |
| 5a | Lazy imports CLI | ✅ Completo | –help < 0,1s | N/A | N/A |
| 5b | `NUMBA_CACHE_DIR` ramdisk | ❌ Faltando | 111 s | **~85-100 s** | Baixa ✓ |
| 7a | `NUMBA_DISABLE_JIT` CI | ✅ Funcional | N/A (sem JIT) | N/A | N/A |
| 7b | Daemon persistente | ❌ Não implementado | 111 s/invocação | 0,001 s/invocação | Alta ⚠️ |

### 8.2 Prioridade de Ações (Custo vs. Benefício)

```
┌─────┬──────────────────────────────────────────────────────────┬──────────┬─────────┐
│  #  │  Ação                                                    │ Impacto  │  Custo  │
├─────┼──────────────────────────────────────────────────────────┼──────────┼─────────┤
│  1  │  Limpar .nbc duplos de hmd_tiv/vmd (Estratégia 4A)       │ −20-40s  │ 1 linha │
│  2  │  NUMBA_CACHE_DIR=/tmp no main.py (Estratégia 5B)         │ −10-30s  │ 3 linhas│
│  3  │  Script geosteering-warmup (Estratégia 3B)               │ UX ++    │ ~50 LOC │
│  4  │  Assinatura explícita @njit em hmd_tiv/vmd (Estr. 4B)    │ +robusto │ ~10 LOC │
│  5  │  AOT parcial (Estratégia 2 híbrida)                      │ −40-50s  │ Alto ⚠️ │
│  6  │  Daemon persistente (Estratégia 7B)                      │ elimina  │ Muito ⚠️│
└─────┴──────────────────────────────────────────────────────────┴──────────┴─────────┘
```

### 8.3 Limite Fundamental

O cold-start de 111 s **não pode ser eliminado** dentro da arquitetura JIT
do Numba 0.61.2, pelas seguintes razões estruturais:

1. `cache=True` já está ativo — o ganho máximo possível já está extraído
2. AOT com `parallel=True` não é suportado — as funções mais críticas
   de performance são incompatíveis com compilação Ahead-of-Time
3. O LLVM backend é sempre necessário em cada novo processo Python para
   converter bitcode em código nativo — sem exceções na API atual do Numba

Em uso contínuo (GUI aberta, CLI em loop, daemon), o pool persistente
`_PERSISTENT_POOL` **já elimina** o overhead nas chamadas 2+. O problema
de 111s é exclusivamente o de **primeiro uso por sessão Python**.

---

## 9. Referências Técnicas

| Documento | Localização | Relevância |
|:----------|:-----------|:-----------|
| Numba User Guide — Caching | [numba.readthedocs.io/caching](https://numba.readthedocs.io) | cache=True internals |
| Numba AOT Compilation | [numba.readthedocs.io/pycc](https://numba.readthedocs.io) | Limitações AOT |
| Sprint v2.29 Back to Basics | `docs/reports/v2.29_2026-05-11.md` | Warmup inline history |
| Sprint v2.27 — Warmup 1 Worker | `docs/CHANGELOG.md` | Histórico 110s→35s |
| Sprint v2.28 — Warmup C+D | `docs/CHANGELOG.md` | Caminhos JIT cobertos |
| Sprint v2.21 — Fix nested prange | `geosteering_ai/simulation/_numba/kernel.py:700-722` | Parallel regression fix |
| Performance Baseline | `docs/PERFORMANCE_BASELINE.md` | Baseline de throughput |

---

*Relatório gerado por análise estática + benchmarks empíricos em 2026-05-12.*
*Nenhuma alteração de código foi realizada — documento de investigação pura.*
