# Relatório — Fase 5 (Single-Level Parallel) + PR Débitos B1/B3/B5/B7

**Data:** 2026-04-05
**Commits anteriores:** `195b69f` (validação final), `44acf2e` (Fase 4)

---

## 1. Resumo Executivo

Esta rodada combina **duas intervenções independentes** no simulador `PerfilaAnisoOmp`:

1. **PR Débitos B1/B3/B5/B7** — quatro correções cirúrgicas de OpenMP hygiene secundária, sem impacto em performance, preservando fidelidade numérica bit-a-bit em `-O0`.
2. **Fase 5 — Eliminação do nested parallelism** — reestrutura o loop paralelo de serial+nested para serial+single-level, eliminando overhead de fork/join aninhado para `ntheta=1` (caso de produção).

Adicionalmente, a **Fase 6** (cache de `commonfactorsMD` por `camadT`) foi **revisada e suspensa**: análise do código revelou que a proposta original contém um **erro conceitual** — `commonfactorsMD` depende de `h0` (profundidade do transmissor, variável em `j`), não apenas de `camadT`. O cache por sentinela proposto na documentação §7.6 copiaria resultados incorretos entre medidas com mesmo `camadT` mas diferentes `h0`, comprometendo a fidelidade física.

---

## 2. PR Débitos B1/B3/B5/B7

### B1 — Cópia redundante eliminada

**Antes:** `fieldsinfreqs_cached_ws` copiava `krJ0J1 = krwJ0J1(:,1)`, `wJ0 = krwJ0J1(:,2)`, `wJ1 = krwJ0J1(:,3)` em variáveis locais, a cada chamada (3 × 201 doubles = 4,8 KB/chamada × 600 chamadas = 2,9 MB/modelo).

**Depois:** Slices `krwJ0J1(:,1)`, `krwJ0J1(:,2)`, `krwJ0J1(:,3)` passadas diretamente como argumentos a `hmd_TIV_optimized_ws` e `vmd_optimized_ws`. Eliminação de 3 variáveis locais.

### B3/D7 — `firstprivate(zrho, cH)` no inner parallel

**Antes:** `zrho` e `cH` declarados `allocatable` no master e listados como `private` no inner `!$omp parallel do` — OpenMP spec define status de alocação indefinido para cópias private de allocatables.

**Depois:** Migrados para `firstprivate(zrho, cH)`, garantindo herança de alocação e valores do master. Mesmo padrão aplicado a `z_rho1/c_H1` no débito D4 anterior.

### B5 — `krwJ0J1` memory leak corrigido

**Antes:** `krwJ0J1(npt, 3)` alocado na linha ~99 sem `deallocate` correspondente. Leak de ~4,8 KB/modelo.

**Depois:** `if (allocated(krwJ0J1)) deallocate(krwJ0J1)` adicionado após o loop paralelo.

### B7 — `contiguous` attribute em `hmd/vmd_optimized_ws`

**Antes:** Dummy arguments `krJ0J1(npt)`, `wJ0(npt)`, `wJ1(npt)`, `Mxdw(npt)`, etc. declarados com tamanho fixo — correto mas não garante ao compilador que as slices recebidas são contíguas.

**Depois:** Migrados para `krJ0J1(:)`, `wJ0(:)`, `wJ1(:)` com atributo `contiguous`. Em Fortran 2008, isso garante que o compilador não gere cópia temporária para slices column-major (que já são contíguas por construção).

### Validação dos Débitos (isolada, antes da Fase 5)

| Teste | Flags | MD5 | Status |
|:------|:------|:---:|:------:|
| T=1 `-O3 -ffast-math` | Produção | `0fc79430142c67f1f2eb16904e87f7e4` | ✅ Idêntico à Fase 4 |
| T=8 `-O3 -ffast-math` | Produção | `0fc79430142c67f1f2eb16904e87f7e4` | ✅ Idêntico à Fase 4 |
| T=1 `-O0 -fno-fast-math` | Validação | `97123697a2e4db34c77cd1d84077b083` | ✅ Bit-exato vs Fase 4 @ -O0 |

---

## 3. Fase 5 — Single-Level Parallel

### Transformação aplicada

**Antes (Fases 2-4):** Nested parallelism com 2 níveis OpenMP:
```fortran
!$omp parallel do schedule(dynamic) num_threads(num_threads_k)  ! outer (ângulos)
  do k = 1, ntheta
    !$omp parallel do schedule(static) num_threads(num_threads_j)  ! inner (medidas)
      do j = 1, nmed(k)
```

**Depois (Fase 5):** Single-level parallel sem nested:
```fortran
do k = 1, ntheta           ! serial (1 iteração para ntheta=1)
  !$omp parallel do schedule(static) num_threads(maxthreads)
    do j = 1, nmed(k)
```

### Mudanças específicas

1. Removido `!$omp parallel do` externo e `!$omp end parallel do` correspondente.
2. Inner parallel usa `maxthreads` diretamente (em vez de `num_threads_j`).
3. `tid = omp_get_thread_num()` direto (sem `omp_get_ancestor_thread_num`).
4. `z_rho1` e `c_H1` são variáveis do master (sem `firstprivate` no outer).

### Validação numérica da Fase 5

| Teste | Flags | Referência | max\|Δ\| | Status |
|:------|:------|:-----------|:--------:|:------:|
| Fase 5 vs Fase 4 @ -O0 | `-O0 -fno-fast-math` | `97123697...` | **0** | ✅ BIT-EXATO |
| Fase 5 vs referência O0 @ -O3 | `-O3 -ffast-math` | — | 4,26e-13 | ✅ Sub-ULP |
| Determinismo T=1,2,4,8 | `-O3 -ffast-math` | — | — | ✅ MD5 idêntico: `3d3c309f...` |

### Benchmark

| Config | Fase 4 (s/modelo) | Fase 5 (s/modelo) | Δ | Throughput |
|:------:|:-----------------:|:-----------------:|:-:|:----------:|
| 1 thread | 0,2393 | 0,2583 | +7,9 % | 13.936 mod/h |
| 8 threads | 0,0618 | 0,0693 | +12,1 % | 51.923 mod/h |

**Análise:** O gfortran 15.2 já otimizava internamente o nested parallelism para `ntheta=1` (um único outer thread, full inner parallel). A Fase 5 não melhora a performance neste caso — o ganho é **código mais limpo e manutenível**, com preparação para multi-ângulo futuro.

---

## 4. Fase 6 — Erro Conceitual Descoberto

### Proposta original (§7.6 da análise de paralelismo)

A documentação propunha um cache por sentinela: quando medidas consecutivas têm o mesmo `camadT`, copiar o resultado anterior de `commonfactorsMD` em vez de recalcular.

### Erro descoberto

A sub-rotina `commonfactorsMD` calcula 6 arrays de saída (Mxdw, Mxup, Eudw, Euup, FEdwz, FEupz) que dependem de:
- **`camadT`** — camada onde está o transmissor (varia lentamente em `j`)
- **`h0`** (= `Tz`) — profundidade exata do transmissor (varia a cada `j`)

Os termos exponenciais usam `h0` diretamente:
```fortran
Mxdw = (exp(-s(:,cT) * (prof(cT) - h0)) + ...) / den
```

Portanto, **mesmo que `camadT` seja igual entre medidas consecutivas, os resultados são diferentes** porque `h0` muda. A cópia proposta no sentinela (`Mxdw_all(:,j) = Mxdw_all(:,j-1)`) é **matematicamente incorreta**.

### Implementação correta (futura)

A Fase 6 correta requer **fatoração dos termos invariantes em `h0`**:
- Os denominadores `den_TM` e `den_TE` dependem apenas de `camadT` (cacheable).
- Os coeficientes multiplicativos de `exp(±s*h0)` e `exp(±u*h0)` dependem apenas de `camadT` (cacheable).
- Os termos em `exp(±s*h0)` e `exp(±u*h0)` dependem de `h0` (devem ser recalculados).

Esta fatoração reduz o custo de `commonfactorsMD` de ~12 `exp()` para ~4 `exp()` por chamada (pré-calculando os 8 coeficientes invariantes). Planejada como **Fase 6b**.

---

## 5. Débitos Residuais

| ID | Descrição | Status |
|:--:|:----------|:------:|
| B2 | `eta` hoisted para `eta_shared` | ✅ Resolvido Fase 4 |
| B4 | Semântica `deallocate(zrho, cH)` após região paralela | 📋 Cosmético |
| B6 | Stride inconveniente em `zrho1(ntheta, nmmax, nf, 3)` | 📋 Otimização L1 |

---

## 6. Arquivos Modificados

| Arquivo | Tipo | Mudanças |
|:--------|:----:|:---------|
| `Fortran_Gerador/PerfilaAnisoOmp.f08` | modificado | B1 (cópia eliminada), B3/D7 (firstprivate inner), B5 (deallocate krwJ0J1), Fase 5 (single-level parallel, tid direto) |
| `Fortran_Gerador/magneticdipoles.f08` | modificado | B7 (contiguous em hmd/vmd_optimized_ws) |
| `docs/ROADMAP.md` | atualizado | Fase 5 ✅, PR Débitos ✅, Fase 6 ⚠️ revisada, débitos B1-B7 status |
| `docs/reference/relatorio_fase5_debitos_fortran.md` | **novo** | Este relatório |
| `docs/reference/analise_paralelismo_cpu_fortran.md` | atualizado | §7.7 resultados empíricos, nota sobre erro conceitual da Fase 6 |
| `docs/reference/documentacao_simulador_fortran.md` | atualizado | Nota sobre single-level parallel e débitos corrigidos |

---

## 7. Próximos Passos

1. **Fase 6b** — Fatoração dos termos invariantes em `h0` de `commonfactorsMD` (semi-cache). Complexidade alta mas ganho estimado de 30-50% no custo de `commonfactorsMD`.
2. **Fase 5b** — Restaurar nested ou `collapse(2)` para `ntheta>1` (multi-ângulo eficiente).
3. **Fase 3b** — Automatic arrays de `fieldsinfreqs` para `thread_workspace` estendido.
4. **Integração Python/DL** — conectar `.dat` validados ao pipeline `geosteering_ai/`.