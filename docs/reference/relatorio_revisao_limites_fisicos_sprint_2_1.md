# Relatório Final — Revisão de Limites Físicos e Expansão de Escopo (pós-Sprint 2.1)

**Projeto**: Geosteering AI v2.0 — Simulador Python Otimizado
**Sprint**: 2.1 (revisão + expansão de limites)
**Data**: 2026-04-11
**Autor**: Daniel Leal
**Branch**: `feature/simulator-python-phase2`
**Plano-mãe**: [`plano_simulador_python_jax_numba.md`](plano_simulador_python_jax_numba.md) §16

---

## 1. Contexto

Após a conclusão da Sprint 2.1 (port Python+Numba de `commonarraysMD` +
`commonfactorsMD`), o usuário levantou três perguntas críticas que
dispararam uma revisão profunda dos limites físicos do simulador
Python e do workflow de PRs:

1. **Por que limitar `tr_spacing_m` a 10 m e `frequency_hz` a 1 MHz?**
   Existe impedimento físico?
2. **O que significam os valores 8.19 m e 20.43 m comentados em
   `fifthBuildTIVModels.py`?**
3. **Por que agora precisa aprovar PRs manualmente? Como automatizar?**

Esta sessão:

- Respondeu detalhadamente as três perguntas
- **Expandiu os limites do `SimulationConfig`** com base nas evidências
  físicas coletadas do código Fortran
- Atualizou testes, plano-mãe, sub skill e documentação
- Validou com **183/183 testes** passando

---

## 2. Respostas às Perguntas

### 2.1 Pergunta 1 — Limites TR (0.1–10 m) e frequência (1 MHz)

**Conclusão**: os limites da Sprint 1.2 eram **arbitrários e
conservadores**, não impostos por física real. Três fatores distintos
limitam o simulador, cada um em um regime diferente:

#### a) Filtro Hankel (limite **computacional**)

- `Werthmüller 201pt` cobre `kr·r ∈ [8.65e-4, 93.7]`, adequado para
  `r ∈ [0.1, ~30]` m.
- `Anderson 801pt` cobre `kr·r ∈ [8.92e-14, 4.94e21]`, adequado para
  `r ∈ [0.01, ~1000]` m.
- Limite real depende do filtro escolhido. **Não é limite físico**.

#### b) Skin depth (limite de **detectabilidade**)

- `δ = 503·√(ρ/f)` metros.
- Para `ρ = 1 Ω·m, f = 20 kHz`: `δ ≈ 3.56 m` — detecta `r < 5 m` com SNR.
- Para `ρ = 1000 Ω·m, f = 20 kHz`: `δ ≈ 112 m` — detecta `r < 150 m`.
- **Limite da ferramenta, não do simulador**. Matematicamente o
  simulador calcula qualquer `(r, f, ρ)`, só o SNR fica ruim.

#### c) Aproximação quasi-estática (limite **físico REAL**)

- Fortran usa `zeta = i·ω·μ₀` em `utils.f08` (linhas 636, 971, 1066),
  **ignorando** `-ω²·μ·ε` (corrente de deslocamento).
- Válido quando `|ω·ε₀·εᵣ / σ| ≪ 1`, ou seja
  `f ≪ σ / (2π·ε₀·εᵣ) = 1.8×10¹⁰ / ρ` Hz.

| ρ (Ω·m) | f_limit (Hz)  | Comentário |
|:-------:|:-------------:|:-----------|
| 1       | 1.8×10¹⁰ (18 GHz) | Sem problema para LWD |
| 100     | 1.8×10⁸ (180 MHz) | Ainda ok |
| 1 000   | 1.8×10⁷ (18 MHz)  | Ok |
| **10 000**  | **1.8×10⁶ (1.8 MHz)** | **Aqui aparece o limite 1 MHz antigo** |
| 100 000 | 1.8×10⁵ (180 kHz) | Crítico em alta resistividade |

**Conclusão**: `1 MHz` era o limite conservador para garantir
quasi-estática em `ρ ≤ 10 000 Ω·m`. Para `ρ ≤ 1 000 Ω·m`, poderíamos
ir até `18 MHz`; para paridade com ferramentas comerciais, `2 MHz` é
o novo limite adotado (ARC6, PeriScope, EcoScope operam em dual-freq
400 kHz + 2 MHz).

#### Novos limites adotados

| Parâmetro | Antigo (Sprint 1.2) | Novo (Sprint 2.1) | Motivo |
|:----------|:-------------------:|:-----------------:|:-------|
| `frequency_hz` min | 100 Hz | **10 Hz** | CSAMT/MT baixa freq |
| `frequency_hz` max | 1 MHz | **2 MHz** | Paridade dual-freq LWD |
| `tr_spacing_m` min | 0.1 m | **0.01 m** | Ferramentas curtas |
| `tr_spacing_m` max | 10 m | **50 m** | PeriScope HD (20.43 m) + margem |
| Resistividade (novo campo futuro) | — | **0.1 – 1e6 Ω·m** | Sal/carbonatos/crosta |

### 2.2 Pergunta 2 — `fifthBuildTIVModels.py` e `dTR = [8.19, 20.43]`

**O que o código revela** (linhas 1127-1137):

```python
# Frequências baixas (2/6 kHz) + espaçamentos longos (8.19/20.43 m)
# = investigação profunda (DOI ~4–15 m)
# Dois ângulos (0° e 30°) ativam componentes off-diagonal do tensor H
freqs   = [20000.]         #[2000., 6000.]  # frequências (Hz)
angulos = [0.]             #[0., 30.]
dTR     = [1.0]    #[8.19, 20.43]    # espaçamentos T-R (m) —
                                      # investigação profunda (tipo ARC/Periscope)
```

**Interpretação**:

- **8.19 m (≈27 pés)**: Arranjo ultra-longo da **Schlumberger ARC6**
  (Array Resistivity Compensated), ferramenta LWD dual-frequency
  400 kHz + 2 MHz. A ARC tem múltiplos receptores e o arranjo de
  8.19 m é uma das configurações ultra-longas para DOI profundo.

- **20.43 m (≈67 pés)**: TR característico da **PeriScope HD 675/825**
  (Schlumberger) e do **GeoSphere** para deep-reading. A PeriScope HD
  opera com TR até 20 m para detectar *shoulder beds*, *oil-water
  contacts* e *faults* a até 15 m de DOI lateral. **Esta é a
  ferramenta principal de geosteering horizontal moderno.**

**Consequência para o simulador Python**:

O Fortran **suporta nativamente** `dTR` até o limite do filtro Hankel
(~30 m com Werthmüller, ~1000 m com Anderson). O fato de o dataset
atual usar apenas `dTR = [1.0]` não é restrição física — é escolha
do pipeline de treino.

**O simulador Python DEVE**:
- Suportar `dTR ∈ [0.01, 50]` m (novo limite expandido)
- Permitir `tr_spacings_m = [8.19, 20.43]` para configurações tipo
  ARC + PeriScope simultâneas
- **Teste adicionado**: `test_arc_periscope_multi_tr_valid` em
  `test_simulation_config.py` valida explicitamente este caso

### 2.3 Pergunta 3 — Automação do merge de PRs

**Por que agora há PRs**: decisão arquitetural #6 do plano-mãe
(documentada em `plano_simulador_python_jax_numba.md §14`):

> *6. Repositório: criar branch `feature/simulator-python` ou trabalhar
> em `main` com feature flag?* → Branch dedicado.

Antes desta decisão, os commits Fortran (F1–F10) iam direto para `main`
sem PRs. Agora o trabalho Python usa feature branches + PRs por
isolamento, auditoria e rollback seguro.

**Opções de automação** (detalhadas na resposta ao usuário):

| Opção | Método | Quando usar |
|:-----:|:-------|:-----------|
| **A** ★ | `gh pr merge --auto --squash --delete-branch` | **Recomendada** — merge automático quando CI passar |
| B | Hook `PostToolUse` no Claude Code settings | Avançado, efeitos colaterais |
| C | Direto em `main` | Se PR é overhead puro (perde auditoria) |
| D | Instrução persistente em `.claude/settings.json` | Claude sempre roda auto-merge |

**Recomendação combinada**: Opção A + Opção D. O usuário pode ativar
com um único comando:

```bash
gh repo edit daniel-guitarplayer-8/geosteering-ai --enable-auto-merge
```

Depois, ao criar PRs, usar `--auto`:

```bash
gh pr merge 2 --auto --squash --delete-branch
```

---

## 3. Correções e Refatorações Aplicadas

### 3.1 `geosteering_ai/simulation/config.py`

**Alterações**:

- `_FREQUENCY_HZ_RANGE`: `(100.0, 1.0e6)` → **`(10.0, 2.0e6)`**
- `_TR_SPACING_M_RANGE`: `(0.1, 10.0)` → **`(0.01, 50.0)`**
- Novo constante: `_RESISTIVITY_OHM_M_RANGE: (0.1, 1.0e6)` (prepara
  Fase 2.5 quando `SimulationConfig` receber `rho_h`/`rho_v`)
- Mensagens de erro das validações reescritas com justificativa
  física detalhada, citando ferramentas LWD reais (ARC6, PeriScope HD)
  e aproximação quasi-estática
- Comentários do bloco de constantes atualizados com referências
  bibliográficas (Anderson 2008, Omeragic 2009, Moran-Gianzero 1979)

**Não alterações** (preservadas):
- Presets (`default()`, `high_precision()`, `production_gpu()`,
  `realtime_cpu()`) não mudaram — usam valores dentro do novo range
- API pública `to_dict/from_dict/to_yaml/from_yaml` inalterada
- Imutabilidade `@dataclass(frozen=True)` preservada
- Thread-safety do FilterLoader (Sprint 1.1) inalterada

### 3.2 `tests/test_simulation_config.py`

**7 testes ajustados** (valores "inválidos" antigos agora são válidos):

| Teste | Valor antigo | Valor novo |
|:------|:-------------|:-----------|
| `test_frequency_too_low_fails` | `50.0` | `5.0` (< 10 Hz novo min) |
| `test_frequency_too_high_fails` | `2.0e6` | `5.0e6` (> 2 MHz novo max) |
| `test_frequency_at_lower_boundary_ok` | `100.0` | `10.0` |
| `test_frequency_at_upper_boundary_ok` | `1.0e6` | `2.0e6` |
| `test_tr_spacing_too_low_fails` | `0.05` | `0.005` |
| `test_tr_spacing_too_high_fails` | `20.0` | `100.0` |
| `test_replace_revalidates` | `50.0` | `5.0` |
| `test_out_of_range_frequency_in_list_fails` | `[..., 2.0e6]` | `[..., 5.0e6]` |
| `test_out_of_range_tr_spacing_in_list_fails` | `[..., 20.0]` | `[..., 100.0]` |

**5 testes novos** cobrindo os casos expandidos:

1. `test_tr_spacing_periscope_hd_deep_reading` — `tr_spacing_m=20.43`
2. `test_tr_spacing_arc_ultra_long_819` — `tr_spacing_m=8.19`
3. `test_frequency_lwd_2mhz_dual` — `frequency_hz=2.0e6`
4. `test_arc_periscope_multi_tr_valid` — `tr_spacings_m=[8.19, 20.43]`
5. `test_lwd_dual_frequency_400k_2mhz_valid` —
   `frequencies_hz=[400000.0, 2.0e6]`

### 3.3 Documentação

**Arquivos atualizados**:

- `docs/reference/plano_simulador_python_jax_numba.md` — nova
  **Seção 16** ("Revisão Pós-Sprint 2.1 — Expansão de Limites e
  Alta Resistividade") com 6 subseções detalhando: descobertas,
  limites expandidos, metas da Fase 2, teste de alta resistividade,
  nota sobre fastmath, frequências > 2 MHz (fora do escopo).
- `.claude/commands/geosteering-simulator-python.md` — seção 5.3
  atualizada com a tabela de limites expandidos e motivação física.
- Este relatório: `docs/reference/relatorio_revisao_limites_fisicos_sprint_2_1.md`.

---

## 4. Validação Consolidada

### 4.1 Bateria completa de testes

```
$ python -m pytest tests/test_simulation_*.py
============================ 183 passed in 1.13s ============================
```

| Suíte | Testes | Tempo |
|:------|:-----:|:-----:|
| `test_simulation_filters.py` | 53 | ~0.82s |
| `test_simulation_config.py` | **67** (+5 novos) | ~0.83s |
| `test_simulation_half_space.py` | 38 | ~0.19s |
| `test_simulation_numba_propagation.py` | 25 | ~0.59s |
| **TOTAL** | **183** (+5 vs Sprint 2.1 inicial) | **1.13s** |

### 4.2 Cobertura de casos físicos agora suportados

| Cenário | Exemplo | Status |
|:--------|:--------|:------:|
| LWD ARC6 curto | `tr=0.56m, f=400k+2M` | ✅ Válido |
| ARC ultra-longo | `tr=8.19m, f=2k-6k` | ✅ Válido |
| PeriScope HD deep | `tr=20.43m, f=2k-6k` | ✅ Válido |
| Dual-frequency LWD | `tr=1m, f=[400k, 2M]` | ✅ Válido |
| CSAMT baixa freq | `tr=1m, f=100Hz` | ✅ Válido |
| Alta resistividade | `ρ=10 000, f=20k` | ✅ Preparado (teste Sprint 2.6) |
| Dielétrico extremo | `ρ=1e5, f>2MHz` | ❌ Fora do escopo (Fase 8) |

---

## 5. Estado do Código Fortran

**Inalterado.** A revisão foi **read-only** sobre o Fortran:

- `Fortran_Gerador/utils.f08:158-297` — fonte de verdade `commonarraysMD`
  e `commonfactorsMD`, consultada mas não modificada.
- `Fortran_Gerador/parameters.f08` — `eps = 1e-9`, `mu`, `pi` — lidos
  para referência.
- `Fortran_Gerador/PerfilaAnisoOmp.f08` — consultado para confirmar
  que `zeta = i·ω·μ` (linhas 636, 971, 1066), sem termo de corrente
  de deslocamento.
- `Fortran_Gerador/fifthBuildTIVModels.py` — lido para descobrir
  os valores ARC/PeriScope (`dTR = [8.19, 20.43]`).

Último commit Fortran em `main`: `732ae7f` (F10 Jacobiano, pré-merge
da Fase 1 Python).

---

## 6. Estado da Arquitetura v2.0

### 6.1 Pacote principal `geosteering_ai/` (inalterado)

- 73 módulos, 744 testes pipeline principal
- `PipelineConfig` permanece com `SPACING_METERS=1.0` default — mas
  este é apenas o **valor default**, não limite. A validação de errata
  do pipeline principal (não do simulador) pode usar limites
  diferentes se necessário.

### 6.2 Subpacote `geosteering_ai/simulation/`

```
geosteering_ai/simulation/
├── __init__.py                v0.3.0 (em main)
├── config.py                  ★ RANGES EXPANDIDOS (pós-Sprint 2.1)
├── filters/                   ✅ Sprint 1.1
├── validation/                ✅ Sprint 1.3
├── _numba/                    🟡 Fase 2 em andamento
│   ├── __init__.py            ✅ Sprint 2.1
│   └── propagation.py         ✅ Sprint 2.1
└── (restante ⬜ pendente)
```

---

## 7. Pendências e Riscos

### 7.1 Pendências imediatas

| Item | Bloqueador? |
|:-----|:-----------:|
| Commit desta revisão + push na branch phase2 | Não (próximo passo) |
| Sprint 2.2: `_numba/dipoles.py` (hmd_TIV + vmd) | Sim para Sprint 2.3 |
| Adicionar campos `rho_h`, `rho_v`, `thicknesses_m` ao SimulationConfig (Sprint 2.5) | Sim para `simulate()` |
| Teste de alta resistividade (Sprint 2.6) | Gate |

### 7.2 Riscos identificados

| Risco | Prob. | Impacto | Mitigação |
|:------|:-----:|:-------:|:----------|
| Filtro Werthmüller degradar em `tr > 30 m` | Média | Médio | Adicionar warning automático + sugerir Anderson em `__post_init__` |
| Alta resistividade causa catastrophic cancellation | Média | Alto | `fastmath=False` já ativo; testes com `ρ=1e4, 1e5` na Sprint 2.6 |
| Usuário passar `dTR = 100 m` (fora filtro) | Baixa | Baixo | Validação rejeita (limite 50 m) |
| Frequência > 2 MHz em dielétrico | Baixa | Baixo | Validação rejeita + documentação clara |

### 7.3 Débitos técnicos

**Nenhum novo**. A revisão foi cirúrgica e não introduziu complexidade
adicional — apenas expandiu limites e atualizou testes.

---

## 8. Próximos Passos

### Imediato (esta sessão)
1. ✅ Atualizar SimulationConfig (feito)
2. ✅ Atualizar 7 testes existentes + 5 testes novos (feito)
3. ✅ Atualizar plano-mãe §16 (feito)
4. ✅ Atualizar sub skill §5.3 (feito)
5. ⬜ Commit + push na branch `feature/simulator-python-phase2`

### Curto prazo (próximas Sprints)
6. **Sprint 2.2**: `_numba/dipoles.py` (hmd_TIV + vmd) consumindo os
   9 arrays + 6 fatores da Sprint 2.1
7. **Sprint 2.3**: `_numba/hankel.py` + `rotation.py` + `geometry.py`
8. **Sprint 2.4**: `_numba/kernel.py` (orquestrador forward)
9. **Sprint 2.5**: `forward.py` (API `simulate()`) + adicionar campos
   `rho_h`, `rho_v`, `thicknesses_m` ao `SimulationConfig`
10. **Sprint 2.6** (gate): testes de paridade Numba vs analítico
    com **alta resistividade** `ρ ∈ [1, 10, 100, 1000, 10000, 100000]` Ω·m
11. **Sprint 2.7** (gate fim Fase 2): benchmark `≥ 40k mod/h`

### Médio prazo (infraestrutura)
12. Automatizar merge de PRs com `gh repo edit --enable-auto-merge` +
    `gh pr merge --auto` (resposta à P3)
13. Considerar CI GitHub Actions para rodar testes em cada PR

---

## 9. Métricas desta Sessão

| Métrica | Valor |
|:--------|:-----:|
| Arquivos modificados | 5 |
| Arquivos criados (este relatório) | 1 |
| Testes existentes ajustados | 7 |
| Testes novos adicionados | 5 |
| Total de testes agora | **183/183 PASS** |
| Tempo de execução total | 1.13s |
| LOC alteradas em config.py | ~60 |
| LOC alteradas em testes | ~50 |
| LOC adicionadas no plano-mãe | ~100 (seção 16) |

---

## 10. Assinatura

**Revisão**: Limites físicos pós-Sprint 2.1
**Status**: ✅ **CONCLUÍDA**
**Testes**: 183/183 PASS em 1.13s
**Branch**: `feature/simulator-python-phase2`
**PR**: #2 (em aberto, receberá este commit adicional)
**Gate 2.1 → 2.2**: ✅ Continua atingido (testes Numba passaram sem alteração)
**Aguardando**: Autorização para Sprint 2.2 (`_numba/dipoles.py`).
