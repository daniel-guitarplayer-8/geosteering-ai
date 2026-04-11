# Filtros Hankel Digitais — Geosteering AI v2.0

Este diretório contém os pesos e abscissas dos **filtros Hankel digitais**
utilizados pelo simulador Python otimizado (`geosteering_ai/simulation/`)
para calcular integrais de Hankel via quadratura digital.

## Contexto matemático

Um filtro Hankel digital aproxima integrais da forma:

```
F(r) = ∫₀^∞ f(kr) · Jν(kr · r) · dkr
```

onde `Jν` é a função de Bessel de primeira espécie de ordem `ν` (0 ou 1)
e `kr` é o número de onda radial. A aproximação por quadratura é:

```
F(r) ≈ (1/r) · Σᵢ₌₀^{N-1} f(aᵢ/r) · wᵢ(ν)
```

com `a = abscissas` e `w = weights_jν` pré-computados e armazenados
nestes arquivos `.npz`. Os 3 filtros aqui presentes são da família
"J₀-J₁ common-base", ou seja, ambos os conjuntos de pesos (J₀ e J₁)
compartilham as mesmas abscissas.

## Artefatos disponíveis

| Arquivo                   | npt | filter_type (Fortran) | Uso                                |
|:--------------------------|:---:|:---------------------:|:-----------------------------------|
| `werthmuller_201pt.npz`   | 201 |         **0 (★)**     | Default — boa precisão banda larga |
| `kong_61pt.npz`           |  61 |           1           | ~3.3× mais rápido, OK para DL      |
| `anderson_801pt.npz`      | 801 |           2           | Máxima precisão, referência        |

O código Fortran equivalente está em
`Fortran_Gerador/filtersv2.f08` nas subrotinas `J0J1Wer`, `J0J1Kong`
e `J0J1And`.

## Formato do arquivo `.npz`

Cada `.npz` contém 4 chaves:

```
abscissas   : np.ndarray(npt,) float64 — kr pontos de quadratura (kr > 0, crescente)
weights_j0  : np.ndarray(npt,) float64 — pesos para integral com J₀
weights_j1  : np.ndarray(npt,) float64 — pesos para integral com J₁
metadata    : np.ndarray(0-d)  str     — JSON serializado com 8 campos
```

Estrutura do JSON em `metadata`:

```json
{
  "filter_name":        "werthmuller_201pt",
  "npt":                201,
  "source_file":        "Fortran_Gerador/filtersv2.f08",
  "source_sha256":      "<hash do arquivo fonte>",
  "fortran_subroutine": "J0J1Wer",
  "extracted_at":       "2026-04-11T...Z",
  "extractor_script":   "scripts/extract_hankel_weights.py",
  "extractor_version":  "1.0.0"
}
```

O campo `source_sha256` permite auditoria de sincronia: se o Fortran for
atualizado sem re-rodar o extrator, o hash ficará desatualizado e o teste
`tests/test_simulation_filters.py` acusará a divergência.

## Como regenerar

Sempre que `Fortran_Gerador/filtersv2.f08` for modificado (novo filtro,
correção de pesos, etc.), execute:

```bash
# Extração completa (regera os 3 .npz)
python scripts/extract_hankel_weights.py

# Verificação de sincronia (sem reescrever)
python scripts/extract_hankel_weights.py --verify

# Debug com logging detalhado
python scripts/extract_hankel_weights.py --verbose
```

Em CI, o `--verify` pode ser usado como gate antes de testes que dependam
dos filtros.

## Carregamento em código Python

```python
from geosteering_ai.simulation.filters import FilterLoader

loader = FilterLoader()

# Por nome canônico
filt = loader.load("werthmuller_201pt")

# Por alias
filt = loader.load("wer")

# Por filter_type Fortran
filt = loader.load("0")

print(filt.abscissas.shape)   # (201,)
print(filt.weights_j0.shape)  # (201,)
print(filt.weights_j1.shape)  # (201,)
print(filt.fortran_filter_type)  # 0
```

Instâncias de `HankelFilter` são **imutáveis** (frozen dataclass, arrays
read-only), portanto podem ser cacheadas e compartilhadas entre threads
sem riscos de race condition.

## Histórico de versões

| Versão | Data       | Descrição                                         |
|:-------|:-----------|:--------------------------------------------------|
| 1.0.0  | 2026-04-11 | Sprint 1.1 — extração inicial dos 3 filtros       |

## Referências bibliográficas

1. **Werthmüller, D.** (2017). "An open-source full 3D electromagnetic
   modeler for 1D VTI media in Python: empymod." *Geophysics*, 82(6),
   WB9-WB19. https://doi.org/10.1190/geo2016-0626.1
2. **Kong, F. N.** (2007). "Hankel transform filters for dipole antenna
   radiation in a conductive medium." *Geophysical Prospecting*, 55(1),
   83-89.
3. **Anderson, W. L.** (1989). "A hybrid fast Hankel transform algorithm
   for electromagnetic modeling." *Geophysics*, 54(2), 263-266.

## Referências internas

- Plano detalhado do simulador Python:
  [`docs/reference/plano_simulador_python_jax_numba.md`](../../../docs/reference/plano_simulador_python_jax_numba.md)
- Documentação do simulador Fortran:
  [`docs/reference/documentacao_simulador_fortran.md`](../../../docs/reference/documentacao_simulador_fortran.md)
- Sub skill Python:
  `.claude/commands/geosteering-simulator-python.md`
- Sub skill Fortran:
  `.claude/commands/geosteering-simulator-fortran.md`
