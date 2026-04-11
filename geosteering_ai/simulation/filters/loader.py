# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/filters/loader.py                              ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Filtros Hankel Digitais                 ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-11 (Sprint 1.1)                                   ║
# ║  Status      : Produção                                                  ║
# ║  Framework   : NumPy 2.x                                                  ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Carrega, cacheia e valida os pesos e abscissas dos filtros Hankel     ║
# ║    digitais a partir dos artefatos .npz gerados pelo script              ║
# ║    `scripts/extract_hankel_weights.py` (que, por sua vez, parseia o     ║
# ║    arquivo Fortran `Fortran_Gerador/filtersv2.f08`).                    ║
# ║                                                                           ║
# ║  CADEIA DE CONFIANÇA (chain-of-trust)                                    ║
# ║    ┌─────────────────────────────────────────────────────────────────┐  ║
# ║    │  Fortran_Gerador/filtersv2.f08  (fonte primária, 5559 linhas)  │  ║
# ║    │          │                                                     │  ║
# ║    │          │  scripts/extract_hankel_weights.py                  │  ║
# ║    │          │  (parser regex + validação + SHA-256 auditável)     │  ║
# ║    │          ▼                                                     │  ║
# ║    │  geosteering_ai/simulation/filters/*.npz                       │  ║
# ║    │          │                                                     │  ║
# ║    │          │  FilterLoader.load(filter_name)                     │  ║
# ║    │          ▼                                                     │  ║
# ║    │  HankelFilter (dataclass imutável, arrays float64)             │  ║
# ║    │          │                                                     │  ║
# ║    │          │  consumo: simulation/_numba/hankel.py              │  ║
# ║    │          │           simulation/_jax/hankel.py                │  ║
# ║    │          ▼                                                     │  ║
# ║    │  ∫ f(kr)·Jν(kr·r) dkr ≈ (1/r)·Σᵢ f(aᵢ/r)·wᵢ(ν)              │  ║
# ║    └─────────────────────────────────────────────────────────────────┘  ║
# ║                                                                           ║
# ║  FILTROS SUPORTADOS (equivalência com Fortran filter_type)              ║
# ║    ┌──────────────────┬──────┬─────────────┬──────────────────────────┐  ║
# ║    │  Nome canônico   │  npt │ filter_type │ Características          │  ║
# ║    ├──────────────────┼──────┼─────────────┼──────────────────────────┤  ║
# ║    │ werthmuller_201  │ 201  │      0 (★)  │ Default Fortran, boa     │  ║
# ║    │                  │      │             │ precisão ampla banda     │  ║
# ║    │ kong_61          │  61  │      1      │ 3.3× mais rápido, boa    │  ║
# ║    │                  │      │             │ precisão para DL         │  ║
# ║    │ anderson_801     │ 801  │      2      │ Máxima precisão, lento   │  ║
# ║    └──────────────────┴──────┴─────────────┴──────────────────────────┘  ║
# ║                                                                           ║
# ║  DESIGN: dataclass imutável                                              ║
# ║    `HankelFilter` é um `@dataclass(frozen=True)` para prevenir mutação  ║
# ║    acidental após carregamento (os arrays NumPy também são marcados      ║
# ║    como `flags.writeable=False`). Isso protege contra bugs sutis onde   ║
# ║    código em outro módulo sobrescreveria os pesos durante loops.        ║
# ║                                                                           ║
# ║  CACHE                                                                    ║
# ║    `FilterLoader.load()` mantém um cache em memória (LRU manual) dos    ║
# ║    filtros já carregados. Chamadas subsequentes com o mesmo nome        ║
# ║    retornam o mesmo objeto, economizando I/O e alocação. Esse cache    ║
# ║    é compartilhado por todas as instâncias `FilterLoader` criadas em   ║
# ║    um mesmo processo (classe-level dict).                               ║
# ║                                                                           ║
# ║  REFERÊNCIAS                                                              ║
# ║    • Werthmüller (2017) Geophysics 82(6) — empymod                      ║
# ║    • Kong (2007)       Geophysical Prospecting 55(1)                    ║
# ║    • Anderson (1989)   Geophysics 54(2), 263-266                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Carregamento e validação dos filtros Hankel digitais.

Este módulo expõe duas entidades públicas:

- :class:`HankelFilter` — dataclass imutável com abscissas, pesos J₀ e J₁.
- :class:`FilterLoader` — fábrica que lê os artefatos .npz e retorna
  instâncias de :class:`HankelFilter` devidamente validadas.

Example:
    Carregamento simples::

        >>> from geosteering_ai.simulation.filters import FilterLoader
        >>> loader = FilterLoader()
        >>> filt = loader.load("werthmuller_201pt")
        >>> filt.npt
        201
        >>> filt.abscissas[0]
        0.000865398089328...

    Integração com backend Numba (Fase 2)::

        from geosteering_ai.simulation.filters import FilterLoader
        loader = FilterLoader()
        filt = loader.load("werthmuller_201pt")
        # passa arrays float64 contíguos para kernel @njit
        H_field = hankel_kernel_numba(
            abscissas=filt.abscissas,
            weights=filt.weights_j1,
            kr_function=my_kr_function,
            r=distance,
        )
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Final

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES DO MÓDULO
# ──────────────────────────────────────────────────────────────────────────────
# Diretório onde os artefatos .npz estão instalados (co-locados com este
# módulo para simplificar a busca e evitar dependência de variáveis de
# ambiente). Resolvido de forma robusta via `Path(__file__).parent`.
_FILTERS_DIR: Final[Path] = Path(__file__).resolve().parent

# Mapeamento de nomes canônicos (snake_case) para arquivos .npz e para
# o `filter_type` equivalente no Fortran (vide Fortran_Gerador/PerfilaAnisoOmp.f08
# comentário em torno da variável `filter_type`).
#
# Esta tabela é a fonte única de verdade para conversão de nomes. Se um
# filtro novo for adicionado no futuro (ex.: Key 401pt), incluí-lo aqui
# e gerar o .npz correspondente via extract_hankel_weights.py.
_FILTER_CATALOG: Final[dict[str, dict[str, object]]] = {
    "werthmuller_201pt": {
        "file": "werthmuller_201pt.npz",
        "npt": 201,
        "fortran_filter_type": 0,
        "description": "Werthmüller 201pt — default do simulador Fortran",
    },
    "kong_61pt": {
        "file": "kong_61pt.npz",
        "npt": 61,
        "fortran_filter_type": 1,
        "description": "Kong 61pt — ~3.3× mais rápido, precisão aceitável",
    },
    "anderson_801pt": {
        "file": "anderson_801pt.npz",
        "npt": 801,
        "fortran_filter_type": 2,
        "description": "Anderson 801pt — máxima precisão, uso em referência",
    },
}

# Aliases curtos aceitos pelo `FilterLoader.load` (conveniência).
# Mapeamento many-to-one para o nome canônico.
_FILTER_ALIASES: Final[dict[str, str]] = {
    # Aliases "curtos"
    "werthmuller": "werthmuller_201pt",
    "wer": "werthmuller_201pt",
    "kong": "kong_61pt",
    "anderson": "anderson_801pt",
    "and": "anderson_801pt",
    # Aliases por filter_type numérico (string)
    "0": "werthmuller_201pt",
    "1": "kong_61pt",
    "2": "anderson_801pt",
}

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# DATACLASS HankelFilter
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class HankelFilter:
    """Representa um filtro Hankel digital (imutável).

    Um filtro Hankel digital aproxima integrais da forma

        F(r) = ∫₀^∞ f(kr) · Jν(kr · r) · dkr

    usando uma quadratura de N pontos previamente otimizada:

        F(r) ≈ (1/r) · Σᵢ₌₀^{N-1} f(aᵢ / r) · wᵢ(ν)

    onde ``a = abscissas`` são os pontos de avaliação (em espaço kr·r
    logaritmicamente espaçado) e ``w = weights_j0`` ou ``weights_j1``
    são os pesos correspondentes à ordem da função de Bessel (ν=0 ou ν=1).

    Ambos os conjuntos de pesos compartilham as mesmas abscissas, pois
    os 3 filtros implementados (Kong, Werthmüller, Anderson) são da
    família "J₀-J₁ compartilhada" (common-base).

    Attributes:
        name: Nome canônico do filtro (ex.: 'werthmuller_201pt').
        abscissas: Pontos de quadratura (shape (npt,), float64, > 0,
            estritamente crescentes).
        weights_j0: Pesos para ∫ f(kr)·J₀(kr·r) dkr (shape (npt,), float64).
        weights_j1: Pesos para ∫ f(kr)·J₁(kr·r) dkr (shape (npt,), float64).
        npt: Número de pontos do filtro (== len(abscissas)).
        fortran_filter_type: Código `filter_type` equivalente no Fortran
            (0=Werthmüller, 1=Kong, 2=Anderson).
        source_sha256: Hash SHA-256 do `filtersv2.f08` no momento da
            extração (para auditoria de sincronia).
        description: Descrição curta do filtro e seus trade-offs.

    Note:
        Todos os arrays são marcados como read-only via
        `arr.setflags(write=False)` no momento da construção pela
        :class:`FilterLoader`. Tentativas de mutação lançarão ValueError.

    Example:
        >>> filt = FilterLoader().load("werthmuller_201pt")
        >>> isinstance(filt, HankelFilter)
        True
        >>> filt.abscissas.flags.writeable
        False
    """

    name: str
    abscissas: np.ndarray
    weights_j0: np.ndarray
    weights_j1: np.ndarray
    npt: int
    fortran_filter_type: int
    source_sha256: str
    description: str = ""

    def __post_init__(self) -> None:
        """Valida invariantes após construção.

        Raises:
            ValueError: Se shapes/tipos/valores não forem consistentes.

        Note:
            O dataclass é ``frozen=True``, então invariantes que dependem
            de arrays são checadas aqui (e não em ``__init__`` padrão).
            Esta validação duplica parte do que `_save_npz` já fez, mas
            é defensiva: protege contra .npz corrompido ou editado à mão.
        """
        # ── Validação de shapes e tipos ─────────────────────────────────
        # Os três arrays DEVEM ter mesmo comprimento e dtype float64.
        # float32 é rejeitado aqui porque introduziria erro numérico
        # visível nos testes de bit-exactness contra Fortran (que usa
        # real(dp) = float64 por default).
        for attr_name in ("abscissas", "weights_j0", "weights_j1"):
            arr = getattr(self, attr_name)
            if not isinstance(arr, np.ndarray):
                raise ValueError(
                    f"{self.name}.{attr_name} deve ser np.ndarray, "
                    f"recebeu {type(arr).__name__}"
                )
            if arr.dtype != np.float64:
                raise ValueError(
                    f"{self.name}.{attr_name} deve ser float64, " f"recebeu {arr.dtype}"
                )
            if arr.shape != (self.npt,):
                raise ValueError(
                    f"{self.name}.{attr_name} deve ter shape ({self.npt},), "
                    f"recebeu {arr.shape}"
                )

        # ── Validação semântica: abscissas > 0 e crescentes ─────────────
        # Filtros Hankel exigem pontos de quadratura estritamente
        # positivos (kr > 0) e ordenados ascendentemente.
        if not np.all(self.abscissas > 0):
            raise ValueError(f"{self.name}: abscissas contêm valores <= 0")
        if not np.all(np.diff(self.abscissas) > 0):
            raise ValueError(f"{self.name}: abscissas não são estritamente crescentes")


# ──────────────────────────────────────────────────────────────────────────────
# CLASSE FilterLoader
# ──────────────────────────────────────────────────────────────────────────────
class FilterLoader:
    """Carregador de filtros Hankel digitais a partir dos .npz.

    Por padrão, lê os artefatos do diretório co-localizado com este módulo
    (`geosteering_ai/simulation/filters/`). Pode ser parametrizado com
    um diretório alternativo para testes ou para overrides de desenvolvimento.

    O carregador mantém um cache de instância (atributo `_instance_cache`)
    e um cache de classe (`_class_cache`) compartilhado entre instâncias.
    Filtros carregados são imutáveis (arrays read-only), então compartilhar
    a mesma instância entre módulos é seguro.

    Attributes:
        filters_dir: Diretório raiz dos artefatos .npz.

    Example:
        Carregamento e uso típico::

            >>> loader = FilterLoader()
            >>> wer = loader.load("werthmuller_201pt")  # primeiro acesso: I/O
            >>> wer_again = loader.load("werthmuller_201pt")  # cache hit
            >>> wer is wer_again
            True

        Listagem de filtros disponíveis::

            >>> loader.available()
            ['werthmuller_201pt', 'kong_61pt', 'anderson_801pt']

        Uso com aliases::

            >>> loader.load("wer").name  # alias de werthmuller_201pt
            'werthmuller_201pt'
    """

    # Cache compartilhado entre instâncias. Chaveado por
    # (caminho_do_diretório, nome_canônico) para permitir múltiplos
    # `FilterLoader` apontando a diretórios distintos sem colisão.
    _class_cache: ClassVar[dict[tuple[str, str], HankelFilter]] = {}

    # Lock que protege inserções concorrentes no `_class_cache`. Sem ele,
    # duas threads chamando `load(name)` simultaneamente no primeiro acesso
    # construiriam dois `HankelFilter` distintos — o segundo sobrepõe o
    # primeiro no dict, quebrando a garantia de identidade `a is b` que os
    # consumidores (kernels Numba/JAX) assumem. Double-checked locking
    # mantém a leitura rápida no caminho feliz (cache hit) e serializa só
    # o cache miss.
    _class_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, filters_dir: Path | str | None = None) -> None:
        """Inicializa o carregador.

        Args:
            filters_dir: Caminho alternativo para o diretório de filtros.
                Se None, usa o diretório co-localizado com este módulo.

        Note:
            O parâmetro `filters_dir` é útil para testes que queiram usar
            artefatos fabricados ou para ambientes com paths customizados.
        """
        self.filters_dir: Path = Path(filters_dir) if filters_dir else _FILTERS_DIR
        if not self.filters_dir.exists():
            raise FileNotFoundError(
                f"Diretório de filtros não encontrado: {self.filters_dir}"
            )

    # ─────────────────────────────────────────────────────────────────────
    # API PÚBLICA
    # ─────────────────────────────────────────────────────────────────────
    def resolve_name(self, name: str) -> str:
        """Resolve alias ou filter_type numérico para o nome canônico.

        Args:
            name: Nome amigável (ex.: 'werthmuller', 'wer', 'kong'),
                nome canônico (ex.: 'werthmuller_201pt') ou string
                numérica ('0', '1', '2') correspondente ao filter_type
                do Fortran.

        Returns:
            Nome canônico (sempre uma chave válida de `_FILTER_CATALOG`).

        Raises:
            KeyError: Se o nome não for reconhecido.
        """
        if name in _FILTER_CATALOG:
            return name
        if name in _FILTER_ALIASES:
            return _FILTER_ALIASES[name]
        raise KeyError(
            f"Filtro '{name}' desconhecido. "
            f"Disponíveis: {sorted(_FILTER_CATALOG.keys())}. "
            f"Aliases: {sorted(_FILTER_ALIASES.keys())}."
        )

    def load(self, name: str) -> HankelFilter:
        """Carrega um filtro Hankel pelo nome (canônico ou alias).

        Args:
            name: Nome do filtro (canônico, alias ou filter_type numérico).

        Returns:
            Instância imutável de :class:`HankelFilter` com os arrays
            prontos para consumo por kernels Numba/JAX.

        Raises:
            KeyError: Se o nome não for reconhecido.
            FileNotFoundError: Se o .npz correspondente não existir.
            ValueError: Se o .npz estiver corrompido ou inconsistente.

        Note:
            O primeiro acesso carrega do disco; acessos subsequentes
            retornam a instância cacheada. O cache é chaveado por
            (caminho do diretório, nome canônico), portanto `FilterLoader`
            s com diretórios distintos não interferem.
        """
        canonical = self.resolve_name(name)
        cache_key = (str(self.filters_dir), canonical)

        # ── Cache hit? (caminho rápido, sem lock) ────────────────────
        # Reuso de instâncias imutáveis é seguro e economiza ~10ms
        # de I/O + validação por chamada. A leitura do dict é atômica
        # sob GIL do CPython, então não precisa de lock no cache hit.
        cached = self._class_cache.get(cache_key)
        if cached is not None:
            return cached

        # ── Cache miss: adquire lock e re-verifica (double-checked) ──
        # Sem o lock, duas threads em cache miss simultâneo construiriam
        # dois `HankelFilter` distintos e o segundo escrito sobreporia
        # o primeiro, quebrando a garantia `a is b` que os consumidores
        # assumem (kernels Numba/JAX chamados de múltiplos workers na
        # Fase 2). A re-verificação dentro do lock garante que apenas
        # uma thread faz o I/O mesmo após passarem ambas pelo `get`
        # inicial sem o lock.
        with FilterLoader._class_lock:
            cached = self._class_cache.get(cache_key)
            if cached is not None:
                return cached

            # ── Cache miss confirmado: ler .npz e construir HankelFilter ─
            catalog_entry = _FILTER_CATALOG[canonical]
            npz_path = self.filters_dir / str(catalog_entry["file"])
            if not npz_path.exists():
                raise FileNotFoundError(
                    f"Artefato .npz do filtro '{canonical}' não encontrado em "
                    f"{npz_path}. Execute scripts/extract_hankel_weights.py "
                    f"para regenerá-lo."
                )

            logger.debug("Carregando filtro %s de %s", canonical, npz_path)
            with np.load(npz_path, allow_pickle=False) as npz:
                # allow_pickle=False é estrito: forçamos que metadata seja
                # serializada como string JSON (cf. extract_hankel_weights.py).
                expected_keys = {
                    "abscissas",
                    "weights_j0",
                    "weights_j1",
                    "metadata",
                }
                missing = expected_keys - set(npz.files)
                if missing:
                    raise ValueError(
                        f"Arquivo {npz_path.name} está incompleto. "
                        f"Chaves faltantes: {sorted(missing)}. "
                        f"Execute scripts/extract_hankel_weights.py para regenerar."
                    )

                abscissas = np.ascontiguousarray(
                    npz["abscissas"],
                    dtype=np.float64,
                )
                weights_j0 = np.ascontiguousarray(
                    npz["weights_j0"],
                    dtype=np.float64,
                )
                weights_j1 = np.ascontiguousarray(
                    npz["weights_j1"],
                    dtype=np.float64,
                )
                metadata = json.loads(str(npz["metadata"]))

            # ── Marca arrays como read-only ──────────────────────────
            # Previne bugs onde outro módulo escreveria nos arrays
            # cacheados, corrompendo todas as chamadas subsequentes.
            abscissas.setflags(write=False)
            weights_j0.setflags(write=False)
            weights_j1.setflags(write=False)

            # ── Constrói dataclass imutável + valida invariantes ─────
            filt = HankelFilter(
                name=canonical,
                abscissas=abscissas,
                weights_j0=weights_j0,
                weights_j1=weights_j1,
                npt=int(catalog_entry["npt"]),
                fortran_filter_type=int(catalog_entry["fortran_filter_type"]),
                source_sha256=str(metadata.get("source_sha256", "")),
                description=str(catalog_entry["description"]),
            )

            self._class_cache[cache_key] = filt
            return filt

    def available(self) -> list[str]:
        """Lista os nomes canônicos dos filtros disponíveis.

        Returns:
            Lista ordenada de nomes canônicos cujo .npz existe em
            `self.filters_dir`. Filtros no catálogo mas sem .npz
            físico são omitidos.
        """
        return [
            name
            for name in _FILTER_CATALOG
            if (self.filters_dir / str(_FILTER_CATALOG[name]["file"])).exists()
        ]

    def clear_cache(self) -> None:
        """Limpa o cache de filtros carregados.

        Note:
            Útil em testes que queiram forçar re-leitura do disco (por
            exemplo, após `scripts/extract_hankel_weights.py` regerar
            um .npz). Afeta todas as instâncias `FilterLoader` do
            processo — o cache é classe-level. A operação é serializada
            via `_class_lock` para evitar corrida com um `load()`
            concorrente em outra thread.
        """
        with FilterLoader._class_lock:
            self._class_cache.clear()


__all__ = ["FilterLoader", "HankelFilter"]
