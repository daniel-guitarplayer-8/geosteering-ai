# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: data/surrogate_data.py                                            ║
# ║  Bloco: 2b — Extracao de Dados para Surrogate Forward Model               ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (ponto unico de verdade)                ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • Extrair pares (rho, H_EM) do .dat para treino do SurrogateNet       ║
# ║    • Selecionar componentes EM configuraveis (Modo A/B/C)                 ║
# ║    • Aplicar decoupling e normalizacao nos canais H                        ║
# ║    • Computar pesos por componente para loss balanceada                    ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig),                                ║
# ║                data/loading.py (EM_COMPONENTS, load_binary_dat)           ║
# ║  Exports: ~4 funcoes/classes — ver __all__                                ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 18.4 (Surrogate Data)                ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.2 (2026-04) — Implementacao inicial (Passo 2, Modo B)            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Extracao de dados para treino do SurrogateNet — forward model neural.

Prepara pares de treino (X_surrogate, Y_surrogate) a partir dos arquivos
.dat do simulador Fortran PerfilaAnisoOmp:

    X_surrogate: (rho_h, rho_v) em log10 scale → shape (N_models, seq_len, 2)
    Y_surrogate: K componentes EM selecionadas → shape (N_models, seq_len, 2*K)

A selecao de componentes eh controlada por config.surrogate_output_components:

  ┌──────────────────────────────────────────────────────────────────────────┐
  │  Modo A (default): ["XX", "ZZ"] → 4 canais (Re+Im x 2)               │
  │  Modo B (geosteering): ["XX", "ZZ", "XZ", "ZX"] → 8 canais           │
  │  Modo C (tensor completo): 9 componentes → 18 canais                  │
  └──────────────────────────────────────────────────────────────────────────┘

Cada componente produz 2 canais: Re(H_ij) e Im(H_ij).
O decoupling (remocao do campo primario Tx-Rx) eh aplicado nas componentes
diagonais (XX, YY, ZZ) conforme constantes ACp e ACx de loading.py.

Ref: docs/ARCHITECTURE_v2.md secao 18.4, Ward & Hohmann (1988) Ch. 4.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig

logger = logging.getLogger(__name__)

# ── Epsilon float32-safe (NUNCA 1e-30) ────────────────────────────────
EPS = 1e-12

# ── Componentes que recebem decoupling (campo primario) ──────────────
# Somente componentes diagonais do tensor H tem acoplamento direto Tx-Rx.
# Componentes cruzadas (off-diagonal) nao recebem decoupling pois o
# campo primario em espaco livre eh diagonal no referencial da ferramenta.
# Ref: data/loading.py apply_decoupling(), CLAUDE.md errata.
_DECOUPLED_COMPONENTS = {"XX", "YY", "ZZ"}

# ── Componentes diagonais do tensor H ─────────────────────────────
# Usadas para classificacao de pesos: diagonais (XX, YY, ZZ) recebem
# peso w_diagonal; off-diagonais (XZ, ZX, etc.) recebem w_cross.
# Coincide com _DECOUPLED_COMPONENTS mas eh semanticamente distinta.
_DIAGONAL_COMPONENTS = {"XX", "YY", "ZZ"}


# ════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════
# Inventario completo de simbolos exportados por este modulo.
# Agrupados semanticamente: dataclass, extracao, pesos.
# ──────────────────────────────────────────────────────────────────────────

__all__ = [
    # ── Dataclass ─────────────────────────────────────────────────────────
    "SurrogateDataset",
    # ── Extracao ──────────────────────────────────────────────────────────
    "extract_surrogate_pairs",
    "get_component_column_indices",
    # ── Pesos ─────────────────────────────────────────────────────────────
    "compute_component_weights",
]


# ════════════════════════════════════════════════════════════════════════════
# SECAO: DATACLASS — SURROGATE DATASET
# ════════════════════════════════════════════════════════════════════════════
# Encapsula os pares (X, Y) extraidos para treino do SurrogateNet,
# junto com metadados sobre componentes e canais selecionados.
# Imutavel apos criacao (frozen=False para permitir normalizacao).
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class SurrogateDataset:
    """Dataset para treino do SurrogateNet — pares (rho, H_EM).

    Encapsula os arrays de entrada (resistividade) e saida (campos EM)
    para treino do forward model neural, junto com metadados sobre
    quais componentes do tensor H estao incluidas.

    Estrutura dos dados:
      ┌──────────────────────────────────────────────────────────────┐
      │  x_rho: (N, seq_len, 2) — [log10(rho_h), log10(rho_v)]   │
      │  y_em:  (N, seq_len, 2*K) — K componentes, Re+Im cada     │
      │                                                              │
      │  Exemplo Modo B (K=4):                                      │
      │    y_em canais: [Re(Hxx), Im(Hxx), Re(Hzz), Im(Hzz),     │
      │                  Re(Hxz), Im(Hxz), Re(Hzx), Im(Hzx)]     │
      └──────────────────────────────────────────────────────────────┘

    Attributes:
        x_rho (np.ndarray): Resistividade em log10 scale, shape (N, L, 2).
            Canal 0 = log10(rho_h), canal 1 = log10(rho_v).
        y_em (np.ndarray): Campos EM selecionados, shape (N, L, 2*K).
            Pares Re/Im intercalados por componente.
        components (List[str]): Lista de componentes incluidas.
            Ex: ["XX", "ZZ", "XZ", "ZX"] para Modo B.
        n_channels (int): Numero total de canais = 2 * len(components).
        channel_names (List[str]): Nomes dos canais para logging/debug.
            Ex: ["Re(Hxx)", "Im(Hxx)", "Re(Hzz)", "Im(Hzz)", ...].
        decoupled (bool): Se True, decoupling foi aplicado nas diagonais.

    Example:
        >>> from geosteering_ai.config import PipelineConfig
        >>> config = PipelineConfig(surrogate_output_components=["XX", "ZZ"])
        >>> ds = extract_surrogate_pairs(data_3d, config)
        >>> assert ds.x_rho.shape == (100, 600, 2)
        >>> assert ds.y_em.shape == (100, 600, 4)

    Note:
        Referenciado em:
            - data/surrogate_data.py: extract_surrogate_pairs()
            - models/surrogate.py: SurrogateNet (treino)
            - tests/test_surrogate.py: TestSurrogateDataset
        Ref: docs/ARCHITECTURE_v2.md secao 18.4.
    """

    x_rho: np.ndarray
    y_em: np.ndarray
    components: List[str]
    n_channels: int
    channel_names: List[str]
    decoupled: bool = True


# ════════════════════════════════════════════════════════════════════════════
# SECAO: FUNCOES AUXILIARES
# ════════════════════════════════════════════════════════════════════════════
# Funcoes de suporte para extracao: mapeamento de indices de coluna,
# geracao de nomes de canais, e calculo de pesos por componente.
# ──────────────────────────────────────────────────────────────────────────


def get_component_column_indices(
    components: List[str],
) -> List[int]:
    """Retorna indices de coluna no .dat para componentes EM selecionadas.

    Mapeia nomes de componentes (ex: "XX", "ZZ", "XZ") para os indices
    de coluna correspondentes no formato 22-colunas do .dat, intercalando
    Re e Im para cada componente.

    Mapeamento (de data/loading.py EM_COMPONENTS):
      ┌──────────────────────────────────────────────────────────────┐
      │  "XX" → cols 4, 5   (Re(Hxx), Im(Hxx))                    │
      │  "XY" → cols 6, 7   (Re(Hxy), Im(Hxy))                    │
      │  "XZ" → cols 8, 9   (Re(Hxz), Im(Hxz))                    │
      │  "YX" → cols 10, 11 (Re(Hyx), Im(Hyx))                    │
      │  "YY" → cols 12, 13 (Re(Hyy), Im(Hyy))                    │
      │  "YZ" → cols 14, 15 (Re(Hyz), Im(Hyz))                    │
      │  "ZX" → cols 16, 17 (Re(Hzx), Im(Hzx))                    │
      │  "ZY" → cols 18, 19 (Re(Hzy), Im(Hzy))                    │
      │  "ZZ" → cols 20, 21 (Re(Hzz), Im(Hzz))                    │
      └──────────────────────────────────────────────────────────────┘

    Args:
        components: Lista de componentes EM.
            Ex: ["XX", "ZZ"] para Modo A, ["XX", "ZZ", "XZ", "ZX"] para B.

    Returns:
        Lista de indices de coluna no .dat.
            Ex: [4, 5, 20, 21] para ["XX", "ZZ"].
            Ex: [4, 5, 20, 21, 8, 9, 16, 17] para ["XX", "ZZ", "XZ", "ZX"].

    Raises:
        ValueError: Se uma componente nao esta em EM_COMPONENTS.

    Example:
        >>> get_component_column_indices(["XX", "ZZ"])
        [4, 5, 20, 21]

    Note:
        Referenciado em:
            - data/surrogate_data.py: extract_surrogate_pairs()
        Ref: data/loading.py EM_COMPONENTS, COL_MAP_22.
    """
    from geosteering_ai.data.loading import EM_COMPONENTS

    indices = []
    for comp in components:
        if comp not in EM_COMPONENTS:
            raise ValueError(
                f"Componente EM '{comp}' invalida. "
                f"Validas: {sorted(EM_COMPONENTS.keys())}"
            )
        re_idx, im_idx = EM_COMPONENTS[comp]
        indices.extend([re_idx, im_idx])
    return indices


def _build_channel_names(components: List[str]) -> List[str]:
    """Gera nomes legiveis para cada canal de saida.

    Args:
        components: Lista de componentes EM (ex: ["XX", "ZZ"]).

    Returns:
        Lista de nomes (ex: ["Re(Hxx)", "Im(Hxx)", "Re(Hzz)", "Im(Hzz)"]).
    """
    names = []
    for comp in components:
        names.append(f"Re(H{comp.lower()})")
        names.append(f"Im(H{comp.lower()})")
    return names


# ════════════════════════════════════════════════════════════════════════════
# SECAO: EXTRACAO DE PARES PARA SURROGATE
# ════════════════════════════════════════════════════════════════════════════
# Funcao principal que extrai (rho, H_EM) do array 22-colunas,
# aplicando decoupling nas componentes diagonais e selecionando
# apenas as componentes configuradas no PipelineConfig.
# ──────────────────────────────────────────────────────────────────────────


def extract_surrogate_pairs(
    data: np.ndarray,
    config: "PipelineConfig",
    *,
    apply_decoup: bool = True,
) -> SurrogateDataset:
    """Extrai pares (rho, H_EM) do array 22-colunas para treino do surrogate.

    Pipeline de extracao:
      ┌──────────────────────────────────────────────────────────────────┐
      │  .dat (22-col) → pares de treino do surrogate:                  │
      │                                                                  │
      │  1. Extrair rho: data[:, :, [2, 3]] → (rho_h, rho_v)          │
      │  2. Aplicar log10 → escala de entrada (clamped [-2, 5])        │
      │  3. Selecionar componentes: config.surrogate_output_components  │
      │  4. Aplicar decoupling nas diagonais (XX, YY, ZZ)              │
      │  5. Retornar SurrogateDataset com metadados                     │
      └──────────────────────────────────────────────────────────────────┘

    Args:
        data: Array 3D (N_models, seq_len, 22) do .dat Fortran.
            Formato 22-colunas conforme COL_MAP_22 em loading.py.
            N_models: numero de modelos geologicos.
            seq_len: pontos de medição por modelo (derivado do .out, default 600).
        config: PipelineConfig com:
            - surrogate_output_components: Lista de componentes EM.
            - spacing_meters: Distancia Tx-Rx para decoupling (default 1.0 m).
        apply_decoup: Se True (default), remove campo primario Tx-Rx
            das componentes diagonais (XX, YY, ZZ).
            ATENCAO: se os dados ja passaram por apply_decoupling() de
            data/loading.py, usar apply_decoup=False para evitar
            decoupling duplo. Os dados brutos do .dat NAO tem decoupling
            — a funcao load_binary_dat() retorna valores raw.

    Returns:
        SurrogateDataset com x_rho (N, L, 2) e y_em (N, L, 2*K).

    Raises:
        ValueError: Se data.ndim != 3 ou data.shape[2] < 22.

    Example:
        >>> import numpy as np
        >>> from geosteering_ai.config import PipelineConfig
        >>> data = np.random.randn(10, 600, 22).astype(np.float32)
        >>> config = PipelineConfig(surrogate_output_components=["XX", "ZZ"])
        >>> ds = extract_surrogate_pairs(data, config)
        >>> assert ds.x_rho.shape == (10, 600, 2)
        >>> assert ds.y_em.shape == (10, 600, 4)

    Note:
        Referenciado em:
            - models/surrogate.py: treino do SurrogateNet
            - tests/test_surrogate.py: TestExtractSurrogatePairs
        Ref: docs/ARCHITECTURE_v2.md secao 18.4.
        Guard numerico: rho clamped a [0.01, 100000] Ohm.m antes de log10.
    """
    if data.ndim != 3:
        raise ValueError(f"data deve ser 3D (N, seq_len, 22), recebido ndim={data.ndim}")
    if data.shape[2] < 22:
        raise ValueError(f"data deve ter >= 22 colunas, recebido {data.shape[2]}")

    components = config.surrogate_output_components
    n_models, seq_len, _ = data.shape
    n_channels = 2 * len(components)

    # ── 1. Extrair resistividade (colunas 2, 3) ──────────────────────
    # rho_h = coluna 2 (Ohm.m), rho_v = coluna 3 (Ohm.m)
    # Clamp para evitar log10(0) e manter range fisico.
    rho_raw = data[:, :, [2, 3]].astype(np.float64)
    rho_clamped = np.clip(rho_raw, 0.01, 100000.0)
    x_rho = np.log10(rho_clamped).astype(np.float32)

    # ── 2. Extrair componentes EM selecionadas ────────────────────────
    col_indices = get_component_column_indices(components)
    y_em = data[:, :, col_indices].astype(np.float32)

    # ── 3. Decoupling: remover campo primario nas diagonais ──────────
    # As componentes diagonais (XX, YY, ZZ) contem o acoplamento direto
    # Tx-Rx que deve ser removido para isolar a resposta da formacao.
    # Componentes cruzadas (XZ, ZX, etc.) nao tem acoplamento direto.
    # Constantes de decoupling (L = spacing_meters):
    #   ACp = -1/(4*pi*L^3)  para XX, YY (planar)
    #   ACx = +1/(2*pi*L^3)  para ZZ (axial)
    if apply_decoup:
        spacing = config.spacing_meters
        acp = -1.0 / (4.0 * np.pi * spacing**3)  # planar (Hxx, Hyy)
        acx = 1.0 / (2.0 * np.pi * spacing**3)  # axial (Hzz)

        # Mapear componentes para constantes de decoupling
        _decoupling_map = {
            "XX": acp,
            "YY": acp,
            "ZZ": acx,
        }

        channel_idx = 0
        for comp in components:
            if comp in _decoupling_map:
                # Decoupling aplicado somente na parte real (Re).
                # A parte imaginaria do campo primario eh zero em
                # frequencia real (campo estatico DC → Im = 0).
                dc_value = _decoupling_map[comp]
                y_em[:, :, channel_idx] -= dc_value  # Re(H) -= AC
                logger.debug("Decoupling aplicado: %s Re -= %.6f", comp, dc_value)
            channel_idx += 2  # pula Re+Im

    channel_names = _build_channel_names(components)

    logger.info(
        "Surrogate dataset extraido: %d modelos, %d pontos, "
        "%d componentes (%s), %d canais",
        n_models,
        seq_len,
        len(components),
        "/".join(components),
        n_channels,
    )

    return SurrogateDataset(
        x_rho=x_rho,
        y_em=y_em,
        components=list(components),
        n_channels=n_channels,
        channel_names=channel_names,
        decoupled=apply_decoup,
    )


# ════════════════════════════════════════════════════════════════════════════
# SECAO: PESOS POR COMPONENTE PARA LOSS BALANCEADA
# ════════════════════════════════════════════════════════════════════════════
# Componentes cruzadas (Hxz, Hzx) tem magnitudes ~10-100x menores
# que as diagonais (Hxx, Hzz) em dip baixo, necessitando pesos maiores
# para evitar que a loss seja dominada pelas diagonais.
# ──────────────────────────────────────────────────────────────────────────


def compute_component_weights(
    config: "PipelineConfig",
) -> np.ndarray:
    """Computa pesos por canal para loss balanceada do surrogate.

    Gera um vetor de pesos com 2*K valores (um por canal Re/Im),
    onde componentes cruzadas recebem peso maior para compensar
    suas magnitudes menores em relacao as diagonais.

    Classificacao das componentes:
      ┌──────────────────────────────────────────────────────────────┐
      │  Diagonais (peso = w_diagonal):  XX, YY, ZZ                │
      │  Cruzadas  (peso = w_cross):     XZ, ZX, XY, YX, YZ, ZY   │
      └──────────────────────────────────────────────────────────────┘

    Args:
        config: PipelineConfig com:
            - surrogate_output_components: Lista de componentes.
            - surrogate_weight_diagonal: Peso para diagonais (default 1.0).
            - surrogate_weight_cross: Peso para cruzadas (default 5.0).

    Returns:
        np.ndarray de shape (2*K,) com pesos por canal.
            Ex: [1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0] para
            components=["XX", "ZZ", "XZ", "ZX"] com w_diag=1.0, w_cross=5.0.

    Example:
        >>> config = PipelineConfig(
        ...     surrogate_output_components=["XX", "ZZ", "XZ", "ZX"],
        ...     surrogate_weight_diagonal=1.0,
        ...     surrogate_weight_cross=5.0,
        ... )
        >>> w = compute_component_weights(config)
        >>> assert w.shape == (8,)

    Note:
        Referenciado em:
            - models/surrogate.py: loss de treino do SurrogateNet
            - tests/test_surrogate.py: TestComputeComponentWeights
        Ref: docs/ARCHITECTURE_v2.md secao 18.4.
    """
    components = config.surrogate_output_components
    w_diag = config.surrogate_weight_diagonal
    w_cross = config.surrogate_weight_cross

    weights = []
    for comp in components:
        w = w_diag if comp in _DIAGONAL_COMPONENTS else w_cross
        weights.extend([w, w])  # Re e Im recebem mesmo peso

    return np.array(weights, dtype=np.float32)
