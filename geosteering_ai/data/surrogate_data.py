# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: data/surrogate_data.py                                            ║
# ║  Bloco: 2b — Extracao de Dados para Surrogate Forward Model               ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · GPU A6000 local (exec)║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (ponto unico de verdade)                ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • Extrair pares (rho, H_EM) do .dat para treino do SurrogateNet       ║
# ║    • Gerar pares ON-THE-FLY via simulador (dispatcher) — sem .dat/.npz    ║
# ║    • Selecionar componentes EM configuraveis (Modo A/B/C)                 ║
# ║    • Aplicar decoupling e normalizacao nos canais H                        ║
# ║    • Computar pesos por componente para loss balanceada                    ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig),                                ║
# ║                data/loading.py (EM_COMPONENTS, load_binary_dat),          ║
# ║                data/synthetic_generator.py (SyntheticDataGenerator)        ║
# ║  Exports: 7 funcoes/classes — ver __all__                                 ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 18.4 (Surrogate Data)                ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.2 (2026-04) — Implementacao inicial (Passo 2, Modo B)            ║
# ║    v2.50  (2026-06) — API on-the-fly (pairs_from_batch/generate/iter)    ║
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
from typing import TYPE_CHECKING, Iterator, List, Literal, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from geosteering_ai.config import PipelineConfig
    from geosteering_ai.data.synthetic_generator import (
        GeneratedBatch,
        SyntheticDataGenerator,
    )

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
    # ── Extracao (offline) ────────────────────────────────────────────────
    "extract_surrogate_pairs",
    "get_component_column_indices",
    # ── Geracao on-the-fly ────────────────────────────────────────────────
    "surrogate_pairs_from_batch",
    "generate_surrogate_dataset",
    "iter_surrogate_batches",
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
        raise ValueError(
            f"data deve ser 3D (N, seq_len, 22), recebido ndim={data.ndim}"
        )
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
# SECAO: GERACAO ON-THE-FLY DE PARES (rho → H) PARA TREINO DO SURROGATE
# ════════════════════════════════════════════════════════════════════════════
# Produtiza o caminho on-the-fly (dívida do Sprint C): gera pares (x_rho, y_em)
# FRESCOS via o simulador batched (dispatcher Sprint B) — sem round-trip .npz e
# sem o laço Python O(N×L) do .dat. Três níveis:
#   • surrogate_pairs_from_batch — ponte GeneratedBatch → SurrogateDataset (vetorizada);
#   • generate_surrogate_dataset  — builder one-shot (amostra + simula + extrai);
#   • iter_surrogate_batches      — stream infinito de batches frescos (refresh por época).
# Reusa a extração TESTADA `extract_surrogate_pairs` (decoupling + log10 + seleção).
# ──────────────────────────────────────────────────────────────────────────


def surrogate_pairs_from_batch(
    batch: "GeneratedBatch",
    config: "PipelineConfig",
    *,
    apply_decoup: bool = True,
) -> SurrogateDataset:
    """Converte um :class:`GeneratedBatch` em pares (rho, H_EM) p/ o surrogate.

    Ponte VETORIZADA entre o gerador sintético e o treino do SurrogateNet: monta
    o array ``(N, L, 22)`` direto do ``H_tensor`` + perfis de resistividade do
    batch (sem o laço Python O(N×L) do ``.dat``) e delega à extração testada
    :func:`extract_surrogate_pairs`.

    Pipeline:
      ┌──────────────────────────────────────────────────────────────────┐
      │  GeneratedBatch → pares de treino do surrogate:                  │
      │                                                                  │
      │  1. Normaliza H p/ a config de referência (TR₀, dip₀, freq₀)    │
      │     → (n_models, n_pos, 9) complex128                            │
      │  2. ρ no receptor por (modelo, posição) — lookup VETORIZADO      │
      │  3. Monta (N, L, 22) float64 (cols 1/2/3 + 9×[Re,Im])          │
      │  4. extract_surrogate_pairs (decoupling + log10 + seleção)       │
      └──────────────────────────────────────────────────────────────────┘

    Args:
        batch: :class:`~geosteering_ai.data.synthetic_generator.GeneratedBatch`
            com ``H_tensor`` (4-D single-config ``(n_models, n_pos, nf, 9)`` ou
            6-D multi-config ``(n_models, nTR, nAng, n_pos, nf, 9)``), ``rho_h``/
            ``rho_v`` ``(n_models, n_layers)``, ``esp`` ``(n_models, n_layers-2)``
            e ``positions_z`` ``(n_pos,)``.
        config: :class:`PipelineConfig` com ``surrogate_output_components`` e
            ``spacing_meters`` (constantes de decoupling).
        apply_decoup: Se ``True`` (default), remove o campo primário Tx-Rx das
            diagonais (XX, YY, ZZ). O gerador produz H **RAW** → aplicar 1× aqui.
            Use ``False`` apenas se o batch já vier com decoupling.

    Returns:
        :class:`SurrogateDataset` com ``x_rho (N, L, 2)`` e ``y_em (N, L, 2*K)``.

    Raises:
        ValueError: Se ``batch.H_tensor`` tiver ``ndim`` ≠ 4 e ≠ 6.

    Example:
        >>> from geosteering_ai.config import PipelineConfig
        >>> from geosteering_ai.data.synthetic_generator import SyntheticDataGenerator
        >>> cfg = PipelineConfig(simulator_backend="numba")
        >>> batch = SyntheticDataGenerator(cfg).generate_batch(
        ...     n_models=8, n_positions=30, n_layers=5, build_dat_22col=False)
        >>> ds = surrogate_pairs_from_batch(batch, cfg)
        >>> assert ds.x_rho.shape == (8, 30, 2)

    Note:
        Produz saída BIT-EXATA ao caminho offline
        (``extract_surrogate_pairs`` sobre o mesmo ``(N, L, 22)``). A escolha da
        config de referência (TR₀/dip₀/freq₀) espelha
        :meth:`SyntheticDataGenerator.to_feature_dataset` e o ``dat_22col``.
        See Also: :func:`generate_surrogate_dataset` (amostra + simula + extrai).
    """
    from geosteering_ai.data.synthetic_generator import _layer_at_batch

    # ── 1. Normaliza H p/ a config de referência (n_models, n_pos, 9) ────────
    # Robustez de ndim (espelha to_feature_dataset): 4-D single-config usa freq0;
    # 6-D multi-config usa (TR0, dip0, freq0) — a MESMA config do dat_22col.
    H = np.asarray(batch.H_tensor)
    if H.ndim == 4:  # (n_models, n_pos, nf, 9)
        H_ref = H[:, :, 0, :]
    elif H.ndim == 6:  # (n_models, nTR, nAng, n_pos, nf, 9)
        H_ref = H[:, 0, 0, :, 0, :]
    else:
        raise ValueError(
            f"batch.H_tensor com ndim inesperado: {H.ndim} (esperado 4 ou 6)."
        )

    n_models, n_pos, _ = H_ref.shape
    positions_z = np.asarray(batch.positions_z, dtype=np.float64)
    n_layers = batch.n_layers

    # ── 2. ρ no receptor por (modelo, posição) — lookup VETORIZADO ───────────
    layer = _layer_at_batch(positions_z, np.asarray(batch.esp), n_layers)
    rho_h_obs = np.take_along_axis(np.asarray(batch.rho_h), layer, axis=1)
    rho_v_obs = np.take_along_axis(np.asarray(batch.rho_v), layer, axis=1)

    # ── 3. Monta (N, L, 22) float64 — colunas 1/2/3 + 9×(Re, Im) ─────────────
    # Cast c128 explícito: preserva a paridade mesmo se o batch vier em complex64
    # (o gerador retorna c128 por default; defensivo p/ batches externos).
    data = np.zeros((n_models, n_pos, 22), dtype=np.float64)
    data[:, :, 1] = positions_z[None, :]
    data[:, :, 2] = rho_h_obs
    data[:, :, 3] = rho_v_obs
    h_c128 = H_ref.astype(np.complex128, copy=False)
    for c in range(9):
        data[:, :, 4 + 2 * c] = h_c128[:, :, c].real
        data[:, :, 5 + 2 * c] = h_c128[:, :, c].imag

    # ── 4. Delega à extração TESTADA (decoupling + log10 + seleção) ──────────
    return extract_surrogate_pairs(data, config, apply_decoup=apply_decoup)


def _resolve_generator(config: "PipelineConfig") -> "SyntheticDataGenerator":
    """Cria um :class:`SyntheticDataGenerator` roteando o backend do ``config``.

    ``simulator_backend='fortran_f2py'`` (default histórico do ``PipelineConfig``)
    NÃO é suportado pelo gerador in-process (o legacy é
    ``Fortran_Gerador/batch_runner.py``); nesse caso roteia p/ ``'auto'``
    (dispatcher Sprint B: JAX GPU ⇄ Numba). Backends ``{numba, jax, auto}``
    passam direto — mantém ``PipelineConfig`` como fonte única de roteamento.

    Args:
        config: :class:`PipelineConfig` com ``simulator_backend``.

    Returns:
        :class:`SyntheticDataGenerator` pronto p/ ``generate_batch``.

    Note:
        Referenciado em :func:`generate_surrogate_dataset` e
        :func:`iter_surrogate_batches`.
    """
    import dataclasses

    from geosteering_ai.data.synthetic_generator import SyntheticDataGenerator

    if config.simulator_backend == "fortran_f2py":
        logger.info(
            "surrogate on-the-fly: simulator_backend='fortran_f2py' não suportado "
            "in-process; roteando p/ 'auto' (dispatcher JAX/Numba)."
        )
        config = dataclasses.replace(config, simulator_backend="auto")
    return SyntheticDataGenerator(config)


def generate_surrogate_dataset(
    config: "PipelineConfig",
    *,
    n_models: int,
    n_positions: int = 600,
    n_layers: int = 5,
    rho_h_range: tuple[float, float] = (1.0, 1000.0),
    rho_v_range: tuple[float, float] = (1.0, 1000.0),
    thickness_range: tuple[float, float] = (1.0, 10.0),
    strategy: Literal["uniform", "log_uniform"] = "log_uniform",
    seed: int = 42,
    frequencies_hz: Optional[Sequence[float]] = None,
    geometry_mode: Literal["templates", "quantize", "per_model"] = "templates",
    n_geometries: Optional[int] = None,
    apply_decoup: bool = True,
    generator: Optional["SyntheticDataGenerator"] = None,
    **gen_kwargs,
) -> SurrogateDataset:
    """Gera ON-THE-FLY um :class:`SurrogateDataset` (amostra + simula + extrai).

    Builder one-shot que produtiza o caminho de geração do surrogate (antes
    hard-rolled no benchmark Fase A): amostra ``n_models`` perfis ρ/geometria,
    roda o simulador batched (via dispatcher) e extrai os pares (x_rho, y_em) —
    tudo VETORIZADO, sem o laço Python O(N×L) nem round-trip ``.npz``.

    Args:
        config: :class:`PipelineConfig` — ``simulator_backend`` roteia o backend
            (``fortran_f2py`` → ``auto``); ``surrogate_output_components`` define K.
        n_models: Número de modelos a gerar.
        n_positions: Posições de medição por modelo. Default 600 (escala produção).
        n_layers: Camadas por modelo (inclui 2 semi-espaços). Default 5.
        rho_h_range: ``(min, max)`` de ρₕ (Ω·m). Default ``(1, 1000)``.
        rho_v_range: ``(min, max)`` de ρᵥ (Ω·m). Default ``(1, 1000)``.
        thickness_range: ``(min, max)`` de espessuras internas (m). Default ``(1, 10)``.
        strategy: ``"log_uniform"`` (default) ou ``"uniform"``.
        seed: Semente p/ reprodutibilidade.
        frequencies_hz: Frequências (Hz). Default ``[config.frequency_hz]``.
        geometry_mode: ``"templates"`` (default) | ``"quantize"`` | ``"per_model"``.
        n_geometries: K geometrias distintas (replicadas) — ativa o caminho
            bucketed rápido no JAX. ``None`` = ``max(1, n_models//32)``.
        apply_decoup: Decoupling nas diagonais (default ``True``; H gerado é RAW).
        generator: :class:`SyntheticDataGenerator` reusável (evita re-warmup JIT).
            ``None`` cria um via :func:`_resolve_generator`.
        **gen_kwargs: Repassados a ``generate_batch`` (ex.: ``jax_chunk_size_models``,
            ``numba_fallback``, ``quantize_step``, ``dip_degs``, ``tr_spacings_m``).

    Returns:
        :class:`SurrogateDataset` com ``x_rho (N, L, 2)`` e ``y_em (N, L, 2*K)``.

    Raises:
        ValueError: Se ``n_models <= 0``.

    Example:
        >>> from geosteering_ai.config import PipelineConfig
        >>> cfg = PipelineConfig(simulator_backend="auto",
        ...                      surrogate_output_components=["XX", "ZZ"])
        >>> ds = generate_surrogate_dataset(cfg, n_models=64, n_positions=600)
        >>> assert ds.x_rho.shape == (64, 600, 2)

    Note:
        Pula a construção do ``.dat`` (``build_dat_22col=False``) — o surrogate
        consome ``H_tensor`` direto. See Also: :func:`iter_surrogate_batches`
        (stream p/ treino on-the-fly por época).
    """
    if n_models <= 0:
        raise ValueError(f"n_models deve ser > 0 (recebido {n_models}).")

    gen = generator if generator is not None else _resolve_generator(config)
    batch = gen.generate_batch(
        n_models=n_models,
        n_positions=n_positions,
        n_layers=n_layers,
        rho_h_range=rho_h_range,
        rho_v_range=rho_v_range,
        thickness_range=thickness_range,
        strategy=strategy,
        seed=seed,
        frequencies_hz=frequencies_hz,
        geometry_mode=geometry_mode,
        n_geometries=n_geometries,
        build_dat_22col=False,  # on-the-fly: consome H_tensor direto, pula o .dat
        **gen_kwargs,
    )
    return surrogate_pairs_from_batch(batch, config, apply_decoup=apply_decoup)


def iter_surrogate_batches(
    config: "PipelineConfig",
    *,
    batch_size: int,
    n_batches: Optional[int] = None,
    seed: int = 0,
    apply_decoup: bool = True,
    generator: Optional["SyntheticDataGenerator"] = None,
    **gen_kwargs,
) -> "Iterator[SurrogateDataset]":
    """Stream ON-THE-FLY de :class:`SurrogateDataset` com modelos FRESCOS por batch.

    Primitivo de streaming p/ treino on-the-fly: a cada batch reamostra modelos
    (``batch_seed = seed + i``) → cada ``SurrogateDataset`` cobre ρ/geometria
    diferentes (dados efetivamente infinitos, sem overfit a um conjunto fixo).
    Reusa um único gerador (1 warmup JIT amortizado).

    Args:
        config: :class:`PipelineConfig` (roteia backend; define K do surrogate).
        batch_size: Modelos por batch (> 0).
        n_batches: Número de batches a produzir. ``None`` (default) = INFINITO —
            o caller DEVE limitar (``itertools.islice`` ou ``for _ in range(E)``).
        seed: Semente base; o batch ``i`` usa ``seed + i`` (determinístico).
        apply_decoup: Decoupling nas diagonais (default ``True``).
        generator: :class:`SyntheticDataGenerator` reusável; ``None`` cria um.
        **gen_kwargs: Repassados a :func:`generate_surrogate_dataset` (ex.:
            ``n_positions``, ``n_layers``, ``n_geometries``, ``frequencies_hz``,
            ``jax_chunk_size_models``).

    Yields:
        :class:`SurrogateDataset` ``(batch_size, L, 2)`` / ``(batch_size, L, 2*K)``.

    Raises:
        ValueError: Se ``batch_size <= 0`` ou ``n_batches < 0``.

    Example:
        >>> import itertools
        >>> from geosteering_ai.config import PipelineConfig
        >>> cfg = PipelineConfig(simulator_backend="auto")
        >>> it = iter_surrogate_batches(cfg, batch_size=128, n_positions=600)
        >>> for epoch in range(3):           # refresh por época
        ...     ds = next(it)                # batch fresco

    Note:
        **Contenção JAX/TF (GPU)**: gere os batches ENTRE épocas (NÃO dentro do
        ``tf.data.map``) — JAX (geração) e TF (treino) competem por VRAM na mesma
        GPU. Prefira ``simulator_backend='auto'`` (o dispatcher cai p/ Numba/CPU
        quando a GPU está ocupada). Espelha a separação de processos do benchmark
        (Fase A JAX × Fase B TF). See Also: :func:`generate_surrogate_dataset`.

        Validação dos argumentos é EAGER (na chamada, não no 1º ``next()``) — esta
        é uma função normal que valida e retorna o gerador interno ``_iter``.
    """
    # ── Validação eager (esta função NÃO é geradora → falha na chamada) ───────
    if batch_size <= 0:
        raise ValueError(f"batch_size deve ser > 0 (recebido {batch_size}).")
    if n_batches is not None and n_batches < 0:
        raise ValueError(f"n_batches deve ser >= 0 ou None (recebido {n_batches}).")

    gen = generator if generator is not None else _resolve_generator(config)

    def _iter() -> "Iterator[SurrogateDataset]":
        i = 0
        while n_batches is None or i < n_batches:
            # Avança a seed por batch → cada batch amostra modelos FRESCOS.
            yield generate_surrogate_dataset(
                config,
                n_models=batch_size,
                seed=seed + i,
                apply_decoup=apply_decoup,
                generator=gen,  # reusa o gerador (1 warmup JIT)
                **gen_kwargs,
            )
            i += 1

    return _iter()


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
