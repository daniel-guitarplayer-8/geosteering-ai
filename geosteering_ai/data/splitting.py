# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: data/splitting.py                                                 ║
# ║  Bloco: 2 — Preparacao de Dados                                           ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (NUNCA globals().get())                  ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • Split por modelo geologico [P1] — NUNCA por amostra                  ║
# ║    • Garantia de zero data leakage entre train/val/test                    ║
# ║    • Preservacao de z_meters para reconstrucao de perfis                   ║
# ║    • Shuffle deterministico via seed para reprodutibilidade                ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig), data/loading.py (AngleGroup)   ║
# ║  Exports: ~4 (DataSplits, split_model_ids, apply_split,                   ║
# ║           split_angle_group)                                               ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 4.2                                   ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (migrado de C23)              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

"""Split de dados por modelo geologico [P1].

Garante que nenhum modelo geologico aparece em mais de um split
(train, val, test), eliminando data leakage entre particoes.

Motivacao fisica:
    Cada modelo geologico gera 600 pontos de profundidade (SEQUENCE_LENGTH).
    Se amostras do MESMO modelo caissem em train e test, a rede
    "memorizaria" o perfil de resistividade daquele modelo, inflando
    metricas artificialmente. O split por modelo garante que a rede
    generaliza para GEOLOGIAS NUNCA VISTAS.

Referencia: docs/ARCHITECTURE_v2.md secao 5.1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

import numpy as np

from geosteering_ai.config import PipelineConfig
from geosteering_ai.data.loading import AngleGroup

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────
# __all__ — Exports publicos deste modulo
# ────────────────────────────────────────────────────────────────────────
__all__ = [
    "DataSplits",
    "split_model_ids",
    "apply_split",
    "split_angle_group",
]


# ════════════════════════════════════════════════════════════════════════
# DATA SPLITS — Container para dados particionados
#
# Resultado do split por modelo geologico. Cada split contem:
#   - x: features EM 3D (n_seq, seq_len=600, n_feat)
#   - y: targets de resistividade 3D (n_seq, seq_len=600, n_tgt)
#   - z: profundidade em metros (NUNCA escalado, usado para plots)
#   - model_ids: quais modelos geologicos pertencem a cada split
#
# z_meters e preservado separadamente para reconstrucao de perfis
# na avaliacao. O scaler NUNCA toca z_meters.
# ════════════════════════════════════════════════════════════════════════

@dataclass
class DataSplits:
    """Resultado do split por modelo geologico.

    Contem arrays 3D para train/val/test com z_meters preservado
    separadamente (NUNCA escalado).

    Attributes:
        x_train: Features de treino, shape (n_train, seq_len, n_feat).
        y_train: Targets de treino, shape (n_train, seq_len, n_tgt).
        z_train: Profundidade treino em metros, shape (n_train, seq_len).
        x_val: Features de validacao.
        y_val: Targets de validacao.
        z_val: Profundidade validacao em metros.
        x_test: Features de teste.
        y_test: Targets de teste.
        z_test: Profundidade teste em metros.
        train_model_ids: IDs dos modelos de treino.
        val_model_ids: IDs dos modelos de validacao.
        test_model_ids: IDs dos modelos de teste.

    Note:
        Referenciado em:
            - data/splitting.py: apply_split() (retorno)
            - data/splitting.py: split_angle_group() (retorno via apply_split)
            - data/pipeline.py: DataPipeline.prepare() (Step 3, destrutured
              para aplicar target_scaling e feature processing)
            - tests/test_data_pipeline.py: TestSplitting.test_apply_split_shapes
        Ref: docs/ARCHITECTURE_v2.md secao 4.2.
        z_train/z_val/z_test: profundidade em metros (NUNCA escalado).
        Bug fix v2.0: z_meters e campo separado (legado misturava no scaler).
        y_train/y_val/y_test: mutaveis — target_scaling e aplicado in-place
        no pipeline.py (DataPipeline.prepare, Step 4).
    """
    x_train: np.ndarray
    y_train: np.ndarray
    z_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    z_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    z_test: np.ndarray
    train_model_ids: Set[int]
    val_model_ids: Set[int]
    test_model_ids: Set[int]


# ════════════════════════════════════════════════════════════════════════
# SPLIT POR MODELO GEOLOGICO [P1]
#
# Principio fundamental: NUNCA split por amostra (row-wise).
# Cada modelo geologico gera multiplas sequencias de 600 pontos.
# O split deve ser feito na granularidade de MODELOS, nao de amostras.
#
# Diagrama — Split Model-Wise (zero leakage):
#
#   Modelos geologicos: [M0, M1, M2, M3, M4, M5, M6, M7, M8, M9]
#                        │    │   │    │    │    │    │    │   │    │
#   shuffle(seed=42):    M3  M7  M1   M9  M0   M5  M4   M8  M2  M6
#                        ├─────────────────────┤├───────┤├───────┤
#                               TRAIN (70%)      VAL(15%) TEST(15%)
#
#   Garantias:
#   ┌──────────────────────────────────────────────────────────────┐
#   │  train_ids ∩ val_ids  = ∅   (zero overlap)                  │
#   │  train_ids ∩ test_ids = ∅   (zero overlap)                  │
#   │  val_ids   ∩ test_ids = ∅   (zero overlap)                  │
#   │  |train| + |val| + |test| = N_MODELS (particao completa)   │
#   │  Cada sequencia pertence a EXATAMENTE UM split              │
#   └──────────────────────────────────────────────────────────────┘
#
#   Contraponto (PROIBIDO — split por amostra):
#     Sequencias do MESMO modelo em train e test → data leakage
#     A rede "memoriza" perfis de resistividade → metricas infladas
# ════════════════════════════════════════════════════════════════════════

def split_model_ids(
    n_models: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[Set[int], Set[int], Set[int]]:
    """Particiona IDs de modelos geologicos em train/val/test.

    Shuffle deterministico por seed. Nenhum modelo compartilhado
    entre particoes (zero data leakage garantido por assertions).

    O shuffle usa np.random.default_rng (gerador moderno do NumPy)
    para reproducibilidade independente do estado global do RNG.

    Args:
        n_models: Total de modelos geologicos.
        train_ratio: Fracao para treino.
        val_ratio: Fracao para validacao.
        test_ratio: Fracao para teste.
        seed: Semente para reproducao.

    Returns:
        Tupla (train_ids, val_ids, test_ids) como sets de inteiros.

    Raises:
        AssertionError: Se as particoes se sobrepuserem.

    Note:
        Referenciado em:
            - data/splitting.py: split_angle_group() (convenience wrapper)
            - data/pipeline.py: DataPipeline.prepare() (via split_angle_group,
              Step 3)
            - data/__init__.py: re-exportado como API publica
            - tests/test_data_pipeline.py: TestSplitting (5 test cases)
        Ref: docs/ARCHITECTURE_v2.md secao 4.2.
        [P1] Split por modelo geologico garante zero data leakage.
        NUNCA split por amostra — amostras do mesmo modelo vazam informacao.
        Ratios default: 70/15/15 (train/val/test) — configuravel via config.
        Seed default: 42 (global_seed do PipelineConfig).
        np.random.default_rng: gerador moderno, independente do estado global.
    """
    # Gerar permutacao deterministica dos IDs de modelos
    rng = np.random.default_rng(seed)
    indices = np.arange(n_models)
    rng.shuffle(indices)

    # Particionar por ratios (train pega o inicio, val o meio, test o fim)
    n_train = int(n_models * train_ratio)
    n_val = int(n_models * val_ratio)

    train_ids = set(indices[:n_train].tolist())
    val_ids = set(indices[n_train : n_train + n_val].tolist())
    test_ids = set(indices[n_train + n_val :].tolist())

    # Validacao: zero overlap — detecta bugs de particionamento
    assert train_ids & val_ids == set(), "Overlap train/val"
    assert train_ids & test_ids == set(), "Overlap train/test"
    assert val_ids & test_ids == set(), "Overlap val/test"

    logger.info(
        "Split modelos: train=%d, val=%d, test=%d (total=%d)",
        len(train_ids), len(val_ids), len(test_ids), n_models,
    )
    return train_ids, val_ids, test_ids


# ────────────────────────────────────────────────────────────────────────
# APPLY SPLIT — Mascara boolean por modelo geologico
#
# Dado um AngleGroup (todos os dados de um angulo theta), aplica
# mascaras booleanas para separar sequencias em train/val/test
# com base no model_id de cada sequencia.
#
# z_meters e preservado separadamente — sera usado para reconstruir
# perfis de profundidade na avaliacao, mas NUNCA passa pelo scaler.
# ────────────────────────────────────────────────────────────────────────

def apply_split(
    angle_group: AngleGroup,
    train_ids: Set[int],
    val_ids: Set[int],
    test_ids: Set[int],
) -> DataSplits:
    """Aplica split por modelo geologico a um AngleGroup.

    Cada sequencia e atribuida ao split do seu modelo geologico.
    z_meters e preservado separadamente (nunca entra no scaler).

    A operacao np.isin cria mascaras booleanas para selecionar
    sequencias pertencentes a cada particao de forma vetorizada.

    Args:
        angle_group: AngleGroup com x, y, z_meters, model_ids.
        train_ids: IDs de modelos para treino.
        val_ids: IDs de modelos para validacao.
        test_ids: IDs de modelos para teste.

    Returns:
        DataSplits com arrays particionados.

    Note:
        Referenciado em:
            - data/splitting.py: split_angle_group() (convenience wrapper)
            - data/pipeline.py: DataPipeline.prepare() (via split_angle_group,
              Step 3)
            - data/__init__.py: re-exportado como API publica
            - tests/test_data_pipeline.py: TestSplitting.test_apply_split_shapes,
              TestSplitting.test_z_meters_preserved
        Ref: docs/ARCHITECTURE_v2.md secao 4.2.
        np.isin: mascara booleana vetorizada (eficiente para grandes arrays).
        z_meters: preservado separadamente — NUNCA escalado pelo scaler.
        Cada sequencia pertence a EXATAMENTE um split (disjuntos).
    """
    model_ids = angle_group.model_ids

    # Mascaras booleanas: cada sequencia pertence a exatamente um split
    train_mask = np.isin(model_ids, sorted(train_ids))
    val_mask = np.isin(model_ids, sorted(val_ids))
    test_mask = np.isin(model_ids, sorted(test_ids))

    return DataSplits(
        x_train=angle_group.x[train_mask],
        y_train=angle_group.y[train_mask],
        z_train=angle_group.z_meters[train_mask],       # z em metros (NUNCA escalado)
        x_val=angle_group.x[val_mask],
        y_val=angle_group.y[val_mask],
        z_val=angle_group.z_meters[val_mask],            # z em metros (NUNCA escalado)
        x_test=angle_group.x[test_mask],
        y_test=angle_group.y[test_mask],
        z_test=angle_group.z_meters[test_mask],          # z em metros (NUNCA escalado)
        train_model_ids=train_ids,
        val_model_ids=val_ids,
        test_model_ids=test_ids,
    )


# ────────────────────────────────────────────────────────────────────────
# CONVENIENCIA — Split completo de um AngleGroup via config
#
# Combina split_model_ids + apply_split em uma unica chamada.
# Valida que model_ids sao contiguos [0, N-1] (requisito do formato
# de dados .dat, onde cada modelo geologico recebe ID sequencial).
# ────────────────────────────────────────────────────────────────────────

def split_angle_group(
    angle_group: AngleGroup,
    config: PipelineConfig,
) -> DataSplits:
    """Conveniencia: split de um AngleGroup usando config.

    Valida que model_ids sao contiguos [0, N-1] e delega para
    split_model_ids + apply_split.

    Args:
        angle_group: AngleGroup a ser particionado.
        config: PipelineConfig com ratios e seed.

    Returns:
        DataSplits com arrays particionados.

    Note:
        Referenciado em:
            - data/pipeline.py: DataPipeline.prepare() (Step 3, ponto
              de entrada para split no pipeline)
        Ref: docs/ARCHITECTURE_v2.md secao 4.2.
        Valida contiguidade [0, N-1] dos model_ids (requisito do formato
        .dat — segregate_by_angle atribui IDs sequenciais).
        Config utilizado: train_ratio, val_ratio, test_ratio, global_seed.
        Wrapper que combina split_model_ids + apply_split em uma unica chamada.
    """
    # Validar contiguidade dos model_ids (requisito do formato .dat)
    actual_ids = set(angle_group.model_ids.tolist())
    n_models = len(actual_ids)
    expected_ids = set(range(n_models))
    if actual_ids != expected_ids:
        raise ValueError(
            f"model_ids devem ser contiguos [0, N-1]. "
            f"Encontrados: {sorted(actual_ids)[:10]}..."
        )
    train_ids, val_ids, test_ids = split_model_ids(
        n_models=n_models,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.global_seed,
    )
    return apply_split(angle_group, train_ids, val_ids, test_ids)
