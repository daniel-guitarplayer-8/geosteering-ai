# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: data/pipeline.py                                                  ║
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
# ║    • Orquestracao completa: raw → split → fit_scaler → tf.data.map        ║
# ║    • Modo OFFLINE: FV+GS+scale aplicados estaticamente (val/test)          ║
# ║    • Modo ON-THE-FLY: noise → FV → GS → scale dentro de tf.data.map      ║
# ║    • DataPipeline como ponto unico de entrada para dados                   ║
# ║    • Bug fix v2.0: GS veem noise (fidelidade LWD), scaler fit em clean    ║
# ║                                                                            ║
# ║  Dependencias: config.py, data/loading.py, data/splitting.py,             ║
# ║                data/feature_views.py, data/geosignals.py, data/scaling.py ║
# ║  Exports: ~3 (PreparedData, DataPipeline, build_train_map_fn)             ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 4.3-4.4                               ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (migrado de C24-C25)          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

"""DataPipeline — Orquestrador da cadeia de dados fisicamente correta.

Dois modos automaticos baseados em config.needs_onthefly_fv_gs:

MODO OFFLINE (FV=identity, GS=off, OU noise=off):
    prepare(): FV+GS+scale offline em train/val/test
    train_map_fn: noise ONLY (sobre dados ja processados)

MODO ON-THE-FLY (FV ou GS ativos COM noise):
    prepare(): fit scaler em clean FV+GS (temporario), val/test offline, train=RAW
    train_map_fn: noise → FV_tf → GS_tf → scale_tf (cadeia completa)

Bug fix v2.0:
    - Resolve double-processing (C22 offline + C24 on-the-fly)
    - z_meters NUNCA entra no scaler
    - Scaling incluido no train_map_fn (legado nao fazia)

Referencia: docs/ARCHITECTURE_v2.md secao 4.3.

.. rubric:: D14 — Interacao Noise x FV x GS (Fidelidade Fisica LWD)

.. code-block:: text

    ┌──────────────────────────────────────────────────────────────────────────┐
    │  INTERACAO NOISE × FV × GS (Fidelidade Fisica LWD)                     │
    ├──────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  CORRETO (on-the-fly, dentro de tf.data.map):                           │
    │    raw Re/Im → noise(σ) → FV(noisy) → GS(noisy) → scale → modelo      │
    │                                    │                                     │
    │                          GS veem ruido ✓ (fidelidade LWD)               │
    │                                                                          │
    │  ERRADO (offline, bug do legado C22-C24):                               │
    │    raw Re/Im → FV(clean) → GS(clean) → scale → noise → modelo          │
    │                                    │                                     │
    │                          GS nunca veem ruido ✗ (bias sistematico)        │
    │                                                                          │
    │  REGRAS:                                                                 │
    │    1. Scaler SEMPRE fitado em dados LIMPOS (FV+GS clean, temporario)    │
    │    2. Val/test transformados offline (sem noise, FV+GS+scale)            │
    │    3. Train permanece raw para noise on-the-fly via tf.data.map()       │
    │    4. NUNCA aplicar noise offline quando FV ou GS estao ativos           │
    └──────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from geosteering_ai.config import PipelineConfig
from geosteering_ai.data.loading import (
    AngleGroup,
    OutMetadata,
    load_dataset,
    parse_out_metadata,
)
from geosteering_ai.data.splitting import DataSplits, split_angle_group, split_model_ids, apply_split
from geosteering_ai.data.feature_views import apply_feature_view, apply_feature_view_tf
from geosteering_ai.data.geosignals import (
    compute_expanded_features,
    compute_geosignals,
    compute_geosignals_tf,
)
from geosteering_ai.data.scaling import (
    apply_target_scaling,
    fit_per_group_scalers,
    fit_scaler,
    make_tf_scaler_fn,
    transform_features,
    transform_per_group,
)

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────
# __all__ — Exports publicos deste modulo
# ────────────────────────────────────────────────────────────────────────
__all__ = [
    "PreparedData",
    "DataPipeline",
]


# ════════════════════════════════════════════════════════════════════════
# RESULTADO DO PIPELINE — Container para dados preparados
#
# PreparedData encapsula todos os arrays prontos para tf.data.Dataset.
# z_meters SEMPRE preservado separadamente (nunca escalado).
#
# Conteudo:
#   - x_train: RAW (se on-the-fly) ou processado+scaled (se offline)
#   - x_val, x_test: SEMPRE processado+scaled (FV+GS+scaler)
#   - y_*: Targets com target_scaling aplicado (log10 por default)
#   - z_*: Profundidade em metros (para reconstrucao de perfis)
#   - scaler_em, scaler_gs: Scalers fitados em dados LIMPOS
#   - metadata: Informacoes do .out (n_models, theta_list, etc.)
# ════════════════════════════════════════════════════════════════════════

@dataclass
class PreparedData:
    """Dados preparados pelo DataPipeline.

    Todos os arrays estao prontos para alimentar tf.data.Dataset.
    z_meters SEMPRE preservado separadamente (nunca escalado).

    Attributes:
        x_train: Features treino (raw se on-the-fly, processado se offline).
        y_train: Targets treino (target-scaled).
        z_train: Profundidade treino em metros (nunca escalado).
        x_val: Features validacao (processado + scaled).
        y_val: Targets validacao (target-scaled).
        z_val: Profundidade validacao em metros.
        x_test: Features teste (processado + scaled).
        y_test: Targets teste (target-scaled).
        z_test: Profundidade teste em metros.
        scaler_em: Scaler fitado em features EM.
        scaler_gs: Scaler fitado em geosinais (ou None).
        n_em_features: Numero de features EM (sem GS).
        expanded_features: Colunas expandidas (se GS ativo).
        metadata: Metadados do .out.

    Note:
        Referenciado em:
            - data/pipeline.py: DataPipeline.prepare() (retorno principal)
            - data/pipeline.py: DataPipeline._prepare_offline() / _prepare_onthefly()
            - training/loop.py: TrainingLoop consome x_train, y_train, z_train
            - inference/pipeline.py: InferencePipeline usa scaler_em, scaler_gs
        Ref: docs/ARCHITECTURE_v2.md secao 4.3 (Data Container).
        Scalers sao fitados em dados LIMPOS (regra absoluta P3).
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
    scaler_em: Optional[Any] = None
    scaler_gs: Optional[Any] = None
    n_em_features: int = 0
    expanded_features: Optional[List[int]] = None
    metadata: Optional[OutMetadata] = None


# ════════════════════════════════════════════════════════════════════════
# DATA PIPELINE — Orquestrador unificado
#
# Ponto unico de entrada para toda a cadeia de dados.
# Encapsula: load → split → FV → GS → scale → tf.data.map
#
# Dois modos de operacao (automaticos via config.needs_onthefly_fv_gs):
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │  MODO OFFLINE (noise=off OU FV=identity sem GS)                   │
# │                                                                     │
# │  prepare():                                                         │
# │    .dat/.out ──→ load ──→ split[P1] ──→ target_scaling             │
# │                                           │                         │
# │    train_clean ──→ FV ──→ GS ──→ fit_scaler ──→ scale ──→ x_train │
# │    val_clean   ──→ FV ──→ GS ──→ scale ──→ x_val                  │
# │    test_clean  ──→ FV ──→ GS ──→ scale ──→ x_test                 │
# │                                                                     │
# │  train_map_fn:                                                      │
# │    x_scaled ──→ noise (se ativo) ──→ x_out                        │
# │                                                                     │
# ├─────────────────────────────────────────────────────────────────────┤
# │  MODO ON-THE-FLY (FV ou GS ativos COM noise)                      │
# │                                                                     │
# │  prepare():                                                         │
# │    .dat/.out ──→ load ──→ split[P1] ──→ target_scaling             │
# │                                           │                         │
# │    train_clean ──→ FV ──→ GS ──→ fit_scaler (temporario, descarta)│
# │    val_clean   ──→ FV ──→ GS ──→ scale ──→ x_val                  │
# │    test_clean  ──→ FV ──→ GS ──→ scale ──→ x_test                 │
# │    x_train = RAW (sem processamento)                               │
# │                                                                     │
# │  train_map_fn (dentro de tf.data.map, CADA batch):                 │
# │    x_raw ──→ noise(A/m) ──→ FV_tf(noisy) ──→ GS_tf(noisy) ──→    │
# │         ──→ scale_tf ──→ x_out                                     │
# │              │                    │                                  │
# │         GS veem noise ✓      scaler fitado em clean ✓              │
# │         (fidelidade LWD)     (distribuicao real)                    │
# │                                                                     │
# │  CADEIA FISICAMENTE CORRETA:                                       │
# │    Noise ANTES de FV+GS → GS calculados sobre dados ruidosos      │
# │    (simula condicoes reais de aquisicao LWD em poco)               │
# └─────────────────────────────────────────────────────────────────────┘
# ════════════════════════════════════════════════════════════════════════

class DataPipeline:
    """Pipeline de dados unificado — cadeia fisicamente correta.

    Orquestra toda a cadeia de preparacao de dados, desde o carregamento
    dos arquivos .dat/.out ate a construcao de funcoes map para tf.data.
    Garante que:
        - Split e feito por modelo geologico [P1], nunca por amostra
        - Scaler e fitado em dados LIMPOS (sem noise)
        - GS veem noise no modo on-the-fly (fidelidade LWD)
        - z_meters e preservado e NUNCA escalado

    Example:
        >>> config = PipelineConfig.robusto()
        >>> pipeline = DataPipeline(config)
        >>> data = pipeline.prepare("/path/to/dat", "/path/to/out")
        >>> map_fn = pipeline.build_train_map_fn(noise_level_var)
        >>> train_ds = train_ds.map(map_fn)
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._families: List[str] = []
        self._expanded_features: List[int] = []
        self._n_em_features: int = config.n_base_features

        # Pre-computar familias de geosinais se ativo
        if config.use_geosignal_features:
            self._families = config.resolve_families()
            self._expanded_features = compute_expanded_features(
                config.input_features, self._families,
            )

    @property
    def is_onthefly(self) -> bool:
        """True se FV/GS devem ser computados on-the-fly (pos-noise)."""
        return self.config.needs_onthefly_fv_gs

    # ──────────────────────────────────────────────────────────────────
    # PREPARE: Carrega → Split → FV+GS (clean) → Fit Scaler → Transform
    #
    # Ponto de entrada principal. Executa a cadeia completa de
    # preparacao de dados, decidindo automaticamente entre modo
    # offline e on-the-fly com base na configuracao.
    #
    # Etapas:
    #   1. Load .dat + parse .out + decoupling (loading.py)
    #   2. Selecionar angulo theta (default: primeiro da lista)
    #   3. Split por modelo geologico [P1] (splitting.py)
    #   4. Target scaling em y (log10 por default)
    #   5-7. Feature processing (offline ou on-the-fly)
    # ──────────────────────────────────────────────────────────────────

    def prepare(
        self,
        dat_path: str,
        out_path: str,
        theta_filter: Optional[float] = None,
    ) -> PreparedData:
        """Executa pipeline completo de preparacao de dados.

        Etapas:
            1. Load .dat + parse .out + decoupling
            2. Selecionar angulo (default: theta=0)
            3. Split por modelo geologico [P1]
            4. Target scaling (log10)
            5. FV+GS em clean → fit scaler (temporario)
            6. Val/Test: FV+GS+scale offline
            7. Train: RAW se on-the-fly, FV+GS+scale se offline

        Args:
            dat_path: Caminho para o .dat.
            out_path: Caminho para o .out.
            theta_filter: Angulo a processar (default: primeiro).

        Returns:
            PreparedData com arrays prontos.

        Note:
            Referenciado em:
                - training/loop.py: TrainingLoop.run() chama prepare() antes do fit
                - data/scaling.py: fit_scaler(), fit_per_group_scalers() invocados aqui
                - data/splitting.py: split_angle_group() garante split por modelo [P1]
                - data/loading.py: load_dataset() + parse_out_metadata() (Steps 1-2)
            Ref: docs/ARCHITECTURE_v2.md secao 4.3 (DataPipeline.prepare).
            A decisao offline vs on-the-fly e automatica via config.needs_onthefly_fv_gs.
        """
        # ── STEP 1: Load .dat + parse .out ────────────────────────────
        angle_groups = load_dataset(dat_path, out_path, self.config)
        metadata = parse_out_metadata(out_path)

        # ── STEP 2: Selecionar angulo theta ───────────────────────────
        if theta_filter is not None:
            group = angle_groups[theta_filter]
        else:
            first_theta = metadata.theta_list[0]
            group = angle_groups[first_theta]
        logger.info("Processando angulo %.1f°", group.theta)

        # ── STEP 3: Split por modelo geologico [P1] ──────────────────
        # Garante zero data leakage entre train/val/test
        splits = split_angle_group(group, self.config)

        # ── STEP 4: Target scaling ────────────────────────────────────
        # Transforma resistividade (Ohm.m) para dominio comprimido
        # Default: log10 comprime [0.3, 10000] → [-0.52, 4.0]
        splits.y_train = apply_target_scaling(splits.y_train, self.config.target_scaling)
        splits.y_val = apply_target_scaling(splits.y_val, self.config.target_scaling)
        splits.y_test = apply_target_scaling(splits.y_test, self.config.target_scaling)

        # ── STEPS 5-7: Feature processing + scaling ──────────────────
        # Decisao automatica: offline vs on-the-fly
        if self.is_onthefly:
            data = self._prepare_onthefly(splits, metadata)
        else:
            data = self._prepare_offline(splits, metadata)

        # Armazenar para build_train_map_fn() acessar scalers
        self._last_prepared = data
        return data

    def _apply_fv_gs(self, x: np.ndarray, raw_data_2d: Optional[np.ndarray] = None) -> np.ndarray:
        """Aplica Feature View e concatena Geosinais (numpy).

        Feature View seleciona/transforma colunas EM relevantes.
        Geosinais computam ratios e diferencas diagnosticas sobre
        as componentes EM (ex: attenuation, phase difference).

        Args:
            x: Features 3D (n_seq, seq_len, n_feat).
            raw_data_2d: Dados raw 2D para geosinais (se None, usa x reshapado).

        Returns:
            Array com FV aplicado e GS concatenados.
        """
        # Feature View — seleciona/transforma colunas EM
        x_fv = apply_feature_view(x, self.config.feature_view)

        # Geosinais — ratios e diferencas diagnosticas sobre EM
        if self.config.use_geosignal_features and self._families:
            n_seq, seq_len, _ = x.shape
            # GS sao computados sobre dados com colunas originais do .dat,
            # nao sobre FV-transformados (preserva relacao fisica entre componentes)
            if raw_data_2d is not None:
                gs = compute_geosignals(raw_data_2d, self._families, self.config.n_columns)
                gs_3d = gs.reshape(n_seq, seq_len, -1)
            else:
                logger.warning(
                    "raw_data_2d nao fornecido para geosinais — "
                    "GS serao computados sobre dados pos-FV (menos preciso)"
                )
                gs_3d = np.zeros((n_seq, seq_len, 0), dtype=x.dtype)
            # Concatenar: [FV features | GS features]
            x_fv = np.concatenate([x_fv, gs_3d], axis=-1)

        return x_fv

    # ──────────────────────────────────────────────────────────────────
    # MODO OFFLINE — FV+GS+scale aplicados estaticamente
    #
    # Usado quando noise=off OU FV=identity sem GS.
    # Todos os splits (train, val, test) recebem o mesmo tratamento:
    #   FV → (GS) → fit_scaler(train) → scale(todos)
    # ──────────────────────────────────────────────────────────────────

    def _prepare_offline(self, splits: DataSplits, metadata: OutMetadata) -> PreparedData:
        """Modo offline: FV+GS+scale aplicados estaticamente.

        Usado quando noise=off OU FV=identity sem GS.
        Todos os splits sao processados de forma identica.
        """
        # FV em todos os splits (seleciona colunas EM relevantes)
        x_train = apply_feature_view(splits.x_train, self.config.feature_view)
        x_val = apply_feature_view(splits.x_val, self.config.feature_view)
        x_test = apply_feature_view(splits.x_test, self.config.feature_view)

        n_em = x_train.shape[-1]

        # Fit scaler em train LIMPO, transform em todos os splits
        if self.config.use_per_group_scalers and self.config.use_geosignal_features:
            # Per-group [P3]: StandardScaler(EM) + RobustScaler(GS)
            scaler_em, scaler_gs = fit_per_group_scalers(x_train, self.config, n_em)
            x_train = transform_per_group(x_train, scaler_em, scaler_gs, n_em)
            x_val = transform_per_group(x_val, scaler_em, scaler_gs, n_em)
            x_test = transform_per_group(x_test, scaler_em, scaler_gs, n_em)
        else:
            # Scaler unico para todas as features
            scaler_em = fit_scaler(x_train, self.config.scaler_type)
            scaler_gs = None
            x_train = transform_features(x_train, scaler_em)
            x_val = transform_features(x_val, scaler_em)
            x_test = transform_features(x_test, scaler_em)

        return PreparedData(
            x_train=x_train.astype(np.float32),
            y_train=splits.y_train.astype(np.float32),
            z_train=splits.z_train,                       # z em metros (NUNCA escalado)
            x_val=x_val.astype(np.float32),
            y_val=splits.y_val.astype(np.float32),
            z_val=splits.z_val,                           # z em metros (NUNCA escalado)
            x_test=x_test.astype(np.float32),
            y_test=splits.y_test.astype(np.float32),
            z_test=splits.z_test,                         # z em metros (NUNCA escalado)
            scaler_em=scaler_em,
            scaler_gs=scaler_gs,
            n_em_features=n_em,
            expanded_features=self._expanded_features or None,
            metadata=metadata,
        )

    # ──────────────────────────────────────────────────────────────────
    # MODO ON-THE-FLY — train permanece RAW, FV+GS no train_map_fn
    #
    # Cadeia fisicamente correta:
    #   noise(raw_em) → FV(noisy) → GS(noisy) → scale
    #
    # O scaler e fitado em dados LIMPOS (FV+GS clean, temporario).
    # Apos capturar as estatisticas, os dados temporarios sao
    # descartados (del). O train permanece RAW para que noise seja
    # aplicado on-the-fly (diferente a cada epoch — curriculum).
    #
    # Val/test sao processados OFFLINE (sem noise).
    # ──────────────────────────────────────────────────────────────────

    def _prepare_onthefly(self, splits: DataSplits, metadata: OutMetadata) -> PreparedData:
        """Modo on-the-fly: train fica RAW, FV+GS no train_map_fn.

        Cadeia on-the-fly: noise(raw_em) → FV(noisy) → GS(noisy) → scale
        Scaler fitado em dados limpos (FV+GS clean, temporario).
        Val/test processados offline.
        """
        # Fit scaler em CLEAN train (FV+GS aplicados temporariamente)
        # Estes dados temporarios so existem para capturar mu/sigma do scaler
        x_train_clean_fv = apply_feature_view(splits.x_train, self.config.feature_view)
        n_em = x_train_clean_fv.shape[-1]

        # Se GS ativo, computar GS em clean para fitar scaler_gs
        scaler_gs = None
        if self.config.use_per_group_scalers and self._families:
            x_train_clean_full = self._apply_fv_gs(splits.x_train)
            scaler_em, scaler_gs = fit_per_group_scalers(
                x_train_clean_full, self.config, n_em,
            )
            del x_train_clean_full  # temporario — descartado apos fit
        else:
            scaler_em = fit_scaler(x_train_clean_fv, self.config.scaler_type)
        del x_train_clean_fv  # temporario — scaler ja capturou estatisticas

        # Val/Test: FV+GS+scale OFFLINE (sem noise)
        x_val_fv = self._apply_fv_gs(splits.x_val) if self._families else apply_feature_view(splits.x_val, self.config.feature_view)
        x_test_fv = self._apply_fv_gs(splits.x_test) if self._families else apply_feature_view(splits.x_test, self.config.feature_view)

        if scaler_gs is not None:
            x_val_fv = transform_per_group(x_val_fv, scaler_em, scaler_gs, n_em)
            x_test_fv = transform_per_group(x_test_fv, scaler_em, scaler_gs, n_em)
        else:
            x_val_fv = transform_features(x_val_fv, scaler_em)
            x_test_fv = transform_features(x_test_fv, scaler_em)

        # Train: permanece RAW (noise+FV+GS+scale serao on-the-fly via train_map_fn)
        return PreparedData(
            x_train=splits.x_train.astype(np.float32),  # RAW! Processado on-the-fly
            y_train=splits.y_train.astype(np.float32),
            z_train=splits.z_train,                       # z em metros (NUNCA escalado)
            x_val=x_val_fv.astype(np.float32),
            y_val=splits.y_val.astype(np.float32),
            z_val=splits.z_val,                           # z em metros (NUNCA escalado)
            x_test=x_test_fv.astype(np.float32),
            y_test=splits.y_test.astype(np.float32),
            z_test=splits.z_test,                         # z em metros (NUNCA escalado)
            scaler_em=scaler_em,
            scaler_gs=scaler_gs,
            n_em_features=n_em,
            expanded_features=self._expanded_features or None,
            metadata=metadata,
        )

    # ──────────────────────────────────────────────────────────────────
    # TRAIN MAP FN — Cadeia on-the-fly via closure TF
    #
    # Constroi funcao para tf.data.Dataset.map() que executa a cadeia
    # fisicamente correta dentro do grafo TF:
    #
    #   x_raw ──→ noise(A/m) ──→ FV_tf(noisy) ──→ GS_tf(noisy)
    #                                                   │
    #                              scale_tf(scaler fitado em clean)
    #                                                   │
    #                                              x_processed, y
    #
    # noise_level_var e compartilhado com o curriculum callback,
    # permitindo rampa 3-fase: limpo → rampa → estavel.
    # ──────────────────────────────────────────────────────────────────

    def build_train_map_fn(
        self,
        noise_level_var: "tf.Variable",
    ) -> Callable:
        """Constroi funcao on-the-fly para tf.data.Dataset.map().

        Cadeia fisicamente correta:
            noise(raw_em) → FV_tf(noisy) → GS_tf(noisy) → scale_tf

        O noise_level_var e um tf.Variable compartilhado com o curriculum
        callback, que controla a intensidade do ruido ao longo do
        treinamento (rampa 3-fase: clean → ramp → stable).

        Args:
            noise_level_var: tf.Variable compartilhado com curriculum callback.

        Returns:
            Funcao (x, y) → (x_processed, y) para tf.data.map().

        Note:
            Referenciado em:
                - training/loop.py: TrainingLoop.run() chama build_train_map_fn()
                - noise/curriculum.py: UpdateNoiseLevelCallback controla noise_level_var
                - data/geosignals.py: compute_geosignals_tf() (Step 3)
                - data/feature_views.py: apply_feature_view_tf() (Step 2)
                - data/scaling.py: make_tf_scaler_fn() (Step 4)
            Ref: docs/ARCHITECTURE_v2.md secao 4.4 (On-the-fly Pipeline).
            Requer prepare() chamado previamente (scalers fitados em dados limpos).
        """
        import tensorflow as tf

        config = self.config
        families = self._families
        expanded_features = self._expanded_features

        # Scalers devem ter sido fitados em prepare()
        if not hasattr(self, "_last_prepared"):
            raise RuntimeError(
                "Chame prepare() antes de build_train_map_fn(). "
                "Os scalers precisam ser fitados em dados limpos primeiro."
            )
        # Converter sklearn scalers para closures TF puras (constantes)
        scale_em_fn = make_tf_scaler_fn(self._last_prepared.scaler_em)
        scale_gs_fn = make_tf_scaler_fn(self._last_prepared.scaler_gs)
        n_em = self._last_prepared.n_em_features

        # ┌──────────────────────────────────────────────────────────────────────────┐
        # │  INTERACAO NOISE × FV × GS (Fidelidade Fisica LWD)                     │
        # ├──────────────────────────────────────────────────────────────────────────┤
        # │                                                                          │
        # │  CORRETO (on-the-fly, dentro de tf.data.map):                           │
        # │    raw Re/Im → noise(σ) → FV(noisy) → GS(noisy) → scale → modelo      │
        # │                                    │                                     │
        # │                          GS veem ruido ✓ (fidelidade LWD)               │
        # │                                                                          │
        # │  ERRADO (offline, bug do legado C22-C24):                               │
        # │    raw Re/Im → FV(clean) → GS(clean) → scale → noise → modelo          │
        # │                                    │                                     │
        # │                          GS nunca veem ruido ✗ (bias sistematico)        │
        # │                                                                          │
        # │  REGRAS:                                                                 │
        # │    1. Scaler SEMPRE fitado em dados LIMPOS (FV+GS clean, temporario)    │
        # │    2. Val/test transformados offline (sem noise, FV+GS+scale)            │
        # │    3. Train permanece raw para noise on-the-fly via tf.data.map()       │
        # │    4. NUNCA aplicar noise offline quando FV ou GS estao ativos           │
        # └──────────────────────────────────────────────────────────────────────────┘

        def train_map_fn(x: tf.Tensor, y: tf.Tensor) -> tuple:
            # ── Step 1: Noise injection (raw Re/Im → noisy Re/Im) ───────────
            # Nivel de ruido controlado por noise_var (curriculum 3-fase)
            # Ruido gaussiano aditivo simula incerteza de medicao LWD.
            # Intensidade controlada por noise_level_var (curriculum).
            if config.use_noise:
                noise = tf.random.normal(
                    shape=tf.shape(x),
                    mean=0.0,
                    stddev=noise_level_var,
                    dtype=tf.float32,
                )
                # Noise aditivo sobre componentes EM raw (A/m)
                x = x + noise

            # ── Step 2: Feature View (noisy Re/Im → FV channels) ────────────
            # Saida: [prefix, z, FV_chan0, FV_chan1, FV_chan2, FV_chan3]
            # FV transforma/seleciona colunas EM APOS noise
            # (noise aplicado sobre dados raw, FV ve dados ruidosos)
            if config.feature_view not in ("identity", "raw"):
                x = apply_feature_view_tf(
                    x, view=config.feature_view, eps=config.eps_tf,
                )

            # ── Step 3: Geosignals (noisy EM → att + phase por familia) ──────
            # Saida: concatena GS channels apos FV channels
            # GS computados APOS noise → GS refletem condicoes LWD reais
            # (attenuation e phase difference calculados sobre sinal ruidoso)
            if config.use_geosignal_features and families:
                gs = compute_geosignals_tf(
                    x, families, expanded_features, eps=config.eps_tf,
                )
                x = tf.concat([x, gs], axis=-1)

            # ── Step 4: Scaling (features normalizadas para modelo) ──────────
            # Usa scaler fitado em dados LIMPOS (regra absoluta)
            # Aplica normalizacao usando estatisticas de dados LIMPOS.
            # Particiona EM e GS para per-group scaling [P3].
            x_em = x[:, :, :n_em]
            x_em = scale_em_fn(x_em)

            if config.use_geosignal_features and x.shape[-1] > n_em:
                x_gs = x[:, :, n_em:]
                x_gs = scale_gs_fn(x_gs)
                x = tf.concat([x_em, x_gs], axis=-1)
            else:
                x = x_em

            return x, y

        return train_map_fn

    # ──────────────────────────────────────────────────────────────────
    # VAL NOISE MAP FN — Noise para dual validation [P2]
    #
    # No modo dual validation, val_noisy_ds recebe noise sobre dados
    # ja processados (FV+GS+scale aplicados offline). Isso permite
    # monitorar a degradacao causada pelo ruido separadamente da
    # convergencia em dados limpos.
    # ──────────────────────────────────────────────────────────────────

    def build_val_noise_map_fn(
        self,
        noise_level_var: "tf.Variable",
    ) -> Callable:
        """Constroi funcao de noise para val_noisy_ds (dual validation P2).

        Aplica apenas noise (sem FV/GS/scale — val ja esta processado).
        Usado para monitorar robustez a ruido durante treinamento.

        Args:
            noise_level_var: tf.Variable compartilhado com curriculum.

        Returns:
            Funcao (x, y) → (x_noisy, y).
        """
        import tensorflow as tf

        def val_noise_fn(x: tf.Tensor, y: tf.Tensor) -> tuple:
            # Noise gaussiano aditivo sobre dados ja normalizados
            noise = tf.random.normal(
                shape=tf.shape(x),
                mean=0.0,
                stddev=noise_level_var,
                dtype=tf.float32,
            )
            return x + noise, y

        return val_noise_fn
