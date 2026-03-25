# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: data/scaling.py                                                   ║
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
# ║    • Target scaling: 8 metodos (log10, sqrt, asinh, etc.)                 ║
# ║    • Feature scalers: 8 tipos (standard, minmax, robust, etc.)            ║
# ║    • Per-group scalers [P3]: StandardScaler(EM) + RobustScaler(GS)        ║
# ║    • TensorFlow scaler fn para on-the-fly dentro de tf.data.map           ║
# ║    • Bug fix v2.0: z_meters preservado (NUNCA escalado pelo scaler)       ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig), scikit-learn                   ║
# ║  Exports: ~8 funcoes (apply_target_scaling, inverse_target_scaling,       ║
# ║           create_scaler, fit_scaler, transform_features,                  ║
# ║           fit_per_group_scalers, transform_per_group, make_tf_scaler_fn)  ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 4.3                                  ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (migrado de C23)             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

"""Scaling — Normalizacao de features e targets.

Scaler fitado em dados LIMPOS (sem noise). z_meters preservado
separadamente e NUNCA entra no scaler EM.

Bug fix v2.0: No legado (C24), x_test_clean era copiado pos-scaling
(zobs normalizado). Agora z_meters e um campo separado em DataSplits.

Referencia: docs/ARCHITECTURE_v2.md secao 5.1.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Tuple

import numpy as np

from geosteering_ai.config import PipelineConfig

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────
# __all__ — Exports publicos deste modulo
# ────────────────────────────────────────────────────────────────────────
__all__ = [
    "apply_target_scaling",
    "inverse_target_scaling",
    "create_scaler",
    "fit_scaler",
    "transform_features",
    "fit_per_group_scalers",
    "transform_per_group",
    "make_tf_scaler_fn",
]


# ════════════════════════════════════════════════════════════════════════
# TARGET SCALING (resistividade → dominio transformado)
#
# Resistividades variam em ordens de magnitude (0.3 a 10000+ Ohm.m).
# Sem transformacao, a rede sofreria com gradientes dominados por
# amostras de alta resistividade. log10 comprime a faixa dinamica.
#
# Metodos disponiveis (8 opcoes, configurados via config.target_scaling):
#
#   ┌──────────┬──────────────────────────────────────┬─────────────────────┐
#   │ Metodo   │ Transformacao (forward)           │ Inversa             │
#   ├──────────┼──────────────────────────────────────┼─────────────────────┤
#   │ log10    │ y' = log10(max(y, eps))          │ y = 10^y'           │
#   │ none     │ y' = y                           │ y = y'              │
#   │ linear   │ y' = y (alias de "none")         │ y = y'              │
#   │ sqrt     │ y' = sqrt(max(y, 0))             │ y = y'^2            │
#   │ cbrt     │ y' = cbrt(y)                     │ y = y'^3            │
#   │ asinh    │ y' = arcsinh(y)                  │ y = sinh(y')        │
#   │ yj       │ Yeo-Johnson (requer estado)      │ (via transformer)   │
#   │ pt       │ PowerTransformer (requer estado)  │ (via transformer)   │
#   └──────────┴──────────────────────────────────────┴─────────────────────┘
#
#   ERRATA: TARGET_SCALING = "log10" (NUNCA "log")
#   ERRATA: eps = 1e-12 para float32 (NUNCA 1e-30)
# ════════════════════════════════════════════════════════════════════════

def apply_target_scaling(
    y: np.ndarray,
    method: str = "log10",
) -> np.ndarray:
    """Aplica transformacao nos targets de resistividade.

    Resistividade (rho, Ohm.m) possui faixa dinamica de ~4 ordens de
    magnitude. A transformacao comprime essa faixa para facilitar o
    aprendizado da rede neural (gradientes mais estaveis).

    Args:
        y: Array de resistividades (Ohm.m), qualquer shape.
        method: Metodo de scaling. Default: "log10" (errata).

    Returns:
        Array transformado com mesma shape.

    Note:
        Referenciado em:
            - data/pipeline.py: DataPipeline.prepare() (Step 4, aplicado
              em y_train, y_val, y_test)
            - data/scaling.py: inverse_target_scaling() (operacao inversa)
            - data/__init__.py: re-exportado como API publica
            - tests/test_data_pipeline.py: TestTargetScaling (3 test cases)
        Ref: docs/ARCHITECTURE_v2.md secao 4.3.
        Default: "log10" (comprime 4 ordens de magnitude de resistividade).
        NUNCA usar "log" (ln) como default — Errata v5.0.15.
        eps = 1e-12 para float32 (NUNCA 1e-30 — Errata v5.0.15).
    """
    if method == "log10":
        # ── log10: Comprime resistividade de [0.1, 10000] Ohm.m para [-1, 4].
        #    Motivacao: Resistividade varia 4+ ordens de magnitude.
        #    Saida: log10(y) onde y = resistividade em Ohm.m.
        #    eps=1e-12 evita log10(0) — ERRATA: NUNCA usar 1e-30.
        return np.log10(np.maximum(y, 1e-12))
    elif method == "none" or method == "linear":
        # ── none/linear: Sem transformacao (identity).
        #    Resistividade permanece em Ohm.m (escala original).
        #    Util para debugging ou quando loss ja e scale-aware.
        return y.copy()
    elif method == "sqrt":
        # ── sqrt: Compressao moderada, preserva monotonicidade.
        #    Comprime [0, 10000] para [0, 100].
        #    Menos agressivo que log10, mas gradientes ainda instáveis
        #    para resistividades muito altas.
        return np.sqrt(np.maximum(y, 0.0))
    elif method == "cbrt":
        # ── cbrt: Compressao mais suave que sqrt, aceita negativos.
        #    Comprime [0, 10000] para [0, 21.5].
        #    Aceita y < 0 (util se targets forem pre-processados).
        return np.cbrt(y)
    elif method == "asinh":
        # ── asinh: Aproxima log para |rho|>>1, linear perto de 0.
        #    Compressao adaptativa: log-like para valores grandes,
        #    linear para valores proximos de zero.
        #    Nao requer clipping (definida para todo R).
        return np.arcsinh(y)
    elif method in ("yj", "pt"):
        raise NotImplementedError(
            f"target_scaling '{method}' requer PowerTransformer com estado "
            "(lambda fitado). Use config.target_scaling='log10' ou implemente "
            "via ScalerRegistry com persistencia do transformer."
        )
    else:
        raise ValueError(f"target_scaling '{method}' invalido")


def inverse_target_scaling(
    y_scaled: np.ndarray,
    method: str = "log10",
) -> np.ndarray:
    """Inverte transformacao de targets para Ohm.m.

    Usada na inferencia para converter predicoes da rede de volta
    ao dominio fisico (resistividade em Ohm.m).

    Args:
        y_scaled: Array transformado.
        method: Metodo usado na transformacao.

    Returns:
        Array em escala original (Ohm.m).

    Note:
        Referenciado em:
            - data/scaling.py: apply_target_scaling() (par forward/inverse)
            - data/__init__.py: re-exportado como API publica
            - tests/test_data_pipeline.py: TestTargetScaling (roundtrip tests)
        Ref: docs/ARCHITECTURE_v2.md secao 4.3.
        Usada na avaliacao (C48+) e InferencePipeline (P6) para converter
        predicoes de volta ao dominio fisico (Ohm.m).
        NUNCA operar no dominio scaled para metricas fisicas — sempre inverter.
    """
    if method == "log10":
        # ── log10 inversa: 10^y' restaura resistividade em Ohm.m.
        #    Input: log10(rho), Output: rho em Ohm.m.
        return np.power(10.0, y_scaled)
    elif method == "none" or method == "linear":
        # ── none/linear inversa: identity (copia).
        return y_scaled.copy()
    elif method == "sqrt":
        # ── sqrt inversa: y'^2 restaura escala original.
        return np.square(y_scaled)
    elif method == "cbrt":
        # ── cbrt inversa: y'^3 restaura escala original.
        return np.power(y_scaled, 3.0)
    elif method == "asinh":
        # ── asinh inversa: sinh(y') restaura escala original.
        return np.sinh(y_scaled)
    else:
        raise ValueError(f"Inversa para '{method}' nao implementada")


# ════════════════════════════════════════════════════════════════════════
# FEATURE SCALERS (sklearn)
#
# Normaliza features de entrada para facilitar convergencia.
# O scaler e SEMPRE fitado em dados LIMPOS (sem noise) para capturar
# a distribuicao real dos sinais EM. z_meters NUNCA passa pelo scaler.
#
# Tipos disponiveis (8 opcoes, configurados via config.scaler_type):
#
#   ┌─────────────┬────────────────────────────────────────────────────┐
#   │ Tipo        │ Descricao                                         │
#   ├─────────────┼────────────────────────────────────────────────────┤
#   │ none        │ Sem scaling (identity)                            │
#   │ standard    │ StandardScaler: (x - mu) / sigma                  │
#   │ minmax      │ MinMaxScaler: (x - min) / (max - min) → [0,1]    │
#   │ robust      │ RobustScaler: (x - median) / IQR                 │
#   │ maxabs      │ MaxAbsScaler: x / max(|x|) → [-1,1]              │
#   │ quantile    │ QuantileTransformer: → distribuicao uniforme       │
#   │ power       │ PowerTransformer: Yeo-Johnson + standardize       │
#   │ normalizer  │ Normalizer: norma L2 por amostra                  │
#   └─────────────┴────────────────────────────────────────────────────┘
#
#   Per-group [P3]: EM usa config.scaler_type (default: standard)
#                   GS usa config.gs_scaler_type (default: robust)
#   Motivacao: GS tem ranges muito diferentes de EM raw (ratios, diffs)
# ════════════════════════════════════════════════════════════════════════

def create_scaler(scaler_type: str = "standard") -> Optional[Any]:
    """Cria scaler sklearn pelo tipo.

    Lazy import: sklearn so e importado quando scaler e criado,
    permitindo que modulos sem sklearn importem scaling.py.

    Args:
        scaler_type: Tipo do scaler (8 opcoes).

    Returns:
        Instancia do scaler, ou None para "none".

    Note:
        Referenciado em:
            - data/scaling.py: fit_scaler() (cria scaler antes de fitar)
            - tests/test_data_pipeline.py: TestScalers.test_all_scaler_types
        Ref: docs/ARCHITECTURE_v2.md secao 4.3.
        Lazy import: sklearn nao e dependencia obrigatoria do pacote.
        Validacao do scaler_type e feita em PipelineConfig.__post_init__().
        8 tipos validos: none, standard, minmax, robust, maxabs, quantile,
        power, normalizer.
    """
    if scaler_type == "none":
        # ── none: Sem scaling (identity). Retorna None.
        #    Usado quando features ja estao normalizadas externamente
        #    ou para debugging sem normalizacao.
        return None
    elif scaler_type == "standard":
        # ── standard: StandardScaler — (x - mu) / sigma.
        #    Centraliza em zero, variancia unitaria.
        #    Default para features EM (componentes do tensor H).
        from sklearn.preprocessing import StandardScaler
        return StandardScaler()
    elif scaler_type == "minmax":
        # ── minmax: MinMaxScaler — (x - min) / (max - min) → [0, 1].
        #    Preserva zeros esparsos. Sensivel a outliers.
        from sklearn.preprocessing import MinMaxScaler
        return MinMaxScaler(feature_range=(0, 1))
    elif scaler_type == "robust":
        # ── robust: RobustScaler — (x - median) / IQR.
        #    Robusto a outliers (usa mediana e interquartil).
        #    Default para geosinais (GS) via gs_scaler_type.
        from sklearn.preprocessing import RobustScaler
        return RobustScaler()
    elif scaler_type == "maxabs":
        # ── maxabs: MaxAbsScaler — x / max(|x|) → [-1, 1].
        #    Preserva zeros. Util para dados esparsos.
        from sklearn.preprocessing import MaxAbsScaler
        return MaxAbsScaler()
    elif scaler_type == "quantile":
        # ── quantile: QuantileTransformer → distribuicao uniforme.
        #    Mapeia para distribuicao uniforme [0, 1] via quantis.
        #    Remove efeito de outliers completamente.
        from sklearn.preprocessing import QuantileTransformer
        return QuantileTransformer(output_distribution="uniform")
    elif scaler_type == "power":
        # ── power: PowerTransformer (Yeo-Johnson + standardize).
        #    Transforma para distribuicao gaussiana via Yeo-Johnson.
        #    Aceita valores negativos (diferente de Box-Cox).
        from sklearn.preprocessing import PowerTransformer
        return PowerTransformer(method="yeo-johnson", standardize=True)
    elif scaler_type == "normalizer":
        # ── normalizer: Normalizer (norma L2 por amostra).
        #    Cada amostra e normalizada para norma unitaria.
        #    Diferente dos outros: opera por AMOSTRA, nao por FEATURE.
        from sklearn.preprocessing import Normalizer
        return Normalizer()
    else:
        raise ValueError(f"scaler_type '{scaler_type}' invalido")


# ────────────────────────────────────────────────────────────────────────
# FIT + TRANSFORM — Operacoes sobre features (2D ou 3D)
#
# REGRA CRITICA: Scaler e fitado em dados LIMPOS (sem noise).
# Isso captura a distribuicao real do sinal EM. Dados ruidosos
# adicionariam variancia espuria ao scaler, distorcendo a normalizacao.
#
#   ┌──────────────────────────────────────────────────────────────┐
#   │  x_train_clean ──→ scaler.fit() ──→ scaler (estatisticas)  │
#   │                                                              │
#   │  x_train_noisy ──→ scaler.transform() ──→ x_train_scaled   │
#   │  x_val_clean   ──→ scaler.transform() ──→ x_val_scaled     │
#   │  x_test_clean  ──→ scaler.transform() ──→ x_test_scaled    │
#   └──────────────────────────────────────────────────────────────┘
# ────────────────────────────────────────────────────────────────────────

def fit_scaler(
    x_train: np.ndarray,
    scaler_type: str = "standard",
) -> Optional[Any]:
    """Fita scaler em dados LIMPOS (sem noise).

    Aceita 2D (n_rows, n_feat) ou 3D (n_seq, seq_len, n_feat).
    z_meters NAO deve estar presente — ja foi separado em DataSplits.

    Para 3D, o array e achatado para (n_rows*seq_len, n_feat) antes
    do fit, garantindo que o scaler veja todas as posicoes de
    profundidade como amostras independentes.

    Args:
        x_train: Features de treino limpas.
        scaler_type: Tipo do scaler.

    Returns:
        Scaler fitado, ou None para "none".

    Note:
        Referenciado em:
            - data/pipeline.py: DataPipeline._prepare_offline() (fit em
              train clean, scaler unico)
            - data/pipeline.py: DataPipeline._prepare_onthefly() (fit em
              train clean FV, temporario)
            - data/scaling.py: fit_per_group_scalers() (fit EM e GS separados)
            - data/__init__.py: re-exportado como API publica
            - tests/test_data_pipeline.py: TestScalers.test_fit_and_transform_3d
        Ref: docs/ARCHITECTURE_v2.md secao 4.3.
        REGRA CRITICA: SEMPRE fitar em dados LIMPOS (sem noise).
        Fit em dados ruidosos introduz variancia espuria — Errata v5.0.15.
        z_meters NUNCA presente (separado em DataSplits.z_train).
    """
    scaler = create_scaler(scaler_type)
    if scaler is None:
        return None

    # Achatamento 3D→2D: (n_seq, seq_len, n_feat) → (n_seq*seq_len, n_feat)
    # Cada ponto de profundidade e tratado como amostra independente
    if x_train.ndim == 3:
        n_seq, seq_len, n_feat = x_train.shape
        flat = x_train.reshape(-1, n_feat)
    elif x_train.ndim == 2:
        flat = x_train
    else:
        raise ValueError(f"x_train deve ser 2D ou 3D, shape: {x_train.shape}")

    scaler.fit(flat)
    logger.info("Scaler '%s' fitado em %d amostras, %d features", scaler_type, flat.shape[0], flat.shape[1])
    return scaler


def transform_features(
    x: np.ndarray,
    scaler: Optional[Any],
) -> np.ndarray:
    """Transforma features com scaler pre-fitado.

    Aceita 2D ou 3D. Se scaler=None, retorna copia.
    Preserva shape original apos transformacao.

    Args:
        x: Features a transformar.
        scaler: Scaler fitado (ou None).

    Returns:
        Array transformado com mesma shape.

    Note:
        Referenciado em:
            - data/pipeline.py: DataPipeline._prepare_offline() (transform
              train/val/test com scaler unico)
            - data/pipeline.py: DataPipeline._prepare_onthefly() (transform
              val/test offline)
            - data/scaling.py: transform_per_group() (transform EM e GS)
            - data/__init__.py: re-exportado como API publica
            - tests/test_data_pipeline.py: TestScalers.test_none_scaler_is_copy
        Ref: docs/ARCHITECTURE_v2.md secao 4.3.
        Quando scaler=None, retorna COPIA (nao referencia) para seguranca.
        Shape preservada: 3D achatado para sklearn, depois restaurado.
    """
    if scaler is None:
        return x.copy()

    original_shape = x.shape
    # Achatamento 3D→2D para compatibilidade com sklearn
    if x.ndim == 3:
        n_seq, seq_len, n_feat = x.shape
        flat = x.reshape(-1, n_feat)
    else:
        flat = x

    scaled = scaler.transform(flat)

    # Restaurar shape original
    if x.ndim == 3:
        scaled = scaled.reshape(original_shape)

    return scaled


# ════════════════════════════════════════════════════════════════════════
# PER-GROUP SCALERS [P3] — EM e GS com scalers separados
#
# Motivacao fisica: features EM (componentes do tensor de campo)
# e geosinais (ratios, diferencas) possuem distribuicoes muito
# diferentes. EM raw sao amplitudes em A/m (~1e-3 a 1e+1), enquanto
# GS sao ratios adimensionais ou diferencas normalizadas (~0 a ~10).
#
# Usar um unico scaler para ambos distorceria a normalizacao:
# - StandardScaler no conjunto EM+GS: media dominada por EM
# - GS ficariam sub-normalizados, perdendo informacao diagnostica
#
#   ┌───────────────────────────────────────────────────────────────────┐
#   │  Features EM (cols 0..n_em-1) ──→ StandardScaler (mu/sigma)     │
#   │  Features GS (cols n_em..end)  ──→ RobustScaler (median/IQR)    │
#   │                                       ↑ robusto a outliers GS   │
#   │  Resultado: [EM_scaled | GS_scaled] concatenados                │
#   └───────────────────────────────────────────────────────────────────┘
# ════════════════════════════════════════════════════════════════════════

def fit_per_group_scalers(
    x_train: np.ndarray,
    config: PipelineConfig,
    n_em_features: int,
) -> Tuple[Optional[Any], Optional[Any]]:
    """Fita scalers separados para features EM e geosinais [P3].

    StandardScaler para EM, RobustScaler para GS (ranges diferentes).
    O scaler EM captura estatisticas das componentes do tensor EM,
    enquanto o scaler GS captura a distribuicao dos geosinais derivados.

    Args:
        x_train: Features limpas 3D (n_seq, seq_len, n_feat).
        config: PipelineConfig com scaler_type e gs_scaler_type.
        n_em_features: Numero de features EM (sem GS).

    Returns:
        (scaler_em, scaler_gs) — scaler_gs pode ser None se nao houver GS.

    Note:
        Referenciado em:
            - data/pipeline.py: DataPipeline._prepare_offline() (per-group
              quando use_per_group_scalers=True e GS ativo)
            - data/pipeline.py: DataPipeline._prepare_onthefly() (per-group
              para fitar scaler clean temporario)
        Ref: docs/ARCHITECTURE_v2.md secao 4.3 (principio P3).
        EM: config.scaler_type (default "standard" — StandardScaler).
        GS: config.gs_scaler_type (default "robust" — RobustScaler).
        SEMPRE fitado em dados LIMPOS (sem noise) — principio P3.
    """
    # Particionar features em grupo EM e grupo GS
    x_em = x_train[:, :, :n_em_features]
    scaler_em = fit_scaler(x_em, config.scaler_type)

    scaler_gs = None
    if x_train.shape[-1] > n_em_features:
        # GS presentes — fitar scaler separado (tipicamente RobustScaler)
        x_gs = x_train[:, :, n_em_features:]
        scaler_gs = fit_scaler(x_gs, config.gs_scaler_type)
        logger.info(
            "Per-group scalers: EM(%s, %d feat), GS(%s, %d feat)",
            config.scaler_type, n_em_features,
            config.gs_scaler_type, x_gs.shape[-1],
        )

    return scaler_em, scaler_gs


def transform_per_group(
    x: np.ndarray,
    scaler_em: Optional[Any],
    scaler_gs: Optional[Any],
    n_em_features: int,
) -> np.ndarray:
    """Transforma features com per-group scalers.

    Aplica scaler_em nas primeiras n_em_features colunas e scaler_gs
    nas colunas restantes (geosinais). Concatena o resultado.

    Args:
        x: Features a transformar.
        scaler_em: Scaler para features EM.
        scaler_gs: Scaler para geosinais.
        n_em_features: Ponto de corte EM|GS.

    Returns:
        Array transformado.

    Note:
        Referenciado em:
            - data/pipeline.py: DataPipeline._prepare_offline() (transform
              train/val/test com per-group scalers)
            - data/pipeline.py: DataPipeline._prepare_onthefly() (transform
              val/test offline com per-group scalers)
        Ref: docs/ARCHITECTURE_v2.md secao 4.3 (principio P3).
        Concatenacao: [EM_scaled | GS_scaled] na ultima dimensao.
        Se scaler_gs=None ou sem GS, retorna apenas EM_scaled.
    """
    # Transformar features EM (componentes do tensor de campo)
    x_em = transform_features(x[:, :, :n_em_features], scaler_em)

    if scaler_gs is not None and x.shape[-1] > n_em_features:
        # Transformar geosinais com scaler separado (RobustScaler)
        x_gs = transform_features(x[:, :, n_em_features:], scaler_gs)
        return np.concatenate([x_em, x_gs], axis=-1)

    return x_em


# ════════════════════════════════════════════════════════════════════════
# TF SCALER FN — para on-the-fly dentro de tf.data.map()
#
# No modo on-the-fly, noise e aplicado em runtime via tf.data.map().
# O scaling tambem deve ser aplicado em TF puro (sem sklearn).
# A solucao: extrair constantes (mean, scale, etc.) do sklearn scaler
# e criar uma closure TF com tf.constant.
#
# Fluxo on-the-fly dentro de tf.data.map:
#   noise(raw_em) → FV_tf(noisy) → GS_tf(noisy) → scale_tf(aqui)
#
#   ┌─────────────────────────────────────────────────────────────────┐
#   │  sklearn scaler (fitado offline em clean)                      │
#   │        │                                                        │
#   │        ▼                                                        │
#   │  Extrai mean_, scale_ (ou center_, scale_ para Robust)        │
#   │        │                                                        │
#   │        ▼                                                        │
#   │  tf.constant(mean_), tf.constant(scale_)                       │
#   │        │                                                        │
#   │        ▼                                                        │
#   │  Closure: lambda x: (x - mean) / (scale + eps)                │
#   │           eps = 1e-12 para evitar divisao por zero              │
#   └─────────────────────────────────────────────────────────────────┘
# ════════════════════════════════════════════════════════════════════════

def make_tf_scaler_fn(
    scaler: Optional[Any],
) -> Callable:
    """Constroi funcao TF que aplica scaling via constantes pre-computadas.

    Extrai mean/scale do sklearn scaler e cria uma closure TF pura
    (sem dependencia de sklearn em runtime). A closure usa tf.constant
    para armazenar as estatisticas, garantindo execucao eficiente
    dentro do grafo TF.

    Scalers suportados para conversao TF:
        - StandardScaler: (x - mean_) / scale_
        - RobustScaler: (x - center_) / scale_
        - MinMaxScaler: (x - data_min_) / data_range_

    Args:
        scaler: Scaler sklearn fitado (StandardScaler, RobustScaler, etc.).

    Returns:
        Funcao (tf.Tensor) -> tf.Tensor que aplica o scaling.

    Note:
        Referenciado em:
            - data/pipeline.py: DataPipeline.build_train_map_fn() (converte
              scaler_em e scaler_gs para closures TF, Step 4 do map_fn)
        Ref: docs/ARCHITECTURE_v2.md secao 4.3.
        Converte sklearn scaler para closure TF pura (constantes pre-computadas).
        eps = 1e-12 evita divisao por zero em features constantes.
        Scaler fitado em dados LIMPOS — constantes capturam distribuicao real.
        MaxAbsScaler, QuantileTransformer, PowerTransformer: fallback identity
        com warning (atributos nao reconhecidos).
    """
    import tensorflow as tf

    if scaler is None:
        return lambda x: x

    if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        # ── StandardScaler: (x - mu) / sigma.
        #    Atributos: mean_ (media por feature), scale_ (desvio padrao).
        #    Resultado: features com media ~0 e variancia ~1.
        mean = tf.constant(scaler.mean_, dtype=tf.float32)
        scale = tf.constant(scaler.scale_, dtype=tf.float32)
        # eps=1e-12 evita divisao por zero em features constantes
        return lambda x: (tf.cast(x, tf.float32) - mean) / (scale + 1e-12)

    elif hasattr(scaler, "center_") and hasattr(scaler, "scale_"):
        # ── RobustScaler: (x - median) / IQR.
        #    Atributos: center_ (mediana), scale_ (interquartil range).
        #    Robusto a outliers. Default para geosinais (gs_scaler_type).
        center = tf.constant(scaler.center_, dtype=tf.float32)
        scale = tf.constant(scaler.scale_, dtype=tf.float32)
        return lambda x: (tf.cast(x, tf.float32) - center) / (scale + 1e-12)

    elif hasattr(scaler, "data_min_") and hasattr(scaler, "data_range_"):
        # ── MinMaxScaler: (x - min) / range → [0, 1].
        #    Atributos: data_min_ (minimo por feature), data_range_ (range).
        #    Sensivel a outliers — nao recomendado para EM raw.
        data_min = tf.constant(scaler.data_min_, dtype=tf.float32)
        data_range = tf.constant(scaler.data_range_, dtype=tf.float32)
        return lambda x: (tf.cast(x, tf.float32) - data_min) / (data_range + 1e-12)

    else:
        logger.warning(
            "Scaler %s nao tem atributos conhecidos — fallback para identity",
            type(scaler).__name__,
        )
        return lambda x: tf.cast(x, tf.float32)
