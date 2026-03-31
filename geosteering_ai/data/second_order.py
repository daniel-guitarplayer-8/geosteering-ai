# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: data/second_order.py                                              ║
# ║  Bloco: 2f — Features de 2o Grau (Estrategia C)                          ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (ponto unico de verdade)                ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • compute_second_order_features(): 6 features derivadas (NumPy)        ║
# ║    • compute_second_order_features_tf(): versao TF para on-the-fly        ║
# ║    • Features: |H|^2, d|H|/dz, Re(H)/Im(H) para H1 e H2                 ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig)                                 ║
# ║  Exports: ~2 funcoes — ver __all__                                        ║
# ║  Ref: Estrategia C (features derivadas da fisica para alta rho)           ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

"""Features de 2o grau — derivadas nao-lineares dos componentes EM.

Computa 6 features adicionais a partir das 4 componentes EM basicas
[Re(H1), Im(H1), Re(H2), Im(H2)] para amplificar sinais fracos em
zonas de alta resistividade.

Motivacao fisica:
    Para f=20 kHz e rho > 100 Ohm.m, o sinal EM bruto (Re, Im)
    se aproxima do acoplamento direto, tornando a inversao ambigua.
    Features de 2o grau extraem informacao nao-linear que permanece
    informativa mesmo quando o sinal bruto eh fraco:

    |H|^2 (potencia):
        Magnifica diferencas de amplitude que seriam sutis em Re/Im.
        Potencia eh proporcional a sigma^2, amplificando contrastes.

    d|H|/dz (gradiente espacial):
        Detecta fronteiras geologicas via variacao espacial.
        Fronteiras causam picos no gradiente mesmo em sinal fraco.

    Re(H)/Im(H) (razao):
        Proporcional a cotan(fase). Captura informacao de fase
        que eh mais robusta que amplitude em alta resistividade.
        Razao compensa parcialmente a perda de amplitude.

    ┌──────────────────────────────────────────────────────────────────┐
    │  6 Features de 2o Grau (Estrategia C)                            │
    │                                                                  │
    │  Canal │ Formula          │ Significado Fisico                   │
    │  ──────┼──────────────────┼──────────────────────────────────────│
    │  0     │ |H1|^2           │ Potencia Hxx (planar)               │
    │  1     │ |H2|^2           │ Potencia Hzz (axial)                │
    │  2     │ d|H1|/dz         │ Gradiente espacial Hxx              │
    │  3     │ d|H2|/dz         │ Gradiente espacial Hzz              │
    │  4     │ Re(H1)/Im(H1)    │ Razao cot(fase) Hxx                 │
    │  5     │ Re(H2)/Im(H2)    │ Razao cot(fase) Hzz                 │
    │                                                                  │
    │  H1 = Hxx (planar), H2 = Hzz (axial)                           │
    │  EPS = 1e-12 (float32 safe, NUNCA 1e-30)                       │
    └──────────────────────────────────────────────────────────────────┘

Dois modos de uso (config.second_order_mode):
    - "feature_view": substitui os 4 canais EM por 6 canais 2o grau
        (como uma Feature View alternativa — FV recebe 4 EM, retorna 6)
    - "postprocess": concatena 6 canais APOS FV+GS mas ANTES do scale
        (features adicionais, nao substitutivas)

Referencia: Relatorio Estrategia C.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Epsilon float32-safe (NUNCA 1e-30) ────────────────────────────────
EPS = 1e-12

# ════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════
__all__ = [
    "compute_second_order_features",
    "compute_second_order_features_tf",
]


# ════════════════════════════════════════════════════════════════════════════
# SECAO: FEATURES DE 2o GRAU (NUMPY)
# ════════════════════════════════════════════════════════════════════════════
# Computa 6 features derivadas a partir dos 4 componentes EM.
# Versao NumPy para uso offline (val/test e fit_scaler).
#
# Input: 4 arrays (Re_H1, Im_H1, Re_H2, Im_H2) shape (...,)
# Output: 6 arrays shape (...,) concatenados no eixo -1
#
# Ref: Estrategia C — features derivadas da fisica.
# ──────────────────────────────────────────────────────────────────────────


def compute_second_order_features(
    x: np.ndarray,
    h1_cols: Tuple[int, int],
    h2_cols: Tuple[int, int],
    eps: float = EPS,
) -> np.ndarray:
    """Computa 6 features de 2o grau a partir de componentes EM.

    Extrai informacao nao-linear dos sinais EM brutos para
    amplificar contrastes em alta resistividade. Produz 6 canais
    adicionais que sao concatenados ao array de features.

    Algoritmo:
        1. Extrai Re/Im de H1 (Hxx) e H2 (Hzz) via h1_cols/h2_cols
        2. Computa |H|^2 = Re^2 + Im^2 (potencia)
        3. Computa d|H|/dz = diff(|H|) / diff(z) (gradiente espacial)
        4. Computa Re/Im (razao proporcional a cot(fase))
        5. Concatena: [|H1|^2, |H2|^2, grad_H1, grad_H2, ratio_H1, ratio_H2]

    Args:
        x: Features 3D (n_seq, seq_len, n_feat) ou 2D (seq_len, n_feat).
            Deve conter Re/Im de H1 e H2 nas posicoes h1_cols/h2_cols.
        h1_cols: Tuple (re_idx, im_idx) para H1 no array x.
        h2_cols: Tuple (re_idx, im_idx) para H2 no array x.
        eps: Epsilon para estabilidade numerica. Default: 1e-12.

    Returns:
        np.ndarray: 6 features de 2o grau, shape (..., 6).
            Canal 0: |H1|^2 (potencia Hxx)
            Canal 1: |H2|^2 (potencia Hzz)
            Canal 2: d|H1|/dz (gradiente espacial Hxx)
            Canal 3: d|H2|/dz (gradiente espacial Hzz)
            Canal 4: Re(H1)/Im(H1) (razao fase Hxx)
            Canal 5: Re(H2)/Im(H2) (razao fase Hzz)

    Example:
        >>> x = np.random.randn(10, 600, 5)
        >>> so = compute_second_order_features(x, h1_cols=(1,2), h2_cols=(3,4))
        >>> so.shape  # (10, 600, 6)

    Note:
        Referenciado em:
            - data/pipeline.py: DataPipeline._apply_second_order() (postprocess)
            - data/feature_views.py: apply_feature_view() (feature_view mode)
            - tests/test_second_order.py: TestSecondOrderFeatures
        Ref: Estrategia C.
        EPS = 1e-12 protege div-by-zero em Re/Im ratio.
        Gradiente usa np.diff com padding de borda (replica primeiro valor).
    """
    is_2d = x.ndim == 2
    if is_2d:
        x = x[np.newaxis, :, :]

    n_seq, seq_len, _ = x.shape

    # ── Extrair componentes EM ────────────────────────────────────────
    re_h1 = x[..., h1_cols[0]]  # Re(Hxx) — (n_seq, seq_len)
    im_h1 = x[..., h1_cols[1]]  # Im(Hxx)
    re_h2 = x[..., h2_cols[0]]  # Re(Hzz)
    im_h2 = x[..., h2_cols[1]]  # Im(Hzz)

    # ── Feature 0,1: Potencia |H|^2 = Re^2 + Im^2 ───────────────────
    # Magnifica diferencas de amplitude. Potencia ~ sigma^2,
    # amplificando contrastes que seriam sutis em Re/Im brutos.
    power_h1 = re_h1**2 + im_h1**2  # |H1|^2 — (n_seq, seq_len)
    power_h2 = re_h2**2 + im_h2**2  # |H2|^2

    # ── Feature 2,3: Gradiente espacial d|H|/dz ──────────────────────
    # Detecta fronteiras geologicas via variacao espacial da magnitude.
    # diff() retorna array de tamanho (seq_len-1), padded com o 1o valor
    # para manter shape consistente (seq_len).
    mag_h1 = np.sqrt(power_h1 + eps)
    mag_h2 = np.sqrt(power_h2 + eps)
    grad_h1 = np.diff(mag_h1, axis=-1, prepend=mag_h1[..., :1])
    grad_h2 = np.diff(mag_h2, axis=-1, prepend=mag_h2[..., :1])

    # ── Feature 4,5: Razao Re/Im ~ cot(fase) ─────────────────────────
    # Captura informacao de fase robusta a perda de amplitude.
    # Em alta rho, amplitude cai mas razao de fase permanece informativa.
    # Denominador: onde |Im| < eps, substitui por eps com sinal original.
    # 1e-30 dentro de sign() eh apenas tiebreaker para im==0.0 exato
    # (forca sign → +1 em vez de 0). NAO eh o eps do projeto (1e-12).
    # Clippado em [-100, 100] para evitar explosao quando Im -> 0.
    safe_im_h1 = np.where(np.abs(im_h1) < eps, np.sign(im_h1 + 1e-30) * eps, im_h1)
    safe_im_h2 = np.where(np.abs(im_h2) < eps, np.sign(im_h2 + 1e-30) * eps, im_h2)
    ratio_h1 = np.clip(re_h1 / safe_im_h1, -100.0, 100.0)
    ratio_h2 = np.clip(re_h2 / safe_im_h2, -100.0, 100.0)

    # ── Concatenar 6 features ─────────────────────────────────────────
    result = np.stack(
        [power_h1, power_h2, grad_h1, grad_h2, ratio_h1, ratio_h2],
        axis=-1,
    ).astype(np.float32)

    if is_2d:
        result = result[0]

    logger.debug(
        "compute_second_order_features: shape=%s, h1_cols=%s, h2_cols=%s",
        result.shape,
        h1_cols,
        h2_cols,
    )

    return result


# ════════════════════════════════════════════════════════════════════════════
# SECAO: FEATURES DE 2o GRAU (TENSORFLOW)
# ════════════════════════════════════════════════════════════════════════════
# Versao TensorFlow para uso on-the-fly dentro de tf.data.map().
# Semantica identica a versao NumPy, mas com operacoes TF
# autodiff-compativeis e lazy import.
#
# Ref: Estrategia C.
# ──────────────────────────────────────────────────────────────────────────


def compute_second_order_features_tf(
    x: "tf.Tensor",
    h1_cols: Tuple[int, int],
    h2_cols: Tuple[int, int],
    eps: float = EPS,
) -> "tf.Tensor":
    """Computa 6 features de 2o grau em TensorFlow (on-the-fly).

    Versao TF de compute_second_order_features(). Semanticamente
    identica, mas usa operacoes TF para compatibilidade com
    tf.data.map() e GradientTape.

    Args:
        x: Tensor 2D (seq_len, n_feat) — um batch element.
        h1_cols: Tuple (re_idx, im_idx) para H1.
        h2_cols: Tuple (re_idx, im_idx) para H2.
        eps: Epsilon float32-safe.

    Returns:
        tf.Tensor: 6 features, shape (seq_len, 6).

    Note:
        Referenciado em:
            - data/pipeline.py: build_train_map_fn() (Step 5 — on-the-fly)
            - tests/test_second_order.py: TestSecondOrderFeaturesTF
        Ref: Estrategia C.
        Lazy import TF. Operacoes autodiff-compativeis.
    """
    import tensorflow as tf

    # ── Extrair componentes EM ────────────────────────────────────────
    re_h1 = x[:, h1_cols[0]]
    im_h1 = x[:, h1_cols[1]]
    re_h2 = x[:, h2_cols[0]]
    im_h2 = x[:, h2_cols[1]]

    # ── Potencia |H|^2 ───────────────────────────────────────────────
    power_h1 = re_h1**2 + im_h1**2
    power_h2 = re_h2**2 + im_h2**2

    # ── Gradiente espacial d|H|/dz ────────────────────────────────────
    mag_h1 = tf.sqrt(power_h1 + eps)
    mag_h2 = tf.sqrt(power_h2 + eps)
    # diff com padding: [0, diff(mag)] — zero no 1o elemento (consistente com NumPy prepend)
    zero = tf.zeros([1], dtype=mag_h1.dtype)
    grad_h1 = tf.concat([zero, mag_h1[1:] - mag_h1[:-1]], axis=0)
    grad_h2 = tf.concat([zero, mag_h2[1:] - mag_h2[:-1]], axis=0)

    # ── Razao Re/Im ───────────────────────────────────────────────────
    # Denominador: onde |Im| < eps, substitui por eps com sinal original.
    # tf.where evita divisao por zero e preserva sinal do Im original.
    safe_im_h1 = tf.where(tf.abs(im_h1) < eps, tf.sign(im_h1 + 1e-30) * eps, im_h1)
    safe_im_h2 = tf.where(tf.abs(im_h2) < eps, tf.sign(im_h2 + 1e-30) * eps, im_h2)
    ratio_h1 = tf.clip_by_value(re_h1 / safe_im_h1, -100.0, 100.0)
    ratio_h2 = tf.clip_by_value(re_h2 / safe_im_h2, -100.0, 100.0)

    # ── Concatenar 6 features ─────────────────────────────────────────
    result = tf.stack(
        [power_h1, power_h2, grad_h1, grad_h2, ratio_h1, ratio_h2],
        axis=-1,
    )

    return result
