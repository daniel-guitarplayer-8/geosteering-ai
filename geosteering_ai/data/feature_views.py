# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: data/feature_views.py                                             ║
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
# ║    • 6 Feature Views com implementacoes numpy e TensorFlow CONSISTENTES   ║
# ║    • Transformacoes sobre componentes EM (Hxx planar, Hzz axial)          ║
# ║    • Bug fix v2.0: ambos backends agora usam log10 (legado: ln vs log10)  ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig)                                 ║
# ║  Exports: ~5 (3 helpers + apply_feature_view + apply_feature_view_tf)     ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 4.3, docs/physics/perspectivas.md    ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial com 6 FVs consistentes       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Feature Views — Transformacoes sobre componentes EM.

6 Feature Views com implementacoes numpy e TensorFlow CONSISTENTES
(mesma base logaritmica, mesmos canais de saida).

Bug fix v2.0: No legado (C22), numpy usava ln e TF usava log10,
e TF dropava canais em H1_logH2 e logH1_logH2. Agora ambos usam log10
e preservam a mesma estrutura de canais.

Referencia: docs/ARCHITECTURE_v2.md secao 5.1, docs/reference/feature_views.md.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from geosteering_ai.config import PipelineConfig

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
# SECAO: CONSTANTES FISICAS E DE CONFIGURACAO
# ════════════════════════════════════════════════════════════════════════════
# Valores criticos validados pela Errata v4.4.5 + v5.0.15.
# Qualquer alteracao DEVE ser aprovada e documentada.
# Ref: docs/physics/errata_valores.md
# ──────────────────────────────────────────────────────────────────────────

# Epsilon seguro para float32 — protege contra underflow em log/divisao.
# NUNCA usar 1e-30 (causa subnormais em float32, gradientes explodidos).
# Ref: Errata v5.0.15, IEEE 754 float32 min normal ≈ 1.175e-38.
EPS = 1e-12

# Feature Views validas — 6 transformacoes sobre componentes EM.
# identity/raw: passthrough (sem transformacao, componentes Re/Im originais).
# H1_logH2: H1 raw (preserva Re/Im planar) + H2 log10 (comprime axial).
# logH1_logH2: ambos log10 (comprime faixa dinamica de ambas componentes).
# IMH1_IMH2_razao: partes imaginarias + razao linear |H1|/|H2|.
# IMH1_IMH2_lograzao: partes imaginarias + log10 da razao |H1|/|H2|.
# Ref: docs/physics/perspectivas.md secao Feature Views.
VALID_VIEWS = {"identity", "raw", "H1_logH2", "logH1_logH2",
               "IMH1_IMH2_razao", "IMH1_IMH2_lograzao"}


# ════════════════════════════════════════════════════════════════════════════
# SECAO: HELPERS MATEMATICOS
# ════════════════════════════════════════════════════════════════════════════
# Funcoes auxiliares para calculo de magnitude, fase e log10 seguro.
# Usadas por ambas versoes (numpy) para garantir consistencia.
# Ref: Errata v5.0.15 — EPS = 1e-12 (NUNCA 1e-30 para float32).
# Magnitude complexa: sqrt(Re^2 + Im^2 + eps) para estabilidade.
# ──────────────────────────────────────────────────────────────────────────

def _magnitude(re: np.ndarray, im: np.ndarray) -> np.ndarray:
    """Magnitude complexa: sqrt(re^2 + im^2 + eps)."""
    return np.sqrt(re ** 2 + im ** 2 + EPS)


def _phase(re: np.ndarray, im: np.ndarray) -> np.ndarray:
    """Fase complexa: arctan2(im, re) em radianos."""
    return np.arctan2(im, re)


def _safe_log10(x: np.ndarray) -> np.ndarray:
    """Log10 seguro com floor em eps."""
    return np.log10(np.maximum(np.abs(x), EPS))


# ════════════════════════════════════════════════════════════════════════════
# SECAO: VERSAO NUMPY — preprocessamento offline (val, test, fit scaler)
# ════════════════════════════════════════════════════════════════════════════
# Aplicada a dados de validacao, teste e ao fit do scaler (dados limpos).
# Nao e usada no treinamento on-the-fly (que usa a versao TensorFlow).
# Ref: docs/ARCHITECTURE_v2.md secao 4.3 — cadeia de dados.
#
#   ┌──────────────────────────────────────────────────────────────────────────┐
#   │  6 Feature Views Canonicas (Ref: docs/physics/perspectivas.md):         │
#   │                                                                          │
#   │  View               │ Canal 0    │ Canal 1  │ Canal 2      │ Canal 3    │
#   │  ───────────────────┼────────────┼──────────┼──────────────┼────────────│
#   │  identity / raw     │ Re(H1)     │ Im(H1)   │ Re(H2)       │ Im(H2)    │
#   │  H1_logH2           │ Re(H1)     │ Im(H1)   │ log10|H2|    │ φ(H2)     │
#   │  logH1_logH2        │ log10|H1|  │ φ(H1)    │ log10|H2|    │ φ(H2)     │
#   │  IMH1_IMH2_razao    │ Im(H1)     │ Im(H2)   │ |H1|/|H2|   │ Δφ        │
#   │  IMH1_IMH2_lograzao │ Im(H1)     │ Im(H2)   │ log10(ratio) │ Δφ        │
#   │                                                                          │
#   │  H1 = Hxx (planar), H2 = Hzz (axial)                                   │
#   │  |H| = sqrt(Re^2 + Im^2 + eps),  phi(H) = arctan2(Im, Re)             │
#   │  eps = 1e-12 (float32 safe — NUNCA 1e-30)                              │
#   │  SEMPRE log10 (NUNCA ln — bug fix v2.0)                                │
#   └──────────────────────────────────────────────────────────────────────────┘
# ──────────────────────────────────────────────────────────────────────────

def apply_feature_view(
    x: np.ndarray,
    view: str = "identity",
    n_prefix: int = 0,
) -> np.ndarray:
    """Aplica Feature View sobre componentes EM (numpy).

    Layout esperado nas ultimas 4 colunas (apos prefix + z):
        [...prefix, z, Re(H1), Im(H1), Re(H2), Im(H2)]

    H1 = Hxx (planar), H2 = Hzz (axial).

    Saida: mesmo shape que entrada, 4 canais EM substituidos.

    Args:
        x: Array 2D (n_rows, n_feat) ou 3D (n_seq, seq_len, n_feat).
        view: Nome da Feature View.
        n_prefix: Colunas prefixo (theta, f_norm) antes de z.

    Returns:
        Array com mesma shape, canais EM transformados.

    Note:
        Referenciado em:
            - data/pipeline.py: DataPipeline._apply_fv_gs() (modo offline)
            - data/pipeline.py: build_train_map_fn() (modo on-the-fly, Step 2)
            - tests/test_data_pipeline.py: TestFeatureViews (9 test cases)
        Ref: docs/ARCHITECTURE_v2.md secao 4.3, docs/physics/perspectivas.md.
        Bug fix v2.0: Legado (C22) usava ln (numpy) vs log10 (TF).
            Agora ambos usam log10 consistentemente.
        Guard numerico: EPS = 1e-12 (NUNCA 1e-30 em float32).
    """
    # ── identity/raw: retorna copia sem transformacao (passthrough) ──
    if view in ("identity", "raw"):
        return x.copy()

    if view not in VALID_VIEWS:
        raise ValueError(f"Feature View '{view}' invalida. Validas: {VALID_VIEWS}")

    result = x.copy()
    original_shape = result.shape

    # ── Flatten 3D→2D para operar uniformemente sobre linhas ──
    if result.ndim == 3:
        n_seq, seq_len, n_feat = result.shape
        result = result.reshape(-1, n_feat)
    elif result.ndim != 2:
        raise ValueError(f"x deve ter 2D ou 3D, shape recebido: {x.shape}")

    # ── Indices das 4 componentes EM (ultimas 4 colunas antes de GS) ──
    # Layout: [prefix...] [z] [Re(H1)] [Im(H1)] [Re(H2)] [Im(H2)] [GS...]
    # em_start aponta para Re(H1), pulando prefix + z_obs
    em_start = n_prefix + 1  # pula prefix + z
    if result.shape[-1] < em_start + 4:
        raise ValueError(
            f"Array tem {result.shape[-1]} colunas mas em_start={em_start} "
            f"requer pelo menos {em_start + 4} (n_prefix={n_prefix}, z=1, 4 EM)"
        )
    re_h1 = result[:, em_start]      # Re(Hxx) — componente planar real
    im_h1 = result[:, em_start + 1]  # Im(Hxx) — componente planar imaginaria
    re_h2 = result[:, em_start + 2]  # Re(Hzz) — componente axial real
    im_h2 = result[:, em_start + 3]  # Im(Hzz) — componente axial imaginaria

    # ── Calculo de magnitude e fase para ambas componentes ──
    mag_h1 = _magnitude(re_h1, im_h1)  # |Hxx| — magnitude planar
    mag_h2 = _magnitude(re_h2, im_h2)  # |Hzz| — magnitude axial
    phi_h1 = _phase(re_h1, im_h1)      # arg(Hxx) — fase planar [rad]
    phi_h2 = _phase(re_h2, im_h2)      # arg(Hzz) — fase axial [rad]

    if view == "H1_logH2":
        # ── H1_logH2: H1 cru preserva SNR em alta atenuacao,
        #    H2 log10-transformado comprime faixa dinamica larga de Hzz.
        #    Saida: [Re(H1), Im(H1), log10|H2|, phi(H2)]
        #    Motivacao fisica: Hzz varia 4+ ordens de magnitude entre
        #    camadas de alta e baixa resistividade. Log10 estabiliza
        #    gradientes e melhora convergencia durante treinamento.
        result[:, em_start + 2] = _safe_log10(mag_h2)
        result[:, em_start + 3] = phi_h2

    elif view == "logH1_logH2":
        # ── logH1_logH2: Ambos H1 e H2 em escala logaritmica.
        #    Saida: [log10|H1|, phi(H1), log10|H2|, phi(H2)]
        #    Motivacao fisica: Magnitude + fase capturam toda informacao
        #    do sinal complexo. Log10 comprime faixa dinamica, tornando
        #    ambas componentes comparaveis em escala para o modelo.
        result[:, em_start]     = _safe_log10(mag_h1)
        result[:, em_start + 1] = phi_h1
        result[:, em_start + 2] = _safe_log10(mag_h2)
        result[:, em_start + 3] = phi_h2

    elif view == "IMH1_IMH2_razao":
        # ── IMH1_IMH2_razao: Partes imaginarias + razao de magnitudes.
        #    Saida: [Im(H1), Im(H2), |H1|/|H2|, phi(H1)-phi(H2)]
        #    Motivacao fisica: Im(H) é mais sensivel a contraste de
        #    resistividade em fronteiras. Razao |H1|/|H2| indica anisotropia
        #    (TIV). Diferenca de fase detecta fronteiras de camada.
        result[:, em_start] = im_h1
        result[:, em_start + 1] = im_h2
        result[:, em_start + 2] = mag_h1 / (mag_h2 + EPS)
        result[:, em_start + 3] = phi_h1 - phi_h2

    elif view == "IMH1_IMH2_lograzao":
        # ── IMH1_IMH2_lograzao: Como razao, mas com log10 da razao.
        #    Saida: [Im(H1), Im(H2), log10(|H1|/|H2|), phi(H1)-phi(H2)]
        #    Motivacao fisica: Log-razao estabiliza variacao quando
        #    contraste de resistividade é muito alto (>100:1).
        #    Preferido para cenarios com camadas de sal ou carbonato.
        result[:, em_start] = im_h1
        result[:, em_start + 1] = im_h2
        result[:, em_start + 2] = _safe_log10(mag_h1 / (mag_h2 + EPS))
        result[:, em_start + 3] = phi_h1 - phi_h2

    # ── Restaura shape 3D se entrada era 3D ──
    if result.ndim == 2 and x.ndim == 3:
        result = result.reshape(original_shape)

    return result


# ════════════════════════════════════════════════════════════════════════════
# SECAO: VERSAO TENSORFLOW — on-the-fly dentro de tf.data.map()
# ════════════════════════════════════════════════════════════════════════════
# Usada no treinamento on-the-fly: noise → FV_tf → GS_tf → scale.
# Semantica IDENTICA a versao numpy: mesmos canais, mesma base log10.
# Import de TensorFlow e lazy (dentro da funcao) para nao forcar
# importacao global em contextos que so usam numpy.
# Ref: docs/ARCHITECTURE_v2.md secao 4.3 — cadeia de dados on-the-fly.
# ──────────────────────────────────────────────────────────────────────────

def apply_feature_view_tf(
    x: "tf.Tensor",
    view: str = "identity",
    n_prefix: int = 0,
    eps: float = EPS,
) -> "tf.Tensor":
    """Aplica Feature View sobre componentes EM (TensorFlow).

    Semantica IDENTICA a versao numpy: mesmos canais, mesma base log10.

    Args:
        x: Tensor 3D (batch, seq_len, n_feat).
        view: Nome da Feature View.
        n_prefix: Colunas prefixo antes de z.
        eps: Epsilon para estabilidade numerica.

    Returns:
        Tensor com mesma shape, canais EM transformados.

    Note:
        Referenciado em:
            - data/pipeline.py: DataPipeline._apply_fv_gs() (modo offline)
            - data/pipeline.py: build_train_map_fn() (modo on-the-fly, Step 2)
            - tests/test_data_pipeline.py: TestFeatureViews (9 test cases)
        Ref: docs/ARCHITECTURE_v2.md secao 4.3, docs/physics/perspectivas.md.
        Bug fix v2.0: Legado (C22) usava ln (numpy) vs log10 (TF).
            Agora ambos usam log10 consistentemente.
        Guard numerico: EPS = 1e-12 (NUNCA 1e-30 em float32).
    """
    import tensorflow as tf

    # ── identity/raw: retorna tensor sem transformacao (passthrough) ──
    if view in ("identity", "raw"):
        return x

    # ── Fatiamento das componentes EM do tensor de entrada ──
    # em_start aponta para Re(H1), pulando prefix + z_obs
    em_start = n_prefix + 1
    tf.debugging.assert_greater_equal(
        tf.shape(x)[-1], em_start + 4,
        message=f"Tensor precisa de >= {em_start + 4} colunas para FV"
    )
    prefix_and_z = x[:, :, :em_start]         # colunas antes das componentes EM
    re_h1 = x[:, :, em_start]                 # Re(Hxx) — componente planar real
    im_h1 = x[:, :, em_start + 1]             # Im(Hxx) — componente planar imaginaria
    re_h2 = x[:, :, em_start + 2]             # Re(Hzz) — componente axial real
    im_h2 = x[:, :, em_start + 3]             # Im(Hzz) — componente axial imaginaria
    tail = x[:, :, em_start + 4:]             # canais GS posteriores (se existirem)

    # ── Calculo de magnitude e fase (TF ops, autodiff-compativel) ──
    mag_h1 = tf.sqrt(re_h1 ** 2 + im_h1 ** 2 + eps)  # |Hxx| — magnitude planar
    mag_h2 = tf.sqrt(re_h2 ** 2 + im_h2 ** 2 + eps)  # |Hzz| — magnitude axial
    phi_h1 = tf.math.atan2(im_h1, re_h1)              # arg(Hxx) — fase planar [rad]
    phi_h2 = tf.math.atan2(im_h2, re_h2)              # arg(Hzz) — fase axial [rad]

    # ── Fator de conversao ln→log10 (constante, nao recomputada por amostra) ──
    log10 = tf.math.log(10.0)

    def _safe_log10_tf(val):
        """Log10 seguro via mudanca de base: log10(x) = ln(x) / ln(10)."""
        return tf.math.log(tf.maximum(tf.abs(val), eps)) / log10

    if view == "H1_logH2":
        # ── H1_logH2: H1 cru preserva SNR em alta atenuacao,
        #    H2 log10-transformado comprime faixa dinamica larga de Hzz.
        #    Saida: [Re(H1), Im(H1), log10|H2|, phi(H2)]
        #    Motivacao fisica: Hzz varia 4+ ordens de magnitude entre
        #    camadas de alta e baixa resistividade. Log10 estabiliza
        #    gradientes e melhora convergencia durante treinamento.
        em = tf.stack([re_h1, im_h1, _safe_log10_tf(mag_h2), phi_h2], axis=-1)

    elif view == "logH1_logH2":
        # ── logH1_logH2: Ambos H1 e H2 em escala logaritmica.
        #    Saida: [log10|H1|, phi(H1), log10|H2|, phi(H2)]
        #    Motivacao fisica: Magnitude + fase capturam toda informacao
        #    do sinal complexo. Log10 comprime faixa dinamica, tornando
        #    ambas componentes comparaveis em escala para o modelo.
        em = tf.stack([
            _safe_log10_tf(mag_h1), phi_h1,
            _safe_log10_tf(mag_h2), phi_h2,
        ], axis=-1)

    elif view == "IMH1_IMH2_razao":
        # ── IMH1_IMH2_razao: Partes imaginarias + razao de magnitudes.
        #    Saida: [Im(H1), Im(H2), |H1|/|H2|, phi(H1)-phi(H2)]
        #    Motivacao fisica: Im(H) é mais sensivel a contraste de
        #    resistividade em fronteiras. Razao |H1|/|H2| indica anisotropia
        #    (TIV). Diferenca de fase detecta fronteiras de camada.
        ratio = mag_h1 / (mag_h2 + eps)
        em = tf.stack([im_h1, im_h2, ratio, phi_h1 - phi_h2], axis=-1)

    elif view == "IMH1_IMH2_lograzao":
        # ── IMH1_IMH2_lograzao: Como razao, mas com log10 da razao.
        #    Saida: [Im(H1), Im(H2), log10(|H1|/|H2|), phi(H1)-phi(H2)]
        #    Motivacao fisica: Log-razao estabiliza variacao quando
        #    contraste de resistividade é muito alto (>100:1).
        #    Preferido para cenarios com camadas de sal ou carbonato.
        ratio = mag_h1 / (mag_h2 + eps)
        em = tf.stack([
            im_h1, im_h2, _safe_log10_tf(ratio), phi_h1 - phi_h2,
        ], axis=-1)

    else:
        raise ValueError(f"Feature View '{view}' invalida. Validas: {VALID_VIEWS}")

    # ── Reconstroi tensor: [prefix+z | EM transformado | tail GS] ──
    return tf.concat([prefix_and_z, em, tail], axis=-1)


# ════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════
# Inventario completo de simbolos exportados por este modulo.
# ──────────────────────────────────────────────────────────────────────────

__all__ = [
    # ── Constantes ────────────────────────────────────────────────────────
    "EPS",
    "VALID_VIEWS",
    # ── Funcoes numpy (preprocessamento offline: val, test, fit scaler) ──
    "apply_feature_view",
    # ── Funcoes TensorFlow (on-the-fly dentro de tf.data.map) ────────────
    "apply_feature_view_tf",
]
