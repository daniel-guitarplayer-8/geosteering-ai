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
VALID_VIEWS = {
    "identity",
    "raw",
    "H1_logH2",
    "logH1_logH2",
    "IMH1_IMH2_razao",
    "IMH1_IMH2_lograzao",
    "second_order",
}

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
#   │  |H| = √(Re² + Im² + ε),  φ(H) = arctan2(Im, Re)                      │
#   │  ε = 1e-12 (float32 safe — NUNCA 1e-30)                                │
#   │  SEMPRE log10 (NUNCA ln — bug fix v2.0)                                │
#   └──────────────────────────────────────────────────────────────────────────┘


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
    return np.sqrt(re**2 + im**2 + EPS)


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
    *,
    h1_cols: "tuple[int, int] | None" = None,
    h2_cols: "tuple[int, int] | None" = None,
) -> np.ndarray:
    """Aplica Feature View sobre componentes EM (numpy).

    Suporta dois modos de operacao:

    1. **Modo posicional (legado, default):** Assume layout contiguous
       [...prefix, z, Re(H1), Im(H1), Re(H2), Im(H2), ...extras].
       H1 e H2 sao identificados pelas 4 colunas imediatamente apos
       prefix + z_obs. Compativel com pipeline P1 baseline.

    2. **Modo explicito (features expandidas):** h1_cols e h2_cols
       especificam as posicoes exatas de (Re, Im) de cada componente
       no array de features extraido. Permite features EM extras
       (Hxy, Hxz, etc.) entre H1 e H2 sem afetar a transformacao.

    H1 = Hxx (planar), H2 = Hzz (axial) no pipeline default.

    As Feature Views transformam APENAS os 4 canais H1/H2 especificados.
    Todas as demais colunas (z_obs, prefix, colunas EM extras, GS) sao
    preservadas intactas.

    Args:
        x: Array 2D (n_rows, n_feat) ou 3D (n_seq, seq_len, n_feat).
        view: Nome da Feature View. Deve estar em VALID_VIEWS.
        n_prefix: Colunas prefixo (theta, f_norm) antes de z.
            Usado apenas no modo posicional (quando h1_cols/h2_cols=None).
        h1_cols: Tupla (re_idx, im_idx) com posicoes de Re(H1) e Im(H1)
            no array de features. Se None, usa modo posicional:
            re_h1 = n_prefix+1, im_h1 = n_prefix+2.
        h2_cols: Tupla (re_idx, im_idx) com posicoes de Re(H2) e Im(H2)
            no array de features. Se None, usa modo posicional:
            re_h2 = n_prefix+3, im_h2 = n_prefix+4.

    Returns:
        np.ndarray: Array com mesma shape, canais H1/H2 transformados
            conforme a Feature View selecionada. Demais colunas intactas.

    Raises:
        ValueError: Se view nao esta em VALID_VIEWS.
        ValueError: Se array tem colunas insuficientes para as posicoes.

    Example:
        >>> # Modo posicional (P1 baseline, 5 features):
        >>> result = apply_feature_view(x, "logH1_logH2")
        >>> # Modo explicito (7 features com Hxy entre Hxx e Hzz):
        >>> result = apply_feature_view(x, "logH1_logH2",
        ...     h1_cols=(1, 2), h2_cols=(5, 6))

    Note:
        Referenciado em:
            - data/pipeline.py: DataPipeline._apply_fv_gs() (modo offline)
            - data/pipeline.py: build_train_map_fn() (modo on-the-fly, Step 2)
            - tests/test_data_pipeline.py: TestFeatureViews,
              TestFeatureViewsExpanded
        Ref: docs/ARCHITECTURE_v2.md secao 4.3, docs/physics/perspectivas.md.
        Bug fix v2.0: Legado (C22) usava ln (numpy) vs log10 (TF).
            Agora ambos usam log10 consistentemente.
        Guard numerico: EPS = 1e-12 (NUNCA 1e-30 em float32).
        v2.0.1: Adicionado h1_cols/h2_cols para features EM expandidas.
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

    # ── Resolucao das posicoes de H1 e H2 ──────────────────────────
    # Dois modos de operacao:
    #
    #   1. Modo posicional (h1_cols=None, h2_cols=None):
    #      Layout contiguous: [prefix..., z, Re(H1), Im(H1), Re(H2), Im(H2), ...]
    #      em_start = n_prefix + 1 (pula prefix + z_obs)
    #      H1 nas posicoes em_start, em_start+1
    #      H2 nas posicoes em_start+2, em_start+3
    #
    #   2. Modo explicito (h1_cols=(re,im), h2_cols=(re,im)):
    #      H1 e H2 em posicoes arbitrarias no array de features.
    #      Permite features EM extras (Hxy, Hxz) entre H1 e H2.
    #      Colunas nao-H1/H2 sao preservadas intactas.
    #
    # Ref: docs/physics/perspectivas.md, PipelineConfig.input_features.
    # ──────────────────────────────────────────────────────────────────
    if (h1_cols is None) != (h2_cols is None):
        raise ValueError(
            "h1_cols e h2_cols devem ser ambos None ou ambos especificados. "
            f"Recebido h1_cols={h1_cols}, h2_cols={h2_cols}"
        )

    if h1_cols is not None and h2_cols is not None:
        # ── Modo explicito: posicoes arbitrarias de H1/H2 ─────────
        re_h1_idx, im_h1_idx = h1_cols
        re_h2_idx, im_h2_idx = h2_cols
        _max_idx = max(re_h1_idx, im_h1_idx, re_h2_idx, im_h2_idx)
        if result.shape[-1] <= _max_idx:
            raise ValueError(
                f"Array tem {result.shape[-1]} colunas mas h1_cols={h1_cols}, "
                f"h2_cols={h2_cols} requerem pelo menos {_max_idx + 1}"
            )
    else:
        # ── Modo posicional (legado): layout contiguous ────────────
        # Layout: [prefix...] [z] [Re(H1)] [Im(H1)] [Re(H2)] [Im(H2)] [GS...]
        em_start = n_prefix + 1  # pula prefix + z
        if result.shape[-1] < em_start + 4:
            raise ValueError(
                f"Array tem {result.shape[-1]} colunas mas em_start={em_start} "
                f"requer pelo menos {em_start + 4} (n_prefix={n_prefix}, z=1, 4 EM)"
            )
        re_h1_idx = em_start
        im_h1_idx = em_start + 1
        re_h2_idx = em_start + 2
        im_h2_idx = em_start + 3

    # ── Extracao dos 4 canais H1/H2 por indice ────────────────────
    re_h1 = result[:, re_h1_idx]  # Re(H1) — componente planar real
    im_h1 = result[:, im_h1_idx]  # Im(H1) — componente planar imaginaria
    re_h2 = result[:, re_h2_idx]  # Re(H2) — componente axial real
    im_h2 = result[:, im_h2_idx]  # Im(H2) — componente axial imaginaria

    # ── Calculo de magnitude e fase para ambas componentes ─────────
    mag_h1 = _magnitude(re_h1, im_h1)  # |H1| — magnitude planar
    mag_h2 = _magnitude(re_h2, im_h2)  # |H2| — magnitude axial
    phi_h1 = _phase(re_h1, im_h1)  # arg(H1) — fase planar [rad]
    phi_h2 = _phase(re_h2, im_h2)  # arg(H2) — fase axial [rad]

    if view == "H1_logH2":
        # ── H1_logH2: H1 cru preserva SNR em alta atenuacao,
        #    H2 log10-transformado comprime faixa dinamica larga de Hzz.
        #    Saida nos canais H2: [log10|H2|, phi(H2)]
        #    Canais H1 preservados. Colunas extras intactas.
        #    Motivacao fisica: Hzz varia 4+ ordens de magnitude entre
        #    camadas de alta e baixa resistividade. Log10 estabiliza
        #    gradientes e melhora convergencia durante treinamento.
        result[:, re_h2_idx] = _safe_log10(mag_h2)
        result[:, im_h2_idx] = phi_h2

    elif view == "logH1_logH2":
        # ── logH1_logH2: Ambos H1 e H2 em escala logaritmica.
        #    Saida: H1→[log10|H1|, phi(H1)], H2→[log10|H2|, phi(H2)]
        #    Colunas extras intactas.
        #    Motivacao fisica: Magnitude + fase capturam toda informacao
        #    do sinal complexo. Log10 comprime faixa dinamica, tornando
        #    ambas componentes comparaveis em escala para o modelo.
        result[:, re_h1_idx] = _safe_log10(mag_h1)
        result[:, im_h1_idx] = phi_h1
        result[:, re_h2_idx] = _safe_log10(mag_h2)
        result[:, im_h2_idx] = phi_h2

    elif view == "IMH1_IMH2_razao":
        # ── IMH1_IMH2_razao: Partes imaginarias + razao de magnitudes.
        #    Saida: H1→[Im(H1), Im(H2)], H2→[|H1|/|H2|, phi(H1)-phi(H2)]
        #    Colunas extras intactas.
        #    Motivacao fisica: Im(H) eh mais sensivel a contraste de
        #    resistividade em fronteiras. Razao |H1|/|H2| indica anisotropia
        #    (TIV). Diferenca de fase detecta fronteiras de camada.
        result[:, re_h1_idx] = im_h1
        result[:, im_h1_idx] = im_h2
        result[:, re_h2_idx] = mag_h1 / (mag_h2 + EPS)
        result[:, im_h2_idx] = phi_h1 - phi_h2

    elif view == "IMH1_IMH2_lograzao":
        # ── IMH1_IMH2_lograzao: Como razao, mas com log10 da razao.
        #    Saida: H1→[Im(H1), Im(H2)], H2→[log10(|H1|/|H2|), phi(H1)-phi(H2)]
        #    Colunas extras intactas.
        #    Motivacao fisica: Log-razao estabiliza variacao quando
        #    contraste de resistividade eh muito alto (>100:1).
        #    Preferido para cenarios com camadas de sal ou carbonato.
        result[:, re_h1_idx] = im_h1
        result[:, im_h1_idx] = im_h2
        result[:, re_h2_idx] = _safe_log10(mag_h1 / (mag_h2 + EPS))
        result[:, im_h2_idx] = phi_h1 - phi_h2

    elif view == "second_order":
        # ── second_order: Features de 2o grau para alta resistividade.
        #    Substitui 4 canais EM por 6 canais derivados:
        #    [|H1|^2, |H2|^2, d|H1|/dz, d|H2|/dz, Re(H1)/Im(H1), Re(H2)/Im(H2)]
        #    Motivacao fisica: amplifica sinais fracos em rho > 100 Ohm.m
        #    onde Re/Im brutos se aproximam do acoplamento direto.
        #    O array de saida tem shape diferente: n_feat-4+6 = n_feat+2 canais.
        from geosteering_ai.data.second_order import compute_second_order_features

        # Reconstruir h1/h2 cols no espaco 2D flat
        _h1 = (re_h1_idx, im_h1_idx)
        _h2 = (re_h2_idx, im_h2_idx)
        so_feats = compute_second_order_features(result, _h1, _h2, eps=EPS)
        # Substituir colunas EM (4) por SO (6): remover 4 colunas, inserir 6
        # Colunas antes do EM + SO + colunas depois do EM
        _em_indices = sorted([re_h1_idx, im_h1_idx, re_h2_idx, im_h2_idx])
        non_em_mask = np.ones(result.shape[-1], dtype=bool)
        non_em_mask[_em_indices] = False
        non_em = result[:, non_em_mask]
        result = np.concatenate([non_em, so_feats], axis=-1)

    # ── Restaura shape 3D se entrada era 3D ──
    if result.ndim == 2 and x.ndim == 3:
        n_seq = original_shape[0]
        result = result.reshape(n_seq, original_shape[1], -1)

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
    *,
    h1_cols: "tuple[int, int] | None" = None,
    h2_cols: "tuple[int, int] | None" = None,
) -> "tf.Tensor":
    """Aplica Feature View sobre componentes EM (TensorFlow).

    Semantica IDENTICA a versao numpy: mesmos canais, mesma base log10.
    Suporta modo posicional (legado) e modo explicito (h1_cols/h2_cols).

    No modo explicito, os 4 canais H1/H2 sao substituidos via
    construcao de lista de colunas, preservando colunas extras
    (Hxy, Hxz, etc.) intactas.

    Args:
        x: Tensor 3D (batch, seq_len, n_feat).
        view: Nome da Feature View.
        n_prefix: Colunas prefixo antes de z (modo posicional).
        eps: Epsilon para estabilidade numerica.
        h1_cols: Tupla (re_idx, im_idx) posicoes de H1 no tensor.
            Se None, usa modo posicional.
        h2_cols: Tupla (re_idx, im_idx) posicoes de H2 no tensor.
            Se None, usa modo posicional.

    Returns:
        Tensor com mesma shape, canais H1/H2 transformados.

    Note:
        Referenciado em:
            - data/pipeline.py: DataPipeline._apply_fv_gs() (modo offline)
            - data/pipeline.py: build_train_map_fn() (modo on-the-fly, Step 2)
            - tests/test_data_pipeline.py: TestFeatureViews,
              TestFeatureViewsExpanded
        Ref: docs/ARCHITECTURE_v2.md secao 4.3, docs/physics/perspectivas.md.
        Bug fix v2.0: Legado (C22) usava ln (numpy) vs log10 (TF).
            Agora ambos usam log10 consistentemente.
        Guard numerico: EPS = 1e-12 (NUNCA 1e-30 em float32).
        v2.0.1: Adicionado h1_cols/h2_cols para features EM expandidas.
    """
    import tensorflow as tf

    # ── identity/raw: retorna tensor sem transformacao (passthrough) ──
    if view in ("identity", "raw"):
        return x

    # ── Validacao parcial h1/h2 ────────────────────────────────────
    if (h1_cols is None) != (h2_cols is None):
        raise ValueError("h1_cols e h2_cols devem ser ambos None ou ambos especificados.")

    # ── Resolucao das posicoes H1/H2 ──────────────────────────────
    _use_explicit = h1_cols is not None and h2_cols is not None
    if _use_explicit:
        re_h1_idx, im_h1_idx = h1_cols
        re_h2_idx, im_h2_idx = h2_cols
    else:
        em_start = n_prefix + 1
        tf.debugging.assert_greater_equal(
            tf.shape(x)[-1],
            em_start + 4,
            message=f"Tensor precisa de >= {em_start + 4} colunas para FV",
        )
        re_h1_idx = em_start
        im_h1_idx = em_start + 1
        re_h2_idx = em_start + 2
        im_h2_idx = em_start + 3

    # ── Extracao dos 4 canais H1/H2 ───────────────────────────────
    re_h1 = x[:, :, re_h1_idx]  # Re(H1) — componente planar real
    im_h1 = x[:, :, im_h1_idx]  # Im(H1) — componente planar imaginaria
    re_h2 = x[:, :, re_h2_idx]  # Re(H2) — componente axial real
    im_h2 = x[:, :, im_h2_idx]  # Im(H2) — componente axial imaginaria

    # ── Calculo de magnitude e fase (TF ops, autodiff-compativel) ──
    mag_h1 = tf.sqrt(re_h1**2 + im_h1**2 + eps)  # |H1| — magnitude planar
    mag_h2 = tf.sqrt(re_h2**2 + im_h2**2 + eps)  # |H2| — magnitude axial
    phi_h1 = tf.math.atan2(im_h1, re_h1)  # arg(H1) — fase planar [rad]
    phi_h2 = tf.math.atan2(im_h2, re_h2)  # arg(H2) — fase axial [rad]

    # ── Fator de conversao ln→log10 (constante, nao recomputada por amostra) ──
    log10 = tf.math.log(10.0)

    def _safe_log10_tf(val):
        """Log10 seguro via mudanca de base: log10(x) = ln(x) / ln(10)."""
        return tf.math.log(tf.maximum(tf.abs(val), eps)) / log10

    # ── Computa os 4 canais transformados para cada FV ─────────────
    if view == "H1_logH2":
        # ── H1_logH2: H1 cru preserva SNR em alta atenuacao,
        #    H2 log10-transformado comprime faixa dinamica larga de Hzz.
        #    Motivacao fisica: Hzz varia 4+ ordens de magnitude entre
        #    camadas de alta e baixa resistividade.
        t_re_h1 = re_h1
        t_im_h1 = im_h1
        t_re_h2 = _safe_log10_tf(mag_h2)
        t_im_h2 = phi_h2

    elif view == "logH1_logH2":
        # ── logH1_logH2: Ambos H1 e H2 em escala logaritmica.
        #    Motivacao fisica: Magnitude + fase capturam toda informacao
        #    do sinal complexo. Log10 comprime faixa dinamica.
        t_re_h1 = _safe_log10_tf(mag_h1)
        t_im_h1 = phi_h1
        t_re_h2 = _safe_log10_tf(mag_h2)
        t_im_h2 = phi_h2

    elif view == "IMH1_IMH2_razao":
        # ── IMH1_IMH2_razao: Partes imaginarias + razao de magnitudes.
        #    Motivacao fisica: Im(H) eh mais sensivel a contraste de
        #    resistividade em fronteiras.
        ratio = mag_h1 / (mag_h2 + eps)
        t_re_h1 = im_h1
        t_im_h1 = im_h2
        t_re_h2 = ratio
        t_im_h2 = phi_h1 - phi_h2

    elif view == "IMH1_IMH2_lograzao":
        # ── IMH1_IMH2_lograzao: Como razao, mas com log10 da razao.
        #    Motivacao fisica: Log-razao estabiliza variacao quando
        #    contraste de resistividade eh muito alto (>100:1).
        ratio = mag_h1 / (mag_h2 + eps)
        t_re_h1 = im_h1
        t_im_h1 = im_h2
        t_re_h2 = _safe_log10_tf(ratio)
        t_im_h2 = phi_h1 - phi_h2

    elif view == "second_order":
        # ── second_order: Features de 2o grau TF (alta resistividade).
        #    Delega para compute_second_order_features_tf().
        #    Retorna tensor com 6 canais SO no lugar dos 4 EM.
        from geosteering_ai.data.second_order import compute_second_order_features_tf

        _h1 = (re_h1_idx, im_h1_idx)
        _h2 = (re_h2_idx, im_h2_idx)
        so_feats = compute_second_order_features_tf(x, _h1, _h2, eps=eps)
        # ── Construir saida: remover 4 EM, concatenar 6 SO ────────
        # Usa tf.gather para selecionar colunas nao-EM sem loop Python.
        # Compativel com graph mode (tf.function) — indices estaticos.
        _em_set = {re_h1_idx, im_h1_idx, re_h2_idx, im_h2_idx}
        n_feat_static = x.shape[-1]
        if n_feat_static is not None:
            non_em_idx = [i for i in range(n_feat_static) if i not in _em_set]
        else:
            # Fallback para shapes desconhecidos (raro)
            non_em_idx = [i for i in range(22) if i not in _em_set]
        if non_em_idx:
            non_em = tf.gather(x, non_em_idx, axis=-1)
            return tf.concat([non_em, so_feats], axis=-1)
        return so_feats

    else:
        raise ValueError(f"Feature View '{view}' invalida. Validas: {VALID_VIEWS}")

    # ── Reconstrucao do tensor com canais transformados ─────────────
    # Modo explicito: substitui canais individuais no tensor completo,
    # preservando todas as colunas extras (Hxy, Hxz, etc.) intactas.
    # Modo posicional: reconstroi via concat [prefix+z | EM | tail].
    if _use_explicit:
        # ── Listar colunas, substituindo apenas H1/H2 ─────────────
        n_feat = x.shape[-1] or tf.shape(x)[-1]
        _replace = {
            re_h1_idx: t_re_h1,
            im_h1_idx: t_im_h1,
            re_h2_idx: t_re_h2,
            im_h2_idx: t_im_h2,
        }
        cols = []
        for i in range(n_feat if isinstance(n_feat, int) else n_feat.numpy()):
            if i in _replace:
                cols.append(tf.expand_dims(_replace[i], axis=-1))
            else:
                cols.append(x[:, :, i : i + 1])
        return tf.concat(cols, axis=-1)
    else:
        # ── Modo posicional: concat [prefix+z | EM 4 | tail] ──────
        prefix_and_z = x[:, :, :re_h1_idx]
        tail = x[:, :, im_h2_idx + 1 :]
        em = tf.stack([t_re_h1, t_im_h1, t_re_h2, t_im_h2], axis=-1)
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
