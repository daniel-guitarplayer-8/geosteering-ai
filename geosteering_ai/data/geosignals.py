# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: data/geosignals.py                                               ║
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
# ║    • 5 familias de geosinais (USD, UAD, UHR, UHA, U3DF) — P4             ║
# ║    • Implementacoes numpy e TensorFlow CONSISTENTES                       ║
# ║    • Cada familia produz 2 canais: atenuacao (dB) + fase (graus)          ║
# ║    • Razoes compensadas em ganho entre componentes do tensor EM            ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig)                                 ║
# ║  Exports: ~6 (FAMILY_DEPS, compute_expanded_features,                    ║
# ║           compute_geosignals, compute_geosignals_tf)                      ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 4.3, docs/physics/perspectivas.md    ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial com 5 familias               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""Geosignals — Sinais geofisicos derivados do tensor EM.

5 familias de geosinais (USD, UAD, UHR, UHA, U3DF) com versoes
numpy e TensorFlow consistentes. Cada familia produz 2 canais:
atenuacao (dB) e phase shift (graus).

Referencia: docs/reference/geosignals.md, docs/physics/perspectivas.md.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from geosteering_ai.config import PipelineConfig
from geosteering_ai.data.loading import EM_COMPONENTS

logger = logging.getLogger(__name__)

# ┌──────────────────────────────────────────────────────────────────────┐
# │  FLUXO DE COMPUTACAO DE GEOSINAIS                                   │
# ├──────────────────────────────────────────────────────────────────────┤
# │  Re/Im(Hxx,Hyy,Hzz,Hxz,Hzx,Hyz,Hzy) → componentes complexas     │
# │       ↓                                                             │
# │  Razao = f(Z1,Z2) × g(Z3,Z4)   (compensada em ganho)              │
# │       ↓                                                             │
# │  Atenuacao = 20·log10(|Razao|)  [dB, clipped ±100]                │
# │  Fase = angle(Razao)            [graus, clipped ±180]              │
# │       ↓                                                             │
# │  Output: 2 canais por familia (att + phase)                        │
# └──────────────────────────────────────────────────────────────────────┘

# ── Epsilon seguro para float32 (NUNCA 1e-30) ──
# Usado em todas as divisoes complexas e log10 para evitar NaN/Inf.
# O valor 1e-12 e adequado para float32 (precision ~7 digitos decimais).
EPS = 1e-12


# ════════════════════════════════════════════════════════════════════════
# DEPENDENCIAS EM POR FAMILIA (D10: Documentacao de Constantes)
# ════════════════════════════════════════════════════════════════════════
# Cada familia de geosinal utiliza um subconjunto especifico das 9
# componentes do tensor EM 3x3 (XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ).
# Apenas as componentes listadas aqui sao necessarias; as demais
# (XY, YX) nao sao usadas por nenhuma familia implementada.
#
# Mapeamento de componente → colunas no .dat de 22 colunas:
#   XX:(4,5)  XY:(6,7)  XZ:(8,9)
#   YX:(10,11) YY:(12,13) YZ:(14,15)
#   ZX:(16,17) ZY:(18,19) ZZ:(20,21)
#
# Cada tupla (re_idx, im_idx) refere-se a parte real e imaginaria.
# ════════════════════════════════════════════════════════════════════════

# ── USD (Symmetrized Directional): Detecta fronteiras de camada
#    Razao dupla: (ZZ+XZ)/(ZZ-XZ) × (ZZ-ZX)/(ZZ+ZX)
#    Sensivel a contraste de resistividade em fronteiras horizontais.
#    Simetrizada para cancelar efeitos de ferramenta (tool rotation).
#
# ── UAD (Anti-symmetrized Directional): Complementar ao USD
#    Razao: (ZZ+XZ)/(ZZ-XZ) × (ZZ+ZX)/(ZZ-ZX)
#    Anti-simetrica — sensivel a direcao (up/down) da fronteira.
#
# ── UHR (Harmonic Resistivity Ratio): Anisotropia horizontal
#    Razao: -2·ZZ / (XX+YY)
#    Compara componente axial (ZZ) com media planar (XX+YY).
#    Sensivel a anisotropia Rv/Rh (resistividade vertical/horizontal).
#
# ── UHA (Harmonic Anisotropy): Anisotropia planar pura
#    Razao: XX / YY
#    Razao entre componentes planares ortogonais.
#    Detecta anisotropia no plano horizontal (fraturas, laminacoes).
#
# ── U3DF (3D Formation): Similar ao USD, plano YZ
#    Razao dupla: (ZZ+YZ)/(ZZ-YZ) × (ZZ-ZY)/(ZZ+ZY)
#    Sensivel a fronteiras no plano perpendicular ao USD.
#    Complementa USD para geometrias 3D complexas.

FAMILY_DEPS: Dict[str, List[str]] = {
    "USD":  ["ZZ", "XZ", "ZX"],   # Symmetrized Directional — fronteiras camada
    "UAD":  ["ZZ", "XZ", "ZX"],   # Anti-sym Directional — direcao fronteira
    "UHR":  ["ZZ", "XX", "YY"],   # Harmonic Resistivity — anisotropia Rv/Rh
    "UHA":  ["XX", "YY"],         # Harmonic Anisotropy — anisotropia planar
    "U3DF": ["ZZ", "YZ", "ZY"],   # 3D Formation — fronteiras plano YZ
}


def compute_expanded_features(
    base_features: List[int],
    families: List[str],
) -> List[int]:
    """Calcula colunas expandidas necessarias para geosinais.

    Dado INPUT_FEATURES base [1,4,5,20,21] e familias ativas,
    retorna a lista expandida com colunas off-diagonal adicionais.

    Args:
        base_features: Colunas base (ex: [1, 4, 5, 20, 21]).
        families: Familias ativas (ex: ["USD", "UHR"]).

    Returns:
        Lista ordenada e deduplicada de indices de colunas.
    """
    needed = set(base_features)
    for fam in families:
        if fam not in FAMILY_DEPS:
            raise ValueError(f"Familia '{fam}' desconhecida. Validas: {list(FAMILY_DEPS)}")
        for comp in FAMILY_DEPS[fam]:
            re_idx, im_idx = EM_COMPONENTS[comp]
            needed.add(re_idx)
            needed.add(im_idx)
    return sorted(needed)


# ════════════════════════════════════════════════════════════════════════
# HELPER: Conversao de Ratio Complexo → Atenuacao/Fase
# ════════════════════════════════════════════════════════════════════════
# Funcao auxiliar que converte um ratio complexo (resultado da razao
# compensada entre componentes EM) nos dois observaveis fisicos:
#   1. Atenuacao (dB): 20·log10(|ratio|), clipped em [-100, +100]
#   2. Phase shift (graus): angle(ratio) em graus, clipped em [-180, +180]
# Usada tanto pela versao numpy quanto como modelo para a versao TF.
# ════════════════════════════════════════════════════════════════════════

def _geosignal_from_ratio_np(
    ratio: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Converte ratio complexo em atenuacao (dB) e phase shift (deg).

    Args:
        ratio: Array complexo (qualquer shape).

    Returns:
        (attenuation_dB, phase_shift_deg).
    """
    att = 20.0 * np.log10(np.abs(ratio) + EPS)
    att = np.clip(att, -100.0, 100.0)
    phase = np.degrees(np.angle(ratio))
    phase = np.clip(phase, -180.0, 180.0)
    return att, phase


# ════════════════════════════════════════════════════════════════════════
# VERSAO NUMPY — Preprocessamento Offline
# ════════════════════════════════════════════════════════════════════════
# Implementacao NumPy pura para uso em preprocessamento offline
# (split, fit_scaler, EDA). Opera sobre arrays 2D (n_rows, 22).
# Produz resultado identico a versao TF (validado em testes).
# O formato de entrada e o .dat bruto do simulador Fortran com 22
# colunas: [zobs, depth, rho1, rho2, Re/Im das 9 componentes EM].
# ════════════════════════════════════════════════════════════════════════

def _build_complex(data: np.ndarray, comp_name: str) -> np.ndarray:
    """Constroi array complexo a partir de Re/Im no data 2D (n_rows, 22)."""
    re_idx, im_idx = EM_COMPONENTS[comp_name]
    return data[:, re_idx] + 1j * data[:, im_idx]


def compute_geosignals(
    raw_data: np.ndarray,
    families: List[str],
    n_columns: int = 22,
) -> np.ndarray:
    """Calcula geosinais a partir de dados brutos do .dat (numpy).

    Output: 2 canais por familia (atenuacao dB, phase shift deg).

    Args:
        raw_data: Array (n_rows, n_columns) do .dat.
        families: Lista de familias (ex: ["USD", "UHR"]).
        n_columns: Numero de colunas (22 ativo, 12 legado).

    Returns:
        Array (n_rows, 2 * len(families)) com [att_1, phase_1, att_2, ...].

    Note:
        Referenciado em:
            - data/pipeline.py: DataPipeline._apply_fv_gs() (modo offline)
            - data/pipeline.py: build_train_map_fn() Step 3 (on-the-fly)
            - tests/test_data_pipeline.py: TestGeosignals (6 test cases)
        Ref: docs/physics/perspectivas.md secao P4.
        Guard numerico: EPS = 1e-12, clipping att [-100,100] dB, phase [-180,180] graus.
    """
    n_rows = raw_data.shape[0]
    result = np.empty((n_rows, 2 * len(families)), dtype=np.float64)

    for i, fam in enumerate(families):
        if fam not in FAMILY_DEPS:
            raise ValueError(f"Familia '{fam}' desconhecida. Validas: {list(FAMILY_DEPS)}")
        # Verificar disponibilidade de colunas para a familia solicitada
        for comp in FAMILY_DEPS[fam]:
            re_idx, im_idx = EM_COMPONENTS[comp]
            if re_idx >= n_columns or im_idx >= n_columns:
                raise ValueError(
                    f"Familia '{fam}' requer componente {comp} "
                    f"(cols {re_idx},{im_idx}), indisponivel com {n_columns} colunas"
                )

        if fam == "USD":
            # ── USD (Symmetrized Directional): Detecta fronteiras de camada.
            #    Formula: (ZZ+XZ)/(ZZ-XZ) × (ZZ-ZX)/(ZZ+ZX)
            #    Sensivel a contraste de resistividade em fronteiras.
            #    Simetrizada para cancelar efeitos de rotacao da ferramenta (tool rotation).
            #    Requer componentes off-diagonal (XZ, ZX) → apenas 22-col.
            #    Saida: [USD_att (dB), USD_phase (graus)]
            zz = _build_complex(raw_data, "ZZ")
            xz = _build_complex(raw_data, "XZ")
            zx = _build_complex(raw_data, "ZX")
            ratio = ((zz + xz) / (zz - xz + EPS)) * ((zz - zx) / (zz + zx + EPS))

        elif fam == "UAD":
            # ── UAD (Anti-symmetrized Directional): Direcao da fronteira.
            #    Formula: (ZZ+XZ)/(ZZ-XZ) × (ZZ+ZX)/(ZZ-ZX)
            #    Anti-simetrica — indica direcao (up/down) da fronteira.
            #    Complementar ao USD: USD detecta presenca, UAD detecta sentido.
            #    Requer componentes off-diagonal (XZ, ZX) → apenas 22-col.
            #    Saida: [UAD_att (dB), UAD_phase (graus)]
            zz = _build_complex(raw_data, "ZZ")
            xz = _build_complex(raw_data, "XZ")
            zx = _build_complex(raw_data, "ZX")
            ratio = ((zz + xz) / (zz - xz + EPS)) * ((zz + zx) / (zz - zx + EPS))

        elif fam == "UHR":
            # ── UHR (Harmonic Resistivity Ratio): Anisotropia Rv/Rh.
            #    Formula: -2·ZZ / (XX+YY)
            #    Compara componente axial (ZZ) com media planar (XX+YY).
            #    Sensivel a anisotropia Rv/Rh (resistividade vertical/horizontal).
            #    Usa apenas componentes diagonais → disponivel em 12-col e 22-col.
            #    Saida: [UHR_att (dB), UHR_phase (graus)]
            zz = _build_complex(raw_data, "ZZ")
            xx = _build_complex(raw_data, "XX")
            yy = _build_complex(raw_data, "YY")
            ratio = -2.0 * zz / (xx + yy + EPS)

        elif fam == "UHA":
            # ── UHA (Harmonic Anisotropy): Anisotropia planar pura.
            #    Formula: XX / YY
            #    Razao entre componentes planares ortogonais.
            #    Detecta anisotropia no plano horizontal (fraturas, laminacoes).
            #    Usa apenas componentes diagonais → disponivel em 12-col e 22-col.
            #    Saida: [UHA_att (dB), UHA_phase (graus)]
            xx = _build_complex(raw_data, "XX")
            yy = _build_complex(raw_data, "YY")
            ratio = xx / (yy + EPS)

        elif fam == "U3DF":
            # ── U3DF (3D Formation): Fronteiras no plano YZ.
            #    Formula: (ZZ+YZ)/(ZZ-YZ) × (ZZ-ZY)/(ZZ+ZY)
            #    Sensivel a fronteiras no plano perpendicular ao USD.
            #    Complementa USD para geometrias 3D complexas (formacoes inclinadas).
            #    Requer componentes off-diagonal (YZ, ZY) → apenas 22-col.
            #    Saida: [U3DF_att (dB), U3DF_phase (graus)]
            zz = _build_complex(raw_data, "ZZ")
            yz = _build_complex(raw_data, "YZ")
            zy = _build_complex(raw_data, "ZY")
            ratio = ((zz + yz) / (zz - yz + EPS)) * ((zz - zy) / (zz + zy + EPS))

        else:
            raise ValueError(f"Familia '{fam}' desconhecida")

        att, phase = _geosignal_from_ratio_np(ratio)
        result[:, 2 * i] = att
        result[:, 2 * i + 1] = phase

    logger.info(
        "Geosinais calculados: familias=%s, shape=%s",
        families, result.shape,
    )
    return result


# ════════════════════════════════════════════════════════════════════════
# VERSAO TENSORFLOW — On-the-fly dentro de tf.data.map()
# ════════════════════════════════════════════════════════════════════════
# Implementacao TF para uso on-the-fly no pipeline de treinamento.
# Semantica IDENTICA a versao numpy (validada em testes unitarios).
# Opera sobre tensores expandidos (batch, seq_len, n_expanded_feat),
# onde n_expanded_feat inclui as colunas off-diagonal adicionais.
# O import de tensorflow e lazy (dentro da funcao) para evitar
# carregar o TF no momento do import do modulo.
# ════════════════════════════════════════════════════════════════════════

def compute_geosignals_tf(
    x_expanded: "tf.Tensor",
    families: List[str],
    expanded_features: List[int],
    eps: float = EPS,
) -> "tf.Tensor":
    """Calcula geosinais on-the-fly (TensorFlow).

    Semantica IDENTICA a versao numpy. Opera sobre tensor expandido
    que contem as colunas off-diagonal necessarias.

    Args:
        x_expanded: Tensor (batch, seq_len, n_expanded_feat).
        families: Familias ativas (ex: ["USD", "UHR"]).
        expanded_features: Lista de colunas .dat presentes no tensor
            (posicao i no tensor = coluna expanded_features[i] do .dat).
        eps: Epsilon para estabilidade numerica.

    Returns:
        Tensor (batch, seq_len, 2 * len(families)).

    Note:
        Referenciado em:
            - data/pipeline.py: DataPipeline._apply_fv_gs() (modo offline)
            - data/pipeline.py: build_train_map_fn() Step 3 (on-the-fly)
            - tests/test_data_pipeline.py: TestGeosignals (6 test cases)
        Ref: docs/physics/perspectivas.md secao P4.
        Guard numerico: EPS = 1e-12, clipping att [-100,100] dB, phase [-180,180] graus.
    """
    import tensorflow as tf

    pi = tf.constant(3.141592653589793, dtype=tf.float32)

    def _get_complex(comp_name: str) -> tf.Tensor:
        """Constroi tensor complexo para componente EM."""
        re_idx, im_idx = EM_COMPONENTS[comp_name]
        re_pos = expanded_features.index(re_idx)
        im_pos = expanded_features.index(im_idx)
        re_val = tf.cast(x_expanded[:, :, re_pos], tf.float32)
        im_val = tf.cast(x_expanded[:, :, im_pos], tf.float32)
        return tf.complex(re_val, im_val)

    def _safe_div(num: tf.Tensor, den: tf.Tensor) -> tf.Tensor:
        """Divisao complexa segura (substitui denominador < eps por eps)."""
        abs_den = tf.abs(den)
        safe_den = tf.where(
            abs_den < eps,
            tf.complex(tf.constant(eps, tf.float32), tf.constant(0.0, tf.float32)),
            den,
        )
        return num / safe_den

    def _ratio_to_att_phase(ratio: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Converte ratio complexo em att (dB) e phase (deg)."""
        log10 = tf.math.log(tf.constant(10.0, dtype=tf.float32))
        abs_ratio = tf.abs(ratio)
        att = 20.0 * tf.math.log(abs_ratio + eps) / log10
        att = tf.clip_by_value(att, -100.0, 100.0)
        phase = tf.math.angle(ratio) * (180.0 / pi)
        phase = tf.clip_by_value(phase, -180.0, 180.0)
        return att, phase

    channels = []
    for fam in families:
        if fam == "USD":
            # ── USD (Symmetrized Directional): Detecta fronteiras de camada.
            #    Formula: (ZZ+XZ)/(ZZ-XZ) × (ZZ-ZX)/(ZZ+ZX)
            #    Sensivel a contraste de resistividade em fronteiras.
            #    Simetrizada para cancelar efeitos de rotacao da ferramenta (tool rotation).
            #    Requer componentes off-diagonal (XZ, ZX) → apenas 22-col.
            #    Saida: [USD_att (dB), USD_phase (graus)]
            zz = _get_complex("ZZ")
            xz = _get_complex("XZ")
            zx = _get_complex("ZX")
            r = _safe_div(zz + xz, zz - xz) * _safe_div(zz - zx, zz + zx)

        elif fam == "UAD":
            # ── UAD (Anti-symmetrized Directional): Direcao da fronteira.
            #    Formula: (ZZ+XZ)/(ZZ-XZ) × (ZZ+ZX)/(ZZ-ZX)
            #    Anti-simetrica — indica direcao (up/down) da fronteira.
            #    Complementar ao USD: USD detecta presenca, UAD detecta sentido.
            #    Requer componentes off-diagonal (XZ, ZX) → apenas 22-col.
            #    Saida: [UAD_att (dB), UAD_phase (graus)]
            zz = _get_complex("ZZ")
            xz = _get_complex("XZ")
            zx = _get_complex("ZX")
            r = _safe_div(zz + xz, zz - xz) * _safe_div(zz + zx, zz - zx)

        elif fam == "UHR":
            # ── UHR (Harmonic Resistivity Ratio): Anisotropia Rv/Rh.
            #    Formula: -2·ZZ / (XX+YY)
            #    Compara componente axial (ZZ) com media planar (XX+YY).
            #    Sensivel a anisotropia Rv/Rh (resistividade vertical/horizontal).
            #    Usa apenas componentes diagonais → disponivel em 12-col e 22-col.
            #    Saida: [UHR_att (dB), UHR_phase (graus)]
            zz = _get_complex("ZZ")
            xx = _get_complex("XX")
            yy = _get_complex("YY")
            r = _safe_div(-2.0 * zz, xx + yy)

        elif fam == "UHA":
            # ── UHA (Harmonic Anisotropy): Anisotropia planar pura.
            #    Formula: XX / YY
            #    Razao entre componentes planares ortogonais.
            #    Detecta anisotropia no plano horizontal (fraturas, laminacoes).
            #    Usa apenas componentes diagonais → disponivel em 12-col e 22-col.
            #    Saida: [UHA_att (dB), UHA_phase (graus)]
            xx = _get_complex("XX")
            yy = _get_complex("YY")
            r = _safe_div(xx, yy)

        elif fam == "U3DF":
            # ── U3DF (3D Formation): Fronteiras no plano YZ.
            #    Formula: (ZZ+YZ)/(ZZ-YZ) × (ZZ-ZY)/(ZZ+ZY)
            #    Sensivel a fronteiras no plano perpendicular ao USD.
            #    Complementa USD para geometrias 3D complexas (formacoes inclinadas).
            #    Requer componentes off-diagonal (YZ, ZY) → apenas 22-col.
            #    Saida: [U3DF_att (dB), U3DF_phase (graus)]
            zz = _get_complex("ZZ")
            yz = _get_complex("YZ")
            zy = _get_complex("ZY")
            r = _safe_div(zz + yz, zz - yz) * _safe_div(zz - zy, zz + zy)

        else:
            raise ValueError(f"Familia '{fam}' desconhecida")

        att, phase = _ratio_to_att_phase(r)
        channels.extend([att, phase])

    if not channels:
        batch_shape = tf.shape(x_expanded)
        return tf.zeros([batch_shape[0], batch_shape[1], 0], dtype=tf.float32)

    return tf.stack(channels, axis=-1)


# ════════════════════════════════════════════════════════════════════════
# __all__ — Exports publicos do modulo (D8)
# ════════════════════════════════════════════════════════════════════════
# Lista explicita de nomes exportados por este modulo.
# Funcoes privadas (_build_complex, _geosignal_from_ratio_np) NAO sao
# exportadas — uso interno apenas.
# ════════════════════════════════════════════════════════════════════════

__all__ = [
    "EPS",
    "FAMILY_DEPS",
    "compute_expanded_features",
    "compute_geosignals",
    "compute_geosignals_tf",
]
