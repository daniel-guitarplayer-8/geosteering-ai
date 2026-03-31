# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODULO: data/boundaries.py                                                ║
# ║  Bloco: 2d — DTB (Distance to Boundary) Labels                            ║
# ║                                                                            ║
# ║  Geosteering AI v2.0 — Inversao 1D de Resistividade via Deep Learning     ║
# ║  Autor: Daniel Leal                                                        ║
# ║  Framework: TensorFlow 2.x / Keras (exclusivo — PyTorch PROIBIDO)         ║
# ║  Ambiente: VSCode + Claude Code (dev) · GitHub CI · Colab Pro+ GPU (exec) ║
# ║  Pacote: geosteering_ai (pip installable)                                 ║
# ║  Config: PipelineConfig dataclass (ponto unico de verdade)                ║
# ║                                                                            ║
# ║  Proposito:                                                                ║
# ║    • detect_boundaries(): detecta fronteiras geologicas a partir de        ║
# ║      perfis de resistividade (mudancas abruptas em rho_h/rho_v)           ║
# ║    • compute_dtb_labels(): calcula DTB_up/DTB_down por ponto de medicao   ║
# ║    • apply_dtb_scaling(): escala DTB (linear, log1p, normalized)          ║
# ║    • inverse_dtb_scaling(): inverte escala DTB para metros                ║
# ║    • build_extended_targets(): concatena [rho_h, rho_v, DTB_up, DTB_down, ║
# ║      rho_up, rho_down] em array 6-canais para output_channels=6          ║
# ║                                                                            ║
# ║  Dependencias: config.py (PipelineConfig)                                 ║
# ║  Exports: ~6 funcoes — ver __all__                                        ║
# ║  Ref: docs/ARCHITECTURE_v2.md secao 5 (Perspectiva 5 — Picasso/DTB)      ║
# ║       docs/physics/perspectivas.md secao P5                               ║
# ║                                                                            ║
# ║  Historico:                                                                ║
# ║    v2.0.0 (2026-03) — Implementacao inicial (P5 DTB)                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

"""DTB (Distance to Boundary) — Labels de distancia a fronteiras geologicas.

Computa labels DTB para cada ponto de medicao em perfis de resistividade.
DTB e a distancia (em metros) do poco ate a fronteira geologica mais
proxima acima (DTB_up) e abaixo (DTB_down).

Fisica:
    Em um modelo geologico 1D com camadas horizontais, fronteiras
    sao definidas por mudancas abruptas de resistividade (contraste).
    O DTB e derivado da posicao de cada ponto em relacao as fronteiras
    detectadas no perfil de resistividade verdadeiro (ground truth).

    DTB_up: distancia ate a fronteira ACIMA (z menor, mais raso)
    DTB_down: distancia ate a fronteira ABAIXO (z maior, mais profundo)

    ┌────────────────────────────────────────────────────────────────┐
    │  Perfil de resistividade e DTB:                                │
    │                                                                │
    │  rho (Ohm.m)     DTB_up (m)     DTB_down (m)                  │
    │  ▲                ▲               ▲                            │
    │  │                │               │                            │
    │  │──┐  Camada A   │╲              │          ╱│                │
    │  │  │(rho=100)    │  ╲            │        ╱  │                │
    │  │  │             │    ╲          │      ╱    │                │
    │  │  │─────────────│──── boundary1 │────╳──────│                │
    │  │  │  Camada B   │    ╱          │      ╲    │                │
    │  │  │  (rho=10)   │  ╱            │        ╲  │                │
    │  │  │             │╱              │          ╲│                │
    │  │  │─────────────│──── boundary2 │────╳──────│                │
    │  │  │  Camada C   │╲              │          ╱│                │
    │  │──┘  (rho=500)  │  ╲            │        ╱  │                │
    │  └────────────────└────────────────└────────────────           │
    │        depth (z)                                               │
    └────────────────────────────────────────────────────────────────┘

    Nota: DTB e clippado em [0, dtb_max_from_picasso] para evitar
    valores muito grandes em camadas espessas. O DOD maximo tipico
    de ferramentas LWD eh ~1.5-3.0 m (Picasso plot).

Cadeia de uso:
    1. detect_boundaries(rho_h, z_obs) → boundary_indices
    2. compute_dtb_labels(z_obs, boundary_indices, dtb_max) → dtb_up, dtb_down
    3. build_extended_targets(rho_h, rho_v, dtb_up, dtb_down, ...) → y_6ch
    4. apply_dtb_scaling(dtb, method, dtb_max) → dtb_scaled

Referencia: docs/physics/perspectivas.md secao P5.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from geosteering_ai.config import PipelineConfig

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ════════════════════════════════════════════════════════════════════════════
# Inventario completo de simbolos exportados por este modulo.
# Agrupados semanticamente para facilitar navegacao.
# ──────────────────────────────────────────────────────────────────────────

__all__ = [
    # ── Deteccao de fronteiras ────────────────────────────────────────
    "detect_boundaries",
    # ── Computacao de DTB labels ──────────────────────────────────────
    "compute_dtb_labels",
    # ── Scaling DTB ───────────────────────────────────────────────────
    "apply_dtb_scaling",
    "inverse_dtb_scaling",
    # ── Targets estendidos (6 canais) ─────────────────────────────────
    "build_extended_targets",
    # ── Pipeline completo ─────────────────────────────────────────────
    "compute_dtb_for_dataset",
]


# ════════════════════════════════════════════════════════════════════════════
# SECAO: DETECCAO DE FRONTEIRAS GEOLOGICAS
# ════════════════════════════════════════════════════════════════════════════
# Fronteiras geologicas sao detectadas a partir de mudancas abruptas
# no perfil de resistividade. O gradiente discreto |diff(rho_h)| e
# comparado com um threshold para identificar transicoes.
#
# Metodo:
#   1. Computa derivada discreta do perfil: grad = diff(rho_h)
#   2. Normaliza pela faixa dinamica local (ou absoluta)
#   3. Identifica picos de |grad| acima do threshold
#   4. Retorna indices de fronteira como boundary_indices
#
# Nota: Usa rho_h (horizontal) pois e mais sensivel a camadas
# horizontais do que rho_v (vertical). Em modelos TIV, rho_h mostra
# transicoes mais abruptas nas fronteiras de camada.
#
# Ref: docs/physics/perspectivas.md secao P5 (DTB as target).
# ──────────────────────────────────────────────────────────────────────────


def detect_boundaries(
    rho_profile: np.ndarray,
    z_obs: np.ndarray,
    *,
    threshold: float = 0.3,
    min_separation: int = 3,
) -> np.ndarray:
    """Detecta fronteiras geologicas em perfil de resistividade.

    Identifica posicoes onde a resistividade muda abruptamente,
    indicando transicao entre camadas geologicas distintas.
    Usa o gradiente normalizado do log10(rho) para ser robusto
    a variacoes de escala (resistividade varia 4+ ordens de magnitude).

    Algoritmo:
        1. Transforma rho para log10 (comprime faixa dinamica)
        2. Calcula gradiente discreto: grad[i] = log_rho[i+1] - log_rho[i]
        3. Identifica pontos com |grad| > threshold
        4. Remove deteccoes muito proximas (< min_separation pontos)
        5. Retorna indices de fronteira ordenados

    Args:
        rho_profile: Perfil de resistividade 1D (seq_len,) em Ohm.m.
            Tipicamente rho_h (col 2 do formato 22-col) por ser mais
            sensivel a fronteiras de camada horizontais.
        z_obs: Profundidade observada 1D (seq_len,) em metros.
            Mesma ordenacao que rho_profile. Usado para calcular
            gradiente em unidades fisicas (Ohm.m/m).
        threshold: Limiar de |grad(log10(rho))| para deteccao.
            Default: 0.3 (equivale a ~2x contraste de resistividade).
            Valores menores detectam transicoes mais sutis.
            Valores maiores so detectam contrastes fortes.
        min_separation: Distancia minima entre fronteiras consecutivas
            em numero de pontos. Default: 3 (evita deteccoes espurias
            em zonas de transicao gradual). Para spacing=1.0 m,
            min_separation=3 corresponde a 3 metros.

    Returns:
        np.ndarray: Indices (int64) das posicoes de fronteira no perfil.
            Shape: (n_boundaries,). Ordenado por posicao crescente.
            Se nenhuma fronteira detectada, retorna array vazio.

    Raises:
        ValueError: Se rho_profile e z_obs tem shapes incompativeis.

    Example:
        >>> rho = np.array([10.0]*50 + [100.0]*50 + [5.0]*50)
        >>> z = np.arange(150, dtype=float)
        >>> boundaries = detect_boundaries(rho, z)
        >>> # boundaries contem ~[49, 99] (transicoes 10->100, 100->5)

    Note:
        Referenciado em:
            - data/boundaries.py: compute_dtb_labels() (usa resultado)
            - data/boundaries.py: compute_dtb_for_dataset() (pipeline completo)
            - tests/test_boundaries.py: TestDetectBoundaries
        Ref: docs/physics/perspectivas.md secao P5.
        Usa log10(rho) para deteccao normalizada por escala.
        threshold=0.3 captura contrastes >= 2x (~100% variacao).
        rho_h preferido sobre rho_v para deteccao de camadas horizontais.
    """
    if rho_profile.shape != z_obs.shape:
        raise ValueError(
            f"rho_profile shape {rho_profile.shape} != z_obs shape {z_obs.shape}"
        )
    if rho_profile.ndim != 1:
        raise ValueError(f"rho_profile deve ser 1D, recebido ndim={rho_profile.ndim}")

    n = len(rho_profile)
    if n < 2:
        return np.array([], dtype=np.int64)

    # ── Transformar para log10 — comprime faixa dinamica ──────────────
    # Resistividade varia de 0.1 a 10000+ Ohm.m (4+ ordens de magnitude).
    # log10 torna a deteccao de contraste independente da escala absoluta:
    #   contraste 10->100 (10x): |grad| = |log10(100) - log10(10)| = 1.0
    #   contraste 1->10 (10x):   |grad| = |log10(10) - log10(1)| = 1.0
    # Sem log10, o segundo contraste seria ignorado (grad = 9 vs 90).
    log_rho = np.log10(np.maximum(rho_profile, 1e-12))

    # ── Gradiente discreto normalizado ────────────────────────────────
    # grad[i] = log_rho[i+1] - log_rho[i]
    # Valor positivo: aumento de resistividade (ex: folhelho -> arenito)
    # Valor negativo: diminuicao (ex: arenito -> folhelho)
    grad = np.abs(np.diff(log_rho))

    # ── Deteccao por threshold ────────────────────────────────────────
    # grad[i] = |log_rho[i+1] - log_rho[i]| → transicao entre i e i+1.
    # Usamos i+1 como indice da fronteira (primeiro ponto da nova camada).
    # Sem +1, o indice cairia no ultimo ponto da camada antiga,
    # causando bias sistematico de 1 ponto em DTB_up/DTB_down.
    candidates = np.where(grad > threshold)[0] + 1

    if len(candidates) == 0:
        return np.array([], dtype=np.int64)

    # ── Remover deteccoes muito proximas ──────────────────────────────
    # Em zonas de transicao gradual, multiplos pontos consecutivos
    # podem ultrapassar o threshold. Mantemos apenas o ponto com
    # maior |grad| em cada janela de min_separation pontos.
    filtered: List[int] = [candidates[0]]
    for c in candidates[1:]:
        if c - filtered[-1] >= min_separation:
            filtered.append(c)
        else:
            # ── Substituir se gradiente maior (pico local) ────────────
            if grad[c] > grad[filtered[-1]]:
                filtered[-1] = c

    boundaries = np.array(filtered, dtype=np.int64)

    logger.debug(
        "detect_boundaries: %d fronteiras detectadas em %d pontos (threshold=%.3f)",
        len(boundaries),
        n,
        threshold,
    )

    return boundaries


# ════════════════════════════════════════════════════════════════════════════
# SECAO: COMPUTACAO DE DTB LABELS
# ════════════════════════════════════════════════════════════════════════════
# Para cada ponto de medicao, computa a distancia ate a fronteira
# geologica mais proxima acima (DTB_up) e abaixo (DTB_down).
#
# Formulas:
#   DTB_up[i]   = z[i] - z[boundary_above(i)]
#   DTB_down[i] = z[boundary_below(i)] - z[i]
#
# Se nao ha fronteira acima/abaixo, DTB = dtb_max (clipping).
# dtb_max tipico: 3.0 m (DOD maximo do Picasso plot).
#
#   ┌──────────────────────────────────────────────────────────────────┐
#   │  Exemplo DTB para 3 camadas:                                     │
#   │                                                                  │
#   │  z (m)   rho      DTB_up   DTB_down   Fronteira                 │
#   │  0.0     100      3.0*     2.0                                  │
#   │  1.0     100      3.0*     1.0                                  │
#   │  2.0     ───── boundary1 ─────        <- fronteira              │
#   │  2.0     10       0.0      1.5                                  │
#   │  3.0     10       1.0      0.5                                  │
#   │  3.5     ───── boundary2 ─────        <- fronteira              │
#   │  3.5     500      0.0      3.0*                                 │
#   │  4.5     500      1.0      3.0*                                 │
#   │                                                                  │
#   │  * = clippado em dtb_max=3.0 m                                  │
#   └──────────────────────────────────────────────────────────────────┘
#
# Ref: docs/physics/perspectivas.md secao P5 (DTB as target).
# ──────────────────────────────────────────────────────────────────────────


def compute_dtb_labels(
    z_obs: np.ndarray,
    boundary_indices: np.ndarray,
    dtb_max: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computa labels DTB_up e DTB_down para cada ponto de medicao.

    Para cada posicao no perfil, calcula a distancia em metros ate
    a fronteira geologica mais proxima acima (DTB_up) e abaixo (DTB_down).
    Valores sao clippados em [0, dtb_max].

    Args:
        z_obs: Profundidade observada 1D (seq_len,) em metros.
            Deve estar em ordem crescente (profundidade aumenta com indice).
        boundary_indices: Indices das fronteiras detectadas por
            detect_boundaries(). Shape: (n_boundaries,).
        dtb_max: Distancia maxima de clipping em metros. Default: 3.0.
            Baseado no DOD maximo tipico de ferramentas LWD (~1.5-3.0 m).
            Valores alem de dtb_max sao saturados (a rede nao consegue
            detectar fronteiras alem do DOI da ferramenta).

    Returns:
        Tuple[np.ndarray, np.ndarray]: (dtb_up, dtb_down), ambos shape (seq_len,).
            dtb_up[i]: distancia ate fronteira acima (z menor) em metros.
            dtb_down[i]: distancia ate fronteira abaixo (z maior) em metros.
            Range: [0.0, dtb_max].

    Example:
        >>> z = np.arange(100, dtype=float)
        >>> boundaries = np.array([30, 70])
        >>> dtb_up, dtb_down = compute_dtb_labels(z, boundaries, dtb_max=3.0)
        >>> dtb_up[30]   # 0.0 (ponto eh fronteira)
        >>> dtb_up[33]   # 3.0 (3m da fronteira, = dtb_max)
        >>> dtb_down[27] # 3.0 (3m da fronteira, = dtb_max)

    Note:
        Referenciado em:
            - data/boundaries.py: compute_dtb_for_dataset() (pipeline completo)
            - data/boundaries.py: build_extended_targets() (constroi y_6ch)
            - tests/test_boundaries.py: TestComputeDTBLabels
        Ref: docs/physics/perspectivas.md secao P5.
        Clipping em dtb_max: fronteiras alem do DOI da ferramenta LWD
        nao sao detectaveis — saturar evita que a rede tente predizer
        distancias alem da capacidade fisica da medida EM.
    """
    n = len(z_obs)

    # ── Caso sem fronteiras: tudo clippado em dtb_max ─────────────────
    if len(boundary_indices) == 0:
        dtb_up = np.full(n, dtb_max, dtype=np.float32)
        dtb_down = np.full(n, dtb_max, dtype=np.float32)
        return dtb_up, dtb_down

    # ── Extrair profundidades das fronteiras ──────────────────────────
    # boundary_z[k] = profundidade da k-esima fronteira em metros
    boundary_z = z_obs[boundary_indices]

    # ── DTB vetorizado via searchsorted ──────────────────────────────
    # np.searchsorted aceita arrays: computa todos os pontos de uma vez.
    # Eliminamos o loop Python para performance com seq_len ate 100000.
    dtb_up = np.full(n, dtb_max, dtype=np.float32)
    dtb_down = np.full(n, dtb_max, dtype=np.float32)

    # ── DTB_up: fronteira acima (maior boundary_z <= z_i) ─────────────
    # searchsorted(side='right'): k[i] = indice do primeiro boundary_z > z_obs[i]
    # A fronteira acima e boundary_z[k[i]-1] se k[i] > 0.
    k_right = np.searchsorted(boundary_z, z_obs, side="right")
    mask_up = k_right > 0
    dtb_up[mask_up] = np.minimum(
        z_obs[mask_up] - boundary_z[k_right[mask_up] - 1],
        dtb_max,
    ).astype(np.float32)

    # ── DTB_down: fronteira abaixo (menor boundary_z >= z_i) ──────────
    # searchsorted(side='left'): k[i] = indice do primeiro boundary_z >= z_obs[i]
    k_left = np.searchsorted(boundary_z, z_obs, side="left")
    mask_down = k_left < len(boundary_z)
    dtb_down[mask_down] = np.minimum(
        boundary_z[k_left[mask_down]] - z_obs[mask_down],
        dtb_max,
    ).astype(np.float32)

    # ── Garantir range [0, dtb_max] ───────────────────────────────────
    dtb_up = np.clip(dtb_up, 0.0, dtb_max)
    dtb_down = np.clip(dtb_down, 0.0, dtb_max)

    logger.debug(
        "compute_dtb_labels: n=%d, n_boundaries=%d, "
        "dtb_up range=[%.3f, %.3f], dtb_down range=[%.3f, %.3f]",
        n,
        len(boundary_indices),
        dtb_up.min(),
        dtb_up.max(),
        dtb_down.min(),
        dtb_down.max(),
    )

    return dtb_up, dtb_down


# ════════════════════════════════════════════════════════════════════════════
# SECAO: DTB SCALING
# ════════════════════════════════════════════════════════════════════════════
# DTB (metros) precisa de scaling diferente de resistividade.
# Resistividade usa log10 (4+ ordens de magnitude).
# DTB eh linear em [0, dtb_max] — log10 nao faz sentido aqui.
#
# Metodos:
#   ┌─────────────┬─────────────────────────────────┬──────────────────────┐
#   │ Metodo      │ Transformacao (forward)          │ Inversa              │
#   ├─────────────┼─────────────────────────────────┼──────────────────────┤
#   │ linear      │ y' = y (metros, sem transform.)  │ y = y'               │
#   │ log         │ y' = log1p(y)                    │ y = expm1(y')        │
#   │ normalized  │ y' = y / dtb_max  -> [0, 1]      │ y = y' x dtb_max    │
#   └─────────────┴─────────────────────────────────┴──────────────────────┘
#
# "linear" (default): preserva interpretabilidade direta em metros.
# "log": comprime range para camadas espessas, similar a log1p(DTB).
# "normalized": normaliza para [0,1] — compativel com sigmoid output.
#
# Ref: docs/physics/perspectivas.md secao P5 (DTB_SCALING).
# ──────────────────────────────────────────────────────────────────────────


def apply_dtb_scaling(
    dtb: np.ndarray,
    method: str = "linear",
    dtb_max: float = 3.0,
) -> np.ndarray:
    """Aplica scaling nos labels DTB.

    DTB (Distance to Boundary) requer scaling diferente de
    resistividade. Enquanto rho usa log10 para comprimir 4+ ordens
    de magnitude, DTB opera em range limitado [0, dtb_max].

    Args:
        dtb: Array de DTB em metros. Qualquer shape. Range: [0, dtb_max].
        method: Metodo de scaling. Default: "linear".
            - "linear": sem transformacao (metros). Preserva interpretabilidade.
            - "log": log1p(dtb) — comprime para camadas espessas.
            - "normalized": dtb/dtb_max -> [0, 1] — compativel com sigmoid.
        dtb_max: DTB maximo em metros. Usado para normalizacao.

    Returns:
        np.ndarray: DTB escalado com mesma shape.

    Raises:
        ValueError: Se method invalido.

    Example:
        >>> dtb = np.array([0.0, 1.5, 3.0])
        >>> apply_dtb_scaling(dtb, "normalized", dtb_max=3.0)
        array([0. , 0.5, 1. ])

    Note:
        Referenciado em:
            - data/boundaries.py: compute_dtb_for_dataset() (pipeline)
            - data/pipeline.py: DataPipeline.prepare() (scaling DTB targets)
            - data/scaling.py: complementa apply_target_scaling (rho-only)
            - tests/test_boundaries.py: TestDTBScaling
        Ref: docs/physics/perspectivas.md secao P5 (DTB_SCALING).
        DTB ja esta clippado em [0, dtb_max] por compute_dtb_labels().
        "linear" e o default — sem transformacao, metros diretos.
    """
    if method == "linear":
        # ── linear: DTB em metros, sem transformacao ──────────────────
        # Preserva interpretabilidade direta: predicao eh em metros.
        # Adequado quando DTB range eh pequeno (0-3m tipicamente).
        return dtb.copy()
    elif method == "log":
        # ── log: log1p(DTB) — compressao logaritmica ─────────────────
        # log1p(0) = 0, log1p(3) = 1.39.
        # Util se DTB varia muito entre camadas finas e espessas.
        # Nao usa log10 pois DTB range eh limitado (nao 4+ ordens).
        return np.log1p(dtb)
    elif method == "normalized":
        # ── normalized: DTB/dtb_max -> [0, 1] ─────────────────────────
        # Normalizacao por range fixo. Compativel com sigmoid na saida.
        # dtb_max vem do Picasso plot (DOD maximo da ferramenta LWD).
        return dtb / max(dtb_max, 1e-12)
    else:
        raise ValueError(
            f"dtb_scaling '{method}' invalido. Validos: 'linear', 'log', 'normalized'"
        )


def inverse_dtb_scaling(
    dtb_scaled: np.ndarray,
    method: str = "linear",
    dtb_max: float = 3.0,
) -> np.ndarray:
    """Inverte scaling de DTB para metros.

    Operacao inversa de apply_dtb_scaling(). Usada na inferencia
    para converter predicoes DTB do modelo de volta para metros.

    Args:
        dtb_scaled: Array de DTB escalado.
        method: Metodo usado na transformacao.
        dtb_max: DTB maximo em metros (para "normalized").

    Returns:
        np.ndarray: DTB em metros com mesma shape.

    Raises:
        ValueError: Se method invalido.

    Example:
        >>> dtb_scaled = np.array([0.0, 0.5, 1.0])
        >>> inverse_dtb_scaling(dtb_scaled, "normalized", dtb_max=3.0)
        array([0. , 1.5, 3. ])

    Note:
        Referenciado em:
            - inference/pipeline.py: InferencePipeline.predict() (inversao DTB)
            - tests/test_boundaries.py: TestDTBScaling (roundtrip)
        Ref: docs/physics/perspectivas.md secao P5.
        Par com apply_dtb_scaling() — roundtrip: f_inv(f(x)) == x.
    """
    if method == "linear":
        return dtb_scaled.copy()
    elif method == "log":
        # ── log inversa: expm1(y') restaura DTB em metros ────────────
        return np.expm1(dtb_scaled)
    elif method == "normalized":
        # ── normalized inversa: y' x dtb_max restaura metros ─────────
        return dtb_scaled * dtb_max
    else:
        raise ValueError(f"Inversa para dtb_scaling '{method}' nao implementada")


# ════════════════════════════════════════════════════════════════════════════
# SECAO: TARGETS ESTENDIDOS (6 CANAIS)
# ════════════════════════════════════════════════════════════════════════════
# Quando output_channels=6 (P5), o target inclui:
#   Canal 0: rho_h (resistividade horizontal, log10-scaled)
#   Canal 1: rho_v (resistividade vertical, log10-scaled)
#   Canal 2: DTB_up (distancia ate fronteira acima, DTB-scaled)
#   Canal 3: DTB_down (distancia ate fronteira abaixo, DTB-scaled)
#   Canal 4: rho_up (resistividade da camada acima, log10-scaled)
#   Canal 5: rho_down (resistividade da camada abaixo, log10-scaled)
#
# rho_up e rho_down sao derivados das fronteiras detectadas:
#   rho_up[i] = rho_h na camada acima da fronteira mais proxima
#   rho_down[i] = rho_h na camada abaixo da fronteira mais proxima
#
# Ref: docs/physics/perspectivas.md secao P5.
# ──────────────────────────────────────────────────────────────────────────


def _compute_boundary_rho(
    rho_h: np.ndarray,
    boundary_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computa rho da camada acima e abaixo para cada ponto.

    Para cada ponto do perfil, identifica a resistividade da camada
    acima e abaixo da fronteira mais proxima. Util para o modelo
    aprender a reconhecer contrastes de resistividade.

    Args:
        rho_h: Perfil de resistividade horizontal 1D (seq_len,).
        boundary_indices: Indices de fronteiras detectadas.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (rho_up, rho_down), shape (seq_len,).
            rho_up[i]: resistividade media da camada acima.
            rho_down[i]: resistividade media da camada abaixo.
            Se nao ha camada acima/abaixo, usa rho_h[i] (self-reference).

    Note:
        Referenciado em:
            - data/boundaries.py: build_extended_targets() (canais 4, 5)
        Os valores rho_up/rho_down sao a resistividade representativa
        de cada camada, computados como a mediana de pontos da camada
        adjacente. Mediana eh robusta a variacao intra-camada.
    """
    n = len(rho_h)
    rho_up = np.copy(rho_h)
    rho_down = np.copy(rho_h)

    if len(boundary_indices) == 0:
        return rho_up, rho_down

    # ── Definir segmentos de camada ───────────────────────────────────
    # Cada segmento [start, end) corresponde a uma camada geologica.
    # Fronteiras delimitam as transicoes entre segmentos.
    segments: List[Tuple[int, int]] = []
    prev = 0
    for b in boundary_indices:
        if b > prev:
            segments.append((prev, int(b)))
        prev = int(b)
    if prev < n:
        segments.append((prev, n))

    # ── Resistividade representativa de cada segmento (mediana) ───────
    seg_rho = []
    for start, end in segments:
        seg_rho.append(float(np.median(rho_h[start:end])))

    # ── Atribuir rho_up e rho_down (vetorizado via slicing) ─────────────
    # Para cada segmento, atribui rho do segmento anterior/seguinte
    # usando slicing numpy em vez de loop ponto-a-ponto.
    for seg_idx, (start, end) in enumerate(segments):
        # ── Camada acima: rho do segmento anterior ────────────────────
        rho_up[start:end] = seg_rho[seg_idx - 1] if seg_idx > 0 else seg_rho[seg_idx]
        # ── Camada abaixo: rho do segmento seguinte ──────────────────
        rho_down[start:end] = (
            seg_rho[seg_idx + 1] if seg_idx < len(segments) - 1 else seg_rho[seg_idx]
        )

    return rho_up, rho_down


def build_extended_targets(
    rho_h: np.ndarray,
    rho_v: np.ndarray,
    dtb_up: np.ndarray,
    dtb_down: np.ndarray,
    rho_up: np.ndarray,
    rho_down: np.ndarray,
) -> np.ndarray:
    """Concatena targets em array 6-canais para output_channels=6.

    Constroi o array de targets estendido para P5 (DTB).
    Layout:
        Canal 0: rho_h (horizontal resistivity)
        Canal 1: rho_v (vertical resistivity)
        Canal 2: DTB_up (distance to boundary above)
        Canal 3: DTB_down (distance to boundary below)
        Canal 4: rho_up (resistivity of layer above)
        Canal 5: rho_down (resistivity of layer below)

    Args:
        rho_h: Resistividade horizontal (seq_len,). Em Ohm.m ou log10.
        rho_v: Resistividade vertical (seq_len,). Em Ohm.m ou log10.
        dtb_up: DTB para cima (seq_len,). Em metros ou scaled.
        dtb_down: DTB para baixo (seq_len,). Em metros ou scaled.
        rho_up: Resistividade da camada acima (seq_len,).
        rho_down: Resistividade da camada abaixo (seq_len,).

    Returns:
        np.ndarray: Shape (seq_len, 6) com todos os canais concatenados.

    Example:
        >>> y_6ch = build_extended_targets(rho_h, rho_v, dtb_up, dtb_down, rho_up, rho_down)
        >>> y_6ch.shape  # (600, 6)

    Note:
        Referenciado em:
            - data/boundaries.py: compute_dtb_for_dataset() (pipeline)
            - data/pipeline.py: DataPipeline.prepare() (DTB integration)
            - tests/test_boundaries.py: TestBuildExtendedTargets
        Ref: docs/physics/perspectivas.md secao P5.
        Layout: [rho_h, rho_v, DTB_up, DTB_down, rho_up, rho_down].
        Todos os arrays devem ter mesma shape (seq_len,).
    """
    return np.stack(
        [rho_h, rho_v, dtb_up, dtb_down, rho_up, rho_down],
        axis=-1,
    ).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
# SECAO: PIPELINE COMPLETO DTB
# ════════════════════════════════════════════════════════════════════════════
# compute_dtb_for_dataset() integra todas as etapas:
#   1. detect_boundaries() para cada modelo geologico
#   2. compute_dtb_labels() com dtb_max
#   3. apply_dtb_scaling() conforme config.dtb_scaling
#   4. _compute_boundary_rho() para rho_up/rho_down
#   5. build_extended_targets() -> y_6ch
#
# Entrada: y_original 3D (n_seq, seq_len, 2) + z_obs
# Saida: y_extended 3D (n_seq, seq_len, 6)
#
# IMPORTANTE: DTB e computado sobre resistividade em Ohm.m ANTES
# do target_scaling (log10). O rho_up/rho_down recebem o mesmo
# scaling que rho_h/rho_v (log10) para consistencia de dominio.
# DTB (metros) recebe scaling separado (config.dtb_scaling).
#
# Ref: docs/physics/perspectivas.md secao P5 (cadeia completa).
# ──────────────────────────────────────────────────────────────────────────


def compute_dtb_for_dataset(
    y: np.ndarray,
    z: np.ndarray,
    config: PipelineConfig,
) -> np.ndarray:
    """Computa DTB labels e targets estendidos para todo o dataset.

    Pipeline completo: para cada sequencia no dataset, detecta
    fronteiras, computa DTB_up/DTB_down, e constroi targets 6-canais.

    IMPORTANTE: Deve ser chamado ANTES de apply_target_scaling().
    Os targets rho_h/rho_v e rho_up/rho_down serao escalados juntos
    pelo target_scaling (log10). DTB recebe scaling separado.

    Cadeia:
        y(n_seq, seq_len, 2) + z(n_seq, seq_len)
            -> detect_boundaries(rho_h[i], z[i]) por sequencia
            -> compute_dtb_labels(z[i], boundaries, dtb_max)
            -> apply_dtb_scaling(dtb, config.dtb_scaling, dtb_max)
            -> _compute_boundary_rho(rho_h[i], boundaries)
            -> build_extended_targets -> y_ext(n_seq, seq_len, 6)

    Args:
        y: Targets originais 3D (n_seq, seq_len, 2) em Ohm.m.
            Canal 0: rho_h, Canal 1: rho_v (ANTES de target_scaling).
        z: Profundidade 2D (n_seq, seq_len) em metros.
        config: PipelineConfig. Atributos usados:
            - config.dtb_max_from_picasso: DTB maximo em metros.
            - config.dtb_scaling: metodo de scaling DTB.
            - config.dtb_boundary_threshold: threshold para deteccao de
              fronteiras em log10(rho). Default: 0.3 (~2x contraste).

    Returns:
        np.ndarray: Targets estendidos 3D (n_seq, seq_len, 6).
            Layout: [rho_h, rho_v, DTB_up, DTB_down, rho_up, rho_down].
            rho_h/rho_v/rho_up/rho_down: em Ohm.m (sem log10).
            DTB_up/DTB_down: escalados por config.dtb_scaling.

    Raises:
        ValueError: Se y.shape[-1] != 2.

    Example:
        >>> config = PipelineConfig.dtb_p5()
        >>> y_ext = compute_dtb_for_dataset(y_raw, z_obs, config)
        >>> y_ext.shape  # (n_seq, 600, 6)

    Note:
        Referenciado em:
            - data/pipeline.py: DataPipeline.prepare() (antes de target_scaling)
            - tests/test_boundaries.py: TestComputeDTBForDataset
        Ref: docs/physics/perspectivas.md secao P5.
        CHAMAR ANTES de apply_target_scaling(). rho channels (0,1,4,5)
        receberao log10 apos retorno. DTB channels (2,3) ja estao scaled.
    """
    if y.ndim != 3 or y.shape[-1] != 2:
        raise ValueError(f"y deve ter shape (n_seq, seq_len, 2), recebido: {y.shape}")
    if z.shape != y.shape[:2]:
        raise ValueError(f"z shape {z.shape} deve ser {y.shape[:2]} (n_seq, seq_len)")

    n_seq, seq_len, _ = y.shape
    dtb_max = config.dtb_max_from_picasso
    dtb_method = config.dtb_scaling
    threshold = config.dtb_boundary_threshold

    y_extended = np.zeros((n_seq, seq_len, 6), dtype=np.float32)

    for i in range(n_seq):
        rho_h_i = y[i, :, 0]  # (seq_len,) em Ohm.m
        rho_v_i = y[i, :, 1]  # (seq_len,) em Ohm.m
        z_i = z[i]  # (seq_len,) em metros

        # ── Step 1: Detectar fronteiras ───────────────────────────────
        boundaries = detect_boundaries(rho_h_i, z_i, threshold=threshold)

        # ── Step 2: Computar DTB labels ───────────────────────────────
        dtb_up, dtb_down = compute_dtb_labels(z_i, boundaries, dtb_max)

        # ── Step 3: Scaling DTB ───────────────────────────────────────
        dtb_up_scaled = apply_dtb_scaling(dtb_up, dtb_method, dtb_max)
        dtb_down_scaled = apply_dtb_scaling(dtb_down, dtb_method, dtb_max)

        # ── Step 4: Resistividade das camadas adjacentes ──────────────
        rho_up, rho_down = _compute_boundary_rho(rho_h_i, boundaries)

        # ── Step 5: Montar target 6-canais ────────────────────────────
        y_extended[i] = build_extended_targets(
            rho_h_i,
            rho_v_i,
            dtb_up_scaled,
            dtb_down_scaled,
            rho_up,
            rho_down,
        )

    logger.info(
        "compute_dtb_for_dataset: n_seq=%d, seq_len=%d, dtb_max=%.1f, "
        "dtb_scaling='%s', y_ext shape=%s",
        n_seq,
        seq_len,
        dtb_max,
        dtb_method,
        y_extended.shape,
    )

    return y_extended
