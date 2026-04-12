# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/postprocess/tilted.py                          ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Feature F7 Antenas Inclinadas         ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-11 (Sprint 2.2)                                   ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : NumPy 2.x                                                  ║
# ║  Dependências: numpy                                                      ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Implementação da feature F7 (Antenas Inclinadas) como               ║
# ║    pós-processamento do tensor H. Projeta o tensor 3×3 para N         ║
# ║    configurações `(β, φ)` simulando ferramentas LWD comerciais com    ║
# ║    antenas não-axiais (ex.: PeriScope HD com 45°, EcoScope com 45°).  ║
# ║                                                                           ║
# ║  FÓRMULA (paridade Fortran PerfilaAnisoOmp.f08:714-730)                 ║
# ║    H_tilted(β, φ) = cos(β)·Hzz + sin(β)·[cos(φ)·Hxz + sin(φ)·Hyz]     ║
# ║                                                                           ║
# ║    onde:                                                                ║
# ║      β = inclinação (graus), 0°=Hzz axial, 90°=plano xz/yz            ║
# ║      φ = azimute (graus), 0°=x, 90°=y, 180°=-x, 270°=-y               ║
# ║                                                                           ║
# ║  ÍNDICES DO TENSOR H (ordem flat 9-componentes)                          ║
# ║    H[..., 0] = Hxx    H[..., 1] = Hxy    H[..., 2] = Hxz               ║
# ║    H[..., 3] = Hyx    H[..., 4] = Hyy    H[..., 5] = Hyz               ║
# ║    H[..., 6] = Hzx    H[..., 7] = Hzy    H[..., 8] = Hzz               ║
# ║                                                                           ║
# ║  CUSTO                                                                    ║
# ║    ~5 mul + 2 add por ponto — desprezível vs forward. Função NumPy    ║
# ║    pura, vetorizada (sem Numba).                                       ║
# ║                                                                           ║
# ║  OPT-IN                                                                   ║
# ║    Ativado por `cfg.use_tilted_antennas=True` + `tilted_configs=tuple`.║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""F7 Antenas Inclinadas — pós-processamento de tensores H.

Projeta o tensor H 3×3 para N configurações de antenas inclinadas
`(β, φ)` com β=inclinação e φ=azimute em graus.

Example:
    Aplicação para 2 configurações tilted::

        >>> import numpy as np
        >>> from geosteering_ai.simulation.postprocess import apply_tilted_antennas
        >>> H = np.zeros((1, 10, 1, 9), dtype=np.complex128)
        >>> H[..., 8] = 1.0 + 0.5j  # Hzz
        >>> H_tilted = apply_tilted_antennas(H, ((0.0, 0.0), (45.0, 0.0)))
        >>> H_tilted.shape
        (2, 1, 10, 1)

Note:
    Para β=0° (Hzz puro), o resultado é exatamente `H_tensor[..., 8]`.
    Para β=90°, φ=0°, o resultado é `H_tensor[..., 2]` (Hxz).
"""
from __future__ import annotations

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Índices das componentes relevantes no tensor H flat (ordem 9-col)
# ──────────────────────────────────────────────────────────────────────────────
# Os tensores armazenados pelo simulador seguem a ordem:
#   [Hxx, Hxy, Hxz, Hyx, Hyy, Hyz, Hzx, Hzy, Hzz]
# F7 só precisa de Hxz, Hyz e Hzz — as 3 componentes da "última coluna"
# do tensor 3×3 (j=z).
_IDX_HXZ: int = 2  # H[..., 2] — linha 1, col 3
_IDX_HYZ: int = 5  # H[..., 5] — linha 2, col 3
_IDX_HZZ: int = 8  # H[..., 8] — linha 3, col 3


def apply_tilted_antennas(
    H_tensor: np.ndarray,
    tilted_configs: tuple[tuple[float, float], ...],
) -> np.ndarray:
    """Aplica projeção F7 Antenas Inclinadas ao tensor H.

    Para cada configuração `(β, φ)` em `tilted_configs`:

        β_rad = deg2rad(β); φ_rad = deg2rad(φ)
        H_tilted[β,φ] = cos(β)·Hzz + sin(β)·[cos(φ)·Hxz + sin(φ)·Hyz]

    Args:
        H_tensor: Array complex128 de shape `(..., 9)` com as 9
            componentes do tensor H (ordem Hxx..Hzz). As dimensões
            prefixos são arbitrárias; o tensor é projetado no final.
        tilted_configs: Tupla de tuplas `(beta_deg, phi_deg)` com
            ângulos em graus. β ∈ [0, 90], φ ∈ [0, 360).

    Returns:
        Array complex128 shape `(n_tilted, *H_tensor.shape[:-1])` onde
        `n_tilted = len(tilted_configs)`. Cada fatia `[i, ...]` contém
        o campo projetado para a configuração `tilted_configs[i]`.

    Raises:
        ValueError: Se `H_tensor.shape[-1] != 9` ou se `tilted_configs`
            for vazio.

    Example:
        Projeção trivial (β=0° → Hzz puro)::

            >>> H = np.zeros((5, 9), dtype=np.complex128)
            >>> H[:, 8] = 1.0 + 0.5j  # preenche só Hzz
            >>> H[:, 2] = 99.0  # Hxz com valor fake para sanidade
            >>> out = apply_tilted_antennas(H, ((0.0, 0.0),))
            >>> out.shape
            (1, 5)
            >>> np.allclose(out[0], H[:, 8])
            True

        Projeção equatorial (β=90°, φ=0° → Hxz puro)::

            >>> H = np.zeros((5, 9), dtype=np.complex128)
            >>> H[:, 2] = 1.0 + 0.5j  # só Hxz
            >>> H[:, 8] = 99.0  # Hzz com valor fake
            >>> out = apply_tilted_antennas(H, ((90.0, 0.0),))
            >>> np.allclose(out[0], H[:, 2])
            True

    Note:
        A fórmula é a mesma usada em PerfilaAnisoOmp.f08:714-730 para
        o flag `use_tilted=1`. Unidade de entrada: graus; conversão
        interna para radianos via `np.deg2rad`.

        Para o layout 22-col do `.dat` (Sprint 2.2 io), este tensor
        projetado é tipicamente escrito em colunas adicionais 22 e 23
        (Re e Im) para cada beam tilted.
    """
    # ── Validação ─────────────────────────────────────────────────
    H_tensor = np.asarray(H_tensor, dtype=np.complex128)
    if H_tensor.shape[-1] != 9:
        raise ValueError(f"H_tensor.shape[-1]={H_tensor.shape[-1]}, esperado 9.")
    if len(tilted_configs) == 0:
        raise ValueError("tilted_configs vazio — nada a projetar.")

    for i, cfg in enumerate(tilted_configs):
        if len(cfg) != 2:
            raise ValueError(
                f"tilted_configs[{i}]={cfg} deve ter 2 elementos " f"(beta_deg, phi_deg)."
            )

    # ── Extração das componentes relevantes ───────────────────────
    # H[..., 2] = Hxz, H[..., 5] = Hyz, H[..., 8] = Hzz
    Hxz = H_tensor[..., _IDX_HXZ]
    Hyz = H_tensor[..., _IDX_HYZ]
    Hzz = H_tensor[..., _IDX_HZZ]

    # ── Loop sobre configurações tilted ───────────────────────────
    # Pré-aloca o array de saída: (n_tilted, *prefix_shape)
    n_tilted = len(tilted_configs)
    prefix_shape = H_tensor.shape[:-1]
    output = np.empty((n_tilted,) + prefix_shape, dtype=np.complex128)

    for i, (beta_deg, phi_deg) in enumerate(tilted_configs):
        beta_rad = np.deg2rad(float(beta_deg))
        phi_rad = np.deg2rad(float(phi_deg))
        cos_beta = np.cos(beta_rad)
        sin_beta = np.sin(beta_rad)
        cos_phi = np.cos(phi_rad)
        sin_phi = np.sin(phi_rad)

        # Fórmula F7 canônica
        output[i] = cos_beta * Hzz + sin_beta * (cos_phi * Hxz + sin_phi * Hyz)

    return output


__all__ = ["apply_tilted_antennas"]
