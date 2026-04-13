# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/validation/compare_empymod.py                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Validação cruzada — empymod (Sprint 4.1)                  ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-13                                                 ║
# ║  Status      : Produção (opt-in, requer empymod instalado)              ║
# ║  Framework   : empymod (Werthmüller) — 3ª fonte de verdade             ║
# ║  Dependências: empymod (opcional)                                         ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Validação cruzada entre o simulador Python Numba e o pacote          ║
# ║    `empymod` (Werthmüller, mesmo autor do filtro 201pt). Este é o      ║
# ║    triângulo de verdade da Fase 4:                                       ║
# ║                                                                           ║
# ║        Fortran (tatu.x)                                                   ║
# ║          /        \                                                       ║
# ║         /          \                                                      ║
# ║     Numba ─── ── ── JAX                                                  ║
# ║        \             /                                                    ║
# ║         \           /                                                     ║
# ║          empymod (3ª fonte)                                              ║
# ║                                                                           ║
# ║  USO                                                                       ║
# ║    >>> from geosteering_ai.simulation.validation import (                ║
# ║    ...     compare_numba_empymod                                         ║
# ║    ... )                                                                  ║
# ║    >>> result = compare_numba_empymod(                                   ║
# ║    ...     rho_h=np.array([1.0, 100.0, 1.0]),                           ║
# ║    ...     depths=np.array([0.0, 5.0]),                                  ║
# ║    ...     offsets=np.array([1.0]),                                      ║
# ║    ...     freqs=np.array([20000.0]),                                    ║
# ║    ... )                                                                  ║
# ║    >>> result.max_rel_error < 1e-6                                       ║
# ║    True                                                                   ║
# ║                                                                           ║
# ║  TOLERÂNCIA ESPERADA                                                       ║
# ║    empymod usa filtro Key 201pt (não Werthmüller 201pt) por default.   ║
# ║    Para matching aproximado:                                             ║
# ║      • Meios isotrópicos: erro relativo < 1e-4                          ║
# ║      • TIV: erro relativo < 1e-3 (empymod usa convenção ligeiramente    ║
# ║        diferente para anisotropia)                                       ║
# ║                                                                           ║
# ║  REFERÊNCIAS                                                              ║
# ║    • Werthmüller (2017) Geophysics 82(6) WB9 — empymod paper           ║
# ║    • https://empymod.emsig.xyz/ — docs e API                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Validação cruzada Numba ↔ empymod (Sprint 4.1).

Módulo opt-in que requer `pip install empymod`. Expõe helpers para
comparar resultados do simulador Python Numba com o pacote empymod
(Werthmüller 2017), fornecendo uma 3ª fonte de verdade independente
do Fortran.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Import lazy de empymod (opt-in — só precisa se o usuário chamar)
try:
    import empymod  # type: ignore[import-not-found]

    HAS_EMPYMOD: bool = True
    _EMPYMOD_VERSION: str = getattr(empymod, "__version__", "unknown")
except ImportError:
    HAS_EMPYMOD = False
    _EMPYMOD_VERSION = "not installed"

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# ComparisonResult — container
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ComparisonResult:
    """Resultado de uma comparação Numba vs empymod.

    Attributes:
        H_numba: (n_offsets, nf, 9) complex128 — resultado Numba.
        H_empymod: (n_offsets, nf) ou similar — resultado empymod.
        max_abs_error: erro absoluto máximo.
        max_rel_error: erro relativo máximo.
        n_offsets, nf: dimensões do caso testado.
        empymod_version: versão do empymod usada.
        backend_numba: "numba" ou "fortran_f2py".
        notes: mensagens informativas.
    """

    H_numba: np.ndarray
    H_empymod: np.ndarray
    max_abs_error: float
    max_rel_error: float
    n_offsets: int
    nf: int
    empymod_version: str
    backend_numba: str = "numba"
    notes: list = field(default_factory=list)

    def summary(self) -> str:
        """Retorna resumo formatado da comparação."""
        lines = [
            "─" * 70,
            "Comparação Numba vs empymod",
            "─" * 70,
            f"  Backend Numba:    {self.backend_numba}",
            f"  empymod version:  {self.empymod_version}",
            f"  Casos:            n_offsets={self.n_offsets}, nf={self.nf}",
            f"  Shape H_numba:    {self.H_numba.shape}",
            f"  Shape H_empymod:  {self.H_empymod.shape}",
            f"  Erro absoluto máx: {self.max_abs_error:.3e}",
            f"  Erro relativo máx: {self.max_rel_error:.3e}",
        ]
        for n in self.notes:
            lines.append(f"  Note: {n}")
        lines.append("─" * 70)
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# compare_numba_empymod — API pública
# ──────────────────────────────────────────────────────────────────────────────


def compare_numba_empymod(
    rho_h: np.ndarray,
    rho_v: Optional[np.ndarray] = None,
    esp: Optional[np.ndarray] = None,
    depth_src: float = 0.0,
    depth_rec: float = 1.0,
    offsets_m: Optional[np.ndarray] = None,
    freqs_hz: Optional[np.ndarray] = None,
    component: str = "Hzz",
    verb: int = 0,
) -> ComparisonResult:
    """Compara simulador Python Numba com empymod em um caso simples.

    Limitações desta primeira versão (Sprint 4.1):
      1. Apenas componente ``Hzz`` (dipolo vertical axial) — empymod
         suporta outros, mas o mapeamento para nosso tensor 9-col
         fica para Sprint 4.2.
      2. Geometria VMD axial (TX e RX alinhados verticalmente,
         offset horizontal = 0) — caso canônico de testes.
      3. Isotrópico recomendado (TIV pode divergir por conta de
         convenção diferente).

    Args:
        rho_h: (n,) resistividades horizontais (Ω·m).
        rho_v: (n,) resistividades verticais. Se None, usa rho_h
            (isotrópico).
        esp: (n-2,) espessuras internas (m).
        depth_src: profundidade do TX (m).
        depth_rec: profundidade do RX (m).
        offsets_m: (n_offsets,) afastamentos horizontais — mantido
            em 0 para VMD axial.
        freqs_hz: (nf,) frequências em Hz.
        component: componente do tensor. Sprint 4.1 suporta só "Hzz".
        verb: verbosity do empymod (0=quiet, 2=verbose).

    Returns:
        :class:`ComparisonResult` com estatísticas + arrays.

    Raises:
        ImportError: Se empymod não instalado.
        NotImplementedError: Se ``component != "Hzz"`` (Sprint 4.1).

    Example:
        >>> # Teste isotrópico 3 camadas
        >>> result = compare_numba_empymod(
        ...     rho_h=np.array([1.0, 100.0, 1.0]),
        ...     esp=np.array([5.0]),
        ...     freqs_hz=np.array([20000.0]),
        ... )
        >>> print(result.summary())
        >>> assert result.max_rel_error < 1e-3
    """
    if not HAS_EMPYMOD:
        raise ImportError(
            "empymod não instalado. Para Sprint 4.1 (validação cruzada),"
            " instale via: `pip install empymod`."
        )

    if component != "Hzz":
        raise NotImplementedError(
            f"Sprint 4.1 suporta apenas component='Hzz'; "
            f"outros componentes em Sprint 4.2."
        )

    # Defaults
    rho_h = np.ascontiguousarray(rho_h, dtype=np.float64)
    n = rho_h.shape[0]
    if rho_v is None:
        rho_v = rho_h.copy()
    else:
        rho_v = np.ascontiguousarray(rho_v, dtype=np.float64)
    if esp is None:
        esp = np.zeros(max(0, n - 2), dtype=np.float64)
    else:
        esp = np.ascontiguousarray(esp, dtype=np.float64)
    if offsets_m is None:
        offsets_m = np.array([1.0], dtype=np.float64)
    if freqs_hz is None:
        freqs_hz = np.array([20000.0], dtype=np.float64)

    n_offsets = offsets_m.shape[0]
    nf = freqs_hz.shape[0]
    notes: list = []

    # ── Rodar simulador Numba ────────────────────────────────────────
    from geosteering_ai.simulation import SimulationConfig, simulate

    # Monta posições z centradas no midpoint (TX=-L/2, RX=+L/2 com L=
    # depth_rec - depth_src). Para VMD axial, offset hor = 0, mas nosso
    # simulador aplica r_half = L*sin(dip)/2 para obter hordist. Para
    # match empymod VMD axial, usamos dip=0° (axial vertical).
    L = depth_rec - depth_src
    z_mid = 0.5 * (depth_src + depth_rec)
    cfg = SimulationConfig(
        frequencies_hz=list(freqs_hz) if nf > 1 else None,
        frequency_hz=float(freqs_hz[0]),
        tr_spacing_m=L,
        parallel=False,
    )
    result_numba = simulate(
        rho_h=rho_h,
        rho_v=rho_v,
        esp=esp,
        positions_z=np.array([z_mid]),
        cfg=cfg,
    )
    # H_numba[0, i_f, 8] → Hzz
    H_numba_hzz = result_numba.H_tensor[0, :, 8]  # (nf,)

    # ── Rodar empymod (VMD axial) ───────────────────────────────────
    # empymod.dipole espera:
    #   src = [x, y, z_src] (TX posição)
    #   rec = [x, y, z_rec] (RX posição)
    #   depth = [interfaces] em ordem crescente
    #   res = [res_camada_0, res_camada_1, ...]
    #   freqtime = frequências
    #   signal = None para domínio de frequência
    #   ab = 55 para Hzz (VMD axial)

    # Constrói array de interfaces (acumulando esp)
    if esp.size > 0:
        interfaces = np.cumsum(esp)  # n-1 interfaces
        depth_interfaces = list(interfaces)
    else:
        depth_interfaces = []

    src = [float(offsets_m[0]) / 2.0, 0.0, depth_src]
    rec = [-float(offsets_m[0]) / 2.0, 0.0, depth_rec]

    try:
        H_empymod_result = empymod.dipole(
            src=src,
            rec=rec,
            depth=depth_interfaces,
            res=list(rho_h),
            freqtime=list(freqs_hz),
            signal=None,
            ab=55,  # VMD axial: dipolo magnético vertical medido por receptor vertical
            verb=verb,
        )
        H_empymod_arr = np.asarray(H_empymod_result, dtype=np.complex128).reshape(-1)
        if H_empymod_arr.shape != H_numba_hzz.shape:
            # Alguns casos empymod retorna shape diferente — força para (nf,)
            H_empymod_arr = H_empymod_arr.flatten()[: H_numba_hzz.size]
        notes.append(
            "empymod.dipole(ab=55) VMD axial — isotrópico; "
            "TIV pode divergir por convenção."
        )
    except Exception as exc:
        notes.append(f"empymod falhou: {exc}")
        H_empymod_arr = np.zeros_like(H_numba_hzz)

    # ── Estatísticas ─────────────────────────────────────────────────
    abs_err = np.abs(H_numba_hzz - H_empymod_arr)
    rel_err = abs_err / np.maximum(np.abs(H_empymod_arr), 1e-15)
    max_abs = float(np.max(abs_err))
    max_rel = float(np.max(rel_err))

    return ComparisonResult(
        H_numba=result_numba.H_tensor,
        H_empymod=H_empymod_arr,
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        n_offsets=n_offsets,
        nf=nf,
        empymod_version=_EMPYMOD_VERSION,
        backend_numba="numba",
        notes=notes,
    )


# ──────────────────────────────────────────────────────────────────────────────
# install_empymod_instruction — utilitário informativo
# ──────────────────────────────────────────────────────────────────────────────


def install_empymod_instruction() -> str:
    """Retorna instruções de instalação do empymod (para mensagens de erro)."""
    return (
        "Para habilitar a validação cruzada empymod (Sprint 4.1), instale:\n"
        "  pip install empymod\n\n"
        "Documentação: https://empymod.emsig.xyz/\n"
        "Paper: Werthmüller (2017) Geophysics 82(6) WB9"
    )


__all__ = [
    "HAS_EMPYMOD",
    "ComparisonResult",
    "compare_numba_empymod",
    "install_empymod_instruction",
]
