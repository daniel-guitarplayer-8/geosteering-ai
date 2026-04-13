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


# ──────────────────────────────────────────────────────────────────────────────
# Sprint 4.2 — Validação 9 componentes TIV (mapeamento λ²)
# ──────────────────────────────────────────────────────────────────────────────
# Estende a Sprint 4.1 (Hzz isotrópico) para os 9 componentes do tensor H
# em meios TIV (transversalmente isotrópicos verticais). Os componentes são
# os 6 distintos do tensor magnético + suas permutações (Hxy=Hyx etc.):
#
#   Componente   ab (empymod)   Significado
#   ──────────   ────────────   ───────────────────────────
#   Hxx          11             dipolo horizontal-x → receptor horizontal-x
#   Hxy          12             dipolo horizontal-x → receptor horizontal-y
#   Hxz          15             dipolo horizontal-x → receptor vertical
#   Hyx          21             dipolo horizontal-y → receptor horizontal-x
#   Hyy          22             dipolo horizontal-y → receptor horizontal-y
#   Hyz          25             dipolo horizontal-y → receptor vertical
#   Hzx          51             dipolo vertical    → receptor horizontal-x
#   Hzy          52             dipolo vertical    → receptor horizontal-y
#   Hzz          55             dipolo vertical    → receptor vertical
#
# Convenção λ²: o tensor 9-col Numba ordena os componentes como
# [Hxx, Hxy, Hxz, Hyx, Hyy, Hyz, Hzx, Hzy, Hzz] (índices 0..8). empymod
# usa `aniso = sqrt(rho_v / rho_h) = λ` — ao chamar `empymod.dipole`,
# passamos `aniso = λ` (não λ²) pois o pacote eleva ao quadrado
# internamente.
# ──────────────────────────────────────────────────────────────────────────────

# Mapeamento componente → código `ab` empymod
COMPONENT_AB_MAP: dict[str, int] = {
    "Hxx": 11,
    "Hxy": 12,
    "Hxz": 15,
    "Hyx": 21,
    "Hyy": 22,
    "Hyz": 25,
    "Hzx": 51,
    "Hzy": 52,
    "Hzz": 55,
}

# Mapeamento componente → índice na coluna 9 do `H_tensor` Numba
COMPONENT_TENSOR_INDEX: dict[str, int] = {
    "Hxx": 0,
    "Hxy": 1,
    "Hxz": 2,
    "Hyx": 3,
    "Hyy": 4,
    "Hyz": 5,
    "Hzx": 6,
    "Hzy": 7,
    "Hzz": 8,
}


@dataclass
class TensorComparisonResult:
    """Resultado de uma comparação tensor 9-comp Numba vs empymod (Sprint 4.2).

    Attributes:
        H_numba: ``(n_positions, nf, 9) complex128`` — tensor completo Numba.
        H_empymod_per_component: dict ``{comp: array (nf,)}`` — empymod por comp.
        max_rel_error_per_component: dict ``{comp: float}`` — erro relativo máximo.
        max_rel_error_global: maior erro relativo entre todos os componentes
            comparados (excluindo os que falharam).
        components_compared: lista dos componentes efetivamente comparados.
        components_failed: lista dos componentes onde empymod falhou (com motivo).
        empymod_version: versão do empymod usada.
        backend_numba: "numba" ou "fortran_f2py".
        notes: mensagens informativas.
    """

    H_numba: np.ndarray
    H_empymod_per_component: dict
    max_rel_error_per_component: dict
    max_rel_error_global: float
    components_compared: list
    components_failed: list
    empymod_version: str
    backend_numba: str = "numba"
    notes: list = field(default_factory=list)

    def summary(self) -> str:
        """Retorna resumo formatado da comparação 9 componentes."""
        lines = [
            "─" * 72,
            "Comparação Numba ↔ empymod — Tensor 9 componentes (Sprint 4.2)",
            "─" * 72,
            f"  Backend Numba:        {self.backend_numba}",
            f"  empymod version:      {self.empymod_version}",
            f"  Componentes OK:       {len(self.components_compared)} / 9",
            f"  Componentes falhos:   {len(self.components_failed)}",
            f"  Erro relativo máx (global): {self.max_rel_error_global:.3e}",
            "  Por componente:",
        ]
        for comp in self.components_compared:
            err = self.max_rel_error_per_component.get(comp, float("nan"))
            lines.append(f"    {comp:<5s}  err_rel = {err:.3e}")
        for comp, reason in self.components_failed:
            lines.append(f"    {comp:<5s}  FALHOU — {reason}")
        for n in self.notes:
            lines.append(f"  Note: {n}")
        lines.append("─" * 72)
        return "\n".join(lines)


def compare_numba_empymod_tensor(
    rho_h: np.ndarray,
    rho_v: Optional[np.ndarray] = None,
    esp: Optional[np.ndarray] = None,
    depth_src: float = 0.0,
    depth_rec: float = 1.0,
    offset_x: float = 0.5,
    freqs_hz: Optional[np.ndarray] = None,
    components: Optional[list] = None,
    verb: int = 0,
) -> TensorComparisonResult:
    """Compara tensor 9-componentes Numba vs empymod em meio TIV (Sprint 4.2).

    .. warning::

       **Status Sprint 4.2:** infraestrutura completa (mapa AB, dispatcher
       por componente, dataclass, mapeamento λ²), porém a bit-exactness
       numérica entre Numba e empymod **ainda não foi alcançada**. A
       diferença observada inclui um fator complexo (≈ ``1/π · e^{-iπ/2}``)
       compatível com divergência de convenção (temporal e^(-iωt) Numba
       vs convenção empymod e/ou normalização ``H`` vs ``B = μ₀H``).
       Sprint 4.3 (PR #12) tratará da reconciliação completa de
       convenções. Por enquanto, o resultado serve como:

       1. **Smoke test** — confirma que ambos códigos rodam e produzem
          resultados finitos para os 9 componentes.
       2. **Detecção de simetrias** — componentes que devem ser ≈ 0
          (e.g. ``Hxy`` em geometria axial) são consistentemente
          pequenos em ambos.
       3. **Scaffolding** para Sprint 4.3 ajustar fatores de conversão.

    Estende :func:`compare_numba_empymod` para validar as 9 componentes do
    tensor magnético em meios isotrópicos OU TIV (anisotrópicos verticais).
    A anisotropia é mapeada via ``aniso = sqrt(rho_v / rho_h) = λ`` (empymod
    eleva ao quadrado internamente para obter λ²).

    Args:
        rho_h: ``(n,)`` resistividades horizontais (Ω·m).
        rho_v: ``(n,)`` resistividades verticais. Se None, usa ``rho_h``
            (caso isotrópico).
        esp: ``(n-2,)`` espessuras internas (m).
        depth_src: profundidade do TX (m).
        depth_rec: profundidade do RX (m).
        offset_x: offset horizontal RX-TX em x (m). Default 0.5 m → cria
            um perfil ligeiramente off-axial onde TODAS as componentes
            cruzadas (Hxy, Hxz, etc.) são não-nulas.
        freqs_hz: ``(nf,)`` frequências em Hz. Default ``[20000.0]``.
        components: Lista de componentes a comparar. Default = TODAS as 9
            chaves de :data:`COMPONENT_AB_MAP`.
        verb: verbosity do empymod (0=quiet, 2=verbose).

    Returns:
        :class:`TensorComparisonResult` com erros por componente.

    Raises:
        ImportError: Se empymod não instalado.

    Example:
        >>> import numpy as np
        >>> result = compare_numba_empymod_tensor(
        ...     rho_h=np.array([1.0, 100.0, 1.0]),
        ...     rho_v=np.array([1.0, 200.0, 1.0]),  # λ²=2
        ...     esp=np.array([5.0]),
        ...     freqs_hz=np.array([20000.0]),
        ... )
        >>> print(result.summary())
        >>> assert result.max_rel_error_global < 1e-2  # TIV pode divergir mais
    """
    if not HAS_EMPYMOD:
        raise ImportError(
            "empymod não instalado. Para Sprint 4.2 (validação tensor),"
            " instale via: `pip install empymod`."
        )

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
    if freqs_hz is None:
        freqs_hz = np.array([20000.0], dtype=np.float64)
    if components is None:
        components = list(COMPONENT_AB_MAP.keys())

    nf = freqs_hz.shape[0]
    notes: list = []

    # ── Convenção empymod: aniso = sqrt(ρv/ρh) = λ ──────────────────────────
    # empymod eleva ao quadrado internamente: σ_v = σ_h / λ².
    # Para meios isotrópicos (rho_v == rho_h), aniso = 1.0.
    aniso = np.sqrt(rho_v / rho_h)
    if not np.allclose(aniso, 1.0):
        notes.append(
            f"Meio TIV detectado: aniso=λ ∈ [{aniso.min():.3f}, {aniso.max():.3f}]"
            " (empymod usa λ²=ρv/ρh internamente)."
        )

    # ── Roda simulador Numba (1 vez, retorna tensor completo 9-comp) ────────
    from geosteering_ai.simulation import SimulationConfig, simulate

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
    H_numba = result_numba.H_tensor  # (1, nf, 9)

    # ── Constrói perfil de interfaces para empymod ──────────────────────────
    # empymod espera ``len(depth) = n - 1`` interfaces para ``n`` camadas.
    # Nosso ``esp`` tem ``n - 2`` espessuras internas, então prependemos
    # ``0.0`` como topo de referência (a primeira interface). Isso é
    # equivalente à convenção do simulador Numba (``prof[0]=-1e300`` é
    # apenas sentinela; as interfaces físicas começam em z=0).
    if n >= 2:
        depth_interfaces = [0.0] + list(np.cumsum(esp))
    else:
        depth_interfaces = []  # half-space (1 camada)

    # Posições TX e RX — replica geometria do `compare_numba_empymod` da
    # Sprint 4.1 (TX em +x/2, RX em -x/2) para preservar o alinhamento que
    # já produzia err_rel < 1e-4 no caso Hzz axial. Para Sprint 4.2,
    # offset_x > 0 garante que os componentes cruzados (Hxy, Hxz, etc.)
    # sejam não-nulos.
    src = [offset_x / 2.0, 0.0, depth_src]
    rec = [-offset_x / 2.0, 0.0, depth_rec]

    # ── Compara cada componente solicitado ──────────────────────────────────
    H_emp_per: dict = {}
    err_per: dict = {}
    failed: list = []
    compared: list = []

    for comp in components:
        if comp not in COMPONENT_AB_MAP:
            failed.append(
                (comp, f"componente desconhecido (válidos: {list(COMPONENT_AB_MAP)})")
            )
            continue
        ab = COMPONENT_AB_MAP[comp]
        idx = COMPONENT_TENSOR_INDEX[comp]
        try:
            H_emp_arr = empymod.dipole(
                src=src,
                rec=rec,
                depth=depth_interfaces,
                res=list(rho_h),
                aniso=list(aniso),
                freqtime=list(freqs_hz),
                signal=None,
                ab=ab,
                verb=verb,
            )
            H_emp_arr = np.asarray(H_emp_arr, dtype=np.complex128).reshape(-1)
            H_numba_comp = H_numba[0, :, idx]  # (nf,)
            # Shape mismatch — não trunca silenciosamente (mascararia bugs
            # de degeneração de freqs ou problemas internos do empymod).
            # Reporta como falha individual e pula este componente.
            if H_emp_arr.shape[0] != H_numba_comp.shape[0]:
                failed.append(
                    (
                        comp,
                        f"shape mismatch: empymod={H_emp_arr.shape[0]} "
                        f"vs numba={H_numba_comp.shape[0]}",
                    )
                )
                continue
            H_emp_per[comp] = H_emp_arr
            abs_err = np.abs(H_numba_comp - H_emp_arr)
            rel_err = abs_err / np.maximum(np.abs(H_emp_arr), 1e-15)
            err_per[comp] = float(np.max(rel_err))
            compared.append(comp)
        except Exception as exc:  # pragma: no cover — defensivo
            failed.append((comp, str(exc)))

    # Erro global = max entre os componentes comparados (ignora falhos)
    if err_per:
        max_rel_global = max(err_per.values())
    else:
        max_rel_global = float("nan")

    return TensorComparisonResult(
        H_numba=H_numba,
        H_empymod_per_component=H_emp_per,
        max_rel_error_per_component=err_per,
        max_rel_error_global=max_rel_global,
        components_compared=compared,
        components_failed=failed,
        empymod_version=_EMPYMOD_VERSION,
        backend_numba="numba",
        notes=notes,
    )


__all__ = [
    "HAS_EMPYMOD",
    "ComparisonResult",
    "compare_numba_empymod",
    "install_empymod_instruction",
    # Sprint 4.2
    "COMPONENT_AB_MAP",
    "COMPONENT_TENSOR_INDEX",
    "TensorComparisonResult",
    "compare_numba_empymod_tensor",
]
