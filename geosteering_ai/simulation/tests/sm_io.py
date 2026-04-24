# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/tests/sm_io.py                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulation Manager — Export .dat / .out (Fortran-compat)   ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-18                                                 ║
# ║  Status      : Produção                                                   ║
# ║  Dependências: numpy, geosteering_ai.simulation.io.binary_dat_multi       ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Reutiliza/exporta os tensores H gerados pelo simulador Python em      ║
# ║    formatos .dat (binário Fortran 22-col) e .out (ASCII de metadados),   ║
# ║    permitindo inspeção com ferramentas do pipeline legado (validar_      ║
# ║    integridade, compare_dats, etc.).                                     ║
# ║                                                                           ║
# ║  FORMATOS SUPORTADOS                                                      ║
# ║    .dat  → binário 22 colunas (1 int32 + 21 float64)                     ║
# ║           ordem Fortran: (itr, kt, fi, jm) × (i, zobs, rho_h, rho_v,    ║
# ║           Re/Im × 9 componentes)                                         ║
# ║    .out  → ASCII 4 linhas: nt nf nmaxmodel | angles | freqs | nmeds      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Export .dat/.out para artefatos do simulador Numba.

Delega ao módulo ``geosteering_ai.simulation.io.binary_dat_multi`` sempre
que possível, mas provê também um escritor .out simplificado e um
fallback mínimo para .dat se o binário oficial não estiver disponível.
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np


def write_out_file(
    path: str,
    n_dips: int,
    n_freqs: int,
    nmaxmodel: int,
    angles: Sequence[float],
    freqs_hz: Sequence[float],
    nmeds_per_angle: Sequence[int],
) -> None:
    """Escreve um .out no padrão Fortran list-directed 4 linhas.

    Args:
        path: Caminho de saída.
        n_dips: Número de ângulos (``nt``).
        n_freqs: Número de frequências (``nf``).
        nmaxmodel: Total de modelos representados.
        angles: Ângulos em graus.
        freqs_hz: Frequências em Hz.
        nmeds_per_angle: Medidas efetivas por ângulo (respeitando 1/cos(θ)).
    """

    def _ff(x: float) -> str:
        """Formata REAL(8) list-directed (largura 18)."""
        ax = abs(float(x))
        k = len(str(int(ax))) if ax >= 1.0 else 1
        dp = max(0, 17 - k)
        return f"   {float(x):.{dp}f}     "

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("".join(f"{v:12d}" for v in [n_dips, n_freqs, nmaxmodel]) + "\n")
        f.write("".join(_ff(a) for a in angles) + "\n")
        f.write("".join(_ff(v) for v in freqs_hz) + "\n")
        f.write("".join(f"{n:12d}" for n in nmeds_per_angle) + "\n")


def write_dat_from_tensor(
    path: str,
    H_stack: np.ndarray,
    z_obs: np.ndarray,
    rho_h_at_obs: Optional[np.ndarray] = None,
    rho_v_at_obs: Optional[np.ndarray] = None,
    model_id_start: int = 1,
) -> None:
    """Escreve .dat binário 22-col a partir do tensor do simulador Python.

    Args:
        path: Caminho de saída.
        H_stack: Shape ``(n_models, nTR, nAng, n_pos, nf, 9)`` complex128.
        z_obs: Shape ``(nAng, n_pos)`` — profundidades do ponto-médio.
        rho_h_at_obs: Shape ``(n_models, nAng, n_pos)`` ou None (zera).
        rho_v_at_obs: idem.
        model_id_start: ID do primeiro modelo (1-based).

    Note:
        Ordem de escrita Fortran: para cada ``(itr, kt, fi, jm)`` escreve 1
        registro (i int32 + 21 float64). Compatível com o leitor
        ``binary_dat_multi.read_dat``.
    """
    H = np.asarray(H_stack)
    if H.ndim != 6:
        raise ValueError(
            f"H_stack deve ter 6 dims (N,nTR,nAng,n_pos,nf,9); got {H.shape}"
        )
    n_models, nTR, nAng, n_pos, nf, n_comp = H.shape
    if n_comp != 9:
        raise ValueError(f"H_stack última dim deve ser 9; got {n_comp}")

    # dtype do registro 22-col
    fields = [("col0", np.int32)] + [(f"col{i}", np.float64) for i in range(1, 22)]
    rec_dtype = np.dtype(fields)

    total_records = n_models * nTR * nAng * nf * n_pos
    buf = np.zeros(total_records, dtype=rec_dtype)

    idx = 0
    for m in range(n_models):
        model_id = model_id_start + m
        for itr in range(nTR):
            for kt in range(nAng):
                for fi in range(nf):
                    for jm in range(n_pos):
                        rec = buf[idx]
                        rec["col0"] = model_id
                        # z_obs: aceitamos (nAng,n_pos) ou (n_pos,)
                        if z_obs.ndim == 2:
                            rec["col1"] = float(z_obs[kt, jm])
                        else:
                            rec["col1"] = float(z_obs[jm])
                        rec["col2"] = (
                            float(rho_h_at_obs[m, kt, jm])
                            if rho_h_at_obs is not None
                            else 0.0
                        )
                        rec["col3"] = (
                            float(rho_v_at_obs[m, kt, jm])
                            if rho_v_at_obs is not None
                            else 0.0
                        )
                        for ic in range(9):
                            val = H[m, itr, kt, jm, fi, ic]
                            rec[f"col{4 + 2 * ic}"] = float(val.real)
                            rec[f"col{5 + 2 * ic}"] = float(val.imag)
                        idx += 1

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    buf.tofile(path)


def compute_nmeds_per_angle(
    tj: float, p_med: float, angles: Sequence[float]
) -> List[int]:
    """Retorna ``nmeds[k] = ceil(tj / (p_med · cos(θ_k)))`` para cada ângulo."""
    import math

    out: List[int] = []
    for a in angles:
        cos_d = max(1e-6, math.cos(math.radians(abs(a))))
        out.append(max(1, int(math.ceil(tj / (p_med * cos_d)))))
    return out


__all__ = ["compute_nmeds_per_angle", "write_dat_from_tensor", "write_out_file"]
