# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/io/tensor_dat.py                               ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : Exportação vetorizada de tensor H → .dat / .out (22-col)   ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Exportadores Fortran-compatíveis        ║
# ║  Versão      : v2.53 (relocado de simulation/tests/sm_io.py p/ produção)  ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-18 (orig.) · 2026-06-02 (relocação produção)       ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : NumPy (structured dtype + broadcast vetorizado)            ║
# ║  Dependências: numpy (obrigatório)                                        ║
# ║  Padrão      : escritor stateless — recebe o tensor 6D já simulado        ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Escreve o tensor H (n_models, nTR, nAng, n_pos, nf, 9) produzido       ║
# ║    pelo simulador Python (Numba ou JAX) em:                               ║
# ║                                                                           ║
# ║      • `.dat`  → binário 22 colunas (1 int32 + 21 float64 = 172 B/rec),   ║
# ║                  layout canônico de `.claude/commands/geosteering-        ║
# ║                  physics.md §4 (idêntico ao `tatu.x` Fortran v10.0).      ║
# ║      • `.out`  → ASCII 4 linhas (nt nf nmaxmodel | ângulos | freqs |      ║
# ║                  nmeds), list-directed Fortran.                           ║
# ║                                                                           ║
# ║    Relocado de `simulation/tests/sm_io.py` (Sprint v2.53) para que        ║
# ║    código de PRODUÇÃO (e.g. a CLI `geosteering-cli simulate --format      ║
# ║    dat`) não dependa de um módulo do subpacote `tests`. O módulo legado   ║
# ║    `sm_io.py` re-exporta destas definições (retrocompat dos testes).      ║
# ║                                                                           ║
# ║  LAYOUT 22-COL (geosteering-physics.md §4 — IMUTÁVEL)                     ║
# ║    ┌─────┬──────────────────────────────────────────────────────────┐    ║
# ║    │ col │  Conteúdo                                                 │    ║
# ║    ├─────┼──────────────────────────────────────────────────────────┤    ║
# ║    │  0  │  meds / model_id (int32) — metadata, NUNCA feature        │    ║
# ║    │  1  │  zobs — profundidade observada (m)            → feature ★  │    ║
# ║    │  2  │  res_h — resistividade horizontal (Ω·m)       → target  ★  │    ║
# ║    │  3  │  res_v — resistividade vertical (Ω·m)         → target  ★  │    ║
# ║    │ 4-5 │  Re/Im(Hxx)                                   → feature ★  │    ║
# ║    │ 6-19│  Re/Im(Hxy,Hxz,Hyx,Hyy,Hyz,Hzx,Hzy)                       │    ║
# ║    │20-21│  Re/Im(Hzz)                                   → feature ★  │    ║
# ║    └─────┴──────────────────────────────────────────────────────────┘    ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    write_dat_from_tensor: tensor 6D → .dat binário 22-col (vetorizado)    ║
# ║    write_out_file       : metadados → .out ASCII list-directed Fortran    ║
# ║    compute_nmeds_per_angle: nmeds[k] = ceil(tj/(p_med·cos θ_k))           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Exportação vetorizada do tensor H do simulador → `.dat`/`.out` 22-col.

Escreve o tensor ``H_stack`` ``(n_models, nTR, nAng, n_pos, nf, 9)`` (saída
de ``simulate_multi(models=...)`` ou ``simulate_batch``) no formato binário
22 colunas canônico (``geosteering-physics.md`` §4), re-legível por
:func:`geosteering_ai.data.loading.load_binary_dat`.

A ordem de escrita é a convenção Fortran ``(m, itr, kt, fi, jm)`` por registro,
bit-a-bit compatível com o leitor ``binary_dat_multi.read_dat`` e com os
artefatos do simulador Fortran ``tatu.x``.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np

__all__ = ["compute_nmeds_per_angle", "write_dat_from_tensor", "write_out_file"]


def write_out_file(
    path: str,
    n_dips: int,
    n_freqs: int,
    nmaxmodel: int,
    angles: Sequence[float],
    freqs_hz: Sequence[float],
    nmeds_per_angle: Sequence[int],
) -> None:
    """Escreve um `.out` no padrão Fortran list-directed 4 linhas.

    O `.out` é o sidecar de metadados do `.dat`: descreve quantos ângulos
    (``nt``), frequências (``nf``) e modelos (``nmaxmodel``) o binário contém,
    além dos vetores de ângulos/frequências e do nº de medidas por ângulo.

    Args:
        path: Caminho de saída do arquivo `.out`.
        n_dips: Número de ângulos (``nt``).
        n_freqs: Número de frequências (``nf``).
        nmaxmodel: Total de modelos representados no `.dat`.
        angles: Ângulos em graus.
        freqs_hz: Frequências em Hz.
        nmeds_per_angle: Medidas efetivas por ângulo (respeitando 1/cos θ).

    Returns:
        None. Efeito colateral: grava o arquivo em ``path`` (cria o diretório
        pai se necessário).

    Raises:
        OSError: Falha de escrita (permissão, disco cheio, caminho inválido).

    Note:
        Layout idêntico a ``binary_dat_multi.export_info_out`` e ao
        ``info*.out`` do ``tatu.x``. Parseável por
        :func:`geosteering_ai.data.loading` (metadados do dataset).

    Example:
        >>> write_out_file(
        ...     "/tmp/info.out", n_dips=1, n_freqs=1, nmaxmodel=3,
        ...     angles=[0.0], freqs_hz=[20000.0], nmeds_per_angle=[16],
        ... )
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
    """Escreve `.dat` binário 22-col a partir do tensor do simulador Python.

    Layout 22-col (``geosteering-physics.md`` §4): cada registro tem 1 int32
    (``col0`` = model_id) + 21 float64 (``col1`` = zobs, ``col2/3`` =
    res_h/res_v, ``col4..21`` = Re/Im das 9 componentes na ordem
    ``Hxx,Hxy,Hxz,Hyx,Hyy,Hyz,Hzx,Hzy,Hzz``). O campo H é gravado **cru**
    (mesmo estado de decoupling do `.dat` do ``tatu.x``).

    Args:
        path: Caminho de saída do `.dat`.
        H_stack: Shape ``(n_models, nTR, nAng, n_pos, nf, 9)`` complex.
        z_obs: Shape ``(nAng, n_pos)`` ou ``(n_pos,)`` — profundidades do
            ponto-médio (m). Broadcast no eixo de modelos/freq.
        rho_h_at_obs: Shape ``(n_models, nAng, n_pos)`` ou None. Resistividade
            horizontal NO ponto de observação (camada que contém cada z).
            ``None`` → ``col2`` fica zerada (use o mapeamento z→camada para
            conformidade com ``geosteering-physics.md`` §4).
        rho_v_at_obs: idem para resistividade vertical (``col3``).
        model_id_start: ID do primeiro modelo (1-based) gravado em ``col0``.

    Returns:
        None. Efeito colateral: grava o `.dat` em ``path`` (cria diretório pai).

    Raises:
        ValueError: ``H_stack`` não tem 6 dims ou última dim ≠ 9.
        OSError: Falha de escrita.

    Note:
        Ordem de escrita Fortran: para cada ``(m, itr, kt, fi, jm)`` escreve
        1 registro (i int32 + 21 float64). Compatível com o leitor
        ``binary_dat_multi.read_dat`` e re-legível por
        :func:`geosteering_ai.data.loading.load_binary_dat` (``n_columns=22``).

        Sprint 15.4 (v2.16): implementação vetorizada com ``np.broadcast_to``
        + ``np.transpose`` — escreve o buffer em ~0.5–2 s (5–30× vs. os 5
        loops Python aninhados anteriores), bit-a-bit idêntica (validado por
        ``tests/test_sm_workers_io.py``).

    Example:
        >>> import numpy as np
        >>> H = np.zeros((2, 1, 1, 4, 1, 9), dtype=np.complex128)
        >>> z = np.linspace(-5.0, 5.0, 4)
        >>> write_dat_from_tensor("/tmp/sim.dat", H, z)
    """
    H = np.asarray(H_stack)
    if H.ndim != 6:
        raise ValueError(
            f"H_stack deve ter 6 dims (N,nTR,nAng,n_pos,nf,9); got {H.shape}"
        )
    n_models, nTR, nAng, n_pos, nf, n_comp = H.shape
    if n_comp != 9:
        raise ValueError(f"H_stack última dim deve ser 9; got {n_comp}")

    # dtype do registro 22-col (1 int32 + 21 float64 = 4 + 168 = 172 bytes/rec)
    fields = [("col0", np.int32)] + [(f"col{i}", np.float64) for i in range(1, 22)]
    rec_dtype = np.dtype(fields)

    total_records = n_models * nTR * nAng * nf * n_pos
    buf = np.zeros(total_records, dtype=rec_dtype)

    # ── Permutação para ordem Fortran (m, itr, kt, fi, jm, ic) ───────────
    # H_stack vem como (m, itr, kt, jm, fi, 9). Fortran espera fi antes de
    # jm na ordem do registro. ``np.transpose`` retorna view (zero-copy) se
    # o resultado for contíguo; aqui forçamos cópia via ``ascontiguousarray``
    # para garantir layout C contíguo necessário pelo broadcast/reshape final.
    H_perm = np.ascontiguousarray(np.transpose(H, (0, 1, 2, 4, 3, 5)))
    H_flat = H_perm.reshape(total_records, 9)

    # ── col0: model_id ────────────────────────────────────────────────────
    model_ids = np.arange(model_id_start, model_id_start + n_models, dtype=np.int32)
    buf["col0"] = np.broadcast_to(
        model_ids[:, None, None, None, None],
        (n_models, nTR, nAng, nf, n_pos),
    ).ravel()

    # ── col1: z_obs (aceita (nAng,n_pos) ou (n_pos,)) ────────────────────
    z_obs = np.asarray(z_obs)
    if z_obs.ndim == 2:
        z_view = z_obs[None, None, :, None, :]  # (1,1,nAng,1,n_pos)
    else:
        z_view = z_obs[None, None, None, None, :]  # (1,1,1,1,n_pos)
    buf["col1"] = np.broadcast_to(z_view, (n_models, nTR, nAng, nf, n_pos)).ravel()

    # ── col2/col3: rho_h_at_obs / rho_v_at_obs (None → zeros) ─────────────
    if rho_h_at_obs is not None:
        buf["col2"] = np.broadcast_to(
            np.asarray(rho_h_at_obs)[:, None, :, None, :],
            (n_models, nTR, nAng, nf, n_pos),
        ).ravel()
    if rho_v_at_obs is not None:
        buf["col3"] = np.broadcast_to(
            np.asarray(rho_v_at_obs)[:, None, :, None, :],
            (n_models, nTR, nAng, nf, n_pos),
        ).ravel()

    # ── col4-21: 9 componentes complexos → (Re, Im) intercalados ─────────
    for ic in range(9):
        buf[f"col{4 + 2 * ic}"] = H_flat[:, ic].real
        buf[f"col{5 + 2 * ic}"] = H_flat[:, ic].imag

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    buf.tofile(path)


def compute_nmeds_per_angle(
    tj: float, p_med: float, angles: Sequence[float]
) -> List[int]:
    """Retorna ``nmeds[k] = ceil(tj / (p_med · cos θ_k))`` para cada ângulo.

    Args:
        tj: Comprimento total do perfil (m).
        p_med: Espaçamento entre medidas verticais (m).
        angles: Ângulos de dip em graus.

    Returns:
        Lista de inteiros — nº de medidas por ângulo (≥ 1), corrigido por
        ``1/cos θ`` (perfis inclinados têm mais medidas ao longo do poço).

    Note:
        ``cos θ`` é limitado a ``1e-6`` para evitar divisão por zero em
        poços horizontais degenerados (θ → 90°).
    """
    import math

    out: List[int] = []
    for a in angles:
        cos_d = max(1e-6, math.cos(math.radians(abs(a))))
        out.append(max(1, int(math.ceil(tj / (p_med * cos_d)))))
    return out
