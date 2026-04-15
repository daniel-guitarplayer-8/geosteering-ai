# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  tests/_fortran_helpers.py                                                ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Helpers de teste — interação com tatu.x (Fortran)          ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-15 (Sprint v1.5.0, PR #21)                        ║
# ║  Status      : Produção (test-only)                                      ║
# ║  Framework   : NumPy + subprocess (tatu.x)                               ║
# ║  Dependências: numpy, pathlib, canonical_models                          ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Helpers reutilizáveis para testes que comparam o simulador Python     ║
# ║    (Numba/JAX) contra o executável Fortran tatu.x, em especial em        ║
# ║    cenários com dip ≠ 0°.                                                ║
# ║                                                                           ║
# ║    Este módulo consolida lógica que estava duplicada entre:              ║
# ║      • benchmarks/bench_multi_vs_fortran.py (função _write_model_in)     ║
# ║      • tests/test_simulation_multi.py (geometria inline para dip=0°)     ║
# ║                                                                           ║
# ║  API PÚBLICA                                                              ║
# ║    write_model_in_multi() — escreve model.in para tatu.x multi-TR/angle ║
# ║    compute_n_pos_for_dip() — calcula n_pos = ceil(tj/(pmed·cos θ))      ║
# ║    compute_pz_for_dip()    — calcula passo TVD: pz = pmed·cos(θ)        ║
# ║                                                                           ║
# ║  REFERÊNCIA FORTRAN                                                       ║
# ║    PerfilaAnisoOmp.f08:670-680 (geometria T-R multi-ângulo)              ║
# ║    Para dip ≠ 0°: Fortran usa pz = p_med·cos(θ) e nmed variável por     ║
# ║    ângulo. Este helper reproduz esse cálculo em Python.                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Helpers de teste para interação com o executável Fortran ``tatu.x``.

Este módulo consolida lógica reutilizável entre ``tests/`` e ``benchmarks/``
para geração de arquivos ``model.in`` e cálculo de geometria T-R compatível
com o simulador Fortran ``PerfilaAnisoOmp.f08`` v10.0.

Uso típico em testes::

    from tests._fortran_helpers import (
        write_model_in_multi, compute_n_pos_for_dip, compute_pz_for_dip,
    )

    n_pos = write_model_in_multi(
        workdir=tmp_path / "fort",
        model_name="oklahoma_3",
        tr_list=[1.0],
        dip_list=[30.0],
        h1=29.5418, tj=120.0, p_med=0.2,
    )
    pz = compute_pz_for_dip(30.0, p_med=0.2)
    positions_z = -29.5418 + np.arange(n_pos) * pz

Note:
    Este módulo é privado (prefixo ``_``) e destinado apenas a testes.
    Não exportar via package principal.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def compute_pz_for_dip(dip_deg: float, *, p_med: float = 0.2) -> float:
    """Calcula o passo TVD em metros para um dado ângulo de dip.

    Fórmula Fortran (PerfilaAnisoOmp.f08): ``pz = p_med * cos(θ_rad)``

    Para dip = 0° (poço vertical): pz = p_med (passo integral ao longo do
    eixo vertical). Para dip > 0°: o passo TVD diminui pelo fator cos(θ),
    pois o passo é medido ao longo do poço inclinado — sua componente
    vertical é reduzida.

    Args:
        dip_deg: Ângulo de inclinação do poço em graus. Range válido:
            [0, 89]. dip=90° produz pz=0, o que é inválido fisicamente.
        p_med: Passo nominal entre medidas ao longo do poço (m).
            Default 0.2 m (resolução LWD padrão).

    Returns:
        Passo TVD (vertical) em metros: ``p_med * cos(θ)``.

    Example:
        >>> round(compute_pz_for_dip(0.0, p_med=0.2), 4)
        0.2
        >>> round(compute_pz_for_dip(30.0, p_med=0.2), 4)
        0.1732
        >>> round(compute_pz_for_dip(60.0, p_med=0.2), 4)
        0.1
    """
    if not 0.0 <= dip_deg < 89.0:
        raise ValueError(
            f"dip_deg={dip_deg}° fora do range válido [0, 89). "
            "Use ângulos < 89° para evitar pz ≈ 0."
        )
    return float(p_med * np.cos(np.deg2rad(dip_deg)))


def compute_n_pos_for_dip(
    dip_deg: float,
    *,
    tj: float = 120.0,
    p_med: float = 0.2,
) -> int:
    """Calcula o número de posições TVD necessárias para cobrir a janela ``tj``.

    Fórmula Fortran: ``n_pos = ceil(tj / (p_med * cos(θ_rad)))``.

    Para dip = 0° (poço vertical): n_pos = ceil(tj/p_med). Para dip > 0°:
    o passo TVD é menor, então são necessárias MAIS posições para cobrir
    a mesma janela vertical. Exemplo: tj=120m, p_med=0.2m:
      • dip=0°  → n_pos = 600
      • dip=30° → n_pos = 693 (~1.15× mais)
      • dip=60° → n_pos = 1200 (2× mais)

    Args:
        dip_deg: Ângulo de inclinação do poço em graus.
        tj: Tamanho da janela de investigação em metros. Default 120.0
            (compatível com ``benchmarks/bench_multi_vs_fortran.py``).
        p_med: Passo nominal entre medidas em metros. Default 0.2.

    Returns:
        Número de posições TVD (int, ≥ 1).

    Example:
        >>> compute_n_pos_for_dip(0.0, tj=120.0, p_med=0.2)
        600
        >>> compute_n_pos_for_dip(30.0, tj=120.0, p_med=0.2)
        694
    """
    pz = compute_pz_for_dip(dip_deg, p_med=p_med)
    return int(np.ceil(tj / pz))


def write_model_in_multi(
    workdir: Path,
    model_name: str,
    tr_list: Iterable[float],
    dip_list: Iterable[float],
    *,
    nf: int = 1,
    freq_hz: float = 20000.0,
    h1: float = 29.5418,
    tj: float = 120.0,
    p_med: float = 0.2,
    filename: str = "bench",
) -> int:
    """Escreve ``model.in`` para ``tatu.x`` suportando multi-TR e multi-ângulo.

    Gera o arquivo ``model.in`` com formato v10.0 esperado pelo executável
    Fortran ``PerfilaAnisoOmp.f08``/``tatu.x``. Suporta cenários:

      • Single-TR / Single-angle
      • Multi-TR × Single-angle
      • Single-TR × Multi-angle
      • Multi-TR × Multi-angle

    A função retorna ``n_pos`` (número de posições TVD usadas pelo Fortran),
    calculado via ``compute_n_pos_for_dip(min(dip_list))`` — o Fortran usa
    o menor ângulo para dimensionar os arrays de saída (maior n_pos garante
    cobertura para todos os outros ângulos).

    Args:
        workdir: Diretório onde ``model.in`` será escrito.
        model_name: Nome do modelo canônico (ex: ``"oklahoma_3"``).
            Deve estar em :func:`get_canonical_model`.
        tr_list: Lista de espaçamentos T-R em metros (nTR valores).
        dip_list: Lista de ângulos de dip em graus (nAngles valores).
        nf: Número de frequências. Default 1 (monofrequência).
        freq_hz: Frequência(s) em Hz. Se ``nf > 1``, replicado nf vezes.
            Para multi-frequência real, ajustar em revisão futura.
        h1: Profundidade do primeiro ponto-médio T-R (m). Default 29.5418
            (compatível com benchmark padrão).
        tj: Tamanho da janela de investigação (m). Default 120.0.
        p_med: Passo nominal entre medidas (m). Default 0.2.
        filename: Prefixo dos arquivos .dat gerados pelo Fortran.
            Default ``"bench"``. O Fortran gerará ``{filename}_TR{i}.dat``
            para cada par T-R.

    Returns:
        ``n_pos`` (int) — número de posições TVD esperado no output.

    Note:
        Lazy import de ``get_canonical_model`` para evitar circular imports
        em módulos de teste.

    Example:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     n_pos = write_model_in_multi(
        ...         Path(tmp), "oklahoma_3",
        ...         tr_list=[1.0], dip_list=[30.0],
        ...         tj=120.0, p_med=0.2,
        ...     )
        >>> n_pos
        693
    """
    from geosteering_ai.simulation.validation.canonical_models import (
        get_canonical_model,
    )

    tr_list = list(tr_list)
    dip_list = list(dip_list)
    if not tr_list:
        raise ValueError("tr_list vazio — forneça ao menos 1 espaçamento T-R.")
    if not dip_list:
        raise ValueError("dip_list vazio — forneça ao menos 1 ângulo de dip.")

    m = get_canonical_model(model_name)

    # ── Monta linhas do modelo.in seguindo PerfilaAnisoOmp.f08 I/O ──────────
    # Layout v10.0:
    #   nf
    #   freq_1, freq_2, ..., freq_nf
    #   ntheta
    #   dip_1, dip_2, ..., dip_ntheta
    #   h1, tj, p_med
    #   nTR
    #   L_1, L_2, ..., L_nTR
    #   filename
    #   n  (número de camadas)
    #   rho_h_1 rho_v_1
    #   rho_h_2 rho_v_2
    #   ...
    #   esp_1, esp_2, ..., esp_{n-2}  (se n > 2)
    #   modelm nmaxmodel  (default 1 1)
    rho_lines = "\n".join(f"{rh} {rv}" for rh, rv in zip(m.rho_h, m.rho_v))
    esp_lines = "\n".join(str(e) for e in m.esp) if len(m.esp) > 0 else ""
    dip_lines = "\n".join(str(d) for d in dip_list)
    tr_lines = "\n".join(str(L) for L in tr_list)

    content = (
        f"{nf}\n" + "\n".join([f"{freq_hz}"] * nf) + "\n"
        f"{len(dip_list)}\n{dip_lines}\n"
        f"{h1}\n{tj}\n{p_med}\n"
        f"{len(tr_list)}\n{tr_lines}\n"
        f"{filename}\n"
        f"{len(m.rho_h)}\n{rho_lines}\n"
    )
    if esp_lines:
        content += esp_lines + "\n"
    content += "1 1\n"

    workdir_path = Path(workdir)
    workdir_path.mkdir(parents=True, exist_ok=True)
    (workdir_path / "model.in").write_text(content)

    # n_pos dimensionado pelo MENOR ângulo (= maior n_pos para cobrir todos)
    return compute_n_pos_for_dip(min(dip_list), tj=tj, p_med=p_med)


__all__ = [
    "compute_n_pos_for_dip",
    "compute_pz_for_dip",
    "write_model_in_multi",
]
