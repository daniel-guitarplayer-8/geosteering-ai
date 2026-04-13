# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/simulation/io/model_in.py                                 ║
# ║  ---------------------------------------------------------------------    ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : Simulador Python — Exportador model.in Fortran            ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-04-11 (Sprint 2.2)                                   ║
# ║  Status      : Produção                                                   ║
# ║  Framework   : stdlib                                                     ║
# ║  Dependências: numpy                                                      ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Exporta um `SimulationConfig` + perfil geológico (rho_h, rho_v,      ║
# ║    thicknesses) para um arquivo `model.in` compatível com o simulador  ║
# ║    Fortran v10.0 (`tatu.x` / `PerfilaAnisoOmp`). O arquivo gerado       ║
# ║    pode ser lido diretamente pelo executável Fortran, permitindo       ║
# ║    validação cruzada dos dois backends.                                 ║
# ║                                                                           ║
# ║  LAYOUT DO MODEL.IN v10.0                                                 ║
# ║    Linhas (na ordem):                                                    ║
# ║      01: nf                        (int)                                ║
# ║      02..nf+1: freq[1..nf]         (float, Hz)                          ║
# ║      nf+2: ntheta                  (int)                                ║
# ║      nf+3..: theta[1..ntheta]      (float, graus)                       ║
# ║      +1: h1                        (altura 1º ponto-médio T-R)         ║
# ║      +1: tj                        (janela investigação)               ║
# ║      +1: pmed                      (passo entre medidas)               ║
# ║      +1: nTR                       (int)                                ║
# ║      +1..nTR: dTR[1..nTR]          (float, m)                           ║
# ║      +1: filename                  (str)                                ║
# ║      +1: n                         (int, nº camadas)                   ║
# ║      +n: rho_h[i] rho_v[i]         (n linhas)                           ║
# ║      +n-2: esp[1..n-2]             (espessuras internas)               ║
# ║      +1: modelm nmaxmodel          ("1 1" default)                     ║
# ║      +1: F5 use_arbitrary_freq     (0=off)                              ║
# ║      +1: F7 use_tilted             (0=off)                              ║
# ║      [F7 cond]: n_tilted, beta[], phi[] (se F7=1)                     ║
# ║      +1: F6 use_compensation       (0=off)                              ║
# ║      [F6 cond]: n_comp_pairs, pares near/far (se F6=1)                ║
# ║      +1: filter_type               (0=Werthmüller, 1=Kong, 2=Anderson)║
# ║                                                                           ║
# ║  OPT-IN                                                                   ║
# ║    Esta função só deve ser chamada quando `cfg.export_model_in=True`.  ║
# ║    Chamar com flag desativada levanta `RuntimeError` para evitar       ║
# ║    escrita acidental no sistema de arquivos.                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Exportador de arquivo `model.in` Fortran-compatível.

Este módulo implementa :func:`export_model_in`, uma função que converte
um `SimulationConfig` + perfil geológico em um arquivo de texto ASCII
idêntico ao formato esperado pelo executável Fortran `tatu.x`.

Example:
    Geração de model.in mínimo::

        >>> import numpy as np
        >>> from geosteering_ai.simulation import SimulationConfig
        >>> from geosteering_ai.simulation.io.model_in import export_model_in
        >>> cfg = SimulationConfig(
        ...     frequency_hz=20000.0,
        ...     export_model_in=True,
        ...     output_dir="/tmp/sim",
        ...     output_filename="test",
        ... )
        >>> rho_h = np.array([1.0, 10.0, 1.0, 10.0, 1.0])   # 5 camadas
        >>> rho_v = np.array([1.0, 10.0, 1.0, 10.0, 1.0])
        >>> thicknesses = np.array([1.5, 2.0, 1.0])          # n-2 = 3
        >>> path = export_model_in(cfg, rho_h, rho_v, thicknesses)

Note:
    O layout segue `PerfilaAnisoOmp.f08` v10.0 (ver header deste módulo).
    Backward-compatible com leitores v9.0 — flags opcionais F5/F6/F7/F10
    ficam no final, e leitores antigos ignoram EOF.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np

from geosteering_ai.simulation.config import SimulationConfig

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Mapeamento Hankel filter → filter_type (paridade Fortran)
# ──────────────────────────────────────────────────────────────────────────────
# Paridade com o flag `filter_type` lido em PerfilaAnisoOmp.f08:
#   0 = Werthmuller 201pt (default)
#   1 = Kong 61pt (3.3× rápido)
#   2 = Anderson 801pt (máxima precisão)
_FILTER_TYPE_MAP: dict[str, int] = {
    "werthmuller_201pt": 0,
    "kong_61pt": 1,
    "anderson_801pt": 2,
}

# Defaults de geometria quando o SimulationConfig não especifica
_DEFAULT_H1 = 29.5418  # altura típica do 1º ponto-médio T-R (m)
_DEFAULT_TJ = 67.8536  # janela de investigação (m)
_DEFAULT_PMED = 0.2  # passo entre medidas (m)
_DEFAULT_THETA_DEG = 0.0  # ângulo de inclinação padrão
_DEFAULT_MODELM = 1  # índice do modelo atual (1-based Fortran)


def export_model_in(
    cfg: SimulationConfig,
    rho_h: np.ndarray,
    rho_v: np.ndarray,
    thicknesses: np.ndarray,
    output_path: Union[Path, str, None] = None,
    *,
    h1: float = _DEFAULT_H1,
    tj: float = _DEFAULT_TJ,
    pmed: float = _DEFAULT_PMED,
    theta_deg: float = _DEFAULT_THETA_DEG,
    modelm: int = _DEFAULT_MODELM,
    nmaxmodel: int = 1,
) -> Path:
    """Escreve um arquivo `model.in` Fortran-compatível.

    Gera um arquivo de texto ASCII que pode ser lido pelo executável
    Fortran `tatu.x` (`PerfilaAnisoOmp`). O layout replica o formato
    v10.0 bit-a-bit, incluindo flags opcionais F5/F6/F7 quando
    ativadas em `cfg`.

    Args:
        cfg: Instância validada de `SimulationConfig`. Deve ter
            `export_model_in=True`, senão `RuntimeError` é levantado
            (fail-fast contra escrita acidental).
        rho_h: Array 1D `(n_layers,)` float64 com resistividades
            horizontais de cada camada em Ω·m.
        rho_v: Array 1D `(n_layers,)` float64 com resistividades
            verticais de cada camada em Ω·m. Deve ter mesmo shape
            que `rho_h`.
        thicknesses: Array 1D `(n_layers - 2,)` float64 com espessuras
            das camadas internas em metros. As camadas 0 e `n-1` são
            semi-espaços infinitos e não entram aqui.
        output_path: Caminho do arquivo de saída. Se `None`, usa
            `{cfg.output_dir}/{cfg.output_filename}.model.in`. Se
            `output_path` não existir, o diretório pai é criado.
        h1: Altura do 1º ponto-médio T-R acima da 1ª interface em
            metros. Default: 29.5418 (paridade com fifthBuildTIVModels).
        tj: Tamanho da janela de investigação em metros. Default:
            67.8536.
        pmed: Passo entre as medidas em metros. Default: 0.2.
        theta_deg: Ângulo de inclinação em graus. Default: 0.0.
        modelm: Índice 1-based do modelo atual no lote. Default: 1.
        nmaxmodel: Número máximo de modelos no lote. Default: 1.

    Returns:
        `pathlib.Path` absoluto do arquivo criado.

    Raises:
        RuntimeError: Se `cfg.export_model_in=False` (proteção contra
            chamadas acidentais).
        ValueError: Se shapes de `rho_h`/`rho_v`/`thicknesses` forem
            incompatíveis.

    Example:
        Modelo 7 camadas para validação contra Fortran::

            >>> cfg = SimulationConfig(
            ...     export_model_in=True,
            ...     output_dir="/tmp/sim",
            ...     output_filename="validacao",
            ... )
            >>> rho_h = np.array([1.0, 80.0, 1.0, 10.0, 1.0, 0.3, 1.0])
            >>> rho_v = np.array([10.0, 80.0, 10.0, 10.0, 10.0, 0.3, 10.0])
            >>> thicknesses = np.array([1.52, 2.35, 2.10, 1.88, 0.92])
            >>> path = export_model_in(cfg, rho_h, rho_v, thicknesses)
            >>> path.name
            'validacao.model.in'
    """
    # ── Proteção fail-fast ─────────────────────────────────────────
    if not cfg.export_model_in:
        raise RuntimeError(
            "export_model_in requer cfg.export_model_in=True. "
            "Use `dataclasses.replace(cfg, export_model_in=True)` para "
            "ativar opt-in."
        )

    # ── Validação de shapes ────────────────────────────────────────
    rho_h = np.asarray(rho_h, dtype=np.float64)
    rho_v = np.asarray(rho_v, dtype=np.float64)
    thicknesses = np.asarray(thicknesses, dtype=np.float64)

    if rho_h.ndim != 1 or rho_v.ndim != 1:
        raise ValueError(
            f"rho_h/rho_v devem ser 1D, recebido shapes " f"{rho_h.shape}/{rho_v.shape}."
        )
    if rho_h.shape != rho_v.shape:
        raise ValueError(f"rho_h.shape={rho_h.shape} != rho_v.shape={rho_v.shape}")
    n_layers = rho_h.shape[0]
    if n_layers < 2:
        raise ValueError(f"n_layers={n_layers} deve ser >= 2 (topo + semi-espaço).")
    expected_thicknesses = n_layers - 2
    if thicknesses.shape[0] != expected_thicknesses:
        raise ValueError(
            f"thicknesses tem shape {thicknesses.shape}, esperado "
            f"({expected_thicknesses},) para n_layers={n_layers}."
        )

    # ── Resolver output path ───────────────────────────────────────
    if output_path is None:
        output_path = Path(cfg.output_dir) / f"{cfg.output_filename}.model.in"
    else:
        output_path = Path(output_path)

    # Cria diretório pai se não existir
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Construção das frequências ─────────────────────────────────
    if cfg.frequencies_hz is not None and len(cfg.frequencies_hz) > 0:
        freqs = list(cfg.frequencies_hz)
    else:
        freqs = [cfg.frequency_hz]
    nf = len(freqs)

    # ── Construção dos espaçamentos T-R ────────────────────────────
    if cfg.tr_spacings_m is not None and len(cfg.tr_spacings_m) > 0:
        dTRs = list(cfg.tr_spacings_m)
    else:
        dTRs = [cfg.tr_spacing_m]
    n_tr = len(dTRs)

    # ── Filter type ────────────────────────────────────────────────
    filter_type = _FILTER_TYPE_MAP.get(cfg.hankel_filter, 0)

    # ── Escrita do arquivo ─────────────────────────────────────────
    # Usamos write() (modo texto, encoding utf-8) em vez de np.savetxt
    # porque precisamos dos comentários `!` à direita (paridade Fortran).
    lines: list[str] = []

    # Bloco 1: frequências
    lines.append(f"{nf:<17d} !número de frequências")
    for i, f in enumerate(freqs, start=1):
        lines.append(f"{f:<17.6f} !frequência {i}")

    # Bloco 2: ângulos (ntheta=1 fixo por enquanto)
    lines.append("1                 !número de ângulos de inclinação")
    lines.append(f"{theta_deg:<17.6f} !ângulo 1")

    # Bloco 3: geometria
    lines.append(f"{h1:<17.6f} !altura do primeiro ponto-médio T-R")
    lines.append(f"{tj:<17.6f} !tamanho da janela de investigação")
    lines.append(f"{pmed:<17.6f} !passo entre as medidas")

    # Bloco 4: T-R
    lines.append(f"{n_tr:<17d} !número de pares T-R")
    for i, d in enumerate(dTRs, start=1):
        lines.append(f"{d:<17.6f} !distância T-R {i}")

    # Bloco 5: nome dos arquivos de saída
    lines.append(f"{cfg.output_filename:<17s} !nome dos arquivos de saída")

    # Bloco 6: camadas
    lines.append(f"{n_layers:<17d} !número de camadas")
    for i in range(n_layers):
        lines.append(
            f"{rho_h[i]:<8.4f} {rho_v[i]:<8.4f} "
            f"!resistividades hor/vert camada {i + 1}"
        )

    # Bloco 7: espessuras
    for i in range(expected_thicknesses):
        if i == 0:
            lines.append(f"{thicknesses[i]:<17.6f} !espessuras das n-2 camadas")
        else:
            lines.append(f"{thicknesses[i]:<17.6f}")

    # Bloco 8: modelm nmaxmodel
    lines.append(
        f"{modelm} {nmaxmodel}             !modelo atual e o número máximo de modelos"
    )

    # Bloco 9: flags F5/F7/F6/Filter
    # F5 (multi-frequência): ativar se nf > 1
    use_arb_freq = 1 if nf > 1 else 0
    lines.append(
        f"{use_arb_freq}                 !F5: use_arbitrary_freq (0=desabilitado)"
    )

    # F7 (antenas inclinadas)
    use_tilted = 1 if cfg.use_tilted_antennas else 0
    lines.append(
        f"{use_tilted}                 !F7: use_tilted_antennas (0=desabilitado)"
    )
    if cfg.use_tilted_antennas and cfg.tilted_configs is not None:
        n_tilted = len(cfg.tilted_configs)
        lines.append(f"{n_tilted:<17d} !n_tilted")
        betas = " ".join(f"{t[0]:.4f}" for t in cfg.tilted_configs)
        phis = " ".join(f"{t[1]:.4f}" for t in cfg.tilted_configs)
        lines.append(f"{betas:<17s} !beta_tilt (graus)")
        lines.append(f"{phis:<17s} !phi_tilt (graus)")

    # F6 (compensação midpoint)
    use_comp = 1 if cfg.use_compensation else 0
    lines.append(f"{use_comp}                 !F6: use_compensation (0=desabilitado)")
    if cfg.use_compensation and cfg.comp_pairs is not None:
        n_comp = len(cfg.comp_pairs)
        lines.append(f"{n_comp:<17d} !n_comp_pairs")
        for near_idx, far_idx in cfg.comp_pairs:
            # Fortran é 1-based, convertemos de 0-based
            lines.append(
                f"{near_idx + 1} {far_idx + 1}               !comp_pair (near, far)"
            )

    # Filtro Hankel
    filter_name_map = {
        0: "Werthmuller",
        1: "Kong",
        2: "Anderson",
    }
    fname = filter_name_map.get(filter_type, "Werthmuller")
    lines.append(f"{filter_type}                 !Filtro: {filter_type}={fname}")

    # ── Grava no arquivo ───────────────────────────────────────────
    content = "\n".join(lines) + "\n"
    output_path.write_text(content, encoding="utf-8")
    logger.debug("model.in gravado em %s (%d linhas)", output_path, len(lines))

    return output_path.resolve()


# ──────────────────────────────────────────────────────────────────────────────
# read_model_in — Parser do formato model.in v10.0 (Sprint 3.3.4+)
# ──────────────────────────────────────────────────────────────────────────────


def read_model_in(path: str | Path) -> dict:
    """Lê um arquivo ``model.in`` v10.0 e retorna dict com parâmetros.

    Parser do mesmo formato que :func:`export_model_in` grava. Permite
    roundtrip: ``export_model_in(cfg, ...) → model.in → read_model_in()
    → dict → simulate()``.

    Args:
        path: Caminho para o arquivo ``model.in``.

    Returns:
        Dict com chaves:
          - ``n_layers`` (int)
          - ``rho_h`` (np.ndarray float64)
          - ``rho_v`` (np.ndarray float64)
          - ``esp`` (np.ndarray float64)
          - ``frequencies_hz`` (list[float])
          - ``tr_spacings_m`` (list[float])
          - ``dip_angles_deg`` (list[float])
          - ``h1`` (float), ``tj`` (float), ``pmed`` (float)
          - ``filename`` (str)
          - ``filter_type`` (int)
          - ``use_arbitrary_freq`` (bool)
          - ``use_tilted`` (bool)
          - ``use_compensation`` (bool)

    Raises:
        FileNotFoundError: Se ``path`` não existe.
        ValueError: Se o formato do arquivo é inválido.

    Example:
        >>> params = read_model_in("model.in")
        >>> params["n_layers"]
        3
        >>> params["rho_h"]
        array([1., 100., 1.])
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"model.in não encontrado: {path}")

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    idx = 0

    def _next_line() -> str:
        nonlocal idx
        if idx >= len(lines):
            raise ValueError(f"model.in truncado na linha {idx + 1}")
        line = lines[idx].strip()
        idx += 1
        # Remove comentários Fortran (tudo após '!')
        if "!" in line:
            line = line[: line.index("!")].strip()
        return line

    try:
        # Frequências
        nf = int(_next_line())
        freqs = [float(_next_line()) for _ in range(nf)]

        # Ângulos
        ntheta = int(_next_line())
        dip_angles = [float(_next_line()) for _ in range(ntheta)]

        # Geometria
        h1 = float(_next_line())
        tj = float(_next_line())
        pmed = float(_next_line())

        # TR spacings
        nTR = int(_next_line())
        tr_spacings = [float(_next_line()) for _ in range(nTR)]

        # Filename
        filename = _next_line()

        # Camadas
        n = int(_next_line())
        rho_h_list = []
        rho_v_list = []
        for _ in range(n):
            parts = _next_line().split()
            rho_h_list.append(float(parts[0]))
            rho_v_list.append(float(parts[1]))
        rho_h = np.array(rho_h_list, dtype=np.float64)
        rho_v = np.array(rho_v_list, dtype=np.float64)

        # Espessuras internas
        esp_list = []
        for _ in range(max(0, n - 2)):
            esp_list.append(float(_next_line()))
        esp = np.array(esp_list, dtype=np.float64)

        # modelm nmaxmodel
        _next_line()  # "1 1" (ignorado)

        # Flags F5, F7, F6, filter_type
        use_arbitrary_freq = int(_next_line()) != 0
        use_tilted = int(_next_line()) != 0
        if use_tilted:
            # Pula linhas de tilted configs (não implementado em detail)
            n_tilted = int(_next_line())
            for _ in range(n_tilted):
                _next_line()  # beta phi pares
        use_compensation = int(_next_line()) != 0
        if use_compensation:
            n_comp = int(_next_line())
            for _ in range(n_comp):
                _next_line()  # near far pares
        filter_type = int(_next_line())

    except (ValueError, IndexError) as exc:
        raise ValueError(
            f"Formato inválido no model.in '{path}' na linha ~{idx}: {exc}"
        ) from exc

    return {
        "n_layers": n,
        "rho_h": rho_h,
        "rho_v": rho_v,
        "esp": esp,
        "frequencies_hz": freqs,
        "tr_spacings_m": tr_spacings,
        "dip_angles_deg": dip_angles,
        "h1": h1,
        "tj": tj,
        "pmed": pmed,
        "filename": filename,
        "filter_type": filter_type,
        "use_arbitrary_freq": use_arbitrary_freq,
        "use_tilted": use_tilted,
        "use_compensation": use_compensation,
    }


# ──────────────────────────────────────────────────────────────────────────────
# simulate_from_model_in — wrapper de conveniência (Sprint 3.3.4+)
# ──────────────────────────────────────────────────────────────────────────────


def simulate_from_model_in(
    path: str | Path,
    output_dir: str | Path = ".",
    export_dat: bool = True,
    export_out: bool = True,
    positions_z: np.ndarray | None = None,
    parallel: bool = True,
):
    """Lê model.in, executa simulação e opcionalmente exporta .dat/.out.

    Wrapper de conveniência que replica o fluxo do simulador Fortran:
    ``model.in → tatu.x → .dat + .out``. Usa o backend Numba (default).

    Args:
        path: Caminho para o arquivo ``model.in``.
        output_dir: Diretório de saída para .dat e .out.
        export_dat: Se True, exporta arquivo .dat binário 22-col.
        export_out: Se True, exporta arquivo .out metadata.
        positions_z: Posições de medição (m). Se None, gera a partir de
            ``h1``, ``tj``, ``pmed`` do model.in.
        parallel: Se True, usa backend paralelo (cache + prange).

    Returns:
        ``SimulationResult`` com o tensor H completo.

    Raises:
        FileNotFoundError: Se ``path`` não existe.
        ValueError: Se o formato do model.in é inválido.

    Example:
        >>> result = simulate_from_model_in("model.in", output_dir="output/")
        >>> result.H_tensor.shape
        (600, 1, 9)
    """
    from geosteering_ai.simulation import SimulationConfig, simulate
    from geosteering_ai.simulation.io.binary_dat import (
        export_binary_dat,
        export_out_metadata,
    )

    params = read_model_in(path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Constrói posições se não fornecidas
    if positions_z is None:
        h1 = params["h1"]
        tj = params["tj"]
        pmed = params["pmed"]
        n_positions = max(1, int(tj / pmed)) if pmed > 0 else 100
        positions_z = np.linspace(h1, h1 + tj, n_positions)

    # Constrói SimulationConfig
    freqs = params["frequencies_hz"]
    cfg = SimulationConfig(
        frequency_hz=freqs[0] if freqs else 20000.0,
        frequencies_hz=freqs if len(freqs) > 1 else None,
        tr_spacing_m=params["tr_spacings_m"][0] if params["tr_spacings_m"] else 1.0,
        parallel=parallel,
        hankel_filter={0: "werthmuller_201pt", 1: "kong_61pt", 2: "anderson_801pt"}.get(
            params["filter_type"], "werthmuller_201pt"
        ),
    )

    # Executa simulação
    dip_deg = params["dip_angles_deg"][0] if params["dip_angles_deg"] else 0.0
    result = simulate(
        rho_h=params["rho_h"],
        rho_v=params["rho_v"],
        esp=params["esp"],
        positions_z=positions_z,
        dip_deg=dip_deg,
        cfg=cfg,
    )

    # Exporta .dat/.out se solicitado
    if export_dat or export_out:
        cfg_export = SimulationConfig(
            export_model_in=False,
            export_binary_dat=export_dat,
            output_dir=str(output_dir),
            output_filename=params.get("filename", "simulation"),
            frequency_hz=cfg.frequency_hz,
            tr_spacing_m=cfg.tr_spacing_m,
            parallel=parallel,
        )
        if export_dat:
            export_binary_dat(
                cfg_export,
                result.H_tensor[np.newaxis, ...],  # (1, n_pos, nf, 9)
                result.z_obs,
                result.rho_h_at_obs,
                result.rho_v_at_obs,
            )
            logger.info("Arquivo .dat exportado em %s", output_dir)
        if export_out:
            export_out_metadata(
                cfg_export,
                n_models=1,
                angulos_deg=np.array(params.get("dip_angles_deg", [0.0])),
                freqs_hz=np.array(params.get("frequencies_hz", [20000.0])),
                nmeds_per_theta=np.array([result.H_tensor.shape[0]]),
            )
            logger.info("Arquivo .out exportado em %s", output_dir)

    return result


__all__ = ["export_model_in", "read_model_in", "simulate_from_model_in"]
