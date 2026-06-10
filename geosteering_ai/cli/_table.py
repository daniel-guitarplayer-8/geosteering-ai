# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  geosteering_ai/cli/_table.py                                            ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : Renderizador de tabela ASCII chave-valor para a CLI        ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : CLI MVP (Sprint v2.53 — backend JAX + tabela de resultado) ║
# ║  Versão      : v2.53                                                      ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Criação     : 2026-06-02                                                 ║
# ║  Status      : Produção — MVP                                             ║
# ║  Framework   : Python stdlib pura (sem tabulate/rich — zero dep nova)     ║
# ║  Dependências: nenhuma (apenas ``os`` para detectar fallback ASCII)       ║
# ║  Padrão      : função pura (entrada = linhas; saída = string)             ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Renderiza os resultados de ``simulate``/``benchmark`` numa tabela      ║
# ║    chave-valor com bordas Unicode box-drawing (fallback ASCII via env     ║
# ║    ``GEOSTEERING_ASCII_TABLE``). Auto-dimensiona as colunas e trunca      ║
# ║    valores largos com ``…``. 100% stdlib — não adiciona dependência.      ║
# ║                                                                           ║
# ║  LAYOUT                                                                   ║
# ║    ┌──────────────────────────────────────────┐                          ║
# ║    │  TÍTULO                                   │   ← título (full-width)   ║
# ║    ├───────────────────┬──────────────────────┤                          ║
# ║    │ chave             │ valor                │   ← linhas k/v            ║
# ║    └───────────────────┴──────────────────────┘                          ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    render_kv_table : (título, linhas) → string da tabela                  ║
# ║    build_result_rows: (stats, hw) → linhas (chave, valor) formatadas      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""Renderizador de tabela ASCII chave-valor da CLI (Sprint v2.53).

Fornece :func:`render_kv_table` (puro, stdlib) e :func:`build_result_rows`
(monta as linhas de resultado de uma simulação/benchmark a partir de um dict
de estatísticas + um dict de hardware). Usado pelos subcomandos ``simulate``
e ``benchmark`` para apresentar throughput, tempo, paralelismo, hardware e
checagem de finitude (NaN/Inf) num quadro legível no terminal.
"""

from __future__ import annotations

import os
from typing import List, Mapping, Optional, Sequence, Tuple

__all__ = ["render_kv_table", "build_result_rows"]

# ──────────────────────────────────────────────────────────────────────────────
# Conjuntos de caracteres de borda — Unicode (default) e ASCII (fallback)
# ──────────────────────────────────────────────────────────────────────────────
# A tabela usa box-drawing Unicode por padrão (consistente com os mega-headers
# D1 do projeto). Ambientes sem suporte UTF-8 (alguns CI, terminais legados,
# pipes para arquivos latin-1) podem forçar ASCII via ``GEOSTEERING_ASCII_TABLE``.
_BORDERS_UNICODE = {
    "tl": "┌",
    "tr": "┐",
    "bl": "└",
    "br": "┘",
    "h": "─",
    "v": "│",
    "ml": "├",
    "mr": "┤",
    "tt": "┬",
    "bt": "┴",
    "cross": "┼",
    "ell": "…",
}
_BORDERS_ASCII = {
    "tl": "+",
    "tr": "+",
    "bl": "+",
    "br": "+",
    "h": "-",
    "v": "|",
    "ml": "+",
    "mr": "+",
    "tt": "+",
    "bt": "+",
    "cross": "+",
    "ell": "...",
}


def _borders() -> dict:
    """Seleciona o conjunto de bordas (ASCII se ``GEOSTEERING_ASCII_TABLE`` set).

    Returns:
        dict: mapa de caracteres de borda (Unicode por default; ASCII se a
            env var ``GEOSTEERING_ASCII_TABLE`` estiver definida como truthy).
    """
    if os.environ.get("GEOSTEERING_ASCII_TABLE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return _BORDERS_ASCII
    return _BORDERS_UNICODE


def _truncate(text: str, width: int, ell: str) -> str:
    """Trunca ``text`` para caber em ``width`` colunas, anexando ``ell``.

    Args:
        text: Texto original.
        width: Largura máxima (nº de caracteres).
        ell: Marca de truncamento (``…`` Unicode ou ``...`` ASCII).

    Returns:
        str: ``text`` intacto se ``len(text) <= width``; senão, recortado e
            terminado em ``ell`` respeitando ``width``.
    """
    if len(text) <= width:
        return text
    keep = max(0, width - len(ell))
    return text[:keep] + ell


def render_kv_table(
    title: str,
    rows: Sequence[Tuple[str, str]],
    *,
    max_value_width: int = 60,
) -> str:
    """Renderiza uma tabela chave-valor com bordas box-drawing.

    Estrutura da saída:
      ┌──────────────────────────────────────────┐
      │  {title}                                  │
      ├───────────────────┬──────────────────────┤
      │ {chave}           │ {valor}              │
      │  ...              │  ...                 │
      └───────────────────┴──────────────────────┘

    Args:
        title: Título exibido na faixa superior (full-width). Truncado se
            exceder a largura total da tabela.
        rows: Sequência de pares ``(chave, valor)``. Ambos convertidos a str.
            Lista vazia retorna apenas a faixa de título.
        max_value_width: Largura máxima da coluna de valores (default 60).
            Valores maiores são truncados com ``…``. A coluna de chaves
            auto-dimensiona pela maior chave.

    Returns:
        str: A tabela completa (múltiplas linhas separadas por ``\\n``,
            sem newline final).

    Raises:
        Não levanta — entradas vazias/None são coeridas a str.

    Note:
        Função PURA (sem efeito colateral, exceto leitura da env var de
        charset). O caller é responsável por ``print`` (D9 — exceção stdout
        de CLI). Ref: :func:`build_result_rows` monta ``rows``.

    Example:
        >>> print(render_kv_table("RESULTADO", [("backend", "numba")]))
        ┌────────────────────┐
        │  RESULTADO         │
        ├──────────┬─────────┤
        │ backend  │ numba   │
        └──────────┴─────────┘
    """
    b = _borders()
    ell = b["ell"]

    # Normaliza pares para str + trunca valores largos.
    norm: List[Tuple[str, str]] = [
        (str(k), _truncate(str(v), max_value_width, ell)) for k, v in rows
    ]

    # ── Auto-dimensiona colunas ──────────────────────────────────────────
    # key_w = maior chave; val_w = maior valor (já truncado). Mínimos de 1
    # evitam larguras zero quando rows está vazio.
    key_w = max((len(k) for k, _ in norm), default=1)
    val_w = max((len(v) for _, v in norm), default=1)
    key_w = max(key_w, 1)
    val_w = max(val_w, 1)

    # Largura interna total entre as bordas externas (para a faixa de título):
    #   ` {key:<key_w} ` + `│` + ` {val:<val_w} ` = (key_w+2) + 1 + (val_w+2)
    inner_w = key_w + val_w + 5

    # Título pode forçar a tabela a alargar; ajusta val_w se o título for maior.
    title_str = str(title)
    if len(title_str) + 4 > inner_w:  # +4 = 2 de padding cada lado mínimo
        val_w = len(title_str) + 4 - (key_w + 5)
        val_w = max(val_w, 1)
        inner_w = key_w + val_w + 5
    title_str = _truncate(title_str, inner_w - 2, ell)

    h, v = b["h"], b["v"]
    left_seg = h * (key_w + 2)
    right_seg = h * (val_w + 2)

    lines: List[str] = []
    # Topo (faixa de título — sem divisão de coluna)
    lines.append(b["tl"] + h * inner_w + b["tr"])
    lines.append(f"{v} {title_str:<{inner_w - 2}} {v}")
    # Separador que introduz a divisão de colunas
    lines.append(b["ml"] + left_seg + b["tt"] + right_seg + b["mr"])
    # Linhas chave-valor
    for k, val in norm:
        lines.append(f"{v} {k:<{key_w}} {v} {val:<{val_w}} {v}")
    # Base
    lines.append(b["bl"] + left_seg + b["bt"] + right_seg + b["br"])
    return "\n".join(lines)


def _fmt_int(n: Optional[int]) -> str:
    """Formata inteiro com separador de milhar (``—`` se None)."""
    return "—" if n is None else f"{int(n):,}"


def _fmt_gb(x: Optional[float]) -> str:
    """Formata gibibytes com 1 casa (``desconhecido`` se None)."""
    return "desconhecido" if x is None else f"{float(x):.1f} GB"


def build_result_rows(
    stats: Mapping[str, object],
    hw: Mapping[str, object],
) -> List[Tuple[str, str]]:
    """Monta as linhas ``(chave, valor)`` de resultado a partir de stats + hw.

    Combina as estatísticas de execução (``stats``) e o inventário de
    hardware (``hw``, de :func:`geosteering_ai.cli._hwinfo.collect_hardware_info`)
    numa lista de pares formatados para :func:`render_kv_table`. Linhas
    específicas do JAX (estratégia, grupos de geometria, GPU) só aparecem
    quando ``stats['backend'] == 'jax'``.

    Args:
        stats: Mapa com (chaves esperadas, todas opcionais salvo backend):
            ``backend`` (str), ``device`` (str), ``throughput_mod_h`` (float),
            ``throughput_std`` (float|None), ``elapsed_s`` (float),
            ``repeat`` (int|None), ``n_models`` (int), ``n_pos`` (int),
            ``n_freqs``/``n_dips``/``n_trs`` (int), ``workers``/``threads``
            (int|None), ``jax_strategy`` (str|None),
            ``n_geometry_groups`` (int|None), ``dtype`` (str),
            ``nan_count``/``inf_count`` (int), ``all_finite`` (bool).
        hw: Mapa de hardware: ``cpu_model``, ``cpu_physical``, ``cpu_logical``,
            ``ram_gb``, ``numba_threads``, ``gpu_name``, ``gpu_vram_gb``,
            ``jax_devices``.

    Returns:
        Lista de pares ``(chave, valor)`` em str, na ordem de exibição.

    Note:
        Tolerante a chaves ausentes (usa ``.get`` com defaults). O caller
        passa o dict de stats que tiver — campos None viram ``—``/``desconhecido``.

    Example:
        >>> rows = build_result_rows(
        ...     {"backend": "numba", "throughput_mod_h": 234665.0},
        ...     {"cpu_model": "Threadripper"},
        ... )
        >>> rows[0]
        ('backend', 'numba')
    """
    backend = str(stats.get("backend", "—"))
    is_jax = backend == "jax"

    n_freqs = stats.get("n_freqs")
    n_dips = stats.get("n_dips")
    n_trs = stats.get("n_trs")
    n_configs: int | None = None
    if isinstance(n_freqs, int) and isinstance(n_dips, int) and isinstance(n_trs, int):
        n_configs = n_freqs * n_dips * n_trs

    rows: List[Tuple[str, str]] = []
    rows.append(("backend", backend))
    rows.append(("device", str(stats.get("device", "—"))))
    # Motivo do fallback (e.g. jax→numba por geometria não-agrupável) — só
    # aparece quando o backend efetivo difere do solicitado (honestidade).
    reason = stats.get("reason")
    if reason:
        rows.append(("motivo (fallback)", str(reason)))

    # ── Performance ──────────────────────────────────────────────────────
    thr = stats.get("throughput_mod_h")
    thr_std = stats.get("throughput_std")
    if isinstance(thr, (int, float)):
        thr_txt = f"{thr:,.0f} mod/h"
        if isinstance(thr_std, (int, float)) and thr_std > 0:
            thr_txt += f" (±{thr_std:,.0f})"
        rows.append(("throughput", thr_txt))
    elapsed = stats.get("elapsed_s")
    if isinstance(elapsed, (int, float)):
        rows.append(("tempo de execução (hot)", f"{elapsed:.3f} s"))
    repeat = stats.get("repeat")
    if isinstance(repeat, int) and repeat > 1:
        rows.append(("repetições (hot)", f"{repeat} (mediana)"))
    # v2.56 — transparência: warmup + total explicam o gap vs `time real`.
    warmup_s = stats.get("warmup_s")
    if isinstance(warmup_s, (int, float)):
        rows.append(("tempo de warmup", f"{warmup_s:.2f} s"))
    total_s = stats.get("total_s")
    if isinstance(total_s, (int, float)):
        rows.append(("tempo total (handler)", f"{total_s:.2f} s"))

    # ── Carga de trabalho ────────────────────────────────────────────────
    rows.append(("n_models", _fmt_int(stats.get("n_models"))))  # type: ignore[arg-type]
    rows.append(("n_pos", _fmt_int(stats.get("n_pos"))))  # type: ignore[arg-type]
    if n_configs is not None:
        rows.append(
            (
                "n_configs (f×dip×TR)",
                f"{n_configs:,} ({n_freqs}×{n_dips}×{n_trs})",
            )
        )
    rows.append(("dtype", str(stats.get("dtype", "—"))))

    # ── Paralelismo (numba) ou estratégia (jax) ──────────────────────────
    if is_jax:
        rows.append(("jax_strategy", str(stats.get("jax_strategy", "bucketed"))))
        n_grp = stats.get("n_geometry_groups")
        if n_grp is not None:
            rows.append(("n_geometry_groups", _fmt_int(n_grp)))  # type: ignore[arg-type]
    else:
        workers = stats.get("workers")
        threads = stats.get("threads")
        rows.append(
            (
                "workers × threads",
                f"{workers if workers else 'auto'} × {threads if threads else 'auto'}",
            )
        )

    # ── Hardware ─────────────────────────────────────────────────────────
    rows.append(("CPU", str(hw.get("cpu_model", "desconhecido"))))
    cpu_phys = hw.get("cpu_physical")
    cpu_log = hw.get("cpu_logical")
    rows.append(
        (
            "CPU cores (fís/lóg)",
            f"{cpu_phys if cpu_phys else '?'} / {cpu_log if cpu_log else '?'}",
        )
    )
    rows.append(("RAM", _fmt_gb(hw.get("ram_gb"))))  # type: ignore[arg-type]
    if is_jax:
        rows.append(("GPU", str(hw.get("gpu_name", "desconhecido"))))
        rows.append(("VRAM", _fmt_gb(hw.get("gpu_vram_gb"))))  # type: ignore[arg-type]

    # ── Saúde numérica (NaN / Inf / finitude) ────────────────────────────
    nan_count = stats.get("nan_count")
    inf_count = stats.get("inf_count")
    all_finite = stats.get("all_finite")
    if nan_count is not None:
        rows.append(("NaNs", _fmt_int(nan_count)))  # type: ignore[arg-type]
    if inf_count is not None:
        rows.append(("Infs", _fmt_int(inf_count)))  # type: ignore[arg-type]
    if all_finite is not None:
        rows.append(("finitude OK", "sim ✓" if all_finite else "NÃO ✗"))

    return rows
