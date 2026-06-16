# -*- coding: utf-8 -*-
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  apps/sim_manager/perspectives/datviewer/service.py                       ║
# ║  ---------------------------------------------------------------------    ║
# ║  Módulo      : DatViewerService — leitura de .dat/.out (somente leitura)  ║
# ║  Projeto     : Geosteering AI v2.0                                        ║
# ║  Subsistema  : SM app — perspectiva Visualizador .dat (Fatia 6h)           ║
# ║  Autor       : Daniel Leal                                                ║
# ║  Status      : Produção                                                   ║
# ║  Dependências: data.loading (load_binary_dat), numpy, pathlib             ║
# ║  ---------------------------------------------------------------------    ║
# ║  FINALIDADE                                                               ║
# ║    Lê artefatos ``.dat`` (binário 22-col Fortran-compat — o formato que o   ║
# ║    grupo "Saída" da Simulação grava via write_dat_from_tensor) + o ``.out`` ║
# ║    ASCII paralelo (metadados). NÃO importa Qt (serviço puro, testável       ║
# ║    headless). NÃO re-simula nem recalcula física — SOMENTE LEITURA. Degrada ║
# ║    gracioso: arquivo ausente/ilegível → DatLoadResult com ``error`` (nunca  ║
# ║    levanta no load).                                                       ║
# ║                                                                           ║
# ║  LAYOUT 22-COL (geosteering-physics.md §4 — IMUTÁVEL)                     ║
# ║    col0=meds/model_id · col1=zobs(m) · col2=res_h(Ω·m) · col3=res_v(Ω·m)   ║
# ║    col4-21 = Re/Im dos 9 componentes (Hxx,Hxy,Hxz,Hyx,Hyy,Hyz,Hzx,Hzy,Hzz) ║
# ║                                                                           ║
# ║  EXPORTS                                                                  ║
# ║    DatViewerService · DatLoadResult · column_labels                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
"""``DatViewerService`` — leitura de ``.dat``/``.out`` (somente leitura, Fatia 6h)."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from geosteering_ai.data.loading import load_binary_dat

logger = logging.getLogger(__name__)

__all__ = ["DatViewerService", "DatLoadResult", "column_labels"]

# Ordem canônica dos 9 componentes do tensor (linha-maior do 3×3 — §4 IMUTÁVEL).
_COMPONENTS_9 = ("Hxx", "Hxy", "Hxz", "Hyx", "Hyy", "Hyz", "Hzx", "Hzy", "Hzz")


def column_labels(n_cols: int) -> List[str]:
    """Rótulos de coluna para a tabela do viewer.

    Args:
        n_cols: nº de colunas do arquivo carregado.

    Returns:
        Para ``n_cols == 22``: a legenda física canônica (meds, zobs, res_h/v,
        Re/Im dos 9 componentes). Para outros formatos: rótulos genéricos
        ``col0..colN`` (não inventa semântica física que não se aplica).
    """
    if n_cols == 22:
        labels = ["meds", "zobs (m)", "res_h (Ω·m)", "res_v (Ω·m)"]
        for comp in _COMPONENTS_9:
            labels.extend([f"Re({comp})", f"Im({comp})"])
        return labels
    return [f"col{i}" for i in range(n_cols)]


@dataclass
class DatLoadResult:
    """Resultado (imutável) de uma tentativa de leitura de ``.dat`` (+``.out``).

    Attributes:
        data: matriz ``(n_rows, n_cols)`` float64, ou ``None`` se falhou.
        n_rows: nº de linhas (registros) lidas.
        n_cols: nº de colunas (22 no formato canônico).
        fmt: formato detectado (``"binário 22-col"`` | ``"ASCII"`` | ``""``).
        out_metadata: conteúdo do ``.out`` paralelo (str) ou ``None`` se ausente.
        dat_path: caminho absoluto do ``.dat`` lido.
        size_mb: tamanho do ``.dat`` em MB.
        error: mensagem de erro PT-BR, ou ``None`` se a leitura teve sucesso.
    """

    data: Optional[np.ndarray]
    n_rows: int
    n_cols: int
    fmt: str
    out_metadata: Optional[str]
    dat_path: str
    size_mb: float
    error: Optional[str]


class DatViewerService:
    """Lê ``.dat``/``.out`` para inspeção (somente leitura — sem Qt, sem física).

    Example:
        >>> svc = DatViewerService()
        >>> res = svc.load("/tmp/sm_output.dat")
        >>> res.n_cols, res.fmt          # doctest: +SKIP
        (22, 'binário 22-col')

    Note:
        Tenta o formato BINÁRIO 22-col primeiro (o que o SM grava); se a leitura
        binária falhar (tamanho não-múltiplo de 172 B/registro), cai para um
        parser ASCII (``np.loadtxt`` — dumps do ``tatu.x``). NUNCA levanta: erros
        viram :attr:`DatLoadResult.error`.
    """

    def load(self, dat_path: str | Path) -> DatLoadResult:
        """Carrega um ``.dat`` (+ ``.out`` paralelo) com degradação graciosa.

        Args:
            dat_path: caminho do arquivo ``.dat``.

        Returns:
            Um :class:`DatLoadResult`. Em falha, ``data is None`` e ``error`` traz
            a mensagem (a View exibe sem crash).
        """
        path = Path(dat_path)
        if not path.exists():
            return DatLoadResult(
                None, 0, 0, "", None, str(path), 0.0, "arquivo .dat não encontrado"
            )

        size_mb = path.stat().st_size / 1e6
        if size_mb > 500:
            logger.warning(
                "DatViewerService: arquivo grande (%.0f MB) — leitura pode demorar.",
                size_mb,
            )

        data, fmt, error = self._read_matrix(path)
        n_rows, n_cols = (data.shape[0], data.shape[1]) if data is not None else (0, 0)
        # Matriz sem registros legíveis (0 linhas/colunas) — ex.: .dat 0-byte ou só
        # whitespace — vira ERRO explícito (degradação graciosa: a View mostra o
        # aviso em vez de uma tabela vazia silenciosa que parece "sucesso").
        if data is not None and (n_rows == 0 or n_cols == 0):
            return DatLoadResult(
                None,
                0,
                0,
                "",
                self._read_out_sibling(path),
                str(path),
                size_mb,
                "arquivo .dat sem registros legíveis (vazio?)",
            )
        out_metadata = self._read_out_sibling(path)
        return DatLoadResult(
            data, n_rows, n_cols, fmt, out_metadata, str(path), size_mb, error
        )

    @staticmethod
    def _looks_binary(path: Path) -> bool:
        """Heurística de conteúdo: o ``.dat`` parece BINÁRIO (vs ASCII numérico)?

        Decide a ORDEM das tentativas de parsing (não a correção — há fallback
        cruzado). Resolve a colisão "tamanho múltiplo de 172 B" que a detecção só
        por tamanho de :func:`load_binary_dat` sofre: um ``.dat`` ASCII cujo tamanho
        casualmente fosse múltiplo de 172 era lido como binário 22-col e exibia LIXO
        silencioso. O int32+float64 do binário quase sempre tem bytes NUL/não-imprimíveis
        no início; ASCII numérico (dígitos/sinais/espaços) não.

        Returns:
            ``True`` se o início do arquivo aparenta ser binário (tenta binário 1º).
        """
        try:
            with open(path, "rb") as fh:
                chunk = fh.read(512)
        except OSError:
            return False
        if not chunk:
            return False  # vazio → deixa o caminho ASCII/empty-guard tratar
        if b"\x00" in chunk:
            return True  # NUL é o discriminador forte (ASCII numérico não tem NUL)
        printable = sum(1 for b in chunk if b in (9, 10, 13) or 32 <= b <= 126)
        return (printable / len(chunk)) < 0.85

    @staticmethod
    def _try_binary(path: Path) -> tuple[Optional[np.ndarray], Optional[Exception]]:
        """Tenta ler como binário 22-col (formato do SM — write_dat_from_tensor)."""
        try:
            return load_binary_dat(str(path), n_columns=22), None
        except (ValueError, OSError) as exc:
            return None, exc

    @staticmethod
    def _try_ascii(path: Path) -> tuple[Optional[np.ndarray], Optional[Exception]]:
        """Tenta ler como ASCII (np.loadtxt — dumps do tatu.x / .dat ASCII).

        ``ndmin=2`` SEMPRE devolve 2-D preservando a orientação real (escalar →
        (1,1); 1 linha → (1,N); 1 coluna → (N,1)) — elimina o crash de array 0-D e
        a desambiguação errada de coluna única do antigo ``reshape(1, -1)``.
        """
        try:
            with warnings.catch_warnings():
                # np.loadtxt emite UserWarning "input contained no data" em arquivos
                # vazios/whitespace — tratamos isso como erro adiante (empty-guard),
                # então silenciamos o ruído no stderr do app em produção.
                warnings.simplefilter("ignore")
                return np.loadtxt(str(path), comments="#", ndmin=2), None
        except Exception as exc:  # noqa: BLE001 — parser ASCII pode falhar de N formas
            return None, exc

    @classmethod
    def _read_matrix(
        cls, path: Path
    ) -> tuple[Optional[np.ndarray], str, Optional[str]]:
        """Lê a matriz escolhendo a ordem por conteúdo (binário/ASCII), com fallback.

        Returns:
            ``(data, fmt, error)`` — ``data`` é ``None`` se ambos os parsers falharem.
        """
        if cls._looks_binary(path):
            data, bin_err = cls._try_binary(path)
            if data is not None:
                return data, "binário 22-col", None
            data, asc_err = cls._try_ascii(path)
            if data is not None:
                return data, "ASCII", None
        else:
            data, asc_err = cls._try_ascii(path)
            if data is not None:
                return data, "ASCII", None
            data, bin_err = cls._try_binary(path)
            if data is not None:
                return data, "binário 22-col", None
        return None, "", f"falha ao ler .dat (binário: {bin_err}; ASCII: {asc_err})"

    @staticmethod
    def _read_out_sibling(path: Path) -> Optional[str]:
        """Lê o ``.out`` paralelo (mesmo basename) como texto, se existir."""
        out_path = path.with_suffix(".out")
        if not out_path.exists():
            return None
        try:
            return out_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None  # .out ilegível — não-fatal (metadados são opcionais)
